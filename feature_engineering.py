import os
import datetime

import pandas as pd
import numpy as np

class Config:
    name_v1 = "lgb baseline"
    model_params = dict(objective="mae",
                        n_estimators=10000,
                        num_leaves=31,
                        random_state=2021,
                        importance_type="gain",
                        colsample_bytree=0.3,
                        learning_rate=0.5
                        )
    fit_params = dict(early_stopping_rounds=100, verbose=100)
    n_fold = 2
    seeds = [2021]
    target_col = ""
    debug = False

# 便利な関数
def reduce_mem_usage(
    df: pd.DataFrame,
    verbose: bool
):
    """iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.  https://qiita.com/kaggle_master-arai-san/items/d59b2fb7142ec7e270a5#reduce_mem_usage
    Args:
        df (pd.DataFrame) : 元のdataframe
        verbose (bool) : 何ギガ減ったか？を表示させるかどうか

    Returns:
        df_out (pd.DataFrame) : 軽くなったdataframe
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    dfs = []
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    dfs.append(df[col].astype(np.int8))
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    dfs.append(df[col].astype(np.int16))
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    dfs.append(df[col].astype(np.int32))
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    dfs.append(df[col].astype(np.int64))
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    dfs.append(df[col].astype(np.float16))
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    dfs.append(df[col].astype(np.float32))
                else:
                    dfs.append(df[col].astype(np.float64))
        else:
            dfs.append(df[col])

    df_out = pd.concat(dfs, axis=1)
    if verbose:
        end_mem = df_out.memory_usage().sum() / 1024**2
        num_reduction = str(100 * (start_mem - end_mem) / start_mem)
        print(
            f'Mem. usage decreased to {str(end_mem)[:3]}Mb:  {num_reduction[:2]}% reduction')
    return df_out

# agg特徴量とz-score特徴量を作成
def aggregation(
    input_df: pd.DataFrame,
    group_key: str,
    group_values: list,
    agg_methods: list
):
    """基本統計量の特徴量を作成 ref:https://github.com/pfnet-research/xfeat/blob/master/xfeat/helper.py
    Args:
        input_df (pd.DataFrame) : 元のdataframe
        group_key (str) : key
        group_values (list) : values
        agg_methods (list) : methods

    Returns:
        output_df.drop(group_key, axis=1) (pd.DataFrame) : 特徴量を追加済のdataframe
    """
    new_df = []
    for agg_method in agg_methods:
        for col in group_values:
            if callable(agg_method):
                agg_method_name = agg_method.__name__
            else:
                agg_method_name = agg_method
            new_col = f"agg_{agg_method_name}_{col}_grpby_{group_key}"
            df_agg = (input_df[[col] + [group_key]
                               ].groupby(group_key)[[col]].agg(agg_method))
            df_agg.columns = [new_col]
            new_df.append(df_agg)

    _df = pd.concat(new_df, axis=1).reset_index()
    output_df = pd.merge(input_df[[group_key]], _df, on=group_key, how="left")
    return output_df.drop(group_key, axis=1)


def get_agg_col_name(
    group_key: str,
    group_values: list,
    agg_methods: list
):
    """get_agg_and_zscore_featuresのためのaggregationで作成した列名を取得
    Args:
        group_key (str) : key
        group_values (list) : values
        agg_methods (list) : methods

    Returns:
       out_cols (list) : aggregationで作成した列名
    """
    out_cols = []
    for group_val in group_values:
        for agg_method in agg_methods:
            out_cols.append(f"agg_{agg_method}_{group_val}_grpby_{group_key}")
    return out_cols


def get_agg_and_zscore_features(
    whole_df: pd.DataFrame,
    group_key: str,
    group_values: list,
    agg_methods: list
):
    """aggregationの実行と追加でz-scoreも取得
    Args:
        whole_df (pd.DataFrame) : 元のdataframe
        group_key (str) : key
        group_values (list) : values
        agg_methods (list) : methods

    Returns:
       output_df (pd.DataFrame) : 特徴量を追加済のdataframe
    """
    # get aggregation
    output_df = aggregation(whole_df, group_key, group_values, agg_methods)

    # get z-score
    z_col_name = get_agg_col_name(group_key, group_values, ["z-score"])
    m_col_name = get_agg_col_name(group_key, group_values, ["mean"])
    s_col_name = get_agg_col_name(group_key, group_values, ["std"])

    output_df[z_col_name] = ((whole_df[group_values].values - output_df[m_col_name].values)
                             / (output_df[m_col_name].values + 1e-8))

    return output_df


def do_agg_funcs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    group_key: str,
    group_values: list,
    agg_methods: list
):
    """agg特徴量とz-score特徴量を作成
    Args:
        train (pd.DataFrame) : 元のtrain data
        test (pd.DataFrame) : 元のtest data
        group_key (str) : key
        group_values (list) : values
        agg_methods (list) : methods

    Returns:
       train_x (pd.DataFrame) : 特徴量を追加済のtrain data
       test_x (pd.DataFrame) : 特徴量を追加済のtest data
    """
    whole_df = pd.concat([train, test]).reset_index(drop=True)
    output_df = pd.DataFrame()
    funcs = [
        get_agg_and_zscore_features,
    ]

    if not funcs:
        return pd.DataFrame(), pd.DataFrame()

    for func in funcs:
        print("start", "\t", func.__name__)
        _df = func(whole_df, group_key, group_values, agg_methods)
        _df = reduce_mem_usage(_df, verbose=False)
        output_df = pd.concat([output_df, _df], axis=1)
    print("")

    train_x = output_df.iloc[:len(train)]
    test_x = output_df.iloc[len(train):].reset_index(drop=True)

    return train_x, test_x

# shift特徴量とdiff特徴量とcumsum特徴量を作成
def get_raw_features(
    input_df: pd.DataFrame,
    raw_cols: list,
    group_key: str,
    group_values_shift: list,
    shift_times: list,
    group_values_diff: list,
    diff_times: list,
    group_values_cumsum: list
):
    """元データの中で特徴量に使う列をそのまま使用
    Args:
        input_df (pd.DataFrame) : 元のdataframe
        raw_cols (list) : 元のdataframeの中でそのまま特徴量に使う列名のlist
        group_key (str) : key
        group_values_shift (list) : values of shift
        shift_times (list) : times of shift
        group_values_diff (list) : values of diff
        diff_times (list) : times of diff
        group_values_cumsum (list) : values of cumsum

    Returns:
       output_df (pd.DataFrame) : 特徴量を追加済のdataframe
    """
    output_df = input_df[raw_cols].copy()
    return output_df


def get_shift_grpby_features(
    input_df: pd.DataFrame,
    raw_cols: list,
    group_key: str,
    group_values_shift: list,
    shift_times: list,
    group_values_diff: list,
    diff_times: list,
    group_values_cumsum: list
):
    """shift特徴量作成
    Args:
        input_df (pd.DataFrame) : 元のdataframe
        raw_cols (list) : 元のdataframeの中でそのまま特徴量に使う列名のlist
        group_key (str) : key
        group_values_shift (list) : values of shift
        shift_times (list) : times of shift
        group_values_diff (list) : values of diff
        diff_times (list) : times of diff
        group_values_cumsum (list) : values of cumsum

    Returns:
       output_df (pd.DataFrame) : 特徴量を追加済のdataframe
    """
    output_df = pd.DataFrame()
    for t in shift_times:
        _df = input_df.groupby(group_key)[group_values_shift].shift(t)
        _df.columns = [
            f'shift={t}_{col}_grpby_{group_key}' for col in group_values_shift]
        output_df = pd.concat([output_df, _df], axis=1)
    return output_df


def get_diff_grpby_features(
    input_df: pd.DataFrame,
    raw_cols: list,
    group_key: str,
    group_values_shift: list,
    shift_times: list,
    group_values_diff: list,
    diff_times: list,
    group_values_cumsum: list
):
    """diff特徴量作成
    Args:
        input_df (pd.DataFrame) : 元のdataframe
        raw_cols (list) : 元のdataframeの中でそのまま特徴量に使う列名のlist
        group_key (str) : key
        group_values_shift (list) : values of shift
        shift_times (list) : times of shift
        group_values_diff (list) : values of diff
        diff_times (list) : times of diff
        group_values_cumsum (list) : values of cumsum

    Returns:
       output_df (pd.DataFrame) : 特徴量を追加済のdataframe
    """
    output_df = pd.DataFrame()
    for t in diff_times:
        _df = input_df.groupby(group_key)[group_values_diff].shift(t)
        _df.columns = [
            f'diff={t}_{col}_grpby_{group_key}' for col in group_values_diff]
        output_df = pd.concat([output_df, _df], axis=1)
    return output_df


def get_cumsum_grpby_features(
    input_df: pd.DataFrame,
    raw_cols: list,
    group_key: str,
    group_values_shift: list,
    shift_times: list,
    group_values_diff: list,
    diff_times: list,
    group_values_cumsum: list
):
    """cumsum特徴量作成
    Args:
        input_df (pd.DataFrame) : 元のdataframe
        raw_cols (list) : 元のdataframeの中でそのまま特徴量に使う列名のlist
        group_key (str) : key
        group_values_shift (list) : values of shift
        shift_times (list) : times of shift
        group_values_diff (list) : values of diff
        diff_times (list) : times of diff
        group_values_cumsum (list) : values of cumsum

    Returns:
       output_df (pd.DataFrame) : 特徴量を追加済のdataframe
    """
    output_df = pd.DataFrame()
    for group_val in group_values_cumsum:
        col_name = f"agg_cumsum_{group_val}_grpby_{group_key}"
        output_df[col_name] = input_df.groupby(group_key)[group_val].cumsum()
    return output_df


def do_shift_diff_cumsum_funcs(
    input_df: pd.DataFrame,
    raw_cols: list,
    group_key: str,
    group_values_shift: list,
    shift_times: list,
    group_values_diff: list,
    diff_times: list,
    group_values_cumsum: list
):
    """shift特徴量とdiff特徴量とcumsum特徴量を作成する関数を実行
    Args:
        input_df (pd.DataFrame) : 元のdataframe
        raw_cols (list) : 元のdataframeの中でそのまま特徴量に使う列名のlist
        group_key (str) : key
        group_values_shift (list) : values of shift
        shift_times (list) : times of shift
        group_values_diff (list) : values of diff
        diff_times (list) : times of diff
        group_values_cumsum (list) : values of cumsum

    Returns:
       output_df (pd.DataFrame) : 特徴量を追加済のdataframe
    """
    output_df = pd.DataFrame()
    funcs = [
        get_raw_features,
        get_shift_grpby_features,
        get_diff_grpby_features,
        get_cumsum_grpby_features
    ]
    for func in funcs:
        print("start", "\t", func.__name__)
        _df = func(input_df, raw_cols, group_key, group_values_shift,
                   shift_times, group_values_diff, diff_times, group_values_cumsum)
        _df = reduce_mem_usage(_df, verbose=False)
        output_df = pd.concat([output_df, _df], axis=1)
    print("")
    return output_df


# この関数で上記全て実行
def feature_engineering(
    train: pd.DataFrame, 
    test: pd.DataFrame,
    group_values_agg: list,
    agg_methods: list,
    raw_cols: list,
    group_key: str,
    group_values_shift: list,
    shift_times: list,
    group_values_diff: list,
    diff_times: list,
    group_values_cumsum: list
):
    """特徴量作成の関数をまとめて実行
    Args:
        train (pd.DataFrame) : train data
        test (pd.DataFrame) : test data
        group_values_agg (list): value of agg,
        agg_methods (list): 集計方法のlist,
        raw_cols (list) : 元のdataframeの中でそのまま特徴量に使う列名のlist
        group_key (str) : key
        group_values_shift (list) : values of shift
        shift_times (list) : times of shift
        group_values_diff (list) : values of diff
        diff_times (list) : times of diff
        group_values_cumsum (list) : values of cumsum

    Returns:
       train_x (pd.DataFrame) : 特徴量を追加済のtrain data
       train_y (pd.DataFrame) : train data の目的変数
       test_x (pd.DataFrame) : 特徴量を追加済のtest data
    """
    # agg特徴量とz-score特徴量を作成
    train_x, test_x = do_agg_funcs(
        train,
        test,
        group_key,
        group_values_agg,
        agg_methods
    )
    
    # shift特徴量とdiff特徴量とcumsum特徴量を作成
    train_x = pd.concat([
        train_x, 
        do_shift_diff_cumsum_funcs(
            train, 
            raw_cols, 
            group_key, 
            group_values_shift, 
            shift_times, 
            group_values_diff, 
            diff_times, 
            group_values_cumsum
        )], axis=1
    )
    
    test_x = pd.concat([
        test_x, 
        do_shift_diff_cumsum_funcs(
            test, 
            raw_cols, 
            group_key, 
            group_values_shift, 
            shift_times, 
            group_values_diff, 
            diff_times, 
            group_values_cumsum
        )], axis=1
    )
    
    # target
    train_y = train[Config.target_col]
    
    return train_x, train_y, test_x

def __main__:
    # 例
    feature_engineering(
        train=train, 
        test=test,
        group_values_agg=["u_in"],
        agg_methods=["mean", "std", "median", "max", "sum"],
        raw_cols=[
            "R",
            "C",
            "time_step",
            "u_in",
            "u_out"
        ],
        group_key="breath_id",
        group_values_shift=["u_in"],
        shift_times=[-1, -2, 1, 2],
        group_values_diff=["u_in"],
        diff_times=[-1, -2, 1, 2],
        group_values_cumsum=["time_step", "u_in", "u_out"]
    )