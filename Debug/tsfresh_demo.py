import xgboost as xgb
import json
import shap
import pingouin
import pandas as pd
from DataAPI.DataFrameAPI import DataFrameAPI
import talib.abstract as ta
from technical import qtpylib
from pandas import DataFrame
import matplotlib.pyplot as pyplot
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from tsfresh.utilities.dataframe_functions import roll_time_series

from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from tsfresh import extract_features,select_features
from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE,MACD_ALGO
from Common.CTime import CTime
from Plot.PlotDriver import CPlotDriver
from Common.CEnum import BSP_TYPE

from tqdm import tqdm
import sys
import talib
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report,roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import joblib
import cupy as cp
from typing import Dict, Generic, List, Optional, TypeVar, Union
from Bi.Bi import CBi
from Seg.Seg import CSeg


def load_by_file():
    df = pd.read_feather(r'E:\freqtrade\data\data\binance\futures\BTC_USDT_USDT-15m-futures.feather')  # 读取完整数据

    df['volume_diff'] =  df['volume'] - df['volume'].shift(1)

    df['atr'] = ta.ATR(df)
    df['volatility_ratio'] = df['atr']/df['close']

    ### 动量指标 Momentum Indicators ###
    df['adx'] = ta.ADX(df)
    df['adxr'] = ta.ADXR(df)
    df['apo'] = ta.APO(df)


    df['aroonosc'] = ta.AROONOSC(df)
    df['bop'] = ta.BOP(df)
    df['cci'] = ta.CCI(df)
    df['cmo'] = ta.CMO(df)
    df['dx'] = ta.DX(df)

    # macd
    macd = ta.MACD(df)
    df['macd'] = macd['macd']
    df['macd_diff'] = df['macd'] - df['macd'].shift(1)
    df['macdsignal'] = macd['macdsignal']
    df['macdsignal_diff'] = df['macdsignal'] - df['macdsignal'].shift(1)
    df['macdhist'] = macd['macdhist']
    df['macdhist_diff'] = df['macdhist'] - df['macdhist'].shift(1)
    # macdext
    macdext = ta.MACDEXT(df)
    df['macdext_macd'] = macdext['macd']
    df['macdext_diff'] = df['macdext_macd'] - df['macdext_macd'].shift(1)
    df['macdext_macdsignal'] = macdext['macdsignal']
    df['macdext_macdsignal_diff'] = df['macdext_macd'] - df['macdext_macd'].shift(1)
    df['macdext_macdhist'] = macdext['macdhist']
    df['macdext_macdhist_diff'] = df['macdext_macd'] - df['macdext_macd'].shift(1)


    # macdfix
    macdfix = ta.MACDFIX(df)
    df['macdfix_macd'] = macdfix['macd']
    df['macdfix_macd_diff'] = df['macdfix_macd'] - df['macdfix_macd'].shift(1)
    df['macdfix_macdsignal'] = macdfix['macdsignal']
    df['macdfix_macdsignal_diff'] = df['macdfix_macdsignal'] - df['macdfix_macdsignal'].shift(1)
    df['macdfix_macdhist'] = macdfix['macdhist']
    df['macdfix_macdhist_diff'] = df['macdfix_macdhist'] - df['macdfix_macdhist'].shift(1)

    df['mfi'] = ta.MFI(df)
    df['minus_di'] = ta.MINUS_DI(df)
    df['mom'] = ta.MOM(df)
    df['plus_di'] = ta.PLUS_DI(df)
    df['roc'] = ta.ROC(df)
    df['rocp'] = ta.ROCP(df)
    df['rocr'] = ta.ROCR(df)
    df['rocr100'] = ta.ROCR100(df)
    df['rsi'] = ta.RSI(df)

    # stoch
    stoch = ta.STOCH(df)
    df['slowk'] = stoch['slowk']
    df['slowd'] = stoch['slowd']
    df['stoch_slowk_solwd_diff'] = df['slowk'] -  df['slowd']

    # stochf
    stochf = ta.STOCHF(df)
    df['stochf_fastk'] = stochf['fastk']
    df['stochf_fastd'] = stochf['fastd']
    df['stochf_fastk_fastd_diff'] = df['stochf_fastk'] -  df['stochf_fastd']


    # stochrsi
    stochrsi = ta.STOCHRSI(df)
    df['stochrsi_fastk'] = stochrsi['fastk']
    df['stochrsi_fastd'] = stochrsi['fastd']

    df['trix'] = ta.TRIX(df)
    df['ultosc'] = ta.ULTOSC(df)
    df['willr'] = ta.WILLR(df)

    ### 交易量指示器 Volume Indicators ###
    df['ad'] = ta.AD(df)
    df['ad_diff'] = ta.AD(df)
    df['adosc'] = ta.ADOSC(df)

    ### 周期指标 Cycle Indicators ###
    df['ht_dcphase'] = ta.HT_DCPHASE(df)
    # ht_phasor
    ht_phasor = ta.HT_PHASOR(df)
    df['inphase'] = ht_phasor['inphase']



    ### 价格转换 Price Transform ###
    df['avgprice'] = ta.AVGPRICE(df)
    df['medprice'] = ta.MEDPRICE(df)
    df['typprice'] = ta.TYPPRICE(df)
    df['wclprice'] = ta.WCLPRICE(df)

    ### 统计函数 Statistic Functions ###
    df['correl'] = ta.CORREL(df)
    df['linearreg'] = ta.LINEARREG(df)
    df['linearreg_angle'] = ta.LINEARREG_ANGLE(df)
    df['linearreg_intercept'] = ta.LINEARREG_INTERCEPT(df)
    df['linearreg_slope'] = ta.LINEARREG_SLOPE(df)
    df['tsf'] = ta.TSF(df)

    ### 重叠研究 Overlap Studies ###
    # 布林线
    bollinger = ta.BBANDS(df)
    df['bb_lowerband'] = bollinger['lowerband']
    df['bb_upperband'] = bollinger['upperband']
    df['bb_middleband'] = bollinger['middleband']
    # dema
    df['dema'] = ta.DEMA(df)
    # ema
    df['ema_7'] = ta.EMA(df, timeperiod=7)
    df['close_ema_7_ratio'] = df['close'] / df['ema_7']

    df['ema_25'] = ta.EMA(df, timeperiod=25)
    df['close_ema_25_ratio'] = df['close'] / df['ema_25']
    df['ema_7_25_ratio'] = df['ema_7'] / df['ema_25']

    df['ema_99'] = ta.EMA(df, timeperiod=99)
    df['close_ema_99_ratio'] = df['close'] / df['ema_99']
    df['ema_7_99_ratio'] = df['ema_7'] / df['ema_99']
    df['ema_25_99_ratio'] = df['ema_25'] / df['ema_99']

    # ht_trendline
    df['ht_trendline'] = ta.HT_TRENDLINE(df)
    # kama 考夫曼均线
    df['kama_7'] = ta.KAMA(df, timeperiod=7)
    df['close_kama_7_ratio'] = df['close'] / df['kama_7']

    df['kama_25'] = ta.KAMA(df, timeperiod=25)
    df['close_kama_25_ratio'] = df['close'] / df['kama_25']
    df['kama_7_25_ratio'] = df['kama_7'] / df['kama_25']

    df['kama_99'] = ta.KAMA(df, timeperiod=99)
    df['close_kama_99_ratio'] = df['close'] / df['kama_99']
    df['kama_7_99_ratio'] = df['kama_7'] / df['kama_99']
    df['kama_25_99_ratio'] = df['kama_25'] / df['kama_99']


    # ma
    df['ma_7'] = ta.MA(df, timeperiod=7)
    df['close_ma_7_ratio'] = df['close'] / df['ma_7']

    df['ma_25'] = ta.MA(df, timeperiod=25)
    df['close_ma_25_ratio'] = df['close'] / df['ma_25']
    df['ma_7_25_ratio'] = df['ma_7'] / df['ma_25']

    df['ma_99'] = ta.MA(df, timeperiod=99)
    df['close_ma_99_ratio'] = df['close'] / df['ma_99']
    df['ma_7_99_ratio'] = df['ma_7'] / df['ma_99']
    df['ma_25_99_ratio'] = df['ma_25'] / df['ma_99']
    # mama
    mama, fama = ta.MAMA(df['close'].values)
    df['mama'] = mama
    df['fama'] = fama
    # midpoint
    df['midpoint'] = ta.MIDPOINT(df)
    # midprice
    df['midprice'] = ta.MIDPRICE(df)
    # sar
    df['sar'] = ta.SAR(df)
    df['close_sar_ratio'] = df['close'] / df['sar']
    # sarext
    df['sarext'] = ta.SAREXT(df)
    df['close_sarext_ratio'] = df['close'] / df['sarext']
    # sma
    df['sma_7'] = ta.SMA(df, timeperiod=7)
    df['close_sma_7_ratio'] = df['close'] / df['sma_7']

    df['sma_25'] = ta.SMA(df, timeperiod=25)
    df['close_sma_25_ratio'] = df['close'] / df['sma_25']
    df['sma_7_25_ratio'] = df['sma_7'] / df['sma_25']

    df['sma_99'] = ta.SMA(df, timeperiod=99)
    df['close_sma_99_ratio'] = df['close'] / df['sma_99']
    df['sma_7_99_ratio'] = df['sma_7'] / df['sma_99']
    df['sma_25_99_ratio'] = df['sma_25'] / df['sma_99']
    # tema
    df['tema_7'] = ta.TEMA(df, timeperiod=7)
    df['close_tema_7_ratio'] = df['close'] / df['tema_7']

    df['tema_25'] = ta.TEMA(df, timeperiod=25)
    df['close_tema_25_ratio'] = df['close'] / df['tema_25']
    df['tema_7_25_ratio'] = df['tema_7'] / df['tema_25']

    df['tema_99'] = ta.TEMA(df, timeperiod=99)
    df['close_tema_99_ratio'] = df['close'] / df['tema_99']
    df['tema_7_99_ratio'] = df['tema_7'] / df['tema_99']
    df['tema_25_99_ratio'] = df['tema_25'] / df['tema_99']

    # trima
    df['trima_7'] = ta.TRIMA(df, timeperiod=7)
    df['close_trima_7_ratio'] = df['close'] / df['trima_7']

    df['trima_25'] = ta.TRIMA(df, timeperiod=25)
    df['close_trima_25_ratio'] = df['close'] / df['trima_25']
    df['trima_7_25_ratio'] = df['trima_7'] / df['trima_25']

    df['trima_99'] = ta.TRIMA(df, timeperiod=99)
    df['close_trima_99_ratio'] = df['close'] / df['trima_99']
    df['trima_7_99_ratio'] = df['trima_7'] / df['trima_99']
    df['trima_25_99_ratio'] = df['trima_25'] / df['trima_99']

    # wma
    df['wma_7'] = ta.WMA(df, timeperiod=7)
    df['close_wma_7_ratio'] = df['close'] / df['wma_7']

    df['wma_25'] = ta.WMA(df, timeperiod=25)
    df['close_wma_25_ratio'] = df['close'] / df['wma_25']
    df['wma_7_25_ratio'] = df['wma_7'] / df['wma_25']

    df['wma_99'] = ta.WMA(df, timeperiod=99)
    df['close_wma_99_ratio'] = df['close'] / df['wma_99']
    df['wma_7_99_ratio'] = df['wma_7'] / df['wma_99']
    df['wma_25_99_ratio'] = df['wma_25'] / df['wma_99']


    df = df.dropna().reset_index(drop=True)
    # stacked = df.stack(dropna=False)

    # df['date_time'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    df = df.sort_values(by='date')
    df['id'] = df.index

    # df['id'] = df.reset_index(drop=True).index + 1  # 加1是为了从1开始编号，而不是从0开始
    # df.to_parquet(r'E:\freqtrade\data\BTC_USDT_USDT-15m-futures.parquet')
    # 转换步骤
    sub_df = df[:2000]
    df_long = (
        df.reset_index(drop=True)
        .melt(
            id_vars=['id',"date"],
            var_name='kind',  # 特征名存入kind列
            value_name='value'
        )
    )
    df_long['value'] = df_long['value'].astype('float32')
    df_long.to_parquet(r'E:\freqtrade\data\BTC_USDT_USDT-15m-long-futures.parquet')
    sub_df_long = (
        sub_df.reset_index(drop=True)
        .melt(
            id_vars=['id',"date"],
            var_name='kind',  # 特征名存入kind列
            value_name='value'
        )
    )
    sub_df_long.to_parquet(r'E:\freqtrade\data\BTC_USDT_USDT-15m-sub-long-futures.parquet')
    return df

def load_extract_features(file_name):
    import pyarrow.parquet as pq

    schema = pq.read_schema(file_name)

    # 获取原始所有列名（含重复）
    col_names = schema.names

    # 2. 去重字段名：保留第一次出现的字段
    seen = set()
    unique_columns = []
    for name in col_names:
        if name not in seen:
            seen.add(name)
            unique_columns.append(name)

    # 3. 用 ParquetFile 读取这些字段（每个字段名只取一次）
    pf = pq.ParquetFile(file_name)
    table = pf.read(columns=unique_columns)

    # 4. 转为 pandas
    features_df = table.to_pandas()
    # print(t1_buy_bsp_extract_features_df.head())

    features_df = features_df.dropna(axis=1)
    features_df = features_df.loc[:,
                                     features_df.nunique() != 1]
    print(features_df.head())
    return features_df

def get_point_label(bsp_df: pd.DataFrame,bsp_academy):
    bsp_label = []
    for _, row in bsp_df.iterrows():
        label = int(row['klu_idx'] in bsp_academy)
        bsp_label.append(label)
    new_bsp_df = bsp_df.set_index('id')
    bsp_y = pd.Series(bsp_label, index=new_bsp_df.index, name='target')
    return bsp_y

def convert_long(bsp_df: pd.DataFrame):
    bsp_df = bsp_df.drop(columns=['date']).sort_values(by='klu_idx').dropna().reset_index(drop=True)
    df_long = (
        bsp_df.reset_index(drop=True)
        .melt(
            id_vars=['id', "klu_idx"],
            var_name='kind',  # 特征名存入kind列
            value_name='value'
        )
    )
    df_long['value'] = df_long['value'].astype('float32')
    return df_long

def load_point_by_file():
    t1_buy_df = pd.read_parquet(r'E:\freqtrade\tsfresh\t1_buy\t1_buy_df.parquet')
    t1_sell_df = pd.read_parquet(r'E:\freqtrade\tsfresh\t1_sell\t1_sell_df.parquet')
    t2_buy_df = pd.read_parquet(r'E:\freqtrade\tsfresh\t2_buy\t2_buy_df.parquet')
    t2_sell_df = pd.read_parquet(r'E:\freqtrade\tsfresh\t2_sell\t2_sell_df.parquet')

    t1_buy_y = pd.read_pickle(r'E:\freqtrade\tsfresh\t1_buy\t1_buy_y.pkl')
    t1_sell_y = pd.read_pickle(r'E:\freqtrade\tsfresh\t1_sell\t1_sell_y.pkl')
    t2_buy_y = pd.read_pickle(r'E:\freqtrade\tsfresh\t2_buy\t2_buy_y.pkl')
    t2_sell_y = pd.read_pickle(r'E:\freqtrade\tsfresh\t2_sell\t2_sell_y.pkl')

    t1_buy_df_long = pd.read_parquet(r'E:\freqtrade\tsfresh\t1_buy\t1_buy_df_long.parquet')
    t1_sell_df_long = pd.read_parquet(r'E:\freqtrade\tsfresh\t1_sell\t1_sell_df_long.parquet')
    t2_buy_df_long = pd.read_parquet(r'E:\freqtrade\tsfresh\t2_buy\t2_buy_df_long.parquet')
    t2_sell_df_long = pd.read_parquet(r'E:\freqtrade\tsfresh\t2_sell\t2_sell_df_long.parquet')

    return t1_buy_df,t1_sell_df,t2_buy_df,t2_sell_df,t1_buy_y,t1_sell_y,t2_buy_y,t2_sell_y,t1_buy_df_long,t1_sell_df_long,t2_buy_df_long,t2_sell_df_long
LINE_TYPE = TypeVar('LINE_TYPE', CBi, CSeg)

def fill_bi_attr(bi: LINE_TYPE,df: pd.DataFrame,index):
    df.loc[index,'bi_macd_area'] = bi.cal_macd_metric(MACD_ALGO.AREA,is_reverse=False)
    df.loc[index,'bi_macd_peak'] =bi.cal_macd_metric(MACD_ALGO.PEAK,is_reverse=False)
    df.loc[index,'bi_macd_full_area'] =bi.cal_macd_metric(MACD_ALGO.FULL_AREA,is_reverse=False)
    df.loc[index,'bi_macd_slope'] =bi.cal_macd_metric(MACD_ALGO.SLOPE,is_reverse=False)
    df.loc[index,'bi_macd_amp'] =bi.cal_macd_metric(MACD_ALGO.AMP,is_reverse=False)
    df.loc[index, 'bi_macd_amount'] = bi.cal_macd_metric(MACD_ALGO.AMOUNT,is_reverse=False)
    df.loc[index, 'bi_macd_volumn'] = bi.cal_macd_metric(MACD_ALGO.VOLUMN,is_reverse=False)
    df.loc[index, 'bi_macd_volumn_avg'] = bi.cal_macd_metric(MACD_ALGO.VOLUMN_AVG,is_reverse=False)
    df.loc[index, 'bi_macd_turnrate_avg'] = bi.cal_macd_metric(MACD_ALGO.TURNRATE_AVG,is_reverse=False)
    df.loc[index, 'bi_macd_diff'] = bi.cal_macd_metric(MACD_ALGO.DIFF,is_reverse=False)
    if bi.pre:
        pre_bi = bi.pre
        df.loc[index, 'bi_macd_area_diff'] =  df.loc[index, 'bi_macd_area'] - pre_bi.cal_macd_metric(MACD_ALGO.AREA,is_reverse=False)
        df.loc[index, 'bi_macd_peak_diff'] = df.loc[index, 'bi_macd_peak'] - pre_bi.cal_macd_metric(MACD_ALGO.PEAK,is_reverse=False)
        df.loc[index, 'bi_macd_full_area_diff'] = df.loc[index, 'bi_macd_full_area'] - pre_bi.cal_macd_metric(MACD_ALGO.FULL_AREA,is_reverse=False)
        df.loc[index, 'bi_macd_slope_diff'] = df.loc[index, 'bi_macd_slope'] - pre_bi.cal_macd_metric(MACD_ALGO.SLOPE,is_reverse=False)
        df.loc[index, 'bi_macd_amp_diff'] =df.loc[index, 'bi_macd_amp'] - pre_bi.cal_macd_metric(MACD_ALGO.AMP,is_reverse=False)
        df.loc[index, 'bi_macd_amount_diff'] =df.loc[index, 'bi_macd_amount'] - pre_bi.cal_macd_metric(MACD_ALGO.AMOUNT,is_reverse=False)
        df.loc[index, 'bi_macd_volumn_diff'] = df.loc[index, 'bi_macd_volumn'] - pre_bi.cal_macd_metric(MACD_ALGO.VOLUMN,is_reverse=False)
        df.loc[index, 'bi_macd_volumn_avg_diff'] = df.loc[index, 'bi_macd_volumn_avg'] - pre_bi.cal_macd_metric(MACD_ALGO.VOLUMN_AVG,is_reverse=False)
        df.loc[index, 'bi_macd_turnrate_avg_diff'] =df.loc[index, 'bi_macd_turnrate_avg'] - pre_bi.cal_macd_metric(MACD_ALGO.TURNRATE_AVG,is_reverse=False)

def cal_bi_momentum(bi):
    bi_last_klu_idx = bi.get_end_klu().idx
    bi_klu_idx = bi.get_begin_klu().idx
    klc = bi.begin_klc
    max_high = bi.get_begin_klu().high
    min_low = bi.get_begin_klu().low
    volume_sum = 0
    while klc and bi_last_klu_idx > bi_klu_idx:
        for klu in klc.lst:
            max_high = max(max_high, klu.high)
            min_low = min(min_low, klu.low)
            volume_sum = volume_sum + klu.volume
            bi_klu_idx = klu.idx
            if bi_klu_idx >= bi_last_klu_idx:
                break
        klc = klc.next
    return (max_high - min_low) * volume_sum
def cal_bi_angle(begin_close, end_close, length):
    return np.arctan((begin_close - end_close) / length) * 180 / np.pi

def get_point_data():
    df = load_by_file()
    code = "sz.000001"
    begin_time = "2018-01-01"
    end_time = None
    data_src = DATA_SRC.BAO_STOCK
    chan_kl_type = KL_TYPE.K_15M
    lv_list = [chan_kl_type]
    config = CChanConfig({
        "trigger_step": True,  # 打开开关！
        "bi_strict": True,
        "skip_step": 0,
        "divergence_rate": float("inf"),
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 0,
        "bs1_peak": False,
        "macd_algo": "peak",
        "bs_type": '1,2,3a,1p,2s,3b',
        "print_warning": True,
        "zs_algo": "normal",
    })

    chan = CChan(
        code=code,
        lv_list=lv_list,
        config=config,
    )
    DataFrameAPI.do_init()
    data_src = DataFrameAPI(code, k_type=chan_kl_type, begin_date=begin_time, end_date=end_time,
                            autype=AUTYPE.QFQ)  # 初始化数据源类
    bsp_dict: Dict[int, T_SAMPLE_INFO] = {}  # 存储策略产出的bsp的特征
    t1_buy_df = pd.DataFrame(columns = df.columns).astype(df.dtypes)
    t1_sell_df = pd.DataFrame(columns = df.columns).astype(df.dtypes)
    t2_buy_df = pd.DataFrame(columns = df.columns).astype(df.dtypes)
    t2_sell_df = pd.DataFrame(columns = df.columns).astype(df.dtypes)
    print(t1_buy_df.dtypes)
    index = -1

    # 跑策略，保存买卖点的特征
    for klu in data_src.get_kl_data_by_df(df):
        index = index + 1
        chan.trigger_load({chan_kl_type: [klu]})
        bsp_list = chan.get_bsp()
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]

        cur_lv_chan = chan[0]

        if last_bsp.klu.idx not in bsp_dict and cur_lv_chan[-2].idx == last_bsp.klu.klc.idx:

            if BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type and last_bsp.bi.pre:
                if last_bsp.is_buy:
                    t1_buy_df_last_idx = len(t1_buy_df)
                    t1_buy_df.loc[t1_buy_df_last_idx] = df.iloc[index]
                    t1_buy_df.loc[t1_buy_df_last_idx,'klu_idx'] = last_bsp.klu.idx
                    klu_idx = last_bsp.klu.idx
                    t1_buy_df.loc[t1_buy_df_last_idx,'bottom_strength'] = df.loc[klu_idx - index + klu_idx:index, ['low']].apply(
                        lambda x: (x.drop(klu_idx).min() - x[klu_idx]) / x[klu_idx]
                        if (x[klu_idx] < x.drop(klu_idx).min())
                        else 0
                    ).item()
                    # 成交量衰减因子
                    t1_buy_df.loc[t1_buy_df_last_idx,'vol_decay'] = df.loc[klu_idx - index + klu_idx:index, ['volume']].apply(
                        lambda x: x[klu_idx] / (x.drop(klu_idx).mean()) if (x[klu_idx] > (x.drop(klu_idx).max())) else 1
                    ).item()

                    # 底分型成交量放大因子
                    t1_buy_df.loc[t1_buy_df_last_idx,'vol_growth'] = df.loc[klu_idx - index + klu_idx:index, ['volume']].apply(
                        lambda x: x[klu_idx] / (x.drop(klu_idx).mean()) if (x[klu_idx] < (x.drop(klu_idx).min())) else 1
                    ).item()
                    t1_buy_df.loc[t1_buy_df_last_idx,'fractal_decay'] = np.exp(-0.1 * (index - klu_idx))
                    t1_buy_df.loc[t1_buy_df_last_idx,'bottom_composite_strength'] = (
                            t1_buy_df.loc[t1_buy_df_last_idx,'bottom_strength'] *  t1_buy_df.loc[t1_buy_df_last_idx,'vol_growth'] *  t1_buy_df.loc[t1_buy_df_last_idx,'fractal_decay']).item()
                    bi = last_bsp.bi
                    pre_bi = last_bsp.bi.pre
                    next_bi = last_bsp.bi.next
                    t1_buy_df.loc[t1_buy_df_last_idx,'bi_angle'] = cal_bi_angle(bi.get_begin_klu().close, bi.get_end_klu().close,
                                                        bi.get_end_klu().idx - bi.get_begin_klu().idx)
                    if pre_bi:
                        t1_buy_df.loc[t1_buy_df_last_idx,'pre_bi_angle'] = cal_bi_angle(pre_bi.get_begin_klu().close,
                                                                pre_bi.get_end_klu().close,
                                                                pre_bi.get_end_klu().idx - pre_bi.get_begin_klu().idx)
                        t1_buy_df.loc[t1_buy_df_last_idx,'pre_now_angle_diff'] = t1_buy_df.loc[t1_buy_df_last_idx,'bi_angle'] -  t1_buy_df.loc[t1_buy_df_last_idx,'pre_bi_angle']

                    bi_momentum_pre = cal_bi_momentum(pre_bi)
                    bi_momentum_curr = cal_bi_momentum(bi)
                    t1_buy_df.loc[t1_buy_df_last_idx,'bi_momentum_pre'] = bi_momentum_pre
                    t1_buy_df.loc[t1_buy_df_last_idx,'bi_momentum_curr'] = bi_momentum_curr
                    t1_buy_df.loc[t1_buy_df_last_idx,'bi_momentum_decay_ratio'] = bi_momentum_curr / bi_momentum_pre
                    t1_buy_df.loc[t1_buy_df_last_idx,'bi_volume_decay'] = bi.get_end_klu().volume / bi.get_begin_klu().volume
                    fill_bi_attr(last_bsp.bi,t1_buy_df,t1_buy_df_last_idx)
                else:
                    t1_sell_df_last_idx = len(t1_sell_df)
                    t1_sell_df.loc[t1_sell_df_last_idx] = df.iloc[index]
                    t1_sell_df.loc[t1_sell_df_last_idx, 'klu_idx'] = last_bsp.klu.idx
                    fill_bi_attr(last_bsp.bi, t1_sell_df, t1_sell_df_last_idx)

            if BSP_TYPE.T2 in last_bsp.type or BSP_TYPE.T2S in last_bsp.type:
                if last_bsp.is_buy:
                    t2_buy_df_last_idx = len(t2_buy_df)
                    t2_buy_df.loc[t2_buy_df_last_idx] = df.iloc[index]
                    t2_buy_df.loc[t2_buy_df_last_idx, 'klu_idx'] = last_bsp.klu.idx
                    fill_bi_attr(last_bsp.bi, t2_buy_df, t2_buy_df_last_idx)
                else:
                    t2_sell_df_last_idx = len(t2_sell_df)
                    t2_sell_df.loc[t2_sell_df_last_idx] = df.iloc[index]
                    t2_sell_df.loc[t2_sell_df_last_idx, 'klu_idx'] = last_bsp.klu.idx
                    fill_bi_attr(last_bsp.bi, t2_sell_df, t2_sell_df_last_idx)

    t1_buy_df = t1_buy_df.astype(df.dtypes)
    t1_sell_df = t1_sell_df.astype(df.dtypes)
    t2_buy_df = t2_buy_df.astype(df.dtypes)
    t2_sell_df = t2_sell_df.astype(df.dtypes)

    bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp()]
    t1_buy_y = get_point_label(t1_buy_df, bsp_academy)
    t1_sell_y = get_point_label(t1_sell_df, bsp_academy)
    t2_buy_y = get_point_label(t2_buy_df, bsp_academy)
    t2_sell_y = get_point_label(t2_sell_df, bsp_academy)

    t1_buy_df.to_parquet(r'E:\freqtrade\tsfresh\t1_buy\t1_buy_df.parquet')
    t1_sell_df.to_parquet(r'E:\freqtrade\tsfresh\t1_sell\t1_sell_df.parquet')
    t2_buy_df.to_parquet(r'E:\freqtrade\tsfresh\t2_buy\t2_buy_df.parquet')
    t2_sell_df.to_parquet(r'E:\freqtrade\tsfresh\t2_sell\t2_sell_df.parquet')

    t1_buy_y.to_pickle(r'E:\freqtrade\tsfresh\t1_buy\t1_buy_y.pkl')
    t1_sell_y.to_pickle(r'E:\freqtrade\tsfresh\t1_sell\t1_sell_y.pkl')
    t2_buy_y.to_pickle(r'E:\freqtrade\tsfresh\t2_buy\t2_buy_y.pkl')
    t2_sell_y.to_pickle(r'E:\freqtrade\tsfresh\t2_sell\t2_sell_y.pkl')

    t1_buy_df_long = convert_long(t1_buy_df)
    t1_buy_df_long.to_parquet(r'E:\freqtrade\tsfresh\t1_buy\t1_buy_df_long.parquet')

    t1_sell_df_long = convert_long(t1_sell_df)
    t1_sell_df_long.to_parquet(r'E:\freqtrade\tsfresh\t1_sell\t1_sell_df_long.parquet')

    t2_buy_df_long = convert_long(t2_buy_df)
    t2_buy_df_long.to_parquet(r'E:\freqtrade\tsfresh\t2_buy\t2_buy_df_long.parquet')

    t2_sell_df_long = convert_long(t2_sell_df)
    t2_sell_df_long.to_parquet(r'E:\freqtrade\tsfresh\t2_sell\t2_sell_df_long.parquet')
    return t1_buy_df,t1_sell_df,t2_buy_df,t2_sell_df,t1_buy_y,t1_sell_y,t2_buy_y,t2_sell_y,t1_buy_df_long,t1_sell_df_long,t2_buy_df_long,t2_sell_df_long


def get_select_features(features_df,extract_features_df,bsp_y,select_features_file,select_features_names_file):
    features = pd.merge(extract_features_df,
                        features_df.set_index('id', drop=True).drop(columns=['date']),
                        how='left', left_index=True, right_index=True)
    features.drop(columns=['klu_idx'], inplace=True)
    selected = select_features(features, bsp_y,fdr_level=0.02)
    selected_feature_names = selected.columns.tolist()
    selected.to_parquet(select_features_file)
    joblib.dump(selected_feature_names,select_features_file)
    for name in selected_feature_names:
        print(name)
    return selected,selected_feature_names

def get_extract_features(df: pd.DataFrame,file_name) -> pd.DataFrame:
    features_ddf = extract_features(
        df,
        column_id="id",
        column_sort="klu_idx",
        column_kind='kind',
        column_value='value',
        n_jobs=10,  # 使用所有worker核心
        disable_progressbar=False  # 显示进度条
    )
    features_ddf = features_ddf.dropna(axis=1, how='all')
    features_ddf = features_ddf.loc[:, features_ddf.nunique() != 1]
    print(features_ddf.head(10))
    print(features_ddf.shape)
    features_ddf.to_parquet(file_name)

def get_origin_features(file_name):
    df = pd.read_parquet(file_name)
    df = df.drop(columns=['date']).sort_values(by='klu_idx').dropna().reset_index(drop=True)
    df_long = (
        df.reset_index(drop=True)
        .melt(
            id_vars=['id', "klu_idx"],
            var_name='kind',  # 特征名存入kind列
            value_name='value'
        )
    )
    df_long['id'] = df_long['id'].astype('int32')
    df_long['klu_idx'] = df_long['klu_idx'].astype('float32')
    df_long['value'] = df_long['value'].astype('float32')
    return df_long

if __name__ == "__main__":

    # t1_buy_df_long = pd.read_parquet(r'E:\freqtrade\data\t1_buy_df_long.parquet')
    # print(t1_buy_df_long.head())

    # t1_buy_df,t1_sell_df,t2_buy_df,t2_sell_df,t1_buy_y,t1_sell_y,t2_buy_y,t2_sell_y,t1_buy_df_long,t1_sell_df_long,t2_buy_df_long,t2_sell_df_long = load_point_by_file()
    t1_buy_df,t1_sell_df,t2_buy_df,t2_sell_df,t1_buy_y,t1_sell_y,t2_buy_y,t2_sell_y,t1_buy_df_long,t1_sell_df_long,t2_buy_df_long,t2_sell_df_long = get_point_data()
    get_extract_features(t1_buy_df_long,r'E:\freqtrade\tsfresh\t1_buy\t1_buy_features_ddf.parquet')
    # get_extract_features(t1_sell_df_long,r'E:\freqtrade\tsfresh\t1_sell\t1_sell_features_ddf.parquet')
    # get_extract_features(t2_buy_df_long,r'E:\freqtrade\tsfresh\t2_buy\t2_buy_features_ddf.parquet')
    # get_extract_features(t2_sell_df_long,r'E:\freqtrade\tsfresh\t2_sell\t2_sell_features_ddf.parquet')

    t1_buy_bsp_extract_features_df = load_extract_features(r'E:\freqtrade\tsfresh\t1_buy\t1_buy_features_ddf.parquet')
    # t1_sell_bsp_extract_features_df = load_extract_features(r'E:\freqtrade\tsfresh\t1_sell\t1_sell_features_ddf.parquet')
    # t2_buy_bsp_extract_features_df = load_extract_features(r'E:\freqtrade\tsfresh\t2_buy\t2_buy_features_ddf.parquet')
    # t2_sell_bsp_extract_features_df = load_extract_features(r'E:\freqtrade\tsfresh\t2_sell\t2_sell_features_ddf.parquet')

    t1_buy_select_df,t1_buy_selected_feature_names = get_select_features(t1_buy_df,t1_buy_bsp_extract_features_df,t1_buy_y,r'E:\freqtrade\tsfresh\t1_buy\t1_buy_selected_features_ddf.parquet',r'E:\freqtrade\tsfresh\t1_buy\t1_buy_selected_features.pkl')
    # t1_sell_select_df,t1_sell_selected_feature_names = get_select_features(t1_sell_df,t1_sell_bsp_extract_features_df,t1_sell_y,r'E:\freqtrade\tsfresh\t1_sell\t1_sell_selected_features_ddf.parquet',r'E:\freqtrade\tsfresh\t1_sell\t1_sell_selected_features.pkl')
    # t2_buy_select_df,t2_buy_selected_feature_names = get_select_features(t1_buy_df,t1_buy_bsp_extract_features_df,t1_buy_y,r'E:\freqtrade\tsfresh\t2_buy\t2_buy_selected_features_ddf.parquet',r'E:\freqtrade\tsfresh\t2_buy\t2_buy_selected_features.pkl')
    # t2_sell_select_df,t2_sell_selected_feature_names = get_select_features(t1_buy_df,t1_buy_bsp_extract_features_df,t1_buy_y,r'E:\freqtrade\tsfresh\t2_sell\t2_sell_selected_features_ddf.parquet',r'E:\freqtrade\tsfresh\t2_sell\t2_sell_selected_features.pkl')

    X = t1_buy_select_df.values  # numpy 数组
    feature_names = t1_buy_select_df.columns.tolist()
    y_values = t1_buy_y.loc[t1_buy_select_df.index].values

    # 4. 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_values, test_size=0.3, random_state=42, stratify=y_values
    )

    X_train_gpu = cp.asarray(X_train)
    y_train_gpu = cp.asarray(y_train)
       # 统计
    y_train_series = pd.Series(y_train)
    y_test_series = pd.Series(y_test)
    print("训练集样本分布：")
    print(y_train_series.value_counts(), "\n")

    print("测试集样本分布：")
    print(y_test_series.value_counts())

    # 5. 定义并训练 XGBoost 模型
    model = XGBClassifier(
        tree_method='hist',
        device='cuda',
        colsample_bytree=0.8,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=3,
        subsample=0.6,
        eval_metric='aucpr', # 关注 Precision-Recall AUC
        random_state=42,
        objective='binary:logistic',
        scale_pos_weight=3  # 初始设为 2.52
    )
    # 待调参范围
    param_grid = {
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'learning_rate': [0.01, 0.05, 0.1],
        'scale_pos_weight': [1.0, 1.5, 2.0, 2.5, 3.0]
    }

    # 分层交叉验证
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1',  # 以 F1 为优化目标
        cv=cv,
        n_jobs=1,
        verbose=2
    )

    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    print("Best F1:", grid.best_score_)
    best_params = grid.best_params_
    final_model = XGBClassifier(**best_params,
                                tree_method='hist',
                                device='cuda',
                                eval_metric='aucpr',  # 关注 Precision-Recall AUC
                                random_state=42,
                                objective='binary:logistic',
                                )
    final_model.fit(X_train_gpu, y_train_gpu)
    # 6. 预测与评估
    X_test_gpu = cp.asarray(X_test)
    y_test_gpu = cp.asarray(y_test)
    y_pred = final_model.predict(X_test_gpu)
    y_proba = final_model.predict_proba(X_test_gpu)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC: {roc_auc:.4f}")

    # load_by_file()



    # df = load_by_file()
    # df['date_time'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    # ddf = dd.from_pandas(df)
    # features_ddf = dd.merge(features_ddf, ddf.set_index('date_time', drop=True), how="left")
    # features_ddf.to_parquet('extracted_features.parquet')
    # data_feat = extract_features(data_roll, column_id='id', column_sort='date')
    # # 对单独标的而言，将日期作为index
    # data_feat.index = [v[1] for v in data_feat.index]
    # data_feat = pd.merge(data_feat, data.set_index('date', drop=True).drop(columns=['code']),
    #                      how='left', left_index=True, right_index=True)
    # data_feat.to_parquet('extracted_features.parquet')

    # print(data_feat.shape)