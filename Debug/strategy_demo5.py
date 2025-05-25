import json
from typing import Dict, TypedDict

import xgboost as xgb
import torch

from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Common.CTime import CTime
from Plot.PlotDriver import CPlotDriver
from chan.DataAPI.DataFrameAPI import DataFrameAPI
import pandas as pd
from pandas import DataFrame
from Common.CEnum import BSP_TYPE
import matplotlib.pyplot as pyplot
from sklearn.metrics import roc_auc_score, average_precision_score,accuracy_score, classification_report, roc_curve, auc
import copy
import numpy as np
from BuySellPoint.BS_Point import CBS_Point
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file
from imblearn.combine import SMOTEENN,SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from tsfresh import extract_features,select_features
from ZS.ZS import CZS
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class T_SAMPLE_INFO(TypedDict):
    feature: CFeatures
    is_buy: bool
    open_time: CTime


def plot(chan, plot_marker,type,is_buy):
    plot_config = {
        "plot_kline": True,
        "plot_bi": True,
        "plot_seg": True,
        "plot_zs": True,
        "plot_bsp": True,
        "plot_marker": True,
    }
    plot_para = {
        "figure": {
            "x_range": 200,
        },
        "marker": {
            "markers": plot_marker
        }
    }
    plot_driver = CPlotDriver(
        chan,
        plot_config=plot_config,
        plot_para=plot_para,
    )
    if type:
        if is_buy:
            plot_driver.save2img(type.name + "_" + "buy" + "_" + "label.png")
        else:
            plot_driver.save2img(type.name + "_" + "sell" + "_" + "label.png")
    else:
        plot_driver.save2img("label.png")

def new_stragety_feature(last_klu, dataframe: DataFrame, index):
    return {
        ### 动量指标 Momentum Indicators ###
        "adx": dataframe['adx'][index],
        "adxr": dataframe['adxr'][index],
        "apo": dataframe['apo'][index],
        "aroondown": dataframe['aroondown'][index],
        "aroonup": dataframe['aroonup'][index],
        "aroonosc": dataframe['aroonosc'][index],
        "bop": dataframe['bop'][index],
        "cci": dataframe['cci'][index],
        "cmo": dataframe['cmo'][index],
        "dx": dataframe['dx'][index],
        "macd": dataframe['macd'][index],
        "macdsignal": dataframe['macdsignal'][index],
        "macdhist": dataframe['macdhist'][index],
        "macd_diff": dataframe['macd_diff'][index],
        "macd_hist_slope": dataframe['macd_hist_slope'][index],
        "macd_hist_std": dataframe['macd_hist_std'][index],
        "mfi": dataframe['mfi'][index],
        "minus_di": dataframe['minus_di'][index],
        "minus_dm": dataframe['minus_dm'][index],
        "mom": dataframe['mom'][index],
        "plus_di": dataframe['plus_di'][index],
        "plus_dm": dataframe['plus_dm'][index],
        "ppo": dataframe['ppo'][index],
        "roc": dataframe['roc'][index],
        "rocp": dataframe['rocp'][index],
        "rocr": dataframe['rocr'][index],
        "rocr100": dataframe['rocr100'][index],
        "rsi": dataframe['rsi'][index],
        "slowk": dataframe['slowk'][index],
        "slowd": dataframe['slowd'][index],
        "stochf_fastk": dataframe['stochf_fastk'][index],
        "stochf_fastd": dataframe['stochf_fastd'][index],
        "stochrsi_fastk": dataframe['stochrsi_fastk'][index],
        "stochrsi_fastd": dataframe['stochrsi_fastd'][index],
        "trix": dataframe['trix'][index],
        "ultosc": dataframe['ultosc'][index],
        "willr": dataframe['willr'][index],

        ### 交易量指示器 Volume Indicators ###
        "ad": dataframe['ad'][index],
        "adosc": dataframe['adosc'][index],
        "obv": dataframe['obv'][index],

        ### 周期指标 Cycle Indicators ###
        "ht_dcperiod": dataframe['ht_dcperiod'][index],
        "ht_dcphase": dataframe['ht_dcphase'][index],

        # ht_phasor
        "inphase": dataframe['inphase'][index],
        "quadrature": dataframe['quadrature'][index],
        # ht_phasor
        "sine": dataframe['sine'][index],
        "leadsine": dataframe['leadsine'][index],
        # "ht_trendmode": dataframe['ht_trendmode'][index],

        ### 价格转换 Price Transform ###
        "avgprice": dataframe['avgprice'][index],
        "medprice": dataframe['medprice'][index],
        "typprice": dataframe['typprice'][index],
        "wclprice": dataframe['wclprice'][index],

        ### 波动性指标 Volatility Indicators ###
        "atr": dataframe['atr'][index],
        "natr": dataframe['natr'][index],
        "trange": dataframe['trange'][index],

        ### 统计函数 Statistic Functions ###
        "beta": dataframe['beta'][index],
        "correl": dataframe['correl'][index],
        "linearreg": dataframe['linearreg'][index],
        "linearreg_angle": dataframe['linearreg_angle'][index],
        "linearreg_intercept": dataframe['linearreg_intercept'][index],
        "linearreg_slope": dataframe['linearreg_slope'][index],
        "stddev": dataframe['stddev'][index],
        "tsf": dataframe['tsf'][index],
        "var": dataframe['var'][index],

        ### 重叠研究 Overlap Studies ###
        # 布林线
        "bb_lowerband": dataframe['bb_lowerband'][index],
        "bb_upperband": dataframe['bb_upperband'][index],
        "bb_middleband": dataframe['bb_middleband'][index],

        # sar
        "sar": dataframe['sar'][index],
        "close_sar_diff": dataframe['close_sar_diff'][index],
        "close_sar_ratio": dataframe['close_sar_ratio'][index],
        # sarext
        "sarext": dataframe['sarext'][index],
        "close_sarext_diff": dataframe['close_sarext_diff'][index],
        "close_sarext_ratio": dataframe['close_sarext_ratio'][index],

        "open_klu_rate": (last_klu.close - last_klu.open) / last_klu.open,
    }

def t1_sell_stragety_feature(last_klu, dataframe: DataFrame, index):
    return {
            ### 动量指标 Momentum Indicators ###
            "adx": dataframe['adx'][index],
            "adxr": dataframe['adxr'][index],
            "apo": dataframe['apo'][index],
            "aroondown": dataframe['aroondown'][index],
            "aroonup": dataframe['aroonup'][index],
            "aroonosc": dataframe['aroonosc'][index],
            "bop": dataframe['bop'][index],
            "cci": dataframe['cci'][index],
            "cmo": dataframe['cmo'][index],
            "dx": dataframe['dx'][index],
            "macd": dataframe['macd'][index],
            "macdsignal": dataframe['macdsignal'][index],
            "macdhist": dataframe['macdhist'][index],
            "macd_diff": dataframe['macd_diff'][index],
            "macd_hist_slope": dataframe['macd_hist_slope'][index],
            "macd_hist_std": dataframe['macd_hist_std'][index],
            "mfi": dataframe['mfi'][index],
            "minus_di": dataframe['minus_di'][index],
            "minus_dm": dataframe['minus_dm'][index],
            "mom": dataframe['mom'][index],
            "plus_di": dataframe['plus_di'][index],
            "plus_dm": dataframe['plus_dm'][index],
            "ppo": dataframe['ppo'][index],
            "roc": dataframe['roc'][index],
            "rocp": dataframe['rocp'][index],
            "rocr": dataframe['rocr'][index],
            "rocr100": dataframe['rocr100'][index],
            "rsi": dataframe['rsi'][index],
            "slowk": dataframe['slowk'][index],
            "slowd": dataframe['slowd'][index],
            "stochf_fastk": dataframe['stochf_fastk'][index],
            "stochf_fastd": dataframe['stochf_fastd'][index],
            "stochrsi_fastk": dataframe['stochrsi_fastk'][index],
            "stochrsi_fastd": dataframe['stochrsi_fastd'][index],
            "trix": dataframe['trix'][index],
            "ultosc": dataframe['ultosc'][index],
            "willr": dataframe['willr'][index],

            ### 交易量指示器 Volume Indicators ###
            "ad": dataframe['ad'][index],
            "adosc": dataframe['adosc'][index],
            "obv": dataframe['obv'][index],

            ### 周期指标 Cycle Indicators ###
            "ht_dcperiod": dataframe['ht_dcperiod'][index],
            "ht_dcphase": dataframe['ht_dcphase'][index],

            # ht_phasor
            "inphase": dataframe['inphase'][index],
            "quadrature": dataframe['quadrature'][index],
            # ht_phasor
            "sine": dataframe['sine'][index],
            "leadsine": dataframe['leadsine'][index],
            # "ht_trendmode": dataframe['ht_trendmode'][index],

            ### 价格转换 Price Transform ###
            "avgprice": dataframe['avgprice'][index],
            "medprice": dataframe['medprice'][index],
            "typprice": dataframe['typprice'][index],
            "wclprice": dataframe['wclprice'][index],

            ### 波动性指标 Volatility Indicators ###
            "atr": dataframe['atr'][index],
            "natr": dataframe['natr'][index],
            "trange": dataframe['trange'][index],

            ### 形态识别 Pattern Recognition ###
            "cdl3blackcrows": dataframe['cdl3blackcrows'][index],
            "cdl3inside": dataframe['cdl3inside'][index],
            "cdl3linestrike": dataframe['cdl3linestrike'][index],
            "cdl3outside": dataframe['cdl3outside'][index],
            "cdl3whitesoldiers": dataframe['cdl3whitesoldiers'][index],
            "cdladvanceblock": dataframe['cdladvanceblock'][index],
            "cdlbelthold": dataframe['cdlbelthold'][index],
            "cdlclosingmarubozu": dataframe['cdlclosingmarubozu'][index],
            "cdldarkcloudcover": dataframe['cdldarkcloudcover'][index],
            "cdldoji": dataframe['cdldoji'][index],
            "cdldojistar": dataframe['cdldojistar'][index],
            "cdldragonflydoji": dataframe['cdldragonflydoji'][index],
            "cdlengulfing": dataframe['cdlengulfing'][index],
            "cdleveningdojistar": dataframe['cdleveningdojistar'][index],
            "cdleveningstar": dataframe['cdleveningstar'][index],
            "cdlgapsidesidewhite": dataframe['cdlgapsidesidewhite'][index],
            "cdlgravestonedoji": dataframe['cdlgravestonedoji'][index],
            "cdlhammer": dataframe['cdlhammer'][index],
            "cdlhangingman": dataframe['cdlhangingman'][index],
            "cdlharami": dataframe['cdlharami'][index],
            "cdlharamicross": dataframe['cdlharamicross'][index],
            "cdlhighwave": dataframe['cdlhighwave'][index],
            "cdlhikkake": dataframe['cdlhikkake'][index],
            "cdlhikkakemod": dataframe['cdlhikkakemod'][index],
            "cdlhomingpigeon": dataframe['cdlhomingpigeon'][index],
            "cdlidentical3crows": dataframe['cdlidentical3crows'][index],
            "cdlinneck": dataframe['cdlinneck'][index],
            "cdlinvertedhammer": dataframe['cdlinvertedhammer'][index],
            "cdllongleggeddoji": dataframe['cdllongleggeddoji'][index],
            "cdllongline": dataframe['cdllongline'][index],
            "cdlmarubozu": dataframe['cdlmarubozu'][index],
            "cdlmatchinglow": dataframe['cdlmatchinglow'][index],
            "cdlmorningdojistar": dataframe['cdlmorningdojistar'][index],
            "cdlmorningstar": dataframe['cdlmorningstar'][index],
            "cdlonneck": dataframe['cdlonneck'][index],
            "cdlpiercing": dataframe['cdlpiercing'][index],
            "cdlrickshawman": dataframe['cdlrickshawman'][index],
            "cdlrisefall3methods": dataframe['cdlrisefall3methods'][index],
            "cdlseparatinglines": dataframe['cdlseparatinglines'][index],
            "cdlshootingstar": dataframe['cdlshootingstar'][index],
            "cdlshortline": dataframe['cdlshortline'][index],
            "cdlspinningtop": dataframe['cdlspinningtop'][index],
            "cdlstalledpattern": dataframe['cdlstalledpattern'][index],
            "cdlsticksandwich": dataframe['cdlsticksandwich'][index],
            "cdltakuri": dataframe['cdltakuri'][index],
            "cdltasukigap": dataframe['cdltasukigap'][index],
            "cdlthrusting": dataframe['cdlthrusting'][index],
            "cdltristar": dataframe['cdltristar'][index],
            "cdlxsidegap3methods": dataframe['cdlxsidegap3methods'][index],

            ### 统计函数 Statistic Functions ###
            "beta": dataframe['beta'][index],
            "correl": dataframe['correl'][index],
            "linearreg": dataframe['linearreg'][index],
            "linearreg_angle": dataframe['linearreg_angle'][index],
            "linearreg_intercept": dataframe['linearreg_intercept'][index],
            "linearreg_slope": dataframe['linearreg_slope'][index],
            "stddev": dataframe['stddev'][index],
            "tsf": dataframe['tsf'][index],
            "var": dataframe['var'][index],

            ### 重叠研究 Overlap Studies ###
            "bb_lowerband": dataframe['bb_lowerband'][index],
            "bb_upperband": dataframe['bb_upperband'][index],
            "bb_middleband": dataframe['bb_middleband'][index],

            "dema": dataframe['dema'][index],
            # ema
            "ema_5": dataframe['ema_5'][index],
            "ema_10": dataframe['ema_10'][index],
            "ema_20": dataframe['ema_20'][index],
            "ema_30": dataframe['ema_30'][index],

            "ht_trendline": dataframe['ht_trendline'][index],
            # kama 考夫曼均线
            "kama_5": dataframe['kama_5'][index],
            "kama_10": dataframe['kama_10'][index],
            "kama_20": dataframe['kama_20'][index],
            "kama_30": dataframe['kama_30'][index],
            # ma
            "ma_20": dataframe['ma_20'][index],
            "ma_20": dataframe['ma_20'][index],
            "ma_20": dataframe['ma_20'][index],
            "ma_30": dataframe['ma_30'][index],

            "mama": dataframe['mama'][index],
            "fama": dataframe['fama'][index],
            "midpoint": dataframe['midpoint'][index],
            "midprice": dataframe['midprice'][index],
            "sar": dataframe['sar'][index],
            "sarext": dataframe['sarext'][index],
            # sma
            "sma_5": dataframe['sma_5'][index],
            "sma_10": dataframe['sma_10'][index],
            "sma_20": dataframe['sma_20'][index],
            "sma_30": dataframe['sma_30'][index],
            # tema
            "tema_5": dataframe['tema_5'][index],
            "tema_10": dataframe['tema_10'][index],
            "tema_20": dataframe['tema_20'][index],
            "tema_30": dataframe['tema_30'][index],
            # t3
            "t3_5": dataframe['t3_5'][index],
            "t3_10": dataframe['t3_10'][index],
            "t3_20": dataframe['t3_20'][index],
            "t3_30": dataframe['t3_30'][index],
            # trima
            "trima_5": dataframe['trima_5'][index],
            "trima_10": dataframe['trima_10'][index],
            "trima_20": dataframe['trima_20'][index],
            "trima_30": dataframe['trima_30'][index],
            # wma
            "wma_30": dataframe['wma_30'][index],
            "wma_20": dataframe['wma_20'][index],
            "wma_10": dataframe['wma_10'][index],
            "wma_5": dataframe['wma_5'][index],
            "open_klu_rate": (last_klu.close - last_klu.open) / last_klu.open,
    }

def cal_bi_angle(begin_close, end_close, length):
    return np.arctan((begin_close - end_close) / length) * 180 / np.pi

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

def cal_zs_in_out_bi_macd_area(zs:CZS,dataframe:pd.DataFrame):
    bi_in = zs.bi_in
    bi_out = zs.bi_out
    bi_in_macd_area = dataframe.iloc[bi_in.get_begin_klu().idx:bi_in.get_end_klu().idx + 1]['macdhist'].abs().sum().item()
    bi_out_macd_area = dataframe.iloc[bi_out.get_begin_klu().idx:bi_out.get_end_klu().idx + 1]['macdhist'].abs().sum().item()
    macd_area_ratio = bi_out_macd_area / bi_in_macd_area
    return bi_in_macd_area,bi_out_macd_area,macd_area_ratio

def t1_buy_stragety_feature(last_klu, last_bsp: CBS_Point, dataframe: DataFrame, index, klu_idx, cur_lv_chan):
    features = {
        ### 动量指标 Momentum Indicators ###
        "adx": dataframe['adx'][index],
        "adxr": dataframe['adxr'][index],
        "bop": dataframe['bop'][index],
        "cci": dataframe['cci'][index],
        "cmo": dataframe['cmo'][index],
        "dx": dataframe['dx'][index],

        "macd_hist_std": dataframe['macd_hist_std'][index],
        "macd_hist_slope": dataframe['macd_hist_slope'][index],
        # "macd_diff": dataframe['macd_diff'][index],
        "mom": dataframe['mom'][index],
        "plus_di": dataframe['plus_di'][index],
        "plus_dm": dataframe['plus_dm'][index],
        "ppo": dataframe['ppo'][index],

        "rsi": dataframe['rsi'][index],
        "stochf_fastk": dataframe['stochf_fastk'][index],
        "stochrsi_fastk": dataframe['stochrsi_fastk'][index],
        "trix": dataframe['trix'][index],

        "willr": dataframe['willr'][index],

        ### 交易量指示器 Volume Indicators ###

        # "adosc": dataframe['adosc'][index],
        # "obv": dataframe['obv'][index],

        ### 周期指标 Cycle Indicators ###
        # "ht_dcperiod": dataframe['ht_dcperiod'][index],
        "ht_dcphase": dataframe['ht_dcphase'][index],

        # ht_phasor
        "sine": dataframe['sine'][index],
        "leadsine": dataframe['leadsine'][index],

        ### 波动性指标 Volatility Indicators ###
        "natr": dataframe['natr'][index],

        ### 统计函数 Statistic Functions ###
        "beta": dataframe['beta'][index],

        ### 重叠研究 Overlap Studies ###
        "bb_pct": dataframe['bb_pct'][index],
        "bb_diff_std": dataframe['bb_diff_std'][index],
        "bb_width_pct": dataframe['bb_width_pct'][index],
        # "bb_break_upper": dataframe['bb_break_upper'][index],
        # "bb_break_lower": dataframe['bb_break_lower'][index],
        "bb_lowerband_slope_5": dataframe['bb_lowerband_slope_5'][index],
        "bb_upperband_slope_5": dataframe['bb_upperband_slope_5'][index],
        "bb_middleband_slope_5": dataframe['bb_middleband_slope_5'][index],
        "bb_upperband_lowerband_slope_diff_5": dataframe['bb_upperband_lowerband_slope_diff_5'][index],
        "bb_upperband_middleband_slope_diff_5": dataframe['bb_upperband_middleband_slope_diff_5'][index],
        "bb_middleband_lowerband_slope_diff_5": dataframe['bb_middleband_lowerband_slope_diff_5'][index],

        # dema
        # "dema": dataframe['dema'][index],
        # 均线
        "close_vs_ema7": dataframe['close_vs_ema7'][index],
        "close_vs_ema25": dataframe['close_vs_ema25'][index],
        "ema7_vs_ema25": dataframe['ema7_vs_ema25'][index],
        "slope_ema7": dataframe['slope_ema7'][index],
        "slope_ema25": dataframe['slope_ema25'][index],
        "close_vs_sma25": dataframe['close_vs_sma25'][index],
        "close_vs_sma99": dataframe['close_vs_sma99'][index],
        "sma25_vs_sma99": dataframe['sma25_vs_sma99'][index],
        "slope_sma25": dataframe['slope_sma25'][index],
        "slope_sma99": dataframe['slope_sma99'][index],
        "close_vs_kama25": dataframe['close_vs_kama25'][index],
        "slope_kama25": dataframe['slope_kama25'][index],
        "close_vs_tema7": dataframe['close_vs_tema7'][index],
        "slope_tema7": dataframe['slope_tema7'][index],

        # ht_trendline
        # "ht_trendline": dataframe['ht_trendline'][index],
        # kama 考夫曼均线
        # "kama_7": dataframe['kama_7'][index],
        "close_kama_7_ratio": dataframe['close_kama_7_ratio'][index],

        # "kama_25": dataframe['kama_25'][index],
        "close_kama_25_ratio": dataframe['close_kama_25_ratio'][index],
        "kama_7_25_ratio": dataframe['kama_7_25_ratio'][index],

        # "kama_99": dataframe['kama_99'][index],
        "close_kama_99_ratio": dataframe['close_kama_99_ratio'][index],
        "kama_7_99_ratio": dataframe['kama_7_99_ratio'][index],
        "kama_25_99_ratio": dataframe['kama_25_99_ratio'][index],

        # ma       ,
        # "ma_7": dataframe['ma_7'][index],
        "close_ma_7_ratio": dataframe['close_ma_7_ratio'][index],

        # "ma_25": dataframe['ma_25'][index],
        "close_ma_25_ratio": dataframe['close_ma_25_ratio'][index],
        "ma_7_25_ratio": dataframe['ma_7_25_ratio'][index],

        # "ma_99": dataframe['ma_99'][index],
        "close_ma_99_ratio": dataframe['close_ma_99_ratio'][index],
        "ma_7_99_ratio": dataframe['ma_7_99_ratio'][index],
        "ma_25_99_ratio": dataframe['ma_25_99_ratio'][index],
        # mama
        "mama": dataframe['mama'][index],
        "fama": dataframe['fama'][index],
        # midpoint
        "midpoint": dataframe['midpoint'][index],
        # midprice
        "midprice": dataframe['midprice'][index],
        # sar
        # "sar": dataframe['sar'][index],
        # "close_sar_ratio": dataframe['close_sar_ratio'][index],
        "sar_distance": dataframe['sar_distance'][index],
        "sar_slope_down": dataframe['sar_slope_down'][index],
        # sarext
        # "sarext": dataframe['sarext'][index],
        # "close_sarext_ratio": dataframe['close_sarext_ratio'][index],
        # sma
        # "sma_7": dataframe['sma_7'][index],
        "close_sma_7_ratio": dataframe['close_sma_7_ratio'][index],

        # "sma_25": dataframe['sma_25'][index],
        "close_sma_25_ratio": dataframe['close_sma_25_ratio'][index],
        "sma_7_25_ratio": dataframe['sma_7_25_ratio'][index],

        # "sma_99": dataframe['sma_99'][index],
        "close_sma_99_ratio": dataframe['close_sma_99_ratio'][index],
        "sma_7_99_ratio": dataframe['sma_7_99_ratio'][index],
        "sma_25_99_ratio": dataframe['sma_25_99_ratio'][index],
        # tema
        # "tema_7": dataframe['tema_7'][index],
        # "close_tema_7_ratio": dataframe['close_tema_7_ratio'][index],
        #
        # "tema_25": dataframe['tema_25'][index],
        # "close_tema_25_ratio": dataframe['close_tema_25_ratio'][index],
        # "tema_7_25_ratio": dataframe['tema_7_25_ratio'][index],
        #
        # "tema_99": dataframe['tema_99'][index],
        # "close_tema_99_ratio": dataframe['close_tema_99_ratio'][index],
        # "tema_7_99_ratio": dataframe['tema_7_99_ratio'][index],
        # "tema_25_99_ratio": dataframe['tema_25_99_ratio'][index],
        #
        # # trima
        # "trima_7": dataframe['trima_7'][index],
        # "close_trima_7_ratio": dataframe['close_trima_7_ratio'][index],
        #
        # "trima_25": dataframe['trima_25'][index],
        # "close_trima_25_ratio": dataframe['close_trima_25_ratio'][index],
        # "trima_7_25_ratio": dataframe['trima_7_25_ratio'][index],
        #
        # "trima_99": dataframe['trima_99'][index],
        # "close_trima_99_ratio": dataframe['close_trima_99_ratio'][index],
        # "trima_7_99_ratio": dataframe['trima_7_99_ratio'][index],
        # "trima_25_99_ratio": dataframe['trima_25_99_ratio'][index],

        # wma
        # "wma_7": dataframe['wma_7'][index],
        # "close_wma_7_ratio": dataframe['close_wma_7_ratio'][index],
        #
        # "wma_25": dataframe['wma_25'][index],
        # "close_wma_25_ratio": dataframe['close_wma_25_ratio'][index],
        # "wma_7_25_ratio": dataframe['wma_7_25_ratio'][index],
        #
        # "wma_99": dataframe['wma_99'][index],
        # "close_wma_99_ratio": dataframe['close_wma_99_ratio'][index],
        # "wma_7_99_ratio": dataframe['wma_7_99_ratio'][index],
        # "wma_25_99_ratio": dataframe['wma_25_99_ratio'][index],
        "open_klu_rate": (last_klu.close - last_klu.open) / last_klu.open,
        "close_slope_5":  dataframe['close_slope_5'][index],
        "close_slope_14":  dataframe['close_slope_14'][index],
        "close_pct_5":  dataframe['close_pct_5'][index],
        "close_mom_10":  dataframe['close_mom_10'][index],
        "volume_slope_14":  dataframe['volume_slope_14'][index],
        "vol_close_slope_div":  dataframe['vol_close_slope_div'][index],
        "volume_pct_5":  dataframe['volume_pct_5'][index],
        "vol_close_pct_div":  dataframe['vol_close_pct_div'][index],
        "volume_mom_10":  dataframe['volume_mom_10'][index],
        "vol_close_mom_div":  dataframe['vol_close_mom_div'][index],
        "rsi_slope_14":  dataframe['rsi_slope_14'][index],
        "rsi_slope_5":  dataframe['rsi_slope_5'][index],
        "rsi_divergence_5":  dataframe['rsi_divergence_5'][index],
        "rsi_divergence_14":  dataframe['rsi_divergence_14'][index],

    }
    features['bottom_strength'] = dataframe.loc[klu_idx - index + klu_idx:index, ['low']].apply(
            lambda x: (x.drop(klu_idx).min() - x[klu_idx]) / x[klu_idx]
            if (x[klu_idx] < x.drop(klu_idx).min())
            else 0
        ).item()
        # 成交量衰减因子
    features['vol_decay'] = dataframe.loc[klu_idx - index + klu_idx:index, ['volume']].apply(
            lambda x: x[klu_idx] / (x.drop(klu_idx).mean()) if (x[klu_idx] > (x.drop(klu_idx).max())) else 1
        ).item()

        # 底分型成交量放大因子
    features['vol_growth'] = dataframe.loc[klu_idx - index + klu_idx:index, ['volume']].apply(
            lambda x: x[klu_idx] / (x.drop(klu_idx).mean()) if (x[klu_idx] < (x.drop(klu_idx).min())) else 1
        ).item()
    features['fractal_decay'] = np.exp(-0.1 * (index - klu_idx))
    features['bottom_composite_strength'] = (
                features['bottom_strength'] * features['vol_growth'] * features['fractal_decay']).item()
    bi = last_bsp.bi
    pre_bi = last_bsp.bi.pre
    next_bi = last_bsp.bi.next
    features['bi_angle'] = cal_bi_angle(bi.get_begin_klu().close, bi.get_end_klu().close,
                                            bi.get_end_klu().idx - bi.get_begin_klu().idx)
    if pre_bi:
        features['pre_bi_angle'] = cal_bi_angle(pre_bi.get_begin_klu().close, pre_bi.get_end_klu().close,
                                                    pre_bi.get_end_klu().idx - pre_bi.get_begin_klu().idx)
        pre_bi_angle = cal_bi_angle(pre_bi.get_begin_klu().close, pre_bi.get_end_klu().close,
                                                    pre_bi.get_end_klu().idx - pre_bi.get_begin_klu().idx)
        features['pre_now_angle_diff'] = features['bi_angle'] - pre_bi_angle

    # bi_momentum_pre = cal_bi_momentum(pre_bi)
    # bi_momentum_curr = cal_bi_momentum(bi)
    # features['bi_momentum_pre'] = bi_momentum_pre
    # features['bi_momentum_curr'] = bi_momentum_curr
    # features['bi_momentum_decay_ratio'] = bi_momentum_curr / bi_momentum_pre
    # features['bi_volume_decay'] = bi.get_end_klu().volume / bi.get_begin_klu().volume

    zs = cur_lv_chan.zs_list[-1]
    # 中枢上沿
    zs_upper_edge = zs.high
    # 中枢下沿
    zs_lower_edge = zs.low
    # 中枢高度
    zs_height = zs_upper_edge - zs_lower_edge
    # 中枢向下价格系数
    zs_downward_price_coeff = (zs_lower_edge-last_klu.close) / zs_height
    zs_downward_price_coeff = max(zs_downward_price_coeff, 0)

    zs_begin_klu_idx = zs.begin_bi.get_begin_klu().idx
    zs_end_klu_idx = zs.end_bi.get_end_klu().idx
    zs_avg_atr = dataframe.iloc[zs_begin_klu_idx:zs_end_klu_idx + 1]['atr'].mean()
    zs_bi_out_avg_atr = dataframe.iloc[zs.bi_out.get_begin_klu().idx:zs.bi_out.get_end_klu().idx + 1]['atr'].mean()
    # 中枢波动率系数
    zs_out_atr_ratio = zs_bi_out_avg_atr/zs_avg_atr

    zs_avg_volume = dataframe.iloc[zs_begin_klu_idx:zs_end_klu_idx + 1]['volume'].mean()
    zs_bi_out_avg_volume = dataframe.iloc[zs.bi_out.get_begin_klu().idx:zs.bi_out.get_end_klu().idx + 1][
        'volume'].mean()
    # 中枢量能系数
    zs_out_volume_ratio = zs_bi_out_avg_volume / zs_avg_volume
    # 中枢时间系数
    zs_time_coeff = np.log(zs_end_klu_idx - zs_begin_klu_idx + 1) / 5

    zs_out_score = (0.4 * zs_downward_price_coeff *100 + 0.25*min(zs_out_volume_ratio,3) * 33.3+0.2*min(zs_out_atr_ratio,2)*50 +0.15*zs_time_coeff*100)
    zs_break_strength = np.clip(zs_out_score,0,100)
    features['zs_break_strength'] = zs_break_strength
    features['zs_time_coeff'] = zs_time_coeff
    features['zs_out_volume_ratio'] = zs_out_volume_ratio
    features['zs_downward_price_coeff'] = zs_downward_price_coeff
    features['zs_out_atr_ratio'] = zs_out_atr_ratio

    bi_in_macd_area,bi_out_macd_area,macd_area_ratio = cal_zs_in_out_bi_macd_area(zs,dataframe)
    # features['bi_in_macd_area'] = bi_in_macd_area
    # features['bi_out_macd_area'] = bi_out_macd_area
    features['macd_area_ratio'] = macd_area_ratio
    return features

def stragety_feature(last_klu):
    return {
            "open_klu_rate": (last_klu.close - last_klu.open) / last_klu.open,
        }

def split_three_phases(data: Dict[int, T_SAMPLE_INFO], ratios: list = [0.7, 0.15, 0.15]) -> tuple:
        keys = list(data.keys())

        total = len(keys)
        # 计算分割点，确保总和不超过100%
        split1 = int(total * ratios[0])
        split2 = split1 + int(total * ratios[1])

        # 三段数据划分
        train = {k: data[k] for k in keys[:split1]}
        val = {k: data[k] for k in keys[split1:split2]}
        test = {k: data[k] for k in keys[split2:]}

        return train, val, test

def write_libsvm(file_name: str, dict: Dict[int, T_SAMPLE_INFO], bsp_academy, feature_meta, feature_idx,
                     plot_marker,selected_feature_names) -> None:
    fid = open(file_name, "w")
    positive = 0
    negative = 0
    for bsp_klu_idx, feature_info in dict.items():
        label = int(bsp_klu_idx in bsp_academy)  # 以买卖点识别是否准确为label
        if label == 1:
            positive += 1
        else:
            negative += 1
        features = []  # List[(idx, value)]
        for feature_name, value in feature_info['feature'].items():
            # if(feature_name in selected_feature_names):
                if feature_name not in feature_meta:
                    cur_feature_idx = feature_idx[-1]
                    feature_meta[feature_name] = cur_feature_idx
                    feature_idx.append(cur_feature_idx + 1)
                features.append((feature_meta[feature_name], value))


        features.sort(key=lambda x: x[0])
        feature_str = " ".join([f"{idx}:{value}" for idx, value in features])
        fid.write(f"{label} {feature_str}\n")
        plot_marker[feature_info["open_time"].to_str()] = (
            "√" if label else "×", "down" if feature_info["is_buy"] else "up")
    fid.close()
    print(file_name, "正样本", positive, "负样本", negative)

def save_libsvm_file(file_name: str, dict: Dict[int, T_SAMPLE_INFO], bsp_academy,):
    buy_flag = None
    if is_buy:
        buy_flag = 'buy'
    else:
        buy_flag = 'sell'
    prifix = type.name + "_" + buy_flag + "_"
    # 生成libsvm样本特征
    bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp()]
    bsp_feature = []
    bsp_label = []
    for bsp_klu_idx, feature_info in bsp_dict.items():
        label = int(bsp_klu_idx in bsp_academy)
        feature_item = {}
        for feature_name, value in feature_info['feature'].items():
            feature_item[feature_name] = value
        bsp_feature.append(feature_item)
        bsp_label.append(label)
    bsp_df = pd.DataFrame(bsp_feature)
    bsp_y = pd.Series(bsp_label, name='target')
    selected = select_features(bsp_df, bsp_y)
    selected_feature_names = selected.columns.tolist()
    for name in selected_feature_names:
        print(name)
    feature_meta = {}  # 特征meta
    feature_idx = [0]
    plot_marker = {}
    train_file_name = prifix + "train.libsvm"
    val_file_name = prifix + "val.libsvm"
    test_file_name = prifix + "test.libsvm"
    #

    # 注释范围start
    train_dict, val_dict, test_dict = split_three_phases(bsp_dict)
    write_libsvm(train_file_name, train_dict, bsp_academy, feature_meta, feature_idx, plot_marker,
                 selected_feature_names)
    write_libsvm(val_file_name, val_dict, bsp_academy, feature_meta, feature_idx, plot_marker, selected_feature_names)
    write_libsvm(test_file_name, test_dict, bsp_academy, feature_meta, feature_idx, plot_marker, selected_feature_names)

    with open(prifix + "feature.meta", "w") as fid:
        # meta保存下来，实盘预测时特征对齐用
        fid.write(json.dumps(feature_meta))

    # 画图检查label是否正确
    # plot(chan, plot_marker, type, is_buy)
    # 注释范围end

def model_tran(chan: CChan, bsp_dict: Dict[int, T_SAMPLE_INFO], type: BSP_TYPE, is_buy: bool, scale_pos_weight):
    buy_flag = None
    if is_buy:
        buy_flag = 'buy'
    else:
        buy_flag = 'sell'
    prifix = type.name + "_" + buy_flag + "_"
    # 生成libsvm样本特征
    bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp()]
    bsp_feature = []
    bsp_label = []
    for bsp_klu_idx, feature_info in bsp_dict.items():
        label = int(bsp_klu_idx in bsp_academy)
        feature_item = {}
        for feature_name, value in feature_info['feature'].items():
            feature_item[feature_name] = value
        bsp_feature.append(feature_item)
        bsp_label.append(label)
    bsp_df = pd.DataFrame(bsp_feature)
    bsp_y = pd.Series(bsp_label, name='target')
    selected = select_features(bsp_df, bsp_y)
    selected_feature_names = selected.columns.tolist()
    for name in selected_feature_names:
        print(name)
    feature_meta = {}  # 特征meta
    feature_idx = [0]
    plot_marker = {}
    train_file_name = prifix + "train.libsvm"
    val_file_name = prifix + "val.libsvm"
    test_file_name = prifix + "test.libsvm"
    #


# 注释范围start
    train_dict, val_dict, test_dict = split_three_phases(bsp_dict)
    write_libsvm(train_file_name, train_dict, bsp_academy, feature_meta, feature_idx, plot_marker,selected_feature_names)
    write_libsvm(val_file_name, val_dict, bsp_academy, feature_meta, feature_idx, plot_marker,selected_feature_names)
    write_libsvm(test_file_name, test_dict, bsp_academy, feature_meta, feature_idx, plot_marker,selected_feature_names)

    with open(prifix + "feature.meta", "w") as fid:
        # meta保存下来，实盘预测时特征对齐用
        fid.write(json.dumps(feature_meta))

    # 画图检查label是否正确
    # plot(chan, plot_marker, type, is_buy)
    # 注释范围end




if __name__ == "__main__":
        """
        本demo主要演示如何记录策略产出的买卖点的特征
        然后将这些特征作为样本，训练一个模型(以XGB为demo)
        用于预测买卖点的准确性
        
        请注意，demo训练预测都用的是同一份数据，这是不合理的，仅仅是为了演示
        """
        code = "sz.000001"
        begin_time = "2018-01-01"
        end_time = None
        data_src = DATA_SRC.BAO_STOCK
        chan_kl_type = KL_TYPE.K_15M
        lv_list = [chan_kl_type]
        print(device)
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
        t1_bsp_buy_dict: Dict[int, T_SAMPLE_INFO] = {}  # 存储策略产出的一类买点bsp的特征
        t1_bsp_sell_dict: Dict[int, T_SAMPLE_INFO] = {}  # 存储策略产出的一类卖点bsp的特征
        t2_bsp_buy_dict: Dict[int, T_SAMPLE_INFO] = {}  # 存储策略产出的一类买点bsp的特征
        t2_bsp_sell_dict: Dict[int, T_SAMPLE_INFO] = {}  # 存储策略产出的一类卖点bsp的特征
        index = -1

        # 跑策略，保存买卖点的特征
        print("跑策略，保存买卖点的特征")

        for klu in data_src.get_kl_data():
            index = index + 1
            chan.trigger_load({chan_kl_type: [klu]})
            bsp_list = chan.get_bsp()
            if not bsp_list:
                continue
            last_bsp = bsp_list[-1]

            cur_lv_chan = chan[0]
            # (BSP_TYPE.T1 not in bsp_list[-1].type and BSP_TYPE.T1P not in bsp_list[-1].type)

            if last_bsp.klu.idx not in bsp_dict and cur_lv_chan[-2].idx == last_bsp.klu.klc.idx:
                bsp_dict[last_bsp.klu.idx] = {
                    "feature": copy.deepcopy(last_bsp.features),
                    "is_buy": last_bsp.is_buy,
                    "open_time": klu.time,
                }
                bsp_feats = new_stragety_feature(klu, data_src.get_df(), index)
                bsp_dict[last_bsp.klu.idx]['feature'].add_feat(bsp_feats)  # 开仓K线特征
                # if (BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type) and last_bsp.bi.pre:
                if (BSP_TYPE.T1 in last_bsp.type) and last_bsp.bi.pre:
                    if last_bsp.is_buy:
                        t1_bsp_buy_dict[last_bsp.klu.idx] = {
                            "feature": CFeatures(),
                            "is_buy": last_bsp.is_buy,
                            "open_time": klu.time,
                        }
                        feature_dict = {}
                        for feature_name, value in  last_bsp.features.items():
                            if feature_name.startswith("bsp1") and (not feature_name.startswith("bsp1p")):
                                feature_dict[feature_name] = value

                        klu_idx = last_bsp.klu.idx
                        # print("index", index, 'klu_idx', last_bsp.klu.idx)
                        # result = data_src.get_df().loc[klu_idx - index + klu_idx:index,
                        #          ['high', 'low', 'close', 'volume']]
                        # print(result)
                        t1_bsp_feats = t1_buy_stragety_feature(klu, last_bsp, data_src.get_df(), index,
                                                               last_bsp.klu.idx, cur_lv_chan)
                        t1_bsp_buy_dict[last_bsp.klu.idx]['feature'].add_feat(feature_dict)
                        t1_bsp_buy_dict[last_bsp.klu.idx]['feature'].add_feat(t1_bsp_feats)
                        # t1_bsp_buy_dict[last_bsp.klu.idx]['feature'].add_feat(bsp_feats)
                    else:
                        t1_bsp_sell_dict[last_bsp.klu.idx] = {
                            "feature": copy.deepcopy(last_bsp.features),
                            "is_buy": last_bsp.is_buy,
                            "open_time": klu.time,
                        }
                        t1_bsp_sell_dict[last_bsp.klu.idx]['feature'].add_feat(bsp_feats)

                if (BSP_TYPE.T2 in last_bsp.type or BSP_TYPE.T2S in last_bsp.type) and last_bsp.bi.pre:
                    if last_bsp.is_buy:
                        t2_bsp_buy_dict[last_bsp.klu.idx] = {
                            "feature": copy.deepcopy(last_bsp.features),
                            "is_buy": last_bsp.is_buy,
                            "open_time": klu.time,
                        }
                        t2_bsp_buy_dict[last_bsp.klu.idx]['feature'].add_feat(bsp_feats)
                    else:
                        t2_bsp_sell_dict[last_bsp.klu.idx] = {
                            "feature": copy.deepcopy(last_bsp.features),
                            "is_buy": last_bsp.is_buy,
                            "open_time": klu.time,
                        }
                        t2_bsp_sell_dict[last_bsp.klu.idx]['feature'].add_feat(bsp_feats)
                # print(last_bsp.klu.time, last_bsp.is_buy)
        print("=============T1 buy start ====================")

        seg = chan[0].seg_list[-1]
        while not seg.is_sure:
            print("end klu",seg.get_end_klu().idx,data_src.get_df().loc[seg.get_end_klu().idx,'date'])
            print("begin klu",seg.get_begin_klu().idx,data_src.get_df().loc[seg.get_begin_klu().idx,'date'])
            seg = seg.pre
        t1_bsp_buy_dict.items()
        print("笔")
        bi = chan[0].bi_list[-1]
        while not bi.is_sure:
            print("end klu",bi.get_end_klu().idx,data_src.get_df().loc[bi.get_end_klu().idx,'date'])
            print("begin klu",bi.get_begin_klu().idx,data_src.get_df().loc[bi.get_begin_klu().idx,'date'])
            bi = bi.pre

        new_t1_bsp_buy_dict:Dict[int, T_SAMPLE_INFO] = {}
        for bsp_klu_idx, feature_info in t1_bsp_buy_dict.items():
            if bsp_klu_idx <= seg.get_end_klu().idx:
                new_t1_bsp_buy_dict[bsp_klu_idx] = feature_info
        # new_t1_bsp_buy_dict = t1_bsp_buy_dict

        model_tran(chan, new_t1_bsp_buy_dict, BSP_TYPE.T1, True, 2.7)
        print("=============T1 buy end ====================")
        # print("=============T1 sell start ====================")
        # model_tran(chan, t1_bsp_sell_dict, BSP_TYPE.T1, False,3.2)
        # print("=============T1 sell end ====================")
        # print("=============T2 buy start ====================")
        # model_tran(chan, t2_bsp_buy_dict, BSP_TYPE.T2, True,2)
        # print("=============T2 buy end ====================")
        # print("=============T2 sell start ====================")
        # model_tran(chan, t2_bsp_sell_dict, BSP_TYPE.T2, False,2.4)
        # print("=============T2 sell end ====================")

