from datetime import datetime

from Common.CEnum import AUTYPE, DATA_FIELD, KL_TYPE
from Common.CTime import CTime
from Common.func_util import kltype_lt_day, str2float
from KLine.KLine_Unit import CKLine_Unit

from .CommonStockAPI import CCommonStockApi
import pandas as pd
from pandas import DataFrame
import talib.abstract as ta
from technical import qtpylib
import numpy as np


def GetColumnNameFromFieldList(fileds: str):
    _dict = {
        "time": DATA_FIELD.FIELD_TIME,
        "open": DATA_FIELD.FIELD_OPEN,
        "high": DATA_FIELD.FIELD_HIGH,
        "low": DATA_FIELD.FIELD_LOW,
        "close": DATA_FIELD.FIELD_CLOSE,
        "volume": DATA_FIELD.FIELD_VOLUME,
    }
    return [_dict[x] for x in fileds.split(",")]


class DataFrameAPI(CCommonStockApi):
    is_connect = None
    dataframe = None

    def __init__(self, code, k_type=KL_TYPE.K_DAY, begin_date=None, end_date=None, autype=AUTYPE.QFQ):
        super(DataFrameAPI, self).__init__(code, k_type, begin_date, end_date, autype)
        self.load_by_file()

    def cal_sar_slope(self, df: DataFrame):
        # 预分配结果列，避免 df.loc 逐行赋值
        df['sar_slope_up'] = 0.0
        df['sar_slope_down'] = 0.0

        # 计算连续趋势的数量（连续相同值的长度）
        trend = df['sar_dir'].values
        trend_len = len(trend)
        consecutive_counts = np.zeros(trend_len, dtype=int)

        count = 0
        for i in range(1, trend_len):
            if trend[i] == trend[i - 1]:
                count += 1
            else:
                count = 1
            consecutive_counts[i] = count

        # 根据趋势计算 sar_slope_up 或 sar_slope_down
        sar = df['sar'].values

        for i in range(1,len(df)):
            cnt = consecutive_counts[i]
            if trend[i] == 0:
                # 下跌趋势
                if i == 0 or cnt <= 1:
                    continue
                elif cnt > 5 and i >= 4:
                    df.at[i, 'sar_slope_down'] = ta.LINEARREG_SLOPE(sar[i - 4:i + 1], timeperiod=5)[-1]
                else:
                    start = i + 1 - cnt
                    if start >= 0:
                        df.at[i, 'sar_slope_down'] = ta.LINEARREG_SLOPE(sar[start:i + 1], timeperiod=int(cnt))[-1]
            else:
                # 上涨趋势
                if i == 0 or cnt <= 1:
                    continue
                elif cnt > 5 and i >= 4:
                    df.at[i, 'sar_slope_up'] = ta.LINEARREG_SLOPE(sar[i - 4:i + 1], timeperiod=5)[-1]
                else:
                    start = i + 1 - cnt
                    if start >= 0:
                        df.at[i, 'sar_slope_up'] = ta.LINEARREG_SLOPE(sar[start:i + 1], timeperiod=int(cnt))[-1]

    def load_by_file(self):
        df = pd.read_feather(r'E:\freqtrade\data\data\binance\futures\BTC_USDT_USDT-15m-futures.feather')  # 读取完整数据

        df['close_slope_5'] = ta.LINEARREG_SLOPE(df['close'], timeperiod=5)
        df['close_slope_14'] = ta.LINEARREG_SLOPE(df['close'], timeperiod=14)
        df['close_pct_5'] = df['close'].pct_change(periods=5)
        df['close_mom_10'] = ta.MOM(df['close'], 10)
        # 趋势类
        df['volume_slope_14'] = ta.LINEARREG_SLOPE(df['volume'], 14)
        df['vol_close_slope_div'] = df['volume_slope_14'] * df['close_slope_14']

        # 高频变化类
        df['volume_pct_5'] = df['volume'].pct_change(periods=5)
        df['vol_close_pct_div'] = df['close_pct_5'] * df['volume_pct_5']
        df["vol_ma5"] = df["volume"].rolling(5).mean()
        df["vol_ma10"] = df["volume"].rolling(10).mean()
        df["vol_ratio"] = df["volume"] / df["vol_ma10"]

        # 动量类
        volume_mom = ta.MOM(df['volume'], 10)
        df['volume_mom_10'] = ta.MOM(df['volume'], 10)
        df['vol_close_mom_div'] = df['volume_mom_10'] * df['close_mom_10']
        ### 动量指标 Momentum Indicators ###
        df['adx'] = ta.ADX(df)
        df['adxr'] = ta.ADXR(df)
        df['apo'] = ta.APO(df)
        # aroon
        aroon = ta.AROON(df)
        df['aroondown'] = aroon['aroondown']
        df['aroonup'] = aroon['aroonup']

        df['aroonosc'] = ta.AROONOSC(df)
        df['bop'] = ta.BOP(df)
        df['cci'] = ta.CCI(df)
        df['cmo'] = ta.CMO(df)
        df['dx'] = ta.DX(df)

        # macd
        macd = ta.MACD(df)
        df['macd'] = macd['macd']
        df['macdsignal'] = macd['macdsignal']
        df['macdhist'] = macd['macdhist']
        df['macd_diff'] = df['macd'] - df['macdsignal']  # 动量差值
        df['macd_hist_slope'] = ta.LINEARREG_SLOPE(df['macdhist'], timeperiod=5)  # 柱状图斜率
        df['macd_hist_std'] = df['macdhist'].rolling(window=5).std()

        df['mfi'] = ta.MFI(df)
        df['minus_di'] = ta.MINUS_DI(df)
        df['minus_dm'] = ta.MINUS_DM(df)
        df['mom'] = ta.MOM(df)
        df['plus_di'] = ta.PLUS_DI(df)
        df['plus_dm'] = ta.PLUS_DM(df)
        df['ppo'] = ta.PPO(df)
        df['roc'] = ta.ROC(df)
        df['rocp'] = ta.ROCP(df)
        df['rocr'] = ta.ROCR(df)
        df['rocr100'] = ta.ROCR100(df)
        df['rsi'] = ta.RSI(df)
        df['rsi_slope_5'] = ta.LINEARREG_SLOPE(df['rsi'], timeperiod=5)
        df['rsi_slope_14'] = ta.LINEARREG_SLOPE(df['rsi'], timeperiod=14)
        df['rsi_divergence_5'] = df['rsi_slope_5'] * df['close_slope_5']
        df['rsi_divergence_14'] = df['rsi_slope_14'] * df['close_slope_14']

        # stoch
        stoch = ta.STOCH(df)
        df['slowk'] = stoch['slowk']
        df['slowd'] = stoch['slowd']

        # stochf
        stochf = ta.STOCHF(df)
        df['stochf_fastk'] = stochf['fastk']
        df['stochf_fastd'] = stochf['fastd']

        # stochrsi
        stochrsi = ta.STOCHRSI(df)
        df['stochrsi_fastk'] = stochrsi['fastk']
        df['stochrsi_fastd'] = stochrsi['fastd']

        df['trix'] = ta.TRIX(df)
        df['ultosc'] = ta.ULTOSC(df)
        df['willr'] = ta.WILLR(df)
        df['willr_slope'] = ta.LINEARREG_SLOPE(df['willr'], timeperiod=5)
        df['willr_std'] = df['willr'].rolling(window=5).std()

        ### 交易量指示器 Volume Indicators ###
        df['ad'] = ta.AD(df)
        df['adosc'] = ta.ADOSC(df)
        df['obv'] = ta.OBV(df)

        ### 周期指标 Cycle Indicators ###
        df['ht_dcperiod'] = ta.HT_DCPERIOD(df)
        df['ht_dcphase'] = ta.HT_DCPHASE(df)
        # ht_phasor
        ht_phasor = ta.HT_PHASOR(df)
        df['inphase'] = ht_phasor['inphase']
        df['quadrature'] = ht_phasor['quadrature']
        # ht_phasor
        ht_sine = ta.HT_SINE(df)
        df['sine'] = ht_sine['sine']
        df['leadsine'] = ht_sine['leadsine']

        df['ht_trendmode'] = ta.HT_TRENDMODE(df)

        ### 价格转换 Price Transform ###
        df['avgprice'] = ta.AVGPRICE(df)
        df['medprice'] = ta.MEDPRICE(df)
        df['typprice'] = ta.TYPPRICE(df)
        df['wclprice'] = ta.WCLPRICE(df)

        ### 波动性指标 Volatility Indicators ###
        df['atr'] = ta.ATR(df)
        df['natr'] = ta.NATR(df)
        df['trange'] = ta.TRANGE(df)

        ### 形态识别 Pattern Recognition ###
        df['cdl2crows'] = ta.CDL2CROWS(df)
        df['cdl3blackcrows'] = ta.CDL3BLACKCROWS(df)
        df['cdl3inside'] = ta.CDL3INSIDE(df)
        df['cdl3linestrike'] = ta.CDL3LINESTRIKE(df)
        df['cdl3outside'] = ta.CDL3OUTSIDE(df)
        df['cdl3whitesoldiers'] = ta.CDL3WHITESOLDIERS(df)
        df['cdlabandonedbaby'] = ta.CDLABANDONEDBABY(df)
        df['cdladvanceblock'] = ta.CDLADVANCEBLOCK(df)
        df['cdlbelthold'] = ta.CDLBELTHOLD(df)
        df['cdlbreakaway'] = ta.CDLBREAKAWAY(df)
        df['cdlclosingmarubozu'] = ta.CDLCLOSINGMARUBOZU(df)
        df['cdlconcealbabyswall'] = ta.CDLCONCEALBABYSWALL(df)
        df['cdlcounterattack'] = ta.CDLCOUNTERATTACK(df)
        df['cdldarkcloudcover'] = ta.CDLDARKCLOUDCOVER(df)
        df['cdldoji'] = ta.CDLDOJI(df)
        df['cdldojistar'] = ta.CDLDOJISTAR(df)
        df['cdldragonflydoji'] = ta.CDLDRAGONFLYDOJI(df)
        df['cdlengulfing'] = ta.CDLENGULFING(df)
        df['cdleveningdojistar'] = ta.CDLEVENINGDOJISTAR(df)
        df['cdleveningstar'] = ta.CDLEVENINGSTAR(df)
        df['cdlgapsidesidewhite'] = ta.CDLGAPSIDESIDEWHITE(df)
        df['cdlgravestonedoji'] = ta.CDLGRAVESTONEDOJI(df)
        df['cdlhammer'] = ta.CDLHAMMER(df)
        df['cdlhangingman'] = ta.CDLHANGINGMAN(df)
        df['cdlharami'] = ta.CDLHARAMI(df)
        df['cdlharamicross'] = ta.CDLHARAMICROSS(df)
        df['cdlhighwave'] = ta.CDLHIGHWAVE(df)
        df['cdlhikkake'] = ta.CDLHIKKAKE(df)
        df['cdlhikkakemod'] = ta.CDLHIKKAKEMOD(df)
        df['cdlhomingpigeon'] = ta.CDLHOMINGPIGEON(df)
        df['cdlidentical3crows'] = ta.CDLIDENTICAL3CROWS(df)
        df['cdlinneck'] = ta.CDLINNECK(df)
        df['cdlinvertedhammer'] = ta.CDLINVERTEDHAMMER(df)
        df['cdlkicking'] = ta.CDLKICKING(df)
        df['cdlkickingbylength'] = ta.CDLKICKINGBYLENGTH(df)
        df['cdlladderbottom'] = ta.CDLLADDERBOTTOM(df)
        df['cdllongleggeddoji'] = ta.CDLLONGLEGGEDDOJI(df)
        df['cdllongline'] = ta.CDLLONGLINE(df)
        df['cdlmarubozu'] = ta.CDLMARUBOZU(df)
        df['cdlmatchinglow'] = ta.CDLMATCHINGLOW(df)
        df['cdlmathold'] = ta.CDLMATHOLD(df)
        df['cdlmorningdojistar'] = ta.CDLMORNINGDOJISTAR(df)
        df['cdlmorningstar'] = ta.CDLMORNINGSTAR(df)
        df['cdlonneck'] = ta.CDLONNECK(df)
        df['cdlpiercing'] = ta.CDLPIERCING(df)
        df['cdlrickshawman'] = ta.CDLRICKSHAWMAN(df)
        df['cdlrisefall3methods'] = ta.CDLRISEFALL3METHODS(df)
        df['cdlseparatinglines'] = ta.CDLSEPARATINGLINES(df)
        df['cdlshootingstar'] = ta.CDLSHOOTINGSTAR(df)
        df['cdlshortline'] = ta.CDLSHORTLINE(df)
        df['cdlspinningtop'] = ta.CDLSPINNINGTOP(df)
        df['cdlstalledpattern'] = ta.CDLSTALLEDPATTERN(df)
        df['cdlsticksandwich'] = ta.CDLSTICKSANDWICH(df)
        df['cdltakuri'] = ta.CDLTAKURI(df)
        df['cdltasukigap'] = ta.CDLTASUKIGAP(df)
        df['cdlthrusting'] = ta.CDLTHRUSTING(df)
        df['cdltristar'] = ta.CDLTRISTAR(df)
        df['cdlunique3river'] = ta.CDLUNIQUE3RIVER(df)
        df['cdlupsidegap2crows'] = ta.CDLUPSIDEGAP2CROWS(df)
        df['cdlxsidegap3methods'] = ta.CDLXSIDEGAP3METHODS(df)

        ### 统计函数 Statistic Functions ###
        df['beta'] = ta.BETA(df)
        df['correl'] = ta.CORREL(df)
        df['linearreg'] = ta.LINEARREG(df)
        df['linearreg_angle'] = ta.LINEARREG_ANGLE(df)
        df['linearreg_intercept'] = ta.LINEARREG_INTERCEPT(df)
        df['linearreg_slope'] = ta.LINEARREG_SLOPE(df)
        df['stddev'] = ta.STDDEV(df)
        df['tsf'] = ta.TSF(df)
        df['var'] = ta.VAR(df)

        ### 重叠研究 Overlap Studies ###
        # 布林线
        upper, middle, lower = ta.BBANDS(df['close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        df['bb_lowerband'] = lower
        df['bb_upperband'] = upper
        df['bb_middleband'] = middle
        df['bb_pct'] = (df['close'] - df['bb_lowerband']) / (df['bb_upperband'] - df['bb_lowerband'])
        df['bb_diff_std'] = (df['close'] - df['bb_middleband']) / (df['bb_upperband'] - df['bb_lowerband'])
        df['bb_width_pct'] = (df['bb_upperband'] - df['bb_lowerband']) / df['bb_middleband']
        df['bb_break_upper'] = (df['close'] > df['bb_upperband']).astype(int)
        df['bb_break_lower'] = (df['close'] < df['bb_lowerband']).astype(int)
        df['bb_lowerband_slope_5'] = ta.LINEARREG_SLOPE(df['bb_lowerband'], timeperiod=5)
        df['bb_upperband_slope_5'] = ta.LINEARREG_SLOPE(df['bb_upperband'], timeperiod=5)
        df['bb_middleband_slope_5'] = ta.LINEARREG_SLOPE(df['bb_middleband'], timeperiod=5)
        df['bb_upperband_lowerband_slope_diff_5'] = df['bb_upperband_slope_5'] - df['bb_lowerband_slope_5']
        df['bb_upperband_middleband_slope_diff_5'] = df['bb_upperband_slope_5'] - df['bb_middleband_slope_5']
        df['bb_middleband_lowerband_slope_diff_5'] = df['bb_middleband_slope_5'] - df['bb_lowerband_slope_5']

        # dema
        df['dema'] = ta.DEMA(df)
        # ema
        df['ema_7'] = ta.EMA(df, timeperiod=7)
        df['close_ema_7_diff'] = df['close'] - df['ema_7']
        df['close_ema_7_ratio'] = df['close'] / df['ema_7']

        df['ema_25'] = ta.EMA(df, timeperiod=25)
        df['close_ema_25_diff'] = df['close'] - df['ema_25']
        df['ema_7_25_diff'] = df['ema_7'] - df['ema_25']
        df['close_ema_25_ratio'] = df['close'] / df['ema_25']
        df['ema_7_25_ratio'] = df['ema_7'] / df['ema_25']

        df['ema_99'] = ta.EMA(df, timeperiod=99)
        df['close_ema_99_diff'] = df['close'] - df['ema_99']
        df['ema_7_99_diff'] = df['ema_7'] - df['ema_99']
        df['ema_25_99_diff'] = df['ema_25'] - df['ema_99']
        df['close_ema_99_ratio'] = df['close'] / df['ema_99']
        df['ema_7_99_ratio'] = df['ema_7'] / df['ema_99']
        df['ema_25_99_ratio'] = df['ema_25'] / df['ema_99']
        df['close_vs_ema7'] = (df['close'] - df['ema_7']) / df['ema_7']
        df['close_vs_ema25'] = (df['close'] - df['ema_25']) / df['ema_25']
        df['close_vs_ema99'] = (df['close'] - df['ema_99']) / df['ema_99']
        df['ema7_vs_ema25'] = (df['ema_7'] - df['ema_25']) / df['ema_25']
        df['ema25_vs_ema99'] = (df['ema_25'] - df['ema_99']) / df['ema_99']
        df['ema7_vs_ema99'] = (df['ema_7'] - df['ema_99']) / df['ema_99']
        df['slope_ema7'] = ta.LINEARREG_SLOPE(df['ema_7'], timeperiod=5)
        df['slope_ema25'] = ta.LINEARREG_SLOPE(df['ema_25'], timeperiod=5)
        df['slope_ema99'] = ta.LINEARREG_SLOPE(df['ema_99'], timeperiod=5)

        # ht_trendline
        df['ht_trendline'] = ta.HT_TRENDLINE(df)
        # kama 考夫曼均线
        df['kama_7'] = ta.KAMA(df, timeperiod=7)
        df['close_kama_7_diff'] = df['close'] - df['kama_7']
        df['close_kama_7_ratio'] = df['close'] / df['kama_7']

        df['kama_25'] = ta.KAMA(df, timeperiod=25)
        df['close_kama_25_diff'] = df['close'] - df['kama_25']
        df['kama_7_25_diff'] = df['kama_7'] - df['kama_25']
        df['close_kama_25_ratio'] = df['close'] / df['kama_25']
        df['kama_7_25_ratio'] = df['kama_7'] / df['kama_25']

        df['kama_99'] = ta.KAMA(df, timeperiod=99)
        df['close_kama_99_diff'] = df['close'] - df['kama_99']
        df['kama_7_99_diff'] = df['kama_7'] - df['kama_99']
        df['kama_25_99_diff'] = df['kama_25'] - df['kama_99']
        df['close_kama_99_ratio'] = df['close'] / df['kama_99']
        df['kama_7_99_ratio'] = df['kama_7'] / df['kama_99']
        df['kama_25_99_ratio'] = df['kama_25'] / df['kama_99']
        df['close_vs_kama7'] = (df['close'] - df['kama_7']) / df['kama_7']
        df['close_vs_kama25'] = (df['close'] - df['kama_25']) / df['kama_25']
        df['close_vs_kama99'] = (df['close'] - df['kama_99']) / df['kama_99']
        df['kama7_vs_kama25'] = (df['kama_7'] - df['kama_25']) / df['kama_25']
        df['kama25_vs_kama99'] = (df['kama_25'] - df['kama_99']) / df['kama_99']
        df['kama7_vs_kama99'] = (df['kama_7'] - df['kama_99']) / df['kama_99']
        df['slope_kama7'] = ta.LINEARREG_SLOPE(df['kama_7'], timeperiod=5)
        df['slope_kama25'] = ta.LINEARREG_SLOPE(df['kama_25'], timeperiod=5)
        df['slope_kama99'] = ta.LINEARREG_SLOPE(df['kama_99'], timeperiod=5)


        # ma
        df['ma_7'] = ta.MA(df, timeperiod=7)
        df['close_ma_7_diff'] = df['close'] - df['ma_7']
        df['close_ma_7_ratio'] = df['close'] / df['ma_7']

        df['ma_25'] = ta.MA(df, timeperiod=25)
        df['close_ma_25_diff'] = df['close'] - df['ma_25']
        df['ma_7_25_diff'] = df['ma_7'] - df['ma_25']
        df['close_ma_25_ratio'] = df['close'] / df['ma_25']
        df['ma_7_25_ratio'] = df['ma_7'] / df['ma_25']

        df['ma_99'] = ta.MA(df, timeperiod=99)
        df['close_ma_99_diff'] = df['close'] - df['ma_99']
        df['ma_7_99_diff'] = df['ma_7'] - df['ma_99']
        df['ma_25_99_diff'] = df['ma_25'] - df['ma_99']
        df['close_ma_99_ratio'] = df['close'] / df['ma_99']
        df['ma_7_99_ratio'] = df['ma_7'] / df['ma_99']
        df['ma_25_99_ratio'] = df['ma_25'] / df['ma_99']
        df['close_vs_ma7'] = (df['close'] - df['ma_7']) / df['ma_7']
        df['close_vs_ma25'] = (df['close'] - df['ma_25']) / df['ma_25']
        df['close_vs_ma99'] = (df['close'] - df['ma_99']) / df['ma_99']
        df['ma7_vs_ma25'] = (df['ma_7'] - df['ma_25']) / df['ma_25']
        df['ma25_vs_ma99'] = (df['ma_25'] - df['ma_99']) / df['ma_99']
        df['ma7_vs_ma99'] = (df['ma_7'] - df['ma_99']) / df['ma_99']
        df['slope_ma7'] = ta.LINEARREG_SLOPE(df['ma_7'], timeperiod=5)
        df['slope_ma25'] = ta.LINEARREG_SLOPE(df['ma_25'], timeperiod=5)
        df['slope_ma99'] = ta.LINEARREG_SLOPE(df['ma_99'], timeperiod=5)
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
        df['close_sar_diff'] = df['close'] - df['sar']
        df['close_sar_ratio'] = df['close'] / df['sar']
        df['sar_distance'] = (df['close'] - df['sar']) / df['close']
        df['sar_dir'] = (df['close'] > df['sar']).astype(int)
        df['sar_reversal'] = df['sar_dir'].diff().fillna(0).abs()
        self.cal_sar_slope(df)
        # sarext
        df['sarext'] = ta.SAREXT(df)
        df['close_sarext_diff'] = df['close'] - df['sarext']
        df['close_sarext_ratio'] = df['close'] / df['sarext']
        # sma
        df['sma_7'] = ta.SMA(df, timeperiod=7)
        df['close_sma_7_diff'] = df['close'] - df['sma_7']
        df['close_sma_7_ratio'] = df['close'] / df['sma_7']

        df['sma_25'] = ta.SMA(df, timeperiod=25)
        df['close_sma_25_diff'] = df['close'] - df['sma_25']
        df['sma_7_25_diff'] = df['sma_7'] - df['sma_25']
        df['close_sma_25_ratio'] = df['close'] / df['sma_25']
        df['sma_7_25_ratio'] = df['sma_7'] / df['sma_25']

        df['sma_99'] = ta.SMA(df, timeperiod=99)
        df['close_sma_99_diff'] = df['close'] - df['sma_99']
        df['sma_7_99_diff'] = df['sma_7'] - df['sma_99']
        df['sma_25_99_diff'] = df['sma_25'] - df['sma_99']
        df['close_sma_99_ratio'] = df['close'] / df['sma_99']
        df['sma_7_99_ratio'] = df['sma_7'] / df['sma_99']
        df['sma_25_99_ratio'] = df['sma_25'] / df['sma_99']
        df['close_vs_sma7'] = (df['close'] - df['sma_7']) / df['sma_7']
        df['close_vs_sma25'] = (df['close'] - df['sma_25']) / df['sma_25']
        df['close_vs_sma99'] = (df['close'] - df['sma_99']) / df['sma_99']
        df['sma7_vs_sma25'] = (df['sma_7'] - df['sma_25']) / df['sma_25']
        df['sma25_vs_sma99'] = (df['sma_25'] - df['sma_99']) / df['sma_99']
        df['sma7_vs_sma99'] = (df['sma_7'] - df['sma_99']) / df['sma_99']
        df['slope_sma7'] = ta.LINEARREG_SLOPE(df['sma_7'], timeperiod=5)
        df['slope_sma25'] = ta.LINEARREG_SLOPE(df['sma_25'], timeperiod=5)
        df['slope_sma99'] = ta.LINEARREG_SLOPE(df['sma_99'], timeperiod=5)

        # tema
        df['tema_7'] = ta.TEMA(df, timeperiod=7)
        df['close_tema_7_diff'] = df['close'] - df['tema_7']
        df['close_tema_7_ratio'] = df['close'] / df['tema_7']

        df['tema_25'] = ta.TEMA(df, timeperiod=25)
        df['close_tema_25_diff'] = df['close'] - df['tema_25']
        df['tema_7_25_diff'] = df['tema_7'] - df['tema_25']
        df['close_tema_25_ratio'] = df['close'] / df['tema_25']
        df['tema_7_25_ratio'] = df['tema_7'] / df['tema_25']

        df['tema_99'] = ta.TEMA(df, timeperiod=99)
        df['close_tema_99_diff'] = df['close'] - df['tema_99']
        df['tema_7_99_diff'] = df['tema_7'] - df['tema_99']
        df['tema_25_99_diff'] = df['tema_25'] - df['tema_99']
        df['close_tema_99_ratio'] = df['close'] / df['tema_99']
        df['tema_7_99_ratio'] = df['tema_7'] / df['tema_99']
        df['tema_25_99_ratio'] = df['tema_25'] / df['tema_99']
        df['close_vs_tema7'] = (df['close'] - df['tema_7']) / df['tema_7']
        df['close_vs_tema25'] = (df['close'] - df['tema_25']) / df['tema_25']
        df['close_vs_tema99'] = (df['close'] - df['tema_99']) / df['tema_99']
        df['tema7_vs_tema25'] = (df['tema_7'] - df['tema_25']) / df['tema_25']
        df['tema25_vs_tema99'] = (df['tema_25'] - df['tema_99']) / df['tema_99']
        df['tema7_vs_tema99'] = (df['tema_7'] - df['tema_99']) / df['tema_99']
        df['slope_tema7'] = ta.LINEARREG_SLOPE(df['tema_7'], timeperiod=5)
        df['slope_tema25'] = ta.LINEARREG_SLOPE(df['tema_25'], timeperiod=5)
        df['slope_tema99'] = ta.LINEARREG_SLOPE(df['tema_99'], timeperiod=5)

        # trima
        df['trima_7'] = ta.TRIMA(df, timeperiod=7)
        df['close_trima_7_diff'] = df['close'] - df['trima_7']
        df['close_trima_7_ratio'] = df['close'] / df['trima_7']

        df['trima_25'] = ta.TRIMA(df, timeperiod=25)
        df['close_trima_25_diff'] = df['close'] - df['trima_25']
        df['trima_7_25_diff'] = df['trima_7'] - df['trima_25']
        df['close_trima_25_ratio'] = df['close'] / df['trima_25']
        df['trima_7_25_ratio'] = df['trima_7'] / df['trima_25']

        df['trima_99'] = ta.TRIMA(df, timeperiod=99)
        df['close_trima_99_diff'] = df['close'] - df['trima_99']
        df['trima_7_99_diff'] = df['trima_7'] - df['trima_99']
        df['trima_25_99_diff'] = df['trima_25'] - df['trima_99']
        df['close_trima_99_ratio'] = df['close'] / df['trima_99']
        df['trima_7_99_ratio'] = df['trima_7'] / df['trima_99']
        df['trima_25_99_ratio'] = df['trima_25'] / df['trima_99']
        df['close_vs_trima7'] = (df['close'] - df['trima_7']) / df['trima_7']
        df['close_vs_trima25'] = (df['close'] - df['trima_25']) / df['trima_25']
        df['close_vs_trima99'] = (df['close'] - df['trima_99']) / df['trima_99']
        df['trima7_vs_trima25'] = (df['trima_7'] - df['trima_25']) / df['trima_25']
        df['trima25_vs_trima99'] = (df['trima_25'] - df['trima_99']) / df['trima_99']
        df['trima7_vs_trima99'] = (df['trima_7'] - df['trima_99']) / df['trima_99']
        df['slope_trima7'] = ta.LINEARREG_SLOPE(df['trima_7'], timeperiod=5)
        df['slope_trima25'] = ta.LINEARREG_SLOPE(df['trima_25'], timeperiod=5)
        df['slope_trima99'] = ta.LINEARREG_SLOPE(df['trima_99'], timeperiod=5)

        # wma
        df['wma_7'] = ta.WMA(df, timeperiod=7)
        df['close_wma_7_diff'] = df['close'] - df['wma_7']
        df['close_wma_7_ratio'] = df['close'] / df['wma_7']

        df['wma_25'] = ta.WMA(df, timeperiod=25)
        df['close_wma_25_diff'] = df['close'] - df['wma_25']
        df['wma_7_25_diff'] = df['wma_7'] - df['wma_25']
        df['close_wma_25_ratio'] = df['close'] / df['wma_25']
        df['wma_7_25_ratio'] = df['wma_7'] / df['wma_25']

        df['wma_99'] = ta.WMA(df, timeperiod=99)
        df['close_wma_99_diff'] = df['close'] - df['wma_99']
        df['wma_7_99_diff'] = df['wma_7'] - df['wma_99']
        df['wma_25_99_diff'] = df['wma_25'] - df['wma_99']
        df['close_wma_99_ratio'] = df['close'] / df['wma_99']
        df['wma_7_99_ratio'] = df['wma_7'] / df['wma_99']
        df['wma_25_99_ratio'] = df['wma_25'] / df['wma_99']
        df['close_vs_wma7'] = (df['close'] - df['wma_7']) / df['wma_7']
        df['close_vs_wma25'] = (df['close'] - df['wma_25']) / df['wma_25']
        df['close_vs_wma99'] = (df['close'] - df['wma_99']) / df['wma_99']
        df['wma7_vs_wma25'] = (df['wma_7'] - df['wma_25']) / df['wma_25']
        df['wma25_vs_wma99'] = (df['wma_25'] - df['wma_99']) / df['wma_99']
        df['wma7_vs_wma99'] = (df['wma_7'] - df['wma_99']) / df['wma_99']
        df['slope_wma7'] = ta.LINEARREG_SLOPE(df['wma_7'], timeperiod=5)
        df['slope_wma25'] = ta.LINEARREG_SLOPE(df['wma_25'], timeperiod=5)
        df['slope_wma99'] = ta.LINEARREG_SLOPE(df['wma_99'], timeperiod=5)


        df['slope'] = ta.LINEARREG_SLOPE(df['close'].values, timeperiod=5)

        stacked = df.stack(dropna=False)

        last_na_all = stacked[stacked.isna()].index[-1]
        print(f"全局最后一个NaN的索引：{last_na_all}")
        self.dataframe = df.iloc[last_na_all[0] + 1:].reset_index(drop=True)

    def get_df(self):
        return self.dataframe

    def get_kl_data(self):
        fields = "time,open,high,low,close,volume"
        timeframe = self.__convert_type()

        for _, row in self.dataframe.iterrows():
            time_str = row['date'].strftime('%Y-%m-%d %H:%M:%S')
            item_data = [
                time_str,
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row['volume']
            ]
            yield CKLine_Unit(self.create_item_dict(item_data, GetColumnNameFromFieldList(fields)), autofix=True)

    def get_kl_data_by_df(self, df: DataFrame):
        fields = "time,open,high,low,close,volume"
        timeframe = self.__convert_type()

        for _, row in df.iterrows():
            time_str = row['date'].strftime('%Y-%m-%d %H:%M:%S')
            item_data = [
                time_str,
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row['volume']
            ]
            yield CKLine_Unit(self.create_item_dict(item_data, GetColumnNameFromFieldList(fields)), autofix=True)

    def SetBasciInfo(self):
        pass

    @classmethod
    def do_init(cls):
        pass

    @classmethod
    def do_close(cls):
        pass

    def __convert_type(self):
        _dict = {
            KL_TYPE.K_DAY: '1d',
            KL_TYPE.K_WEEK: '1w',
            KL_TYPE.K_MON: '1M',
            KL_TYPE.K_5M: '5m',
            KL_TYPE.K_15M: '15m',
            KL_TYPE.K_30M: '30m',
            KL_TYPE.K_60M: '1h',
        }
        return _dict[self.k_type]

    def parse_time_column(self, inp):
        if len(inp) == 10:
            year = int(inp[:4])
            month = int(inp[5:7])
            day = int(inp[8:10])
            hour = minute = 0
        elif len(inp) == 17:
            year = int(inp[:4])
            month = int(inp[4:6])
            day = int(inp[6:8])
            hour = int(inp[8:10])
            minute = int(inp[10:12])
        elif len(inp) == 19:
            year = int(inp[:4])
            month = int(inp[5:7])
            day = int(inp[8:10])
            hour = int(inp[11:13])
            minute = int(inp[14:16])
        else:
            raise Exception(f"unknown time column from TradingView:{inp}")
        return CTime(year, month, day, hour, minute, auto=not kltype_lt_day(self.k_type))

    def create_item_dict(self, data, column_name):
        for i in range(len(data)):
            data[i] = self.parse_time_column(data[i]) if i == 0 else str2float(data[i])
        return dict(zip(column_name, data))
