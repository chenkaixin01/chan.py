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

    def load_by_file(self):
        df = pd.read_feather(r'E:\freqtrade\data\data\binance\futures\BTC_USDT_USDT-15m-futures.feather')  # 读取完整数据

        ### 动量指标 Momentum Indicators ###
        df ['adx'] = ta.ADX(df)
        df ['adxr'] = ta.ADXR(df)
        df ['apo'] = ta.APO(df)
        # aroon
        aroon = ta.AROON(df)
        df ['aroondown'] = aroon['aroondown']
        df ['aroonup'] = aroon['aroonup']

        df ['aroonosc'] = ta.AROONOSC(df)
        df ['bop'] = ta.BOP(df)
        df ['cci'] = ta.CCI(df)
        df ['cmo'] = ta.CMO(df)
        df ['dx'] = ta.DX(df)

        # macd
        macd = ta.MACD(df)
        df ['macd'] = macd['macd']
        df ['macdsignal'] = macd['macdsignal']
        df ['macdhist'] = macd['macdhist']
        # macdext
        macdext = ta.MACDEXT(df)
        df ['macdext_macd'] = macdext['macd']
        df ['macdext_macdsignal'] = macdext['macdsignal']
        df ['macdext_macdhist'] = macdext['macdhist']
        # macdfix
        macdfix = ta.MACDFIX(df)
        df ['macdfix_macd'] = macdfix['macd']
        df ['macdfix_macdsignal'] = macdfix['macdsignal']
        df ['macdfix_macdhist'] = macdfix['macdhist']

        df ['mfi'] = ta.MFI(df)
        df ['minus_di'] = ta.MINUS_DI(df)
        df ['minus_dm'] = ta.MINUS_DM(df)
        df ['mom'] = ta.MOM(df)
        df ['plus_di'] = ta.PLUS_DI(df)
        df ['plus_dm'] = ta.PLUS_DM(df)
        df ['ppo'] = ta.PPO(df)
        df ['roc'] = ta.ROC(df)
        df ['rocp'] = ta.ROCP(df)
        df ['rocr'] = ta.ROCR(df)
        df ['rocr100'] = ta.ROCR100(df)
        df ['rsi'] = ta.RSI(df)

        # stoch
        stoch = ta.STOCH(df)
        df ['slowk'] = stoch['slowk']
        df ['slowd'] = stoch['slowd']

        # stochf
        stochf = ta.STOCHF(df)
        df ['stochf_fastk'] = stochf['fastk']
        df ['stochf_fastd'] = stochf['fastd']

        # stochrsi
        stochrsi = ta.STOCHRSI(df)
        df ['stochrsi_fastk'] = stochrsi['fastk']
        df ['stochrsi_fastd'] = stochrsi['fastd']

        df ['trix'] = ta.TRIX(df)
        df ['ultosc'] = ta.ULTOSC(df)
        df ['willr'] = ta.WILLR(df)

        ### 交易量指示器 Volume Indicators ###
        df ['ad'] = ta.AD(df)
        df ['adosc'] = ta.ADOSC(df)
        df ['obv'] = ta.OBV(df)

        ### 周期指标 Cycle Indicators ###
        df ['ht_dcperiod'] = ta.HT_DCPERIOD(df)
        df ['ht_dcphase'] = ta.HT_DCPHASE(df)
        # ht_phasor
        ht_phasor = ta.HT_PHASOR(df)
        df ['inphase'] = ht_phasor['inphase']
        df ['quadrature'] = ht_phasor['quadrature']
        # ht_phasor
        ht_sine = ta.HT_SINE(df)
        df ['sine'] = ht_sine['sine']
        df ['leadsine'] = ht_sine['leadsine']

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
        bollinger = ta.BBANDS(df)
        df['bb_lowerband'] = bollinger['lowerband']
        df['bb_upperband'] = bollinger['upperband']
        df['bb_middleband'] = bollinger['middleband']
        df['bb_close_lowerband_ratio'] = df['close'] / bollinger['lowerband']
        df['bb_close_upperband_ratio'] = df['close'] / bollinger['upperband']
        df['bb_close_middleband_ratio'] = df['close'] / bollinger['middleband']
        df['bb_upperband_lowerband_ratio'] = bollinger['upperband'] / bollinger['lowerband']
        df['bb_upperband_middleband_ratio'] = bollinger['upperband'] / bollinger['middleband']
        df['bb_middleband_lowerband_ratio'] = bollinger['middleband'] / bollinger['lowerband']
        df['bb_lowerband_slope_5'] = ta.LINEARREG_SLOPE(df['bb_lowerband'], timeperiod=5)
        df['bb_upperband_slope_5'] = ta.LINEARREG_SLOPE(df['bb_upperband'], timeperiod=5)
        df['bb_middleband_slope_5'] = ta.LINEARREG_SLOPE(df['bb_middleband'], timeperiod=5)
        df['bb_upperband_lowerband_slope_diff_5'] = df['bb_upperband_slope_5'] - df['bb_lowerband_slope_5']
        df['bb_upperband_middleband_slope_diff_5'] = df['bb_upperband_slope_5'] - df['bb_middleband_slope_5']
        df['bb_middleband_lowerband_slope_diff_5'] = df['bb_middleband_slope_5'] - df['bb_lowerband_slope_5']
        df['bb_lowerband_slope_20'] = ta.LINEARREG_SLOPE(df['bb_lowerband'], timeperiod=20)
        df['bb_upperband_slope_20'] = ta.LINEARREG_SLOPE(df['bb_upperband'], timeperiod=20)
        df['bb_middleband_slope_20'] = ta.LINEARREG_SLOPE(df['bb_middleband'], timeperiod=20)
        df['bb_upperband_lowerband_slope_diff_20'] = df['bb_upperband_slope_20'] - df['bb_lowerband_slope_20']
        df['bb_upperband_middleband_slope_diff_20'] = df['bb_upperband_slope_20'] - df['bb_middleband_slope_20']
        df['bb_middleband_lowerband_slope_diff_20'] = df['bb_middleband_slope_20'] - df['bb_lowerband_slope_20']

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

        df['slope'] = ta.LINEARREG_SLOPE(df['close'].values, timeperiod=5)

        stacked = df.stack(dropna=False)

        last_na_all = stacked[stacked.isna()].index[-1]
        print(f"全局最后一个NaN的索引：{last_na_all}")
        self.dataframe = df.iloc[last_na_all[0]+1:].reset_index(drop=True)


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

    def get_kl_data_by_df(self,df:DataFrame):
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
