from datetime import datetime

import ccxt

from Common.CEnum import AUTYPE, DATA_FIELD, KL_TYPE
from Common.CTime import CTime
from Common.func_util import kltype_lt_day, str2float
from KLine.KLine_Unit import CKLine_Unit
from datetime import datetime, timedelta

from .CommonStockAPI import CCommonStockApi


def GetColumnNameFromFieldList(fileds: str):
    _dict = {
        "time": DATA_FIELD.FIELD_TIME,
        "open": DATA_FIELD.FIELD_OPEN,
        "high": DATA_FIELD.FIELD_HIGH,
        "low": DATA_FIELD.FIELD_LOW,
        "close": DATA_FIELD.FIELD_CLOSE,
    }
    return [_dict[x] for x in fileds.split(",")]


class CCXT(CCommonStockApi):
    is_connect = None

    def __init__(self, code, k_type=KL_TYPE.K_DAY, begin_date=None, end_date=None, autype=AUTYPE.QFQ):
        super(CCXT, self).__init__(code, k_type, begin_date, end_date, autype)
        self.exchange = ccxt.binance({
            'proxies': {
                'http': 'http://127.0.0.1:8890',
                'https': 'http://127.0.0.1:8890'
            }
        })

    def get_kl_data(self):
        fields = "time,open,high,low,close,volume"
        # 获取当前时间
        now = datetime.now()

        # 计算15天前的时间
        fifteen_days_ago = now - timedelta(days=15)

        since_date = self.exchange.parse8601(fifteen_days_ago.timestamp())
        data = self.exchange.fetch_ohlcv("BTC/USDT:USDT", "15m", since=since_date)

        for item in data:
            time_obj = datetime.fromtimestamp(item[0] / 1000)
            time_str = time_obj.strftime('%Y-%m-%d %H:%M:%S')
            item_data = [
                time_str,
                item[1],
                item[2],
                item[3],
                item[4],
                item[5]
            ]
            yield CKLine_Unit(self.create_item_dict(item_data, GetColumnNameFromFieldList(fields)), autofix=True)
    def get_now_kl_data(self):
        fields = "time,open,high,low,close,volume"
        # 获取当前时间
        now = datetime.now()

        # 计算15天前的时间
        fifteen_days_ago = now - timedelta(days=15)

        since_date = self.exchange.parse8601(fifteen_days_ago.timestamp())
        data = self.exchange.fetch_ohlcv("BTC/USDT:USDT", "15m", since=since_date)

        for item in data:
            time_obj = datetime.fromtimestamp(item[0] / 1000)
            time_str = time_obj.strftime('%Y-%m-%d %H:%M:%S')
            item_data = [
                time_str,
                item[1],
                item[2],
                item[3],
                item[4],
                item[5]
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
