from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, BSP_TYPE, DATA_SRC, FX_TYPE, KL_TYPE
from chan.DataAPI.ccxt import CCXT
if __name__ == "__main__":
    """
    一个极其弱智的策略，只交易一类买卖点，底分型形成后就开仓，直到一类卖点顶分型形成后平仓
    只用做展示如何自己实现策略，做回测用~
    """
    code = "sz.000001"
    begin_time = "2025-01-01"
    end_time = None
    data_src = DATA_SRC.BAO_STOCK
    lv_list = [KL_TYPE.K_DAY]

    config = CChanConfig({
        "trigger_step": True,  # 打开开关！
        "divergence_rate": 0.8,
        "min_zs_cnt": 1,
    })
    data_src = CCXT(code, k_type=KL_TYPE.K_15M, begin_date=begin_time, end_date=end_time,
                            autype=AUTYPE.QFQ)
    data_src.get_kl_data()
