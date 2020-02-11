from part1_Funs import *
from part4_Funs import *
from Funs_HXH3 import *

# ---------------------


# --------------------------------------- 1. 提取rawData ------------------------------------- #
data_path = '../data/'
result_path = '../result/'

logging.info('Load rawData for analysis ...')
rawData = pkl_read_write(data_path + 'rawData.pkl', 'read')

columns = ['中证红利全收益', '3-5年国开债指数', '货基指数', '景安短融']
start_dt1 = '20120101'
end_dt1 = '20191231'
tradeInfo = rawData.loc[start_dt1:end_dt1, columns]

del rawData

# 处理景安短融数据
asset_ret = nav2return(tradeInfo)
asset_ret.loc[:'20130901', '景安短融'] = asset_ret.loc[:'20130901', '货基指数']
asset_nav = nav2return(asset_ret, type=2)

dfPlot(asset_nav[['货基指数', '景安短融']])
dfPlot(asset_nav)
tradeInfo['景安短融'] = asset_nav['景安短融']

# ---------------------------------- 2. 计算权重 ------------------------------------ #
# 如果多资产选择，就这样处理
start_dt = '20130101'
end_dt = tradeInfo.index.max()
tradingdate = pkl_read_write(data_path + 'trading_date.pkl', 'read')
tradingdate = tradingdate.ix[(tradingdate['Date'] >= start_dt) & (tradingdate['Date'] <= end_dt), :]

# date list
quarter_list = getPeriodFisrtLastDates(tradingdate, period='Quarter')
month_list = getPeriodFisrtLastDates(tradingdate, period='Month')
daily_list = tradingdate['Date'].tolist()


# FOF回测里转成这种格式，应该取数更快  Ben@20190628
tmpRet = nav2return(tradeInfo[['中证红利全收益', '3-5年国开债指数', '货基指数', '景安短融']])

columns1 = ['中证红利全收益', '3-5年国开债指数']
columns2 = ['中证红利全收益', '3-5年国开债指数', '货基指数']
columns3 = ['中证红利全收益', '3-5年国开债指数', '景安短融']

ret1 = tmpRet[columns1].copy()
ret2 = tmpRet[columns2].copy()
ret3 = tmpRet[columns3].copy()

chgDates = daily_list

# -------------------------- 测试目标风险 ------------------------------ #
# 1. 计算资产风险
asset_risk = cal_asset_risk(ret1, chgDates, risk_type='normal')
dfPlot(asset_risk)

# 2. 风险调整
# asset_risk_adjusted = adjust_risk(asset_risk, adjust_fct=None)

# 3. 计算权重
risk_target = 0.04
stock_limit = 1

# 测试权重计算
daily_target_wts1 = cal_risk_target_wts(asset_risk, risk_target=0.04, stock_limit=stock_limit)
dfPlot(daily_target_wts1)

daily_target_wts2 = cal_risk_target_wts(asset_risk, risk_target=0.06, stock_limit=stock_limit)
dfPlot(daily_target_wts2)

daily_target_wts3 = cal_risk_target_wts(asset_risk, risk_target=0.04, stock_limit=0.2)
dfPlot(daily_target_wts3)

daily_target_wts4 = cal_risk_target_wts(asset_risk, risk_target=0.06, stock_limit=0.2)
dfPlot(daily_target_wts4)

daily_target_wts5 = cal_risk_target_wts(asset_risk, risk_target=0.04, stock_limit=0.3)
dfPlot(daily_target_wts5)


adjust_fct = pkl_read_write(data_path + 'adjust_fct.pkl', 'read')
adjust_fct = adjust_fct.rename(columns={'中证红利': '中证红利全收益'})

adjust_fct = adjust_fct.loc[asset_risk.index.tolist(), :]
asset_risk_adjusted = adjust_risk(asset_risk, adjust_fct=adjust_fct)

comp = pd.concat([asset_risk['中证红利全收益'], asset_risk_adjusted['中证红利全收益']], axis=1)
comp.columns = ['中证红利波动率', '中证红利波动率-估值调整']
dfPlot(comp)

# 计算估值的权重
daily_target_wts2_1 = cal_risk_target_wts(asset_risk_adjusted, risk_target=0.04, stock_limit=1)
dfPlot(daily_target_wts2_1)

# 测试含有货币基金的方案, 虽然含有货币基金但是没有加入货币基金相关信息
daily_target_wts3_1 = cal_risk_target_wts_wmf(asset_risk, risk_target=0.04, stock_limit=1)
dfPlot(daily_target_wts3_1)

daily_target_wts3_2 = cal_risk_target_wts_wmf(asset_risk, risk_target=0.04, stock_limit=0.2)
dfPlot(daily_target_wts3_2)

# ---------------------- 测试风险预算 ---------------------------- #
daily_target_wts4_1 = cal_risk_parity_wts(ret1, chgDates, risk_budget=[3, 1], stock_limit=1)
dfPlot(daily_target_wts4_1)

daily_target_wts4_2 = cal_risk_parity_wts(ret1, chgDates, risk_budget=[3, 1], stock_limit=0.2)
dfPlot(daily_target_wts4_2)

daily_target_wts5_1 = cal_risk_parity_wts_wmf(ret1, chgDates, risk_budget=[3, 1], stock_limit=1)
dfPlot(daily_target_wts5_1)

daily_target_wts5_2 = cal_risk_parity_wts_wmf(ret1, chgDates, risk_budget=[3, 1], stock_limit=0.2)
dfPlot(daily_target_wts5_2)


# 1. 风险预算模型0.04
print(daily_target_wts1.tail(3))

# 2. 低风险短融
print(daily_target_wts5_2.tail(3))

# -------------------------------------------------------------------------

# 回测参数配置并保留
daily_target_wts5_1 = daily_target_wts5_1.rename(columns={'货币基金': '景安短融'})

bkt_config = {
	'chgDates': quarter_list,
	'weights': daily_target_wts5_1,
	'trade_info': tradeInfo[daily_target_wts5_1.columns.tolist()]
}

pkl_read_write(data_path + 'bkt_config_wmf.pkl', 'write', bkt_config)



