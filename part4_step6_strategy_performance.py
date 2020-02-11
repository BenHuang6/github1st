from part4_Funs import *
from Funs_HXH3 import *
# LogConfig('part4_result_analysis.log')

plt.switch_backend('agg')   # 设置不显示图像
# plt.switch_backend('TkAgg')  # 默认可以显示图像

# --------------------------------------------------------
# 获取结果数据
data_path = '../data/'
result_path = '../result/'

bkt_result = pkl_read_write(result_path + 'bkt_result_wmf' + '.pkl', 'read')
navRecord = bkt_result['navRecord']
tradeRecord = bkt_result['tradeRecord']
portfolioRecord = bkt_result['portfolioRecord']

port_nav = DataFrame(Series(navRecord, name='port_nav'))

# port_nav = pd.read_excel('port_nav.xlsx', index_col=0)
# port_nav.index = port_nav.index.map(lambda x: str(x))

# ----------------------------------------- 打包成一个函数 -------------------------- #
# ---------  日收益特征 ---------- #
port_ret = nav2return(port_nav)
port_ret.columns = ['port_ret']
ret_char = daily_ret_characteristic(port_ret * 1e2)

# --------- 年度/月度收益 ---------- #
port_yearly_ret = nav2yearlyreturn(port_nav)
port_monthly_ret1 = nav2monthlyreturn(port_nav)
port_monthly_ret2 = nav2periodreturn(port_nav, period='Month')

# --------- 持有期收益 ------------ #
port_yearly_ret = cal_holding_return(port_nav)

# --------- 相对基准的超额收益和超额回撤 ----------- #

logging.info('Load rawData for analysis ...')
rawData = pkl_read_write(data_path + 'rawData.pkl', 'read')

columns = ['3-5年国开债指数']
dat = rawData[columns].copy()
del rawData

dat = dat.loc['20130101':'20191231', :]
bond_nav = dat / dat.iloc[0, :]

combo_nav = pd.concat([port_nav, bond_nav], axis=1, sort=True)
dfPlot(combo_nav)

# 计算超额收益
excess_ret = nav2return((nav2return(combo_nav.iloc[:, 0]) - nav2return(combo_nav.iloc[:, 1])), 2)
combo_nav['excess_ret'] = excess_ret
dfPlot(combo_nav)

# 计算超额收益最大回撤
relative_drawback = cal_relative_drawback(combo_nav.iloc[:, :2])
relative_drawback_max = cal_relative_maxdrawback(combo_nav.iloc[:, :2])
relative_drawback_max_yearly = cal_yearly_relative_drawback(combo_nav.iloc[:, :2])
relative_drawback_max_yearly = relative_drawback_max_yearly.rename(columns=
			{'startDate': '回撤开始日期', 'endDate': '回撤结束日期', 'drawback': '年度最大回撤'})

# 绘图
relative_drawback_1 = relative_drawback[['endDate', 'drawback']]
relative_drawback_1.index = relative_drawback_1['endDate'].tolist()
relative_drawback_1 = relative_drawback_1['drawback']
dfPlot(relative_drawback_1)

# ----------------- 下行风险 --------------- #
dvol = downside_vol(port_ret) * np.sqrt(244)

# ----------------- 年度波动率/年度最大回撤 ------------- #
port_nav3 = port_nav.copy()
port_nav3['port_nav3'] = port_nav['port_nav']
res = nav2yearlystats(port_nav3)


ann1 = strategyAnalysis()
res1 = ann1.Get_BasicIndictors(nav2return(port_nav))
ann1.Plot()


# ------------------------------- porfolioRecord数据 ---------------------------------- #
columns = ['中证红利全收益', '3-5年国开债指数', '景安短融']
columns = columns + ['现金']

actionDates = list(Series(list(portfolioRecord.keys())).sort_values())
portfolio_weights = DataFrame(0, index=actionDates, columns=columns)
for actionDate in actionDates:
	print(actionDate)
	tmp = portfolioRecord[actionDate]
	portfolio_weights.loc[actionDate, :] = \
		pd.concat([tmp.position['weight'] * (1 - tmp.cash / tmp.asset), Series(tmp.cash / tmp.asset, index=['现金'])])

dfPlot(portfolio_weights)

# 权重对比和更新
weightStackPlot(portfolio_weights)

# 计算换手率
tradeHist = DataFrame()
for tradeDay in list(tradeRecord.keys()):
	tradeHist = pd.concat([tradeHist, tradeRecord[tradeDay]], sort=True)

# 月度换手率, 计算s单边换手
tradeDates = tradeHist['Date'].unique().tolist()
turnover = Series(index=tradeDates)
tradeDates = tradeDates[1:]
for tradeDate in tradeDates:
	tmp_asset = portfolioRecord[tradeDate].asset
	tmp = tradeHist.loc[tradeHist['Date'] == tradeDate, :].copy()    # True/False索引用loc
	turnover[tradeDate] = tmp['tradeAmt'].sum() / 2 / tmp_asset

dfPlot(turnover)















