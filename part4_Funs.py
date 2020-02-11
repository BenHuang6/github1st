# from Funs_HXH3 import *
from part1_Funs import *
import scipy.optimize as sco

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  	# 指定默认字体 SimHei为黑体
mpl.rcParams['axes.unicode_minus'] = False  	# 用来正常显示负号

def get_tradingdate(w):

	res = {
		'Data': None,
		'ErrorCode': 0,
		'message': ''
	}

	if w.isconnected():
		rawData_D = w.tdays('1991-01-01', '2021-01-01', 'Period=D')
		rawData_W = w.tdays('1991-01-01', '2021-01-01', 'Period=W')
		rawData_M = w.tdays('1991-01-01', '2021-01-01', 'Period=M')
		rawData_Q = w.tdays('1991-01-01', '2021-01-01', 'Period=Q')

		res['ErrorCode'] = rawData_D.ErrorCode | rawData_W.ErrorCode | rawData_M.ErrorCode | rawData_Q.ErrorCode

		if ~res['ErrorCode']:
			actionDates = list(Series(rawData_D.Times).apply(lambda c: c.strftime('%Y%m%d')))
			dat = DataFrame(rawData_D.Data, columns=actionDates, index=['Date']).T
			dat['Date'] = dat['Date'].apply(lambda x: x.strftime("%Y%m%d"))

			Weeks = list((DataFrame(rawData_W.Data, index=['Date']).T)['Date'].apply(lambda x: x.strftime("%Y%m%d")))
			Months = list((DataFrame(rawData_M.Data, index=['Date']).T)['Date'].apply(lambda x: x.strftime("%Y%m%d")))
			Quarters = list((DataFrame(rawData_Q.Data, index=['Date']).T)['Date'].apply(lambda x: x.strftime("%Y%m%d")))

			# 判断是否是月末   # 最后把整个系统这里，应该这个数据是已经存在的     ！！！

			dat['isWeek'] = 0
			dat['isMonth'] = 0
			dat['isQuarter'] = 0

			dat.ix[dat['Date'].isin(Weeks), 'isWeek'] = 1
			dat.ix[dat['Date'].isin(Months), 'isMonth'] = 1
			dat.ix[dat['Date'].isin(Quarters), 'isQuarter'] = 1

		else:
			res['message'] = 'Fetching Data Error '
			print(res['message'])
			dat = DataFrame()
	else:
		res['message'] = 'Wind is not connected!'
		print(res['message'])
		dat = DataFrame()

	res['Data'] = dat

	return res


def getChgDates(trading_date, Period='Month'):

	column = 'is' + Period
	trading_date = trading_date.ix[:, ['Date', column]].copy()

	# 避免后面出现out of index range
	if trading_date.iloc[-1, -1] == 1:
		trading_date = trading_date.iloc[:-1, :]

	actionDates = trading_date['Date'].tolist()
	Period_End = trading_date.ix[trading_date[column] == 1, :].index.tolist()
	Period_Start = [actionDates[actionDates.index(td)+1] for td in Period_End]
	Period_Start = [actionDates[0]] + Period_Start

	chgDates = Period_Start

	return chgDates





def weightStackPlot(weights):

	x = np.array(range(0, weights.shape[0]))
	y = weights.values.astype('float32')

	XTicks = Series(np.linspace(0, weights.shape[0] - 1, 5)).astype('int').tolist()

	plt.stackplot(x, y.T)
	plt.xticks(XTicks, list(weights.index[XTicks]))
	plt.legend(weights.columns.tolist())
	plt.show()




# 计算换手率，该换手率计算略微复杂，简单可以不考虑净值
def calTurnOverBasedWeights(wts, tickerRet, tickerNV):
	'''
	:param wts: weights, a dataframe with chgDates as index
	:param tickerRet: monthly returns with chgDates as index
	:param tickerNV: netvalue, a dataframe with chgDates as index
	:return:
	'''

	if (wts.shape[0] != tickerNV.shape[0]) or (wts.shape[0] != tickerRet.shape[0]):
		print('Length of Weights, NetValue and Return at chgDates Do NOT Match!')
		return []

	dat = pd.concat([wts, tickerNV], axis=1)

	tmp1 = dat.ix[:-1, :-1].values * np.tile((dat.ix[:-1, -1].values)[:, None], (1, wts.shape[1]))
	tmp2 = dat.ix[1:, :-1].values * np.tile((dat.ix[1:, -1].values)[:, None], (1, wts.shape[1]))
	delta_wts = np.abs(tmp1 * (1 + tickerRet.ix[1:, :].values) - tmp2).sum(axis=1)

	turnover = 2 * delta_wts[:, None] / (tickerNV.ix[:-1, :].values + tickerNV.ix[1:, :].values)
	turnover_df = DataFrame(turnover, index=wts.index.tolist()[1:], columns=['TurnOver'])
	return turnover_df



def TDRet2PeriodRet(ret, chgDates):
	'''

	:param ret: daily returns
	:param chgDates: chgDates
	:return: period returns in each period interval
	'''

	periodRet = DataFrame(index=chgDates, columns=ret.columns)
	actionDates = ret.index.tolist()
	for chgDate in chgDates:
		if chgDates.index(chgDate) == 0:
			periodRet.ix[chgDate, :] = 0
		else:
			tmp = ret.ix[chgDates[chgDates.index(chgDate) - 1]:chgDate, :].copy()
			tmp.ix[0, :] = 0
			periodRet.ix[chgDate, :] = (((tmp + 1).cumprod()).ix[-1, :] - 1)

	return periodRet


def getRiskBudgetWeights(ret, chgDates, vol_bond, LBW_S=20, LBW_L=40):

	'''
	:param ret: 收益率序列
	:param chgDates: 调仓时间
	:param vol_bond: 波动率限制
	:param LBW_S: 短窗口
	:param LBW_L: 长窗口
	:return: 权重
	'''

	wts = DataFrame(0, index=chgDates, columns=['Stock', 'Bond'])
	risk = DataFrame(0, index=chgDates, columns=['Stock', 'Bond'])
	volatility = DataFrame(0, index=chgDates, columns=['Stock', 'Bond'])

	actionDates = ret.index.tolist()

	for chgDate in chgDates:

		# 计算历史波动率
		ind = actionDates.index(chgDate)
		if ind < LBW_S:
			vol = 0.5 * ret.ix[:ind, :].std() * np.sqrt(244) + \
				  0.5 * ret.ix[:ind, :].std() * np.sqrt(244)
		elif (ind >= LBW_S) & (ind < LBW_L):
			vol = 0.5 * ret.ix[(ind - LBW_S):ind, :].std() * np.sqrt(244) + \
				  0.5 * ret.ix[:ind, :].std() * np.sqrt(244)
		else:
			vol = 0.5 * ret.ix[(ind-LBW_S):ind, :].std() * np.sqrt(244) + \
				  0.5 * ret.ix[(ind-LBW_L):ind, :].std() * np.sqrt(244)

		# 通过波动率判断股债权重
		if vol['Stock'] <= vol_bond:
			wts.ix[chgDate, :] = [1, 0]   # 全仓股票
		elif (vol['Stock'] > vol_bond) & (vol['Bond'] >= vol_bond):
			# 暂定这个比例20:80
			wts.ix[chgDate, 'Stock'] = 0.2
			wts.ix[chgDate, 'Bond'] = 0.8
			logging.warning('Volatility of Bond >= 0.04')
		else:
			wts.ix[chgDate, 'Stock'] = (vol_bond - vol['Bond']) / (vol['Stock'] - vol['Bond'])
			wts.ix[chgDate, 'Bond'] = 1 - wts.ix[chgDate, 'Stock']

		# 这里其实没太大必要算risk, volatility
		risk.ix[chgDate, :] = wts.ix[chgDate, :] * vol
		volatility.ix[chgDate, :] = vol

	return wts, risk, volatility


def calTotSEofTRC(x, SIGMA):
	product = np.dot(SIGMA, x)
	TRC = x * product
	totSE = 0
	for i in range(0, SIGMA.shape[0]):
		for j in range(0, SIGMA.shape[0]):
			totSE = totSE + 1e12 * (TRC[i] - TRC[j]) ** 2
			# The original TRC is too small, which easily makes the optimization terminate. So multiple 1e12
	return totSE

# optimize risk parity
def optRiskParityByTotSE(SIGMA, calTotSEofTRC):

	fun = lambda x: calTotSEofTRC(x, SIGMA)
	x0 = (1 / np.sqrt(np.diag(SIGMA))) / (1 / np.sqrt(np.diag(SIGMA))).sum()   # 1 / np.sqrt(np.diag(SIGMA))  # m * [1. / m, ]
	m = SIGMA.shape[0]
	cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
	bnds = tuple((0, 1) for x in range(m))
	opts = sco.minimize(fun, x0, method='SLSQP', bounds=bnds, constraints=cons)
	wts = opts.x
	# TRC = wts * np.dot(SIGMA, wts)
	return Series(wts, index=list(SIGMA.columns), name='weights')

def getRiskParityWeights(ret, chgDates, LBW=60):

	wts = DataFrame(0, index=chgDates, columns=['Stock', 'Bond'])
	risk = DataFrame(0, index=chgDates, columns=['Stock', 'Bond'])
	volatility = DataFrame(0, index=chgDates, columns=['Stock', 'Bond'])
	actionDates = ret.index.tolist()

	for chgDate in chgDates:

		# 计算历史波动率
		ind = actionDates.index(chgDate)
		if ind < LBW:
			SIGMA = ret.ix[:ind, :].cov()
		else:
			SIGMA = ret.ix[(ind - LBW):ind, :].cov()

		wts.ix[chgDate, :] = optRiskParityByTotSE(SIGMA)
		vol = ret.ix[(ind - 60):ind, :].std() * np.sqrt(244)
		risk.ix[chgDate, :] = wts.ix[chgDate, :] * vol     # 这里是有点问题,两者加起来不是总的组合风险  Ben @20190703
		volatility.ix[chgDate, :] = vol

	return wts, risk, volatility


def get28Weights(ret, chgDates):

	wts = DataFrame(0, index=chgDates, columns=['Stock', 'Bond'])
	risk = DataFrame(0, index=chgDates, columns=['Stock', 'Bond'])
	volatility = DataFrame(0, index=chgDates, columns=['Stock', 'Bond'])
	actionDates = ret.index.tolist()

	for chgDate in chgDates:

		# 计算历史波动率
		ind = actionDates.index(chgDate)
		vol = ret.ix[(ind - 60):ind, :].std() * np.sqrt(244)

		wts.ix[chgDate, 'Stock'] = 0.2
		wts.ix[chgDate, 'Bond'] = 0.8

		risk.ix[chgDate, :] = wts.ix[chgDate, :] * vol
		volatility.ix[chgDate, :] = vol

	return wts, risk, volatility


# 计算年度收益

def NV2YearlyReturn(dat, start_year, end_year):

	if isinstance(dat, Series):
		dat = DataFrame(dat)

	index_list = list(Series(np.arange(start_year, end_year + 1)).astype('str'))

	yearly_ret = DataFrame(0, columns=dat.columns, index=index_list)

	for year in yearly_ret.index.tolist():
		dat_seg = dat.ix[(year+'0101'): (year+'1231'), :]
		yearly_ret.ix[year, :] = (dat_seg.ix[-1, :] / dat_seg.ix[0, :] - 1)

	return yearly_ret


def dfBarPlot(dat,
			  figsize=(10, 6),
			  title_str='',
			  ylabel='',
			  ygrid=False,
			  show=True,
			  save_path_name=''):

	if dat.shape[1] > 6:
		logging.warning('data columns > 6, please separate.')
		return None

	dat_len = dat.shape[0]
	colors = ['lightskyblue', 'yellowgreen', 'b', 'm', 'g', 'k']  	# 最多6个
	columns = dat.columns.tolist()

	XTicks = Series(np.arange(0, dat_len)).astype('int').tolist()

	plt.figure(figsize=figsize)
	for i in range(len(columns)):
		plt.bar(np.array(range(dat_len)) + (1-0.2) / len(columns) * (i - int(len(columns) / 2)), dat.iloc[:, i],
				width=(1-0.2) / len(columns) + 0.01, facecolor=colors[i], edgecolor='white', label=columns[i], lw=1)

	plt.xticks(XTicks, list(dat.index[XTicks]))
	plt.legend(loc="upper left")   # label的位置在左上，没有这句会找不到label去哪了

	if ygrid:
		plt.grid(axis='y', linestyle='-.')
	plt.title(title_str)
	plt.ylabel(ylabel)

	if show:
		plt.show()
	else:
		plt.savefig(save_path_name + '.png')
		plt.close()


def calRollingSharpe2(dat, rollingPeriod=60, fct=244):

	dat = DataFrame(dat)
	rolling_ann_mean = dat.rolling(window=rollingPeriod).mean() * fct
	rolling_ann_std = dat.rolling(window=rollingPeriod).std() * (fct ** (1/2))

	rolling_sharpe = rolling_ann_mean / rolling_ann_std
	return rolling_sharpe


def StockBondAlocate(stock_columns, bond_columns, rawData,
					 start_dt, end_dt, trading_date, Period='M'):

	columns = stock_columns + bond_columns

	dat = rawData[columns].copy()
	del rawData

	# 去掉nan
	dat = dat.dropna(how='any')

	# 获取调仓时点
	if Period == 'M':
		Period = 'Month'
	elif Period == 'Q':
		Period = 'Quarter'
	trading_date = trading_date.ix[(trading_date['Date'] >= start_dt) & (trading_date['Date'] <= end_dt), :]

	# 防止start_dt, end_dt为非交易日
	start_dt = trading_date.index.tolist()[0]
	end_dt = trading_date.index.tolist()[-1]

	chgDates = getChgDates(trading_date, Period=Period)

	# 方法及参数设置
	methods = ['risk_budget', 'risk_parity', 'allocate_28']
	vol_bond = 0.04

	# 获取权重和回测
	k = 0
	tot_num = len(stock_columns) * len(bond_columns) * len(methods)
	strategyPerformance = dict()

	for method in methods:

		strategyNavs = DataFrame()
		strategyIndicators = DataFrame()
		strategyWeights = DataFrame()
		strategyRisks = DataFrame()
		strategyVolatility = DataFrame()

		for i in range(len(stock_columns)):
			for j in range(len(bond_columns)):

				k = k + 1
				logging.info(repr(k) + '/' + repr(tot_num) + ' ' + method + ' ' + stock_columns[i] + ' + ' + bond_columns[j])

				StockBondPair = {
					'Stock': stock_columns[i],
					'Bond': bond_columns[j]
				}

				tmpDat = dat[list(StockBondPair.values())].rename(
					columns={StockBondPair['Stock']: 'Stock', StockBondPair['Bond']: 'Bond'})
				tmpRet = NV2Ret(tmpDat)

				# method 1: 风险预算, 用20day+40day合计volatility
				if method == 'risk_budget':
					if Period == 'Month':
						LBW_S = 20
						LBW_L = 40
					elif Period == 'Quarter':
						LBW_S = 20  #  60
						LBW_L = 40  # 120
					wts, risk, volatility = getRiskBudgetWeights(tmpRet, chgDates, vol_bond, LBW_S=LBW_S, LBW_L=LBW_L)

				elif method == 'risk_parity':
					LBW = 60
					wts, risk, volatility = getRiskParityWeights(tmpRet, chgDates, LBW=LBW)

				elif method == 'allocate_28':
					wts, risk, volatility = get28Weights(tmpRet, chgDates)

				# ------- 策略回测 ------- #
				config = {
						'chgDates': chgDates,
						'weights': wts,
						'totCost': 0
					}

				# -------不同策略，只需要输入不同权重和调仓时间
				tmp = dat[list(StockBondPair.values())].rename(columns={StockBondPair['Stock']: 'Stock', StockBondPair['Bond']: 'Bond'})
				nav_input = tmp.ix[start_dt:end_dt, :] / tmp.ix[start_dt, :]

				if np.any(nav_input.isnull()):
					logging.error('nav_input Exists NaN, Please Check and re-run!')
				else:
					logging.info('nav_input DOES NOT exists NaN, continue to BackTest.')

				# 获取每个月的月收益
				tickerRet = TDRet2PeriodRet(NV2Ret(nav_input), chgDates).ix[wts.index.tolist(), :]

				res = simpleBKT(nav_input, config)

				# 计算净值
				tmpNav = NV2Ret(res['ret'], 2)
				tmpNav.columns = [StockBondPair['Stock'] + ' + ' + StockBondPair['Bond']]
				strategyNavs = pd.concat([strategyNavs, tmpNav], axis=1, sort=True)

				# 权重
				tmpWeights = res['weights']
				tmpWeights['Strategy'] = StockBondPair['Stock'] + ' + ' + StockBondPair['Bond']
				strategyWeights = pd.concat([strategyWeights, tmpWeights], sort=True)

				# 风险
				tmpRisks = risk
				tmpRisks['Strategy'] = StockBondPair['Stock'] + ' + ' + StockBondPair['Bond']
				strategyRisks = pd.concat([strategyRisks, tmpRisks], sort=True)

				# 资产波动率
				tmpVolatility = volatility
				tmpVolatility['Strategy'] = StockBondPair['Stock'] + ' + ' + StockBondPair['Bond']
				strategyVolatility = pd.concat([strategyVolatility, tmpVolatility], sort=True)

				# 指标
				tmpIndicators = res['indicators']
				tmpIndicators.columns = [StockBondPair['Stock'] + ' + ' + StockBondPair['Bond']]

				# 计算月平均换手率
				if Period == 'Month':
					fct = 12
				elif Period == 'Quarter':
					fct = 4

				tmpIndicators.ix['TurnOver', :] = \
					round(float(calTurnOverBasedWeights(wts, tickerRet, tmpNav.ix[wts.index.tolist(), :]).mean()), 3) * fct  # 年化换手率
				tmpIndicators.index = ['总收益率', '年化收益率', '年化波动率', '夏普比率', '最大回撤', 'calmar比率', '年化换手率']
				strategyIndicators = pd.concat([strategyIndicators, tmpIndicators], axis=1)

		strategyPerformance[method] = {
			'strategyNaVs': 		strategyNavs,
			'strategyWeights': 		strategyWeights,
			'strategyRisks': 		strategyRisks,
			'strategyVolatility':	strategyVolatility,
			'strategyIndicators': 	strategyIndicators,
			'chgDates': 			chgDates
		}

	return strategyPerformance





