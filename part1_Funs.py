# 导包
import pickle
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import logging
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
from arch import arch_model
from datetime import datetime, timedelta

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA
import statsmodels.api as sm
import scipy.stats as scs



# -------------------------------------- Funs: Tools ---------------------------------- #

# --- .pkl文件读写函数
def pkl_read_write(filename, flag, input_data=None):
	'''
	:param filename: 读写文件名
	:param flag: 读/写
	:param input_data: 写入数据
	'''
	if flag == 'read':
		try:
			pkl_file = open(filename, 'rb')
			data = pickle.load(pkl_file)
			pkl_file.close()
			return data
		except:
			logging.warning(filename + ' does not exist in current path')
			return DataFrame()
	elif flag == 'write':
		pkl_file = open(filename, 'wb')
		pickle.dump(input_data, pkl_file)
		pkl_file.close()
		return True

# --- 日志函数
def LogConfig(filename):

	user_path = '../log/' + datetime.today().strftime('%Y%m%d')
	folder = os.path.exists(user_path)
	if not folder:
		os.makedirs(user_path)

	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s %(levelname) -2s: %(message)s',
		filename=user_path + '/' + filename,
		filemode='w'
	)

	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s %(levelname) -8s: %(message)s')
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)

# --- 创建文件夹路径
def checkdir(pth):
	'''
	:param pth: 路径（绝对路径和相对路径）
	:return: 返回该路径
	'''
	user_path = pth
	folder = os.path.exists(user_path)
	if not folder:
		os.makedirs(user_path)

	return user_path


def dfPlot(dat, figsize=(10, 6), title_str=[], num=8):

	'''
	:param dat: a dataframe to draw
	:param title_str: title
	:return: figure
	'''

	dat = DataFrame(dat)
	XTicks = Series(np.linspace(0, dat.shape[0] - 1, num)).astype('int').tolist()

	dat.plot(figsize=figsize)
	plt.xticks(XTicks, list(dat.index[XTicks]))
	if isinstance(title_str, str):
		plt.title(title_str)
	plt.grid()
	plt.show()

def dfPlotSave(dat, path_filename, title_str=[], num=8):

	'''
	:param dat: a dataframe to draw
	:param title_str: title
	:return: figure
	'''


	dat = DataFrame(dat)
	XTicks = Series(np.linspace(0, dat.shape[0] - 1, num)).astype('int').tolist()

	plt.figure(figsize=(10, 6))
	plt.plot(dat)
	plt.xticks(XTicks, list(dat.index[XTicks]))
	if isinstance(title_str, str):
		plt.title(title_str)

	plt.grid()
	plt.legend(dat.columns.tolist())
	plt.savefig(path_filename + '.png')
	plt.close()


# 绘制利差图
def diffDensityPlot(mu, var, title, path):

	x = np.arange(-4 * np.sqrt(var) + mu, 4 * np.sqrt(var) + mu, 1)
	y = norm.pdf(x, mu, np.sqrt(var))

	# 确定刻度线和标点
	if int(mu / 3) < 10:
		threeValues = [-20, 0, int(2 * mu / 3)]
	else:
		threeValues = [0, int(mu / 30) * 10, int(2 * mu / 30) * 10]

	probs = norm.cdf(threeValues + [mu], mu, np.sqrt(var)).round(2) * 100

	pdfHeights = norm.pdf(threeValues + [mu], mu, np.sqrt(var))
	heightDelta = (pdfHeights[-1] - pdfHeights[0]) / 3
	threeHeights = [pdfHeights[0] + heightDelta, pdfHeights[1] + heightDelta, pdfHeights[2] + heightDelta]


	plt.figure(figsize=(10, 5))
	plt.plot(x, y)

	for i in range(len(threeValues)):
		plt.plot([threeValues[i]] * 2, [0, threeHeights[i]], 'k')
		plt.text(threeValues[i], threeHeights[i], repr(probs[i])[:4] + '%', color='r')
		plt.text(threeValues[i] + 1, 0.00, repr(threeValues[i]))

	plt.title(title + ': $\mu$=%.1f, $\sigma$=%.1f ' % (mu, np.sqrt(var)))
	plt.xlabel('x(BP)')
	plt.ylabel('Probability density')
	plt.grid()
	# plt.show()

	plt.savefig(path + '/' + title + '.png')
	plt.close()

	return True


def dfDoubleYPlot(dat, num=8):

	colnames = dat.columns.tolist()
	fig = plt.figure()
	ax = fig.add_subplot(111)

	lns1 = ax.plot(range(dat.shape[0]), dat.ix[:, 0], 'b-', label=colnames[0])

	ax2 = ax.twinx()
	lns2 = ax2.plot(range(dat.shape[0]), dat.ix[:, -1], 'r-', label=colnames[-1])

	# added these three lines
	lns = lns1 + lns2
	labs = [l.get_label() for l in lns]
	ax.legend(lns, labs)

	ax.grid()
	ax.set_xlabel('Date')
	ax.set_ylabel(colnames[0])
	ax2.set_ylabel(colnames[-1])

	XTicks = Series(np.linspace(0, dat.shape[0] - 1, num)).astype('int').tolist()
	ax.set_xticks(XTicks)
	ax.set_xticklabels(list(dat.index[XTicks]))
	plt.show()

	return True

def dfDoubleYPlotSave(dat, path_filename, num=8):

	colnames = dat.columns.tolist()

	fig = plt.figure(figsize=(10, 6))
	ax = fig.add_subplot(111)

	lns1 = ax.plot(range(dat.shape[0]), dat.ix[:, 0], 'b-', label=colnames[0])

	ax2 = ax.twinx()
	lns2 = ax2.plot(range(dat.shape[0]), dat.ix[:, -1], 'r-', label=colnames[-1])

	# added these three lines
	lns = lns1 + lns2
	labs = [l.get_label() for l in lns]
	ax.legend(lns, labs, loc=0)

	ax.grid()
	ax.set_xlabel('Date')
	ax.set_ylabel(colnames[0])
	ax2.set_ylabel(colnames[-1])

	XTicks = Series(np.linspace(0, dat.shape[0] - 1, num)).astype('int').tolist()
	ax.set_xticks(XTicks)
	ax.set_xticklabels(list(dat.index[XTicks]))

	plt.savefig(path_filename + '.png')
	plt.close()

	return True



def get_wsd(w, varCodes, indicators,  varNames, start_dt, end_dt, Adj='', Period='D'):

	res = {
		'Data': None,
		'ErrorCode': 0,
		'Message': ''
	}

	if isinstance(indicators, str):        # Ben @20190510  把字符型Indicators转成list
		indicators = [indicators]

	if (len(varCodes) > 1) & (len(indicators) > 1):
		res['Message'] = 'varCodes and indicators are both Mulit-dimensional, Please use Loop!'
		print(res['Message'])
		dat = DataFrame()

	if w.isconnected():
		if len(Adj) > 0:
			rawData = w.wsd(varCodes, indicators, start_dt, end_dt, 'Period=' + Period, 'PriceAdj=' + Adj)
		else:
			rawData = w.wsd(varCodes, indicators, start_dt, end_dt, 'Period='+Period)
		res['ErrorCode'] = rawData.ErrorCode

		if ~rawData.ErrorCode:
			actionDates = list(Series(rawData.Times).apply(lambda c: c.strftime('%Y%m%d')))
			dat = DataFrame(rawData.Data, columns=actionDates, index=varNames).T
		else:
			res['Message'] = 'Fetching Data Error '
			print(res['Message'])
			dat = DataFrame()
	else:
		res['Message'] = 'Wind is not Connected!'
		print(res['Message'])
		dat = DataFrame()

	res['Data'] = dat

	return res


def get_edb(w, varCodes, varNames, start_dt, end_dt):

	res = {
		'Data': None,
		'ErrorCode': 0,
		'Message': ''
	}

	if w.isconnected():
		rawData = w.edb(varCodes, start_dt, end_dt)    # 去掉用前值填充 "Fill=Previous"   Ben @20190510
		res['ErrorCode'] = rawData.ErrorCode

		if ~rawData.ErrorCode:
			actionDates = list(Series(rawData.Times).apply(lambda c: c.strftime('%Y%m%d')))
			if (len(actionDates) > 1):
				dat = DataFrame(rawData.Data, columns=actionDates, index=varNames).T
			else:
				dat = DataFrame(rawData.Data, index=actionDates, columns=varNames)
				# 注意只有一个数据的时候，数据结构，可能不能转置，还需要check   Ben @20190123
		else:
			res['Message'] = 'Wind API Fetching Data Error '
			print(res['Message'])
			dat = DataFrame()

	else:
		res['Message'] = 'Wind is not Connected!'
		print(res['Message'])
		dat = DataFrame()

	res['Data'] = dat

	return res



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


# -------------------------------------- Funs: VaR Related ---------------------------------- #
# --- 三种方法计算VaR
def estimateVaR(dat, k, probs):
	'''
	:param dat: 到期收益率数据，Series格式，单位是百分位  注意去掉前端NaN
	:param k: step k 差分
	:param probs: 估计VaR的概率值
	:return: 返回三种方法估计的VaR取值
	'''

	if np.any(dat.isnull()):
		print('Data Exists NaN!')
		return []
	# -+-
	# plt.figure()
	# plt.hist(dat, bins=100)
	# plt.title('Data Histogram')
	# plt.show()
	# -+-

	# step k 差分值分布
	diff = Series(dat[k:, ].values - dat[:(-k), ].values, index=dat[k:, ].index)

	diff = diff * 100  # 转成BP

	# -+-
	# k阶差分分布
	# plt.figure()
	# plt.hist(diff, bins=100, density=True)
	# plt.title('DIFF (' + repr(k) + ') Histogram')
	# plt.ylabel('Freq')
	# -+-

	t = np.linspace(diff.min(), diff.max(), 1000)

	# kernel density of residuals
	kde = plt.mlab.GaussianKDE(diff)

	# Method 1: 通过正太分布sigma计算
	sigma_1 = diff.std()
	VaR_Norm = []
	VaR_index = []
	for prob in probs:
		if prob < 1:
			VaR_Norm.append(norm.ppf(prob) * sigma_1)
		else:
			VaR_Norm.append(diff.max())
		VaR_index.append('P = ' + repr(prob))

	VaR_Norm = DataFrame(VaR_Norm, index=VaR_index, columns=['Norm'])

	# Method 2: 通过核函数计算
	kdeValue = kde(t)
	kdeCumValue = (t[1] - t[0]) * kdeValue.cumsum()

	VaR_Kde = []
	for prob in probs:
		if prob < 1:
			tmp = t[np.where(np.abs(kdeCumValue - prob) == np.abs(kdeCumValue - prob).min())[0]][0]
			VaR_Kde.append(tmp)
		else:
			VaR_Kde.append(diff.max())

	VaR_Kde = DataFrame(VaR_Kde, index=VaR_index, columns=['Kde'])

	# Method 3: 通过数据quantile计算
	s = diff.sort_values()
	qts = s.rank() / len(s)

	VaR_Qts = []
	for prob in probs:
		if prob < 1:
			tmp = s[np.where(np.abs(qts - prob) == np.abs(qts - prob).min())[0]][0]
			VaR_Qts.append(tmp)
		else:
			VaR_Qts.append(diff.max())

	VaR_Qts = DataFrame(VaR_Qts, index=VaR_index, columns=['Qts'])

	# 合并三种估计方法
	VaR_Res = pd.concat([VaR_Norm, VaR_Kde, VaR_Qts], axis=1)

	return VaR_Res

# --- 计算区间内VaR
def estimatePeriodVaR(dat, period, probs):
	'''
	:param dat: 到期收益率数据，Series格式，单位是百分位  注意去掉前端NaN
	:param period: T1~TN日
	:param probs: 置信概率
	:return: 返回T1~TN日的VaR最大值
	'''

	DVaR = DataFrame()
	for i in range(period):
		tmp = estimateVaR(dat, i+1, probs)
		tmp = DataFrame(tmp.max(axis=1), columns=['T' + repr(i+1)])
		DVaR = pd.concat([DVaR, tmp], axis=1)

	VaR_Res = DVaR.max(axis=1)

	return VaR_Res


# --- 利用GARCH得到的异方差计算VaR
def GARCH_VaRCalculation(garch_std, probs, steps):
	'''
	:param garch_std: GARCH模型计算得到的异方差
	:param probs: 置信概率
	:param steps: 预测步数
	:return: 预测步数对应的VaR
	'''

	res = DataFrame()

	for step in steps:
		tmp = []
		tmp_index = []
		for prob in probs:
			tmp.append(norm.ppf(prob) * garch_std * np.sqrt(step))  # VaR(k) = sqrt(k) * VaR
			tmp_index.append('P = ' + repr(prob))

		tmp = DataFrame(tmp, index=tmp_index, columns=['Step_' + repr(step)])
		res = pd.concat([res, tmp], axis=1)

	return res

# --- 利用GARCH(1,1)估算异方差即波动率
def GARCH_VaR(dat, probs, steps):
	'''
	:param dat: 到期收益率数据，Series格式，单位是百分位  注意去掉前端NaN
	:param probs: 置信概率
	:param steps: 预测步数
	:return: VaR
	'''

	k = 1
	diff = Series(dat[k:, ].values - dat[:(-k), ].values, index=dat[k:, ].index)
	diff = diff * 100  # 转成BP

	# 1. 拟合GARCH(1,1)模型
	am = arch_model(diff)
	res = am.fit()

	# 2. 模型预测
	forcast = res.forecast(horizon=5)
	pred_var = forcast.variance.ix[-1, :]

	pred_std_h1 = pred_var[0] ** (1/2)
	res = GARCH_VaRCalculation(pred_std_h1, probs, steps)

	return res


def GARCH_Prediction(dat, path, filename):

	k = 1
	diff = Series(dat[k:, ].values - dat[:(-k), ].values, index=dat[k:, ].index)
	diff = diff * 100  # 转成BP

	# 1. 拟合GARCH(1,1)模型
	am = arch_model(diff)
	res = am.fit()

	# 2. 模型预测
	forcast = res.forecast(horizon=10)
	pred_var = forcast.variance.ix[-1, :]

	pred_conditional_volatility = Series(len(res.conditional_volatility) * [np.nan],
										 index=res.conditional_volatility.index)
	pred_var.index = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10']
	pred_conditional_volatility = pd.concat([pred_conditional_volatility, pred_var ** (1 / 2)])

	XTicks = Series(np.linspace(0, diff[-240:].shape[0] - 1, 8)).astype('int').tolist()

	plt.figure(figsize=(10, 6))
	plt.plot(diff[-240:], label='diff_data')
	plt.plot(range(240), res.conditional_volatility[-240:], label='conditional_volatility')
	plt.plot(range(250), pred_conditional_volatility[-250:], 'r--', label='pred_volatility')

	# plt.plot(x,vol_pre,'.r',label='predict_volatility')
	plt.legend(loc=0)
	plt.xticks(XTicks, list(diff[-240:].index[XTicks]))
	plt.grid()
	plt.title(filename + ' Volatility')
	# plt.show()

	plt.savefig(path + filename + '.png')
	plt.close()

	return res, pred_var


def diffPlot(dat, path, multiplier=100):
	'''
	:param dat: dat is nx2 dataframe, the benchmark is at the end
	:param path: path to save image
	:return:
	'''

	colnames = dat.columns.tolist()
	filename = path + colnames[0] + '_' + colnames[1]
	# plot difference
	datDiff = (dat.ix[:, 0] - dat.ix[:, 1]) * multiplier

	XTicks = Series(np.linspace(0, dat.shape[0] - 1, 8)).astype('int').tolist()

	plt.figure(figsize=(10, 6))

	plt.subplot(211)
	plt.plot(dat)
	plt.xticks(XTicks, list(dat.index[XTicks]))
	plt.grid()
	plt.title(colnames[0] + ', ' + colnames[1])
	plt.legend(colnames)

	plt.subplot(212)
	plt.bar(range(len(datDiff)), datDiff)
	plt.xticks(XTicks, list(datDiff.index[XTicks]))
	plt.legend([colnames[0] + ' - ' + colnames[1]])

	plt.savefig(filename + '.png')
	plt.close()

	return True


def tsPlot(y, lags=30):

	if not isinstance(y, pd.Series):
		y = Series(y)

	XTicks = Series(np.linspace(0, y.shape[0] - 1, 5)).astype('int').tolist()

	figsize=(10, 8)
	fig = plt.figure(figsize=figsize)
	layout = (3, 2)

	ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
	acf_ax = plt.subplot2grid(layout, (1, 0))
	pacf_ax = plt.subplot2grid(layout, (1, 1))
	qq_ax = plt.subplot2grid(layout, (2, 0))
	pp_ax = plt.subplot2grid(layout, (2, 1))

	y.plot(ax=ts_ax)
	ts_ax.grid()
	ts_ax.set_xticks(XTicks)
	ts_ax.set_xticklabels(list(y.index[XTicks]))

	ts_ax.set_title('Time Series Analysis Plots')

	plot_acf(y, lags=lags, ax=acf_ax)
	plot_pacf(y, lags=lags, ax=pacf_ax)

	sm.qqplot(y, line='s', ax=qq_ax)#QQ图检验是否是正太分布
	qq_ax.set_title('QQ Plot')
	qq_ax.grid()
	scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)
	pp_ax.grid()

	plt.tight_layout()
	plt.show()

	return True


# 提取16年的日历日数据
def getEveryDay(begin_date,end_date):
	date_list = []
	begin_date = datetime.strptime(begin_date, "%Y%m%d")
	end_date = datetime.strptime(end_date,"%Y%m%d")
	while begin_date <= end_date:
		date_str = begin_date.strftime("%Y%m%d")
		date_list.append(date_str)
		begin_date += timedelta(days=1)
	return date_list


# 计算分位数并画图
def diffHistPct(dat, curValue, pct, title, filename):

	plt.figure(figsize=(10,5))
	(n, bins, patches) = plt.hist(dat, bins=50, density=True)
	plt.plot([curValue] * 2, [0, n.max()], 'r')
	plt.text(curValue, n.max(), repr(round(curValue))[:4] + 'BP (' + repr(pct * 100)[:4] + '%' + ')')
	plt.title(title)
	plt.xlabel('x(BP)')
	plt.ylabel('Empirical Density')
	plt.grid()

	plt.savefig(filename + '.png')
	plt.close()

	return True



def tsARMAPredictPlotSave(dat, path_filename, order_k=5, num=8):

	if len(dat) > 300:
		tmp = dat[-300:].copy()
	else:
		tmp = dat.copy()

	arma = ARMA(tmp, order=(order_k, 0)).fit(disp=-1)
	predict = arma.predict(start=0, end=len(tmp)+4)

	# 画图起始点
	start = 50
	XTicks = Series(np.linspace(start, len(tmp) - 1, num)).astype('int').tolist()

	plt.figure(figsize=(10, 6))
	plt.plot(range(start, len(tmp)), tmp[start:], 'k', label=tmp.name)
	plt.plot(predict[len(tmp):], 'r--', label=tmp.name + ' Predict')
	plt.legend(loc='best')
	plt.xticks(XTicks, list(tmp.index[XTicks]))

	plt.title(tmp.name + ' Prediction')
	plt.grid()

	plt.savefig(path_filename + '.png')
	plt.close()


def tsDataFrameARMAPredictPlotSave(dat, path_filename, order_k=5, num=8):

	# 去掉nan
	dat = dat.dropna(axis=0, how='any')

	if dat.shape[0] > 300:
		tmp = dat[-300:].copy()
	elif (dat.shape[0] < 300) & (dat.shape[0] > 30):
		tmp = dat.copy()
	else:
		print('Data length is not Enough! ')
		return []

	tmp = DataFrame(tmp)
	predict = DataFrame()
	for col in tmp.columns.tolist():
		arma = ARMA(tmp[col], order=(order_k, 0)).fit(disp=-1)
		predict_tmp = arma.predict(start=0, end=tmp.shape[0]+4)
		predict = pd.concat([predict, predict_tmp], axis=1)

	predict.columns = tmp.columns.tolist()

	# 画图起始点
	start = 100 if tmp.shape[0] > 200 else 30


	XTicks = Series(np.linspace(start, tmp.shape[0] - 1, num)).astype('int').tolist()
	colnames = tmp.columns.tolist()
	colors = ['g', 'b', 'c', 'm', 'k', 'r']

	plt.figure(figsize=(10, 6))

	for i in range(0, len(colnames)):
		plt.plot(range(start, tmp.shape[0]), tmp.ix[start:, colnames[i]], colors[i % 6])

	for i in range(0, len(colnames)):
		plt.plot(predict.ix[tmp.shape[0]:, colnames[i]], colors[i % 6] + '--')

	plt.legend(colnames)
	plt.xticks(XTicks, list(tmp.index[XTicks]))

	plt.grid()

	plt.savefig(path_filename + '.png')
	plt.close()






