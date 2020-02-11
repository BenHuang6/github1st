from Funs_HXH3 import *
import sympy

from WindPy import *
w.start()

# 双Y轴有新函数

def dfDoubleYPlot_Revised(s1, s2, labels=['s1', 's2']):

    date_list = s1.index.tolist() + s2.index.tolist()
    date_list = Series(date_list).drop_duplicates().sort_values().tolist()

    x_range_s1 = [date_list.index(tmp) for tmp in s1.index.tolist()]
    x_range_s2 = [date_list.index(tmp) for tmp in s2.index.tolist()]

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    lns1 = ax1.plot(np.array(x_range_s1), s1.values, label=labels[0])
    ax1.set_ylabel(labels[0])
    ax1.set_title(labels[0] + " & " + labels[1])

    ax2 = ax1.twinx()
    lns2 = ax2.plot(np.array(x_range_s2), s2.values, 'r', label=labels[1])
    ax2.set_xlim([0, len(date_list)])
    ax2.set_ylabel(labels[1])


    # added these three lines
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)

    ax1.grid()
    ax1.set_xlabel('Date')

    XTicks = Series(np.linspace(0, len(date_list) - 1, 8)).astype('int').tolist()
    ax1.set_xticks(XTicks)
    ax1.set_xticklabels(list(Series(date_list)[XTicks]))

    plt.show()

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
	ax.legend(lns, labs, loc=0)

	ax.grid()
	ax.set_xlabel('Date')
	ax.set_ylabel(colnames[0])
	ax2.set_ylabel(colnames[-1])

	XTicks = Series(np.linspace(0, dat.shape[0] - 1, num)).astype('int').tolist()
	ax.set_xticks(XTicks)
	ax.set_xticklabels(list(dat.index[XTicks]))
	plt.show()

	return True


sigma1 = 1
sigma2 = 20
sigma_target = 4

# 目标风险

w1, w2 = sympy.symbols("w1 w2")
eq = [w1 + w2 - 1,
	  (w1 ** 2) * (sigma1 ** 2) + (w2 ** 2) * (sigma2 ** 2) - sigma_target ** 2]
var = [w1, w2]

res = sympy.solve(eq, var)

print(res)

w1_s, w2_s = (25/26 - np.sqrt(79)/26, 1/26 + np.sqrt(79)/26)
sigma_s_2 = (w1_s ** 2) * (sigma1 ** 2) + (w2_s ** 2) * (sigma2 ** 2)
sigma_s = np.sqrt(sigma_s_2)
print(sigma_s)

# 风险平价
w1, w2 = sympy.symbols("w1 w2")
eq = [w1 + w2 - 1,
	  (w1 ** 2) * (sigma1 ** 2) - (w2 ** 2) * (sigma2 ** 2)]
var = [w1, w2]

res = sympy.solve(eq, var)

print(res)


## ------------------ 目标波动率和风险预算模型 权重研究 -------------- #

r_sigma = np.linspace(4, 50, 100)
r_p = 1
r_t = 4

# 1.风险预算
w2_rp = 1 / (1 + r_sigma * np.sqrt(1 / r_p))

# 2. 目标波动率r_t=4
w2_rt_part1 = 1 / (1 + r_sigma ** 2)
w2_rt_part2 = np.sqrt((1 / (1 + r_sigma ** 2)) ** 2 + (r_t ** 2 - 1) / (1 + r_sigma ** 2))

w2_rt_1 = w2_rt_part1 + w2_rt_part2
w2_rt_2 = w2_rt_part1 - w2_rt_part2


# 画图
plt.figure(figsize=(8, 6))

plt.plot(r_sigma, w2_rp, label='w_rp')
plt.plot(r_sigma, w2_rt_1, label='w_rt')
plt.plot(r_sigma, w2_rt_part1, label='w_rt_part1')
plt.plot(r_sigma, w2_rt_part2, label='w_rt_part2')

plt.legend()
plt.grid()
plt.show()


# 目标波动率r_t=6

def cal_rt_rp_wts(r_t, r_p, r_sigma_max):

	r_sigma = np.linspace(r_t, r_sigma_max, 100)

	# 1.风险预算
	w2_rp = 1 / (1 + r_sigma * np.sqrt(1 / r_p))

	# 2. 目标波动率r_t=4
	w2_rt_part1 = 1 / (1 + r_sigma ** 2)
	w2_rt_part2 = np.sqrt((1 / (1 + r_sigma ** 2)) ** 2 + (r_t ** 2 - 1) / (1 + r_sigma ** 2))

	w2_rt = w2_rt_part1 + w2_rt_part2

	return r_sigma, w2_rt, w2_rp


r_sigma_4, w2_rt_4, w2_rp_4 = cal_rt_rp_wts(r_t=4, r_p=1, r_sigma_max=50)
r_sigma_6, w2_rt_6, w2_rp_6 = cal_rt_rp_wts(r_t=6, r_p=1, r_sigma_max=50)
r_sigma_8, w2_rt_8, w2_rp_8 = cal_rt_rp_wts(r_t=8, r_p=1, r_sigma_max=50)

# 画图
plt.figure(figsize=(8, 6))

plt.plot(r_sigma_4, w2_rt_4, label='w_rt_4')
plt.plot(r_sigma_4, w2_rp_4, label='w_rp_4')

plt.legend()
plt.grid()
plt.show()


# 比较不同波动率
plt.figure(figsize=(8, 6))

plt.plot(r_sigma_4, w2_rt_4, label='rt=4')
plt.plot(r_sigma_6, w2_rt_6, label='rt=6')
plt.plot(r_sigma_8, w2_rt_8, label='rt=8')

plt.legend()
plt.grid()
plt.show()

# 实际债和股票的波动率
data_path = '../data/'
dat = pkl_read_write(data_path + 'rawData.pkl', 'read')

columns = ['3-5年国开债指数', '中证100全收益']
dat = dat.loc['20120104':, columns]

dat = dat / dat.iloc[0, :]
dfPlot(dat)

dfDoubleYPlot(dat)


ret = nav2return(dat)
rolling_vol = ret.rolling(window=240).std() * np.sqrt(244)
dfPlot(rolling_vol)
dfPlot(rolling_vol['中证100全收益'])
dfPlot(rolling_vol['3-5年国开债指数'])


# 双Y轴
dfDoubleYPlot_Revised(rolling_vol['3-5年国开债指数'], rolling_vol['中证100全收益'],
					  labels=['3-5年国开债指数', '中证100全收益'])

# 计算比值
rolling_vol['r_sigma'] = rolling_vol['中证100全收益'] / rolling_vol['3-5年国开债指数']
dfPlot(rolling_vol['r_sigma'])

# 去掉空值
rolling_vol = rolling_vol.dropna(how='any')

#
r_sigma = rolling_vol['r_sigma'].copy()
r_p = 10
r_t = 4

# 1.风险预算
ws_rp = 1 / (1 + r_sigma * np.sqrt(1 / r_p))

# 2. 目标波动率r_t=4
ws_rt_part1 = 1 / (1 + r_sigma ** 2)
ws_rt_part2 = np.sqrt((1 / (1 + r_sigma ** 2)) ** 2 + (r_t ** 2 - 1) / (1 + r_sigma ** 2))

ws_rt = ws_rt_part1 + ws_rt_part2

# 比较不同波动率
XTicks = Series(np.linspace(0, ws_rt.shape[0] - 1, 10)).astype('int').tolist()

plt.figure(figsize=(8, 6))

plt.plot(ws_rp, label='ws_rp')
plt.plot(ws_rt, label='ws_rt')
plt.xticks(XTicks, list(ws_rt.index[XTicks]))
plt.legend()
plt.grid()
plt.show()


tmp = pd.concat([rolling_vol, ws_rp, ws_rt], axis=1)
tmp.columns = ['3-5年国开债指数', '中证100全收益', '股债波动率比值', '风险预算股票权重', '目标风险股票权重']
tmp.index = tmp.index.map(lambda x: datetime.strptime(x, '%Y%m%d'))
tmp = tmp.dropna(how='any')

result_path = '../result/'
tmp.to_excel(result_path + 'wts2.xlsx')

# r_t = 4 与实际vol_target = 4% 有一些差别，不能等同；风险预算符合的更好。

# ----- 计算实际权重值 ------ #
chgDates = ws_rt.index.tolist()
risk_budget = [r_p, 1]
stock_limit = 1
ret = ret[['中证100全收益', '3-5年国开债指数']]

actual_wts_rp = cal_risk_parity_wts(ret, chgDates, risk_budget=risk_budget, stock_limit=stock_limit)

asset_risk = cal_asset_risk(ret, chgDates, risk_type='normal')
actual_wts_rt = cal_risk_target_wts(asset_risk, risk_target=0.04, stock_limit=1)

combo_wts = pd.concat([ws_rp, actual_wts_rp['中证100全收益'], ws_rt, actual_wts_rt['中证100全收益']], axis=1, sort=True)
combo_wts.columns = ['theo_wts_rp', 'actual_wts_rp', 'theo_wts_rt', 'actual_wts_rt']
combo_wts = combo_wts.iloc[1:, :]


dfPlot(combo_wts[['theo_wts_rp', 'actual_wts_rp']], title_str='risk parity: [10, 1] ')
dfPlot(combo_wts[['theo_wts_rt', 'actual_wts_rt']], title_str='risk target: r_t = 4, risk_target = 4% ')

dfPlot(combo_wts[['actual_wts_rp', 'actual_wts_rt']], title_str='risk parity: [10, 1], risk target: risk_target = 4% ')


# 比较资产价格走势和权重走势

dat2 = dat.loc[(combo_wts.index[0]):, ].copy()
dat2 = dat2 / dat2.iloc[0, :]

dfPlot(dat2)

tmp2 = pd.concat([dat2, combo_wts[['actual_wts_rp', 'actual_wts_rt']]], axis=1, sort=True)
tmp2.index = tmp2.index.map(lambda x: datetime.strptime(x, '%Y%m%d'))

result_path = '../result/'
tmp2.to_excel(result_path + '资产走势和权重走势比较.xlsx')

# ----- 回测表现 ----- #
start_date = '20130101'
end_date = dat.index.max()

tradingdate = get_tradingdate_api(w, end_date)['Data']
tradingdate = tradingdate.ix[(tradingdate['Date'] >= start_date) & (tradingdate['Date'] <= end_date), :]

# date list
quarter_list = getPeriodFisrtLastDates(tradingdate, period='Quarter')
month_list = getPeriodFisrtLastDates(tradingdate, period='Month')
daily_list = tradingdate['Date'].tolist()

# 设置股债比例
daily_wts_rp = DataFrame(combo_wts['actual_wts_rp'])
daily_wts_rp.columns = ['中证100全收益']
daily_wts_rp['3-5年国开债指数'] = 1 - daily_wts_rp['中证100全收益']

daily_wts_rt = DataFrame(combo_wts['actual_wts_rt'])
daily_wts_rt.columns = ['中证100全收益']
daily_wts_rt['3-5年国开债指数'] = 1 - daily_wts_rt['中证100全收益']


daily_wts_set = [daily_wts_rp, daily_wts_rt]

res_nav = DataFrame()

for wts in daily_wts_set:

	wts_columns = wts.columns.tolist()

	# 回测参数配置并保留
	bkt_config = {
		'chgDates': quarter_list,
		'weights': wts,
		'trade_info': dat[wts_columns].copy()
	}


	# ----- 回测 -----
	# 导入数据, chgDates, weights, trade_info
	chgDates = bkt_config['chgDates']
	weights = bkt_config['weights']
	trade_info = bkt_config['trade_info']

	core1 = SimpleBKTCore_V1()
	core1.print_info()
	core1.set_switch_variables(temporary_adjust_flag=True)

	# 配置回测需要的数据
	# core1.data_bkt_configuration(trade_info, weights, chgDates=chgDates)
	core1.data_bkt_configuration(trade_info, weights.ix[chgDates, :])
	core1.strategy_bkt()

	port_nav = Series(core1.navRecord, name='+'.join(wts_columns))

	res_nav = pd.concat([res_nav, port_nav], axis=1, sort=True)

res_nav.columns = ['rp', 'rt']

dfPlot(res_nav)

# 计算指标
ann = strategyAnalysis()
comp1_indicator = ann.Get_BasicIndictors(nav2return(res_nav)).T
column_rep = {
	'tot_return': '总回报',
	'ann_return': '年化收益率',
	'ann_vol': '年化波动率',
	'sharpe_ratio': '夏普比率',
	'maxdrawdown': '最大回撤',
	'calmar_ratio': '卡曼比率'
}

comp1_indicator = comp1_indicator.rename(columns=column_rep)
comp1_indicator.to_excel(result_path + '回测指标.xlsx')
print(comp1_indicator)

# 绘制回测曲线
drawback_info = cal_drawback(res_nav)

port_drawback = drawback_info['drawback']
port_maxdrawback = drawback_info['maxdrawback_info']

dfPlot(port_drawback)






