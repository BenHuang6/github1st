from Funs_HXH3 import *

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# --------------------------
from WindPy import *
w.start()


def get_mutli_excess_return(ret, ref_ret):

	excess_return = DataFrame()

	if isinstance(ret, Series):
		ret = DataFrame(ret)

	if isinstance(ref_ret, Series):
		ref_ret = DataFrame(ref_ret)

	ret_columns = ret.columns.tolist()
	ref_ret_columns = ref_ret.columns.tolist()

	for i in range(ret.shape[1]):

		j = i

		if ref_ret.shape[1] == 1:
			j = 0

		tmp = ret.iloc[:, i] - ref_ret.iloc[:, j]
		tmp.name = ret_columns[i] + ' - ' + ref_ret_columns[j]
		excess_return = pd.concat([excess_return, tmp], axis=1, sort=True)

	return excess_return

# ---------------------------------------------------------


result_path = '../result/'
# 提取基金信息
dat2 = pd.read_excel(result_path + 'FOF基金筛选.xlsx')

fund_codes = list(dat2['基金代码'])

fund_attr = {
    'name_official': '基金名称',
    'fund_setupdate': '成立日期',
    'fund_existingyear': '成立年限',
    'fund_benchmark': '业绩基准',
    'fund_changeofbenchmark': '业绩基准变更',
    'fund_mgrcomp': '基金公司',
    'fund_fundmanager': '现任基金经理',
    'fund_predfundmanager': '历任基金经理'
}


fund_info = get_wss(w, fund_codes, list(fund_attr.keys()))['Data']
fund_info = fund_info.rename(columns=fund_attr)
fund_info['基金代码'] = fund_info.index.tolist()

fund_info = pd.merge(fund_info, dat2[['资产类别与风格', '基金代码', '基金类型']], on=['基金代码'])

fund_info['成立日期'] = fund_info['成立日期'].apply(lambda x: x.strftime('%Y%m%d'))
fund_info['成立年限'] = round(fund_info['成立年限'].astype('float32'), 2)

fund_info = fund_info[['资产类别与风格', '基金类型', '基金名称', '基金代码', '成立日期', '成立年限',
                       '基金公司', '现任基金经理', '历任基金经理', '业绩基准', '业绩基准变更']]

fund_info.to_excel(result_path + '基金筛选结果.xlsx')


# 提取2012年以来基金和指数表现, 指数用全收益
start_dt = '20120101'
end_dt = '20200131'

others = 'PriceAdj=F'
fund_nav = get_wsd(w, fund_info['基金代码'].tolist(), ['NAV_adj'],  fund_info['基金名称'].tolist(),
							start_dt, end_dt, others)['Data']

fund_nav = fund_nav / fund_nav.iloc[0, :]


index_code = ['H00905.CSI', 'H00300.CSI', 'H00922.CSI', 'CBA00101.CS',
			  'H00942.CSI', 'H00933.CSI', 'SPX.GI', 'IXIC.GI']
index_name = ['中证500全收益', '沪深300全收益', '中证红利全收益', '中债综合财富指数',
			  '中证内地消费主题全收益', '中证医药卫生全收益', '标普500', '纳斯达克指数']

index_close = get_wsd(w, index_code, ['close'], index_name, start_dt, end_dt)['Data']
index_nav = index_close / index_close.iloc[0, :]

# 成长风格
growth_columns = ['富国天惠成长混合(LOF)', '富国中证500指数增强(LOF)']
index_columns = ['中证500全收益']
growth_nav = pd.concat([fund_nav[growth_columns], index_nav[index_columns]], axis=1, sort=True)
dfPlot(growth_nav)

# 超额收益曲线
growth_ret = nav2return(growth_nav)
growth_exret = get_mutli_excess_return(growth_ret.iloc[:, :2], growth_ret.iloc[:, -1])
growth_exnav = nav2return(growth_exret, 2)
dfPlot(growth_exnav)

growth_nav_all = pd.concat([growth_nav, growth_exnav], axis=1, sort=True)
dfPlot(growth_nav_all)

# 年度超额收益
growth_yearly_ret = nav2periodreturn(growth_nav, 'Year')['ret_table']
growth_yearly_ret.index = growth_yearly_ret.index.map(lambda x: x[:4])
growth_yearly_ret = growth_yearly_ret * 1e2
growth_yearly_ret = growth_yearly_ret.iloc[:-1, :]

dfBarPlot(growth_yearly_ret, ylabel='%', ygrid=True)


# 价值风格
value_columns = ['汇添富价值精选混合', '富国沪深300指数增强']
index_columns = ['沪深300全收益']
value_nav = pd.concat([fund_nav[value_columns], index_nav[index_columns]], axis=1, sort=True)
dfPlot(value_nav)

# 超额收益曲线
value_ret = nav2return(value_nav)
value_exret = get_mutli_excess_return(value_ret.iloc[:, :2], value_ret.iloc[:, -1])
value_exnav = nav2return(value_exret, 2)
dfPlot(value_exnav)

value_nav_all = pd.concat([value_nav, value_exnav], axis=1, sort=True)
dfPlot(value_nav_all)

# 年度超额收益
value_yearly_ret = nav2periodreturn(value_nav, 'Year')['ret_table']
value_yearly_ret.index = value_yearly_ret.index.map(lambda x: x[:4])
value_yearly_ret = value_yearly_ret * 1e2
value_yearly_ret = value_yearly_ret.iloc[:-1, :]

dfBarPlot(value_yearly_ret, ylabel='%', ygrid=True)




# 债券
bond_columns = ['易方达稳健收益债券B', '易方达安心回报债券A', '工银双利债券A']
index_columns = ['中债综合财富指数']
bond_nav = pd.concat([fund_nav[bond_columns], index_nav[index_columns]], axis=1, sort=True)
dfPlot(bond_nav)

# 超额收益曲线
bond_ret = nav2return(bond_nav)
bond_exret = get_mutli_excess_return(bond_ret.iloc[:, 2], bond_ret.iloc[:, -1])
bond_exnav = nav2return(bond_exret, 2)
dfPlot(bond_exnav)

bond_nav_all = pd.concat([bond_nav.iloc[:, 2:4], bond_exnav], axis=1, sort=True)
dfPlot(bond_nav_all)

# 年度超额收益
bond_yearly_ret = nav2periodreturn(bond_nav.iloc[:, 2:4], 'Year')['ret_table']
bond_yearly_ret.index = bond_yearly_ret.index.map(lambda x: x[:4])
bond_yearly_ret = bond_yearly_ret * 1e2
bond_yearly_ret = bond_yearly_ret.iloc[:-1, :]

dfBarPlot(bond_yearly_ret, ylabel='%', ygrid=True)





# 消费
consumption_columns = ['易方达消费行业股票']
index_columns = ['中证内地消费主题全收益']
consumption_nav = pd.concat([fund_nav[consumption_columns], index_nav[index_columns]], axis=1, sort=True)
dfPlot(consumption_nav)

# 超额收益曲线
consumption_ret = nav2return(consumption_nav)
consumption_exret = get_mutli_excess_return(consumption_ret.iloc[:, 0], consumption_ret.iloc[:, -1])
consumption_exnav = nav2return(consumption_exret, 2)
dfPlot(consumption_exnav)

consumption_nav_all = pd.concat([consumption_nav, consumption_exnav], axis=1, sort=True)
dfPlot(consumption_nav_all)

# 年度超额收益
consumption_yearly_ret = nav2periodreturn(consumption_nav, 'Year')['ret_table']
consumption_yearly_ret.index = consumption_yearly_ret.index.map(lambda x: x[:4])
consumption_yearly_ret = consumption_yearly_ret * 1e2
consumption_yearly_ret = consumption_yearly_ret.iloc[:-1, :]

dfBarPlot(consumption_yearly_ret, ylabel='%', ygrid=True)




# 医药
medical_columns = ['汇添富医药保健混合']
index_columns = ['中证医药卫生全收益']
medical_nav = pd.concat([fund_nav[medical_columns], index_nav[index_columns]], axis=1, sort=True)
dfPlot(medical_nav)


# 超额收益曲线
medical_ret = nav2return(medical_nav)
medical_exret = get_mutli_excess_return(medical_ret.iloc[:, 0], medical_ret.iloc[:, -1])
medical_exnav = nav2return(medical_exret, 2)
dfPlot(medical_exnav)

medical_nav_all = pd.concat([medical_nav, medical_exnav], axis=1, sort=True)
dfPlot(medical_nav_all)

# 年度超额收益
medical_yearly_ret = nav2periodreturn(medical_nav, 'Year')['ret_table']
medical_yearly_ret.index = medical_yearly_ret.index.map(lambda x: x[:4])
medical_yearly_ret = medical_yearly_ret * 1e2
medical_yearly_ret = medical_yearly_ret.iloc[:-1, :]

dfBarPlot(medical_yearly_ret, ylabel='%', ygrid=True)







# 标普和纳指
index_columns = ['标普500', '纳斯达克指数']
oversea_nav = index_nav[index_columns].copy()
dfPlot(oversea_nav)

oversea_yearly_ret = nav2periodreturn(oversea_nav, 'Year')['ret_table']
oversea_yearly_ret.index = oversea_yearly_ret.index.map(lambda x: x[:4])
oversea_yearly_ret = oversea_yearly_ret * 1e2
oversea_yearly_ret = oversea_yearly_ret.iloc[:-1, :]

dfBarPlot(oversea_yearly_ret, ylabel='%', ygrid=True)





# 合起来，算表现指标
como_nav = pd.concat([fund_nav, index_nav], axis=1)

ann = strategyAnalysis()
como_indicator = ann.Get_BasicIndictors(nav2return(como_nav))
como_indicator = como_indicator.T
column_rep = {
	'tot_return': '总回报',
	'ann_return': '年化收益率',
	'ann_vol': '年化波动率',
	'sharpe_ratio': '夏普比率',
	'maxdrawdown': '最大回撤',
	'calmar_ratio': '卡曼比率'
}

como_indicator = como_indicator.rename(columns=column_rep)

como_indicator.to_excel(result_path + '基金与指数表现.xlsx')


# ---- 计算风险预算权重 ----- #
# 如果多资产选择，就这样处理
start_date = '20130101'
end_date = como_nav.index.max()

tradingdate = get_tradingdate_api(w, end_date)['Data']
tradingdate = tradingdate.ix[(tradingdate['Date'] >= start_date) & (tradingdate['Date'] <= end_date), :]

# date list
quarter_list = getPeriodFisrtLastDates(tradingdate, period='Quarter')
month_list = getPeriodFisrtLastDates(tradingdate, period='Month')
daily_list = tradingdate['Date'].tolist()


# FOF回测里转成这种格式，应该取数更快  Ben@20190628
combo_ret = nav2return(como_nav)

columns1 = ['富国天惠成长混合(LOF)', '汇添富价值精选混合', '工银双利债券A']  # 主动组合
columns2 = ['富国中证500指数增强(LOF)', '富国沪深300指数增强', '工银双利债券A']  # 增强组合
columns3 = ['中证500全收益', '沪深300全收益', '中债综合财富指数']  # 基准指数组合

# 后续加行业和海外

ret1 = combo_ret[columns1].copy()
ret2 = combo_ret[columns2].copy()
ret3 = combo_ret[columns3].copy()

chgDates = daily_list

# 计算权重
daily_target_wts1_1 = cal_risk_parity_wts(ret1, chgDates, risk_budget=[1, 1, 1], stock_limit=1)
dfPlot(daily_target_wts1_1)
# 二级债基波动率还是相对比较大，因此2/8比例

daily_target_wts2_1 = cal_risk_parity_wts(ret2, chgDates, risk_budget=[1, 1, 1], stock_limit=1)
dfPlot(daily_target_wts2_1)

daily_target_wts3_1 = cal_risk_parity_wts(ret3, chgDates, risk_budget=[1, 1, 1], stock_limit=1)
dfPlot(daily_target_wts3_1)

# 增加用指数做的权重，然后用主动基金回测项目
daily_target_wts3_2 = daily_target_wts3_1.rename(columns={'中证500全收益': '富国天惠成长混合(LOF)',
														  '沪深300全收益': '汇添富价值精选混合'  #,
														  })  # '中债综合财富指数': '工银双利债券A'
dfPlot(daily_target_wts3_2)
# dfPlot(como_nav[['富国天惠成长混合(LOF)', '汇添富价值精选混合', '工银双利债券A']])
#
# dfPlot(como_nav[['中债综合财富指数', '工银双利债券A']])

# 添加行业指数和美股
columns4 = ['富国天惠成长混合(LOF)', '汇添富价值精选混合', '易方达消费行业股票', '汇添富医药保健混合', '中债综合财富指数']  # 基准指数组合
columns5 = ['富国天惠成长混合(LOF)', '汇添富价值精选混合', '标普500', '纳斯达克指数', '中债综合财富指数']  # 基准指数组合
columns6 = ['富国天惠成长混合(LOF)', '汇添富价值精选混合', '易方达消费行业股票',
			'标普500', '纳斯达克指数', '汇添富医药保健混合', '中债综合财富指数']  # 基准指数组合

ret4 = combo_ret[columns4].copy()
ret5 = combo_ret[columns5].copy()
ret6 = combo_ret[columns6].copy()


daily_target_wts4_1 = cal_risk_parity_wts(ret4, chgDates, risk_budget=[1, 1, 1, 1, 1], stock_limit=1)
dfPlot(daily_target_wts4_1)

daily_target_wts5_1 = cal_risk_parity_wts(ret5, chgDates, risk_budget=[1, 1, 1, 1, 1], stock_limit=1)
dfPlot(daily_target_wts5_1)

daily_target_wts6_1 = cal_risk_parity_wts(ret6, chgDates, risk_budget=[1, 1, 1, 1, 1, 1, 1], stock_limit=1)
dfPlot(daily_target_wts6_1)



# ---- 目标风险 ----
# columns_b1 = ['富国天惠成长混合(LOF)', '工银双利债券A']  # 主动组合
# columns_b2 = ['汇添富价值精选混合', '工银双利债券A']  # 主动组合
columns_b3 = ['富国天惠成长混合(LOF)', '中债综合财富指数']  # 主动组合
columns_b4 = ['汇添富价值精选混合', '中债综合财富指数']  # 主动组合

columns_b5 = ['中证500全收益', '中债综合财富指数']  # 主动组合
columns_b6 = ['沪深300全收益', '中债综合财富指数']  # 主动组合

ret_b3 = combo_ret[columns_b3].copy()
ret_b4 = combo_ret[columns_b4].copy()

ret_b5 = combo_ret[columns_b5].copy()
ret_b6 = combo_ret[columns_b6].copy()


asset_risk = cal_asset_risk(ret_b3, daily_list, risk_type='normal')
daily_target_wts_b3 = cal_risk_target_wts(asset_risk, risk_target=0.06, stock_limit=1)
dfPlot(daily_target_wts_b3)

asset_risk = cal_asset_risk(ret_b4, daily_list, risk_type='normal')
daily_target_wts_b4 = cal_risk_target_wts(asset_risk, risk_target=0.06, stock_limit=1)
dfPlot(daily_target_wts_b4)

asset_risk = cal_asset_risk(ret_b5, daily_list, risk_type='normal')
daily_target_wts_b5 = cal_risk_target_wts(asset_risk, risk_target=0.06, stock_limit=1)
dfPlot(daily_target_wts_b5)

asset_risk = cal_asset_risk(ret_b6, daily_list, risk_type='normal')
daily_target_wts_b6 = cal_risk_target_wts(asset_risk, risk_target=0.06, stock_limit=1)
dfPlot(daily_target_wts_b6)


# 目前看b4是比较好的


daily_wts_set = [daily_target_wts1_1, daily_target_wts2_1, daily_target_wts3_1, daily_target_wts3_2,
				 daily_target_wts4_1, daily_target_wts5_1, daily_target_wts6_1,
				 daily_target_wts_b3, daily_target_wts_b4, daily_target_wts_b5, daily_target_wts_b6]

res_nav = DataFrame()

for wts in daily_wts_set:

	wts_columns = wts.columns.tolist()

	# 回测参数配置并保留
	bkt_config = {
		'chgDates': quarter_list,
		'weights': wts,
		'trade_info': como_nav[wts_columns].copy()
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

	# # 查看策略表现
	# # port_nav.index = port_nav.index.map(lambda x: datetime.strftime(x, '%Y%m%d'))
	# port_indicator = ann.Get_BasicIndictors(nav2return(DataFrame(port_nav)))
	# print(port_indicator)
	# dfPlot(port_nav)

	res_nav = pd.concat([res_nav, port_nav], axis=1, sort=True)


pkl_read_write(result_path + 'res_nav_1_1_006.pkl', 'write', res_nav)


# 比较1：主动、增强、基准
sel_columns1 = ['富国天惠成长混合(LOF)+汇添富价值精选混合+工银双利债券A',
			   '富国中证500指数增强(LOF)+富国沪深300指数增强+工银双利债券A',
			   '中证500全收益+沪深300全收益+中债综合财富指数',]
comp1_nav = res_nav[sel_columns1].copy()

dfPlot(comp1_nav)

ann = strategyAnalysis()
comp1_indicator = ann.Get_BasicIndictors(nav2return(comp1_nav)).T
comp1_indicator = comp1_indicator.rename(columns=column_rep)
comp1_indicator.to_excel(result_path + 'comp1_indicator_1_1.xlsx')
print(comp1_indicator)


# 比较2：基准+主动, 基准
sel_columns2 = ['富国天惠成长混合(LOF)+汇添富价值精选混合+中债综合财富指数',
			   '中证500全收益+沪深300全收益+中债综合财富指数']
comp2_nav = res_nav[sel_columns2].copy()

dfPlot(comp2_nav)

ann = strategyAnalysis()
comp2_indicator = ann.Get_BasicIndictors(nav2return(comp2_nav)).T
comp2_indicator = comp2_indicator.rename(columns=column_rep)
comp2_indicator.to_excel(result_path + 'comp2_indicator_1_1.xlsx')
print(comp2_indicator)


# 比较3：风险平价，混合
sel_columns3 = ['富国天惠成长混合(LOF)+汇添富价值精选混合+易方达消费行业股票+汇添富医药保健混合+中债综合财富指数',
            	'富国天惠成长混合(LOF)+汇添富价值精选混合+标普500+纳斯达克指数+中债综合财富指数',
       			'富国天惠成长混合(LOF)+汇添富价值精选混合+易方达消费行业股票+标普500+纳斯达克指数+汇添富医药保健混合+中债综合财富指数',]
comp3_nav = res_nav[sel_columns3].copy()

dfPlot(comp3_nav)

ann = strategyAnalysis()
comp3_indicator = ann.Get_BasicIndictors(nav2return(comp3_nav)).T
comp3_indicator = comp3_indicator.rename(columns=column_rep)
comp3_indicator.to_excel(result_path + 'comp3_indicator_1_1.xlsx')
print(comp3_indicator)

# 比较4:目标风险
sel_columns4 = ['汇添富价值精选混合+中债综合财富指数',
				'中证500全收益+中债综合财富指数',
       			'沪深300全收益+中债综合财富指数']
comp4_nav = res_nav[sel_columns4].copy()

dfPlot(comp4_nav)

ann = strategyAnalysis()
comp4_indicator = ann.Get_BasicIndictors(nav2return(comp4_nav)).T
comp4_indicator = comp4_indicator.rename(columns=column_rep)
comp4_indicator.to_excel(result_path + 'comp4_indicator_1_1_006.xlsx')
print(comp4_indicator)











# 保存核心结果
bkt_result = {
	'navRecord': core1.navRecord,
	'tradeRecord': core1.tradeRecord,
	'portfolioRecord': core1.portfolioRecord
}

pkl_read_write(result_path + 'bkt_result_active.pkl', 'write', bkt_result)


































































