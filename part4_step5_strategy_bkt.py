from part1_Funs import *
from part4_Funs import *
from part4_index_info import *
from Funs_HXH3 import *

plt.switch_backend('agg')   # 设置不显示图像
# plt.switch_backend('TkAgg')  # 默认可以显示图像

# ----------------------------------------- 回测 -------------------------------------- #
data_path = '../data/'
result_path = '../result/'

# 导入数据, chgDates, weights, trade_info
bkt_config = pkl_read_write(data_path + 'bkt_config_wmf.pkl', 'read')
chgDates = bkt_config['chgDates']
weights = bkt_config['weights']
trade_info = bkt_config['trade_info']

dfPlot(weights)

# 调整trade_info格式
# trade_info1 = trade_info.copy()
#
# tmp_trade_info = DataFrame()
# for column in trade_info1.columns.tolist():
#
# 	tmp = trade_info1[[column]].copy()
# 	tmp['Date'] = tmp.index.tolist()
#
# 	tmp = tmp.rename(columns={column: 'Value'})
# 	tmp['WindName'] = column
#
# 	tmp_trade_info = pd.concat([tmp_trade_info, tmp], sort=True)
#
#
# tmp_trade_info = tmp_trade_info[['Date', 'WindName', 'Value']]
# tmp_trade_info = tmp_trade_info.sort_values(by=['Date', 'WindName']).reset_index(drop=True)

core1 = SimpleBKTCore_V1()
core1.print_info()
core1.set_switch_variables(temporary_adjust_flag=True)

# 配置回测需要的数据
# core1.data_bkt_configuration(trade_info, weights, chgDates=chgDates)
core1.data_bkt_configuration(trade_info, weights.ix[chgDates, :])
core1.strategy_bkt()

port_nav = Series(core1.navRecord, name='portfolio_nav')
# dfPlot(port_nav)
port_nav.index = port_nav.index.map(lambda x: datetime.strptime(x, '%Y%m%d'))
# port_nav.to_excel(result_path + 'class_bkt_nav2.xlsx')

# 查看持仓和交易信息
# core1.tradeRecord['20190701']
# core1.portfolioRecord['20190701'].position

# 保存核心结果
bkt_result = {
	'navRecord': core1.navRecord,
	'tradeRecord': core1.tradeRecord,
	'portfolioRecord': core1.portfolioRecord
}

pkl_read_write(result_path + 'bkt_result_wmf.pkl', 'write', bkt_result)










