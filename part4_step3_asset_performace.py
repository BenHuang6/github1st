from part1_Funs import *
from part4_Funs import *
from part4_index_info import *
from Funs_HXH3 import *


LogConfig('part4_analyze_asset_data.log')

# --------------------------------------- 1. 提取rawData ------------------------------------- #
data_path = '../data/'
result_path = '../result/' + (datetime.today()).strftime('%Y%m%d') + '/'
checkdir(result_path)

logging.info('Load rawData for analysis ...')
rawData = pkl_read_write(data_path + 'rawData.pkl', 'read')

columns = [
	'3-5年国开债指数',
	'沪深300全收益', '中证红利全收益'
]

dat = rawData[columns].copy()
del rawData

# 去掉nan
dat = dat.dropna(how='any')


# 提取dat里，起始时间最近的时间
# start_dt = '20130109'
# end_dt = '20190517'

start_dt = dat.index.tolist()[0]
end_dt = dat.index.tolist()[-1]

dat = dat.ix[start_dt:end_dt, :]
date_str = start_dt + '_' + end_dt

# dat -> nav
nav = dat / dat.ix[0, :]

# 1. 绘图
logging.info('Draw asset nav curves.')
dfPlotSave(nav, 'asset_nav', result_path + 'asset_nav_' + date_str)

# 2. 计算各个指标
logging.info('Calculate asset performance indicators.')
ann = strategyAnalysis()
asset_indicators = ann.Get_BasicIndictors(nav2return(nav))
asset_indicators.index = ['总收益率', '年化收益率', '年化波动率', '夏普比率', '最大回撤', 'calmar比率']
asset_indicators.T.to_excel(result_path + 'asset_indicators_' + date_str + '.xlsx')

# 3. 年度收益对比
logging.info('Calculate asset yealy returns.')
yearly_ret = nav2yearlyreturn(nav) * 1e2

dfBarPlot(yearly_ret.iloc[:, :3], ylabel='%', show=False,
		  save_path_name=result_path + 'asset_yearly_return1_' + (start_dt[:4] + '_' + end_dt[:4]))

# dfBarPlot(yearly_ret.iloc[:, 3:], ylabel='%', show=False,
# 		  save_path_name=result_path + 'asset_yearly_return2_' + (start_dt[:4] + '_' + end_dt[:4]))

# 4. Rolling Sharpe / vol
logging.info('Calculate asset rolling sharpe and std.')
rollingPeriod = 244
fct = 244
rolling_sharpe = calRollingSharpe2(nav2return(nav), rollingPeriod=rollingPeriod, fct=fct)
rolling_std = nav2return(nav).rolling(window=rollingPeriod).std() * np.sqrt(fct)

dfPlotSave(rolling_sharpe.iloc[:, :3], result_path + 'asset_rolling_sharpe1_' + date_str, 'asset rolling sharpe')
# dfPlotSave(rolling_sharpe.iloc[:, 3:], result_path + 'asset_rolling_sharpe2_' + date_str, 'asset rolling sharpe')

dfPlotSave(rolling_std.iloc[:, :3], result_path + 'asset_rolling_std1_' + date_str, 'asset rolling std')
# dfPlotSave(rolling_std.iloc[:, 3:], result_path + 'asset_rolling_std2_' + date_str, 'asset rolling std')

# 保留结果
logging.info('Collect info of asset performance and save result.')
asset_res = {
	'nav': 				nav,
	'indicators': 		asset_indicators,
	'yearly_ret': 		yearly_ret,
	'rolling_sharpe': 	rolling_sharpe,
	'rolling_std': 		rolling_std
}

pkl_read_write(result_path + 'asset_res_' + date_str + '.pkl', 'write', asset_res)





