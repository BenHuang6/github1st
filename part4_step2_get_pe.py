# from part4_Funs import *
from part4_index_info import *
from WindPy import *
from Funs_HXH3 import *

w.start()

data_path = '../data/'

# 添加风险调整
# 计算PE_TTM
hour = datetime.now().hour
lastestDate = (datetime.today() - timedelta(days=1)) if hour < 22 else datetime.today()
lastestDate = lastestDate.strftime('%Y%m%d')


varCodes = ['000016.SH', '000903.SH', '000300.SH', '000905.SH', '000922.CSI']
indicator = ['pe_ttm']
varNames = ['上证50', '中证100', '沪深300', '中证500', '中证红利']
start_dt = '2007-01-01'
end_dt = lastestDate

dat = get_wsd(w, varCodes, indicator,  varNames, start_dt, end_dt)['Data']
dfPlot(dat)

adjust_fct = cal_adjust_fct(dat)
# adjust_fct = adjust_fct.loc['20130101':'20191231', :]
dfPlot(adjust_fct)
pkl_read_write(data_path + 'adjust_fct.pkl', 'write', adjust_fct)

# quantile = cal_quantile_alongtime(dat)
# dfPlot(quantile)



