from part4_Funs import *
from part4_index_info import *
from WindPy import *
from Funs_HXH3 import *

LogConfig('part4_get_asset_data.log')


# --------------------------------------- 1. 提取数据 ------------------------------------- #
w.start()
logging.info('Start to get Index / Fund data ...')
data_path = '../data/'

#  如果低于晚上10点，用前一天作为最新数据，否则用当天
hour = datetime.now().hour
lastestDate = (datetime.today() - timedelta(days=1)) if hour < 22 else datetime.today()
lastestDate = lastestDate.strftime('%Y%m%d')

# 开始提取数据
logging.info('Current Date: ' + lastestDate)

start_dt = '20070101'
end_dt = lastestDate

# 指数 or 基金
Index = Index_Info.ix[Index_Info['ProductType'] == '指数', :]
Fund = Index_Info.ix[Index_Info['ProductType'] == '基金', :]

# 1. 提取指数行情数据
logging.info('Get Index daily trading data ... ')

if not Index.empty:
	Index_RawData = get_wsd(w, Index['WindCode'].tolist(), ['close'],  Index['Name'].tolist(),
							start_dt, end_dt)['Data']
else:
	Index_RawData = DataFrame()


# 2. 提取基金净值数据
logging.info('Get Fund adjusted Nav data ... ')
if not Fund.empty:
	others = 'PriceAdj=F'
	Fund_RawData = get_wsd(w, Fund['WindCode'].tolist(), ['NAV_adj'],  Fund['Name'].tolist(),
							start_dt, end_dt, others)['Data']  # 基金后复权应该更合适？  Ben@20190608
else:
	Fund_RawData = DataFrame()

# 合并指数和基金数据
logging.info('Combine Index and Fund Data.')
rawData = pd.concat([Index_RawData, Fund_RawData], axis=1, sort=True)


# 检查上市后是否存在NaN，如有，是Wind问题
i = 0
for column in rawData.columns.tolist():
	tmp = rawData.ix[list(Index_Info.ix[Index_Info['Name'] == column, 'StartDate'])[0]:, column].copy()
	if np.any(tmp.isnull()):
		logging.warning('<' + column + '> Exists NaN，Please Check it! ')
		i = i + 1

if i == 0:
	logging.info('All data DO NOT include NaN, they are CLEAR.')


# ------------------ 提取外部csv数据 ------------------- #
# 提取MSCI100数据
logging.info('Load Local Data <MSCI100>')

MSCI100 = pd.read_csv(data_path + 'MSCI100.csv', encoding='gbk')
MSCI100.columns = ['Date', '漂亮100', '漂亮100全收益']
MSCI100.ix[:, 'Date'] = MSCI100.ix[:, 'Date'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d').strftime('%Y%m%d'))   # '%d/%b/%y'
# MSCI100.ix[2226:, 'Date'] = MSCI100.ix[2226:, 'Date'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d').strftime('%Y%m%d'))
MSCI100.index = MSCI100['Date'].tolist()
MSCI100 = MSCI100.drop(columns=['Date'])


# ------------------ 合并内外部数据 -------------------- #
logging.info('Combine Online and Local data ... ')
rawData = pd.concat([rawData, MSCI100], axis=1, join_axes=[rawData.index], sort=True)

logging.info('Combine Finished, Save data!')
pkl_read_write(data_path + 'rawData.pkl', 'write', rawData)


# ------------------ 获取日期数据 --------------------- #
trading_date = get_tradingdate_api(w, lastestDate)['Data']
pkl_read_write(data_path + 'trading_date.pkl', 'write', trading_date)














# rawData = pkl_read_write('rawData.pkl', 'read')

# rawData_len2 = rawData.shape[0]
# if rawData_len2 != rawData_len1:
# 	print([rawData_len2, rawData_len1], '合并后数据维度改变，注意！')

# 删掉NaN
# rawData = rawData.dropna(how='any')




# ------------- 写成一个更新的函数

# dat = pkl_read_write(data_path + localData + '.pkl', 'read')
#
# 	if dat.empty:
# 		datLastDate = dataFetchInfo[localData]['start_dt']
# 	else:
# 		# 处理三天内出现NaN的情况    Ben @20190510
# 		nan_row = dat.ix[-3:, :].isnull().any(axis=1)
# 		if nan_row.sum() > 0:
# 			nan_Date = nan_row[nan_row == True].index.min()
# 			dat = dat.ix[dat.index < nan_Date, :]
#
# 		datLastDate = dat.index.max()
#
# 	if datLastDate < lastestDate:
#
# 		logging.info('Download New Data of <' + localData + '> from Wind API ... ')
#
# 		start_dt = datLastDate
# 		end_dt = lastestDate
#
# 		varCodes = dataFetchInfo[localData]['varCodes']
# 		varNames = dataFetchInfo[localData]['varNames']
#
# 		try:
# 			if dataFetchInfo[localData]['fun'] == 'edb':
# 				tmp = get_edb(w, varCodes, varNames, start_dt, end_dt)
# 			elif dataFetchInfo[localData]['fun'] == 'wsd':
# 				tmp = get_wsd(w, varCodes, [dataFetchInfo[localData]['indicator']],  varNames, start_dt, end_dt)
#
# 			if ~tmp['ErrorCode']:
# 				new_dat = tmp['Data']
#
# 				if dat.empty:
# 					dat = pd.concat([dat, new_dat])  # 数据填充为后续数据处理环节，这里是取rawData
# 				else:
# 					dat = pd.concat([dat, new_dat.ix[1:, :]])
#
# 				# 更新并存储数据
# 				pkl_read_write(data_path + localData + '.pkl', 'write', dat)
# 				logging.info('New data is Downloaded and Combined!')
#
# 			else:
# 				logging.warning(tmp['Message'])
# 				pass
#
# 		except:
# 			logging.error('Data Download Error! Check!')
#
# 	else:
# 		logging.info('Local data is the Latest, No need to Update.')








