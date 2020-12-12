import random
import pandas as pd
import os
import numpy as np
import datetime


def getThreshold(stock_id_list, daily_path):
    rise_pct = []
    for stk_id in stock_id_list:
        string = daily_path+'/'+stk_id+'.csv'
        daily_data = pd.read_csv(string, sep = '\t')
        diff = list(daily_data.open.diff(-1))
        stk_rise_pct = diff[:len(diff)-1] / daily_data.open[1:] * 100
        rise_pct.extend(stk_rise_pct)
        # print(f'{stk_id} finished!\n')
    UP_parm = np.quantile(rise_pct, 0.67)
    DOWN_parm = np.quantile(rise_pct, 0.33)
    return UP_parm, DOWN_parm

## add the classfication label to a stock for each date
def daily_label(stock_id, daily_path, UP_parm, DOWN_parm):
    try:
        string = daily_path+'/'+stock_id+'.csv'
        # daily_eg = pd.read_csv(string)
        daily_eg = pd.read_csv(string, sep = '\t')
    except:
        print('file not exist!')
    diff = list(daily_eg.open.diff(-1))
    diff = [float('inf')] + diff[:len(diff)-1]
    daily_eg['rise_pct'] = diff / daily_eg.open*100
    daily_eg['label'] = 'PRESERVE'
    daily_eg.loc[daily_eg['rise_pct']>UP_parm, 'label'] = 'UP'
    daily_eg.loc[daily_eg['rise_pct']<DOWN_parm, 'label'] = 'DOWN'
    daily_eg['date'] = pd.to_datetime(daily_eg['trade_date'], format='%Y%m%d')
    return daily_eg.loc[daily_eg['trade_date']<20201014]


## 没有nan版本的300只沪深股票dataloader
def buildDataloader_v2(stock_id_list, daily_path, news_path):
    ## 用来算某一天之前有多少天的新闻
    def count_smaller(lt, num):
        res = [i for i in lt if i < num]
        return len(res)
    N = len(stock_id_list)
    dataloader = pd.DataFrame()
    for ct,stock_id in enumerate(stock_id_list):
      daily_data = daily_label(stock_id,daily_path)
      ## date_list and int_date_list 是该股票有交易的日期
      date_list = [i.strftime("%Y-%m-%d") for i in daily_data.date]
      int_date_list = [i for i in daily_data.trade_date]
      date_list.reverse()
      int_date_list.reverse()
      df_id = pd.DataFrame()
      ## news_date_list：该股票有新闻的日期的列表
      news_date_list = [int(date[:4]+date[5:7]+date[8:10]) for date in date_list if os.path.isfile(news_path+f'{stock_id}/{date}.txt')]
      entity = set()
      for label_date in int_date_list:
          id = count_smaller(news_date_list,label_date) ## 在label_date之前有新闻的天数
          cur_pct = list(daily_data.loc[daily_data['trade_date']==label_date].rise_pct)
          if id >= 10:
              prev10days_news = news_date_list[(id-10):id] ## 记下前10条新闻的日期
              prev10days_str = str(prev10days_news)
              if prev10days_str not in entity:  ## 若已经有记录过这10条新闻，则跳过
                  entity.add(prev10days_str)
                  prev_news_path = []
                  for date in prev10days_news:
                    date  = str(date)
                    prev_news_path.append(f'{stock_id}/{date[:4]}-{date[4:6]}-{date[6:8]}.txt')
                  df_id = df_id.append(pd.DataFrame([stock_id]+[label_date]+prev_news_path+cur_pct).transpose())
      dataloader = dataloader.append(df_id)
      print('stock:', stock_id, ' finished!')
      print(N-ct-1, ' stocks left.')
      print('------------------------------')
    dataloader.columns = ['stock_id','label_date','day-1','day-2','day-3','day-4','day-5','day-6','day-7','day-8',
                          'day-9','day-10','rise_pct']
    dataloader.reset_index(inplace = True, drop = True)
    ## set label according to the rise percentage
    UP_parm = np.quantile(dataloader.rise_pct, 2/3)
    Down_parm = np.quantile(dataloader.rise_pct, 1/3)
    dataloader['label'] = 'PRESERVE'
    dataloader.loc[dataloader['rise_pct']>UP_parm, 'label'] = 'UP'
    dataloader.loc[dataloader['rise_pct']<Down_parm, 'label'] = 'DOWN'
    dataloader.drop(columns = 'rise_pct', inplace=True)
    return dataloader



def write_dataloader(dataloader, Ptrain, Pcv, outputpath):
    random.seed(50)
    dataloader = dataloader.sort_values(by='label_date')
    
    n = dataloader.shape[0]
    train_num = int(n*Ptrain)
    cv_num = int(n*Pcv)

    train = dataloader[:train_num]
    cv = dataloader[train_num:train_num+cv_num]
    test = dataloader[cv_num+train_num:]
    
    if not os.path.isdir(outputpath):
        os.mkdir(outputpath)
    train.to_csv(outputpath+'/train_data.csv', index=False)
    cv.to_csv(outputpath+'/cv_data.csv', index=False)
    test.to_csv(outputpath+'/test_data.csv', index=False)
    print('Successfully created training, cv and test dataset!')
    # return train, cv, test



GOOGLE_DRIVE_PATH_AFTER_MYDRIVE = 'stockDataFromTushare545'
GOOGLE_DRIVE_PATH = os.path.join('drive', 'My Drive', GOOGLE_DRIVE_PATH_AFTER_MYDRIVE)
# print(os.listdir(GOOGLE_DRIVE_PATH+'/daily/'))
df_dataloader2 = buildDataloader_v2(stock_id_list, daily_path, news_path)
write_dataloader(df_dataloader2, Ptrain, Pcv, outputpath)
df_dataloader2.label.value_counts()

# Successfully created training, cv and test dataset!
# PRESERVE    11095
# UP          11094
# DOWN        11093