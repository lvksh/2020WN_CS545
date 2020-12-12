import multiprocessing as mp
import pandas as pd
import os
import numpy as np
import time 
def createDir(id2name, outputPath):
    # create the directories 
    # outputPath: "stockNews/"
    
    os.mkdir(outputPath)
    
    for stockid in id2name.keys():
        os.makedirs(outputPath + "/{}".format(stockid))
    pass

def assign(id_, newsFiles, id2name, stockNewsCount):
    # write corresponding news to corresponding txt file
    stockNewsDateCount = {}
    for csvFile in newsFiles: # 600 days of news file
        # csvFile: sina-2020-10-10.csv
        if csvFile[-3:] != 'csv':
            continue;
        date = csvFile[-14:-4] # 2020-10-10

        # open file
        newsFile = pd.read_csv(csvFile, sep = '\t')
        for tup in newsFile.itertuples(): 
            if csvFile[-22:-15] == 'general':
                title = tup[1]
                content = tup[2]
            else:
                title = tup[2]
                content = tup[3]
            if pd.isna(content):
                content = ' '
            if pd.isna(title):
                title = ' '
            content = content.strip().replace(u'\u3000', u' ').replace(u'\xa0', u' ').replace('\n', ' ')
            if id_ in title or id_[2:] in title or id2name[id_] in title or \
               id_ in content or id_[2:] in content or id2name[id_] in content:
                f = open(outputPath + f'/{id_}/{date}.txt', 'a+')
                f.write(title + ' ' + content + '\n')
                f.close()

                stockNewsDateCount[date] = stockNewsDateCount.get(date, 0) + 1 # maintain the count of news for each stock

    stockNewsCount[id_] = stockNewsDateCount
    pass

def process(idList, stockNewsCount):
    for id_ in idList:
        assign(id_, newsFiles, id2name, stockNewsCount)
    pass

###########################
# Defining variables
# change this path to fit yours directory
dataPath = '/Users/lvkunsheng/PycharmProjects/cs545Finals/stockDataFromTushare'
outputPath = dataPath + '/stockNews'
mappingPath = dataPath + '/stockid2name.csv'

id2name= pd.read_csv(mappingPath, sep = '\t')
id2name = {tup[1]:tup[2] for tup in id2name.itertuples()}

newsFiles = [dataPath + '/generalNews/' + csvFile for csvFile in os.listdir(dataPath + '/generalNews')] + \
            [dataPath + '/sinaNews/' + csvFile for csvFile in os.listdir(dataPath + '/sinaNews')] + \
            [dataPath + '/thsNews/' + csvFile for csvFile in os.listdir(dataPath + '/thsNews')] + \
            [dataPath + '/eastmoneyNews/' + csvFile for csvFile in os.listdir(dataPath + '/eastmoneyNews')] + \
            [dataPath + '/ycjNews/' + csvFile for csvFile in os.listdir(dataPath + '/ycjNews')] + \
            [dataPath + '/wsNews/' + csvFile for csvFile in os.listdir(dataPath + '/wsNews')]


# start 
# creating directories
createDir(id2name, outputPath)

# initializing variables for parallel processing
id_list = list(id2name.keys())
nb_process = int(mp.cpu_count()) - 1
#nb_process = 7
l = list(np.array_split(id_list, nb_process))
l = [x.tolist() for x in l]

stockNewsCount = mp.Manager().dict() # count how many news on a certain date for a certain stock

process_list = [mp.Process(target=process, args = (idList,stockNewsCount)) for idList in l]

time1=time.time()
for p in process_list:
    p.start()

for p in process_list:
    p.join()

time2=time.time()
print('Cost time: ' + str(time2 - time1) + 's')
# Cost time: 1346.320482969284s
# for 300 stocks and 4000 news files
######################
# some analysis of the assign result

stockNewsCount = dict(stockNewsCount)
cnt = 0
for key in stockNewsCount.keys():
    cnt += np.sum(list(stockNewsCount[key].values()))
print("Altogether {} news for 300 stocks".format(cnt))
# Altogether 163251 news for 300 stocks

print("Averagely {} news for each stock".format(cnt/300))
# Averagely 544.17 news for each stock

#count	300.000000
#mean	544.170000
#std	1145.446034
#min	48.000000
#25%	144.750000
#50%	260.500000
#75%	542.750000
#max	15046.000000
