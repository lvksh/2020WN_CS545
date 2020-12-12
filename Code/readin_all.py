import pandas as pd
import numpy as np 
from sys import getsizeof
def path2tensor(row, max_news_cnt = 4, embedded_size = 768):
    filePath = dataPath + '/stockNewsVecBert'
    tensor = []
    for i in range(3, 3+10):
        if pd.isna(row[i]): # no news on this date
                # Pads with all 0
            sentence = [[0]*embedded_size]*max_news_cnt
        else:
            path = filePath + '/' + row[i]
            sentence = np.genfromtxt(path, delimiter=',')
            sentence = sentence.reshape((-1,embedded_size)) # avoid being (300,)
            # pad with 0s
            if len(sentence) > max_news_cnt:
                sentence = sentence[:max_news_cnt]
            elif len(sentence) < max_news_cnt:
                sentence = np.append(sentence, np.array([[0]*embedded_size]*(max_news_cnt-len(sentence))), axis = 0)
            else:
                pass
        tensor.append(sentence)
    return np.array(tensor)

dataPath = '/Users/lvkunsheng/PycharmProjects/cs545Finals/stockDataFromTushare'
for s in ['cv', 'test', 'train']:
    dataloaderFile = dataPath + f'/dataloader/{s}_data_full.csv'
    df = pd.read_csv(dataloaderFile)
    ts = np.array(list(map(path2tensor, df.itertuples())))
    print(f"{s} shape: ",ts.shape)
    print(f"{s} takes up {round(getsizeof(ts) / 1024**2, 2)} MB")
    np.save(dataPath + f'/dataloader/{s}_bert_np_full', ts, allow_pickle = False)
    del ts