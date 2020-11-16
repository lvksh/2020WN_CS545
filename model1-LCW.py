import pickle, argparse, os, sys
from sklearn.metrics import accuracy_score
import numpy as np
import random
import torch
import torch.nn as nn
import torch.functional as F
import thulac # Chinese word splitting tool
import gensim
import pickle
import re


import pickle, argparse, os, sys
from sklearn.metrics import accuracy_score
import numpy as np
import random
import torch
import torch.nn as nn
import torch.functional as F
import thulac # Chinese word splitting tool
import gensim
import pickle
import re
import pandas as pd
dataPath = '/Users/lvkunsheng/PycharmProjects/cs545Finals/stockDataFromTushare'
word2vecPath =  dataPath + '/ChineseWord2Vec/sgns.financial.word'
stopwordsPath = dataPath + '/Stopwords/stopwords.pkl'

class InputDataset(torch.utils.data.Dataset):
    def __init__(self, dataloaderFile, dataPath, word2vecPath, stopwordsPath, embedded_size, max_news_cnt):
        # read in the dataloader file and store it in class
        # make sure some rows that have no news at all are discarded 
        self.df_dataloader = pd.read_csv(dataloaderFile)
        self.embedded_size = embedded_size
        self.max_news_cnt  = max_news_cnt
        self.dataPath = dataPath # prefix of all the files
        self.thu1 = thulac.thulac(seg_only=True)   # filter out meaningless word
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vecPath, binary=False)
        with open(stopwordsPath, 'rb') as handle:
            self.stopwords = pickle.load(handle)

        
    def __getitem__(self, index):
        # read in one line of data
        # return x, y
        # [['今天', '股市', '又', '降', '了'], sentence2, ..., sentence10], label
        # [vector of corpus1, vector of corpus2, ..., vector of corpus10], label
        def clean_sentence(s, stopwords)->[str]:
            text_split = self.thu1.cut(s, text=True)
            w_l = []
            for word in text_split:
                if word not in stopwords:
                    w_l.append(word)
            return w_l
        
        def clean_file(f_name,stopwords)->[str]:
            f = open(f_name)
            txt = f.read()
            txt = re.sub(r'(\(image_address="https|image_address="http)?:\/\/(\w|\.|\/|\?|\=|\&|\%\\)*\b', '', txt)
            txt = re.sub(r'\\u[a-zA-Z0-9]*\b', '', txt)
            txt = txt.strip().rstrip()
            tl = txt.split('\n')
            f.close()
            s_l = []
            for t in tl:
                s_l.append(clean_sentence(t,stopwords))
            return s_l
        
        def vectorize_s(s:[str],model)->'vec':
            ct = 0
            vec = np.zeros((self.embedded_size,))
            for word in s:
                if word in model.vocab:
                    ct+=1
                    vec+=model.get_vector(word)
            if ct!=0:
                vec=vec/ct
            return vec

        def vectorize_sl(s_l:[[str]],model)->'vec':
            v_l = []
            for s in s_l:
                vec  = vectorize_s(s,model)
                if vec[0]!= np.nan:
                    v_l.append(vec)
            # pad with 0s
            v_l.extend([[0]*self.embedded_size]*(self.max_news_cnt-len(v_l)))
            return v_l    
        
        row = self.df_dataloader.iloc[index, :]
        corpus = []
        for i in range(2, 12): # for each date
            if pd.isna(row[i]): # no news on this date
                # Pads with all 0
                corpus.append([[0]*self.embedded_size]*self.max_news_cnt)
            else:
                sentence = clean_file(self.dataPath + '/' + row[i],
                                      self.stopwords)
                
                corpus.append(vectorize_sl(sentence,self.word2vec))
                
        return np.array(corpus), row[-1]
    def __len__(self):
        return self.df_dataloader.shape[0]
    

    
# test code for data loader:
############################
dataPath = '/Users/lvkunsheng/PycharmProjects/cs545Finals/stockDataFromTushare' # only change this
word2vecPath =  dataPath + '/ChineseWord2Vec/sgns.financial.word'
stopwordsPath = dataPath + '/Stopwords/stopwords.pkl'
dataloaderFile = dataPath + '/dataloader/train_data.csv'
input_data = InputDataset('/Users/lvkunsheng/Documents/545Final/2020WN_CS545/dataloader/train_data.csv',
                          dataPath,
                          word2vecPath,
                          stopwordsPath)
train_loader = torch.utils.data.DataLoader(dataset=input_data,
                                           batch_size=5, 
                                           shuffle=False)
for i, (X, label) in enumerate(train_loader):
    break
print(X.shape)
# [5, 10, 127, 300]
# batch_size, seq_len, max_news, embedded_size

############################

############################
# Model Building
############################
class baseLineRnnModel(nn.Module):
    def __init__(self, embedded_size, max_news, batch_size, seq_len, hidden_size, num_layers, num_classes):
        super(baseLineRnnModel, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedded_size = embedded_size
        self.batch_size = batch_size
        self.seq_len = seq_len 
        self.hidden_size = hidden_size
        self.max_news = max_news
        ################
        # News-level attention
        ################
        self.Wn = nn.Linear(in_features = self.embedded_size, out_features = 1, bias = True) # 1*300
        self.sigmoid = nn.Sigmoid()
        ################
        
        ################
        # GRU
        ################
        self.gru = nn.GRU(input_size=embedded_size,  # The number of expected features in the input x, which is embedded size 
                          hidden_size=hidden_size, # The number of features in the hidden state h, which is the output dim
                          num_layers=num_layers,  # Number of recurrent layers
                          batch_first=True, # (batch, seq, feature)
                          bidirectional=True, # bidirectional GRU, concatenate two directions
                          dropout=0.3) # dropout
        ################
        
        ################
        # Temporal attention
        ################
        self.Wh = nn.Linear(in_features = 2 * self.hidden_size, out_features = 1, bias = True)
        ################
        
        ################
        # Discriminative Network
        ################
        self.fc = nn.Linear(in_features = 2 * self.hidden_size, out_features = num_classes, bias = True)
        ################
    def forward(self, X):
        # X.shape: [5, 10, 4, 300]
        # batch_size, seq_len, max_news, embedded_size
        
        ################
        # News-level attention
        ################
        Ut = self.sigmoid(self.Wn(X.float())) # Wn(X)/Ut is [5, 10, 4, 1]
        at = nn.Softmax(dim = 3)(Ut.reshape((self.batch_size, self.seq_len, 1, self.max_news))) # [5, 10, 1, 4]
        # [5, 10, 1, 4] * [5, 10, 4, 300] = [5, 10, 1, 300] -> [5, 10, 300]
        dt = torch.matmul(at, X.float()).reshape((self.batch_size, self.seq_len, self.embedded_size))
        ################
        
        ################
        # GRU
        ################
        # input of shape (batch, seq_len, input_size)
        # output (all ht) of shape (batch, seq_len, num_directions * hidden_size)
        # h_n (the last ht) of shape (batch, num_layers * num_directions,  hidden_size)
        h, _ = self.gru(dt) 
        ################
        
        ################
        # Temporal attention
        ################
        o_i = sigmoid_att(self.Wh(h)) # [batch, seq_len, 1]
        beta_i = nn.Softmax(dim = 1)(o_i).reshape((self.batch_size, 1, -1)) # [batch, 1, seq_len]
        # [batch, 1, seq_len] * [batch, seq_len, 2 * hidden_size] = [batch, 1, 2*hidden_size]
        V = torch.matmul(beta_i, h).reshape((self.batch_size, -1)) #  [batch, 2*hidden_size]
        ################
        
        ################
        # Discriminative Network
        ################
        output = self.fc(V)
        ################
        
        return output

model = baseLineRnnModel(embedded_size = 300,
                         max_news = 4,
                         hidden_size = 300,
                         batch_size = 5,
                         seq_len = 10,
                         num_layers = 1,
                         num_classes = 3)
print(X.shape)
y = model(X)
print(y.shape)