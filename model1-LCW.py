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


class InputDataset(torch.utils.data.Dataset):
    def __init__(self, dataloaderFile, dataPath, word2vecPath, stopwordsPath):
        # read in the dataloader file and store it in class
        # make sure some rows that have no news at all are discarded 
        self.df_dataloader = pd.read_csv(dataloaderFile)
        self.df_dataloader = self.df_dataloader.iloc[:,1:]
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
        def clean_sentence(s,stopwords)->[str]:
            text_split = thu1.cut(s, text=True)
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
            vec = np.zeros((300,))
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
            return v_l    
        
        row = self.df_dataloader.iloc[index, :]
        corpus = []
        for i in range(1, row.shape[0] - 2): # for each date
            if pd.isna(row[i]): # no news on this date
                corpus.append([])
            else:
                sentence = clean_file(self.dataPath + '/' + row[i],
                                      self.stopwords)
                
                corpus.append(vectorize_sl(sentence,self.word2vec))
        return corpus, row[-1]
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
X
############################

############################
# Model Building
############################
class baseLineRnnModel(nn.Module):
    def __init__(self, embedded_size, hidden_size, num_layers, num_classes):
        super(baseLineRnnModel, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedded_size = embedded_size
        
        ################
        # news attention
        ################
        self.Wn = torch.tensor((1,self.embedded_size), requires_grad = True) # 1*300
        ################
        
        ################
        # GRU
        ################
        self.gru = nn.GRU(input_size=embedded_size,  # The number of expected features in the input x, which is embedded size 
                          hidden_size=hidden_size, # The number of features in the hidden state h, which is the output dim
                          num_layers=1,  # Number of recurrent layers
                          batch_first=True, # (batch, seq, feature)
                          bidirectional=True, # bidirectional GRU, concatenate two directions
                          dropout=0.3) # dropout
        ################
        
    def forward(self, X):
        # 10 * 300 * Lt * batch_size 
        # nt: Lt * batch_size * embedded_size
        
        ################
        # news attention
        ################
        Ut = nn.Sigmoid(self.Wn * nt)
        at = nn.Softmax(Ut)
        dt = nt * at # 300*1
        ################
        
        ################
        # GRU
        ################
        
        
        ################
        loss = None
        return loss
