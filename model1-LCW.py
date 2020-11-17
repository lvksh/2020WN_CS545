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
# dataPath = '/Users/lvkunsheng/PycharmProjects/cs545Finals/stockDataFromTushare'
# word2vecPath =  dataPath + '/ChineseWord2Vec/sgns.financial.word'
# stopwordsPath = dataPath + '/Stopwords/stopwords.pkl'

class InputDataset(torch.utils.data.Dataset):
    def __init__(self, dataPath, dataloaderFile, stopwordsPath, embedded_size, max_news_cnt):
        # read in the dataloader file and store it in class
        # make sure some rows that have no news at all are discarded 
        self.df_dataloader = pd.read_csv(dataloaderFile)
        self.embedded_size = embedded_size
        self.max_news_cnt  = max_news_cnt
        self.dataPath = dataPath # prefix of all the files
        self.thu1 = thulac.thulac(seg_only=True)   # filter out meaningless word
        # self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vecPath, binary=False)
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
            if len(v_l) > self.max_news_cnt:
                v_l = v_l[:self.max_news_cnt]
            else:
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
                
                corpus.append(vectorize_sl(sentence,word2vec))
        
        label = row[-1]
        if label == 'UP':
            label = 0
        elif label == 'DOWN':
            label = 1
        elif label == 'PRESERVE':
            label = 2
            
        return np.array(corpus), label
    def __len__(self):
        return self.df_dataloader.shape[0]
    

    
# test code for data loader:
############################
dataPath = "/Users/lvkunsheng/PycharmProjects/cs545Finals/stockDataFromTushare" # only change this
word2vecPath =  dataPath + '/ChineseWord2Vec/sgns.financial.word'
stopwordsPath = dataPath + '/Stopwords/stopwords.pkl'
dataloaderFile = dataPath + '/dataloader/train_data.csv'
if 'word2vec' not in dir():
    print("Loading the word2vec model, it takes time, don't worry.")
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vecPath, binary=False)
input_data = InputDataset(dataPath,
                          dataloaderFile,
                          stopwordsPath,
                          embedded_size = 300,
                          max_news_cnt = 4)
train_loader = torch.utils.data.DataLoader(dataset=input_data,
                                           batch_size=5, 
                                           shuffle=False)

for i, (X, label) in enumerate(train_loader):
    break
print(X.shape)
# [5, 10, 4, 300]
# batch_size, seq_len, max_news, embedded_size

############################

############################
# Model Building
############################
class HAN(nn.Module):
    def __init__(self, embedded_size, max_news, batch_size, seq_len, hidden_size, num_layers, num_classes):
        super(HAN, self).__init__()
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
                          dropout=0) # dropout
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
        at = nn.Softmax(dim = 3)(Ut.reshape((-1, self.seq_len, 1, self.max_news))) # [5, 10, 1, 4]
        # [5, 10, 1, 4] * [5, 10, 4, 300] = [5, 10, 1, 300] -> [5, 10, 300]
        dt = torch.matmul(at, X.float()).reshape((-1, self.seq_len, self.embedded_size))
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
        o_i = self.sigmoid(self.Wh(h)) # [batch, seq_len, 1]
        beta_i = nn.Softmax(dim = 1)(o_i).reshape((-1, 1, self.seq_len)) # [batch, 1, seq_len]
        # [batch, 1, seq_len] * [batch, seq_len, 2 * hidden_size] = [batch, 1, 2*hidden_size]
        V = torch.matmul(beta_i, h).reshape((-1, 2*self.hidden_size)) #  [batch, 2*hidden_size]
        ################
        
        ################
        # Discriminative Network
        ################
        output = self.fc(V)
        ################
        
        return output


# test code for model module:
############################    
model = HAN(embedded_size = 300,
             max_news = 4,
             hidden_size = 300,
             batch_size = 5,
             seq_len = 10,
             num_layers = 1,
             num_classes = 3)
print(X.shape)
y = model(X)
print(y.shape)
# # torch.Size([17, 10, 4, 300])
# # torch.Size([17, 3])
############################

# Notes:  
# 1. matmul automaticlly do batch matrix multiplication: (*, n, m) * (*, m, p) = (*, n, p)  
# 2. Linear layers will keep the dimension in the front and only transform the last dimension: (*, n) -> (*, m)  
# 3. sigmoid is element-wise  
# 4. softmax need to decide which dimension to sum over 
# 5. when specifying shape in forward, try to set -1 for batch_size. Then we don't have to worry about drop_last in dataloader. 

############################

############################
# Model Training 
############################
# Initiation
dataPath = '/Users/lvkunsheng/PycharmProjects/cs545Finals/stockDataFromTushare' # change this
word2vecPath =  dataPath + '/ChineseWord2Vec/sgns.financial.word'
stopwordsPath = dataPath + '/Stopwords/stopwords.pkl'
dataloaderPath_train = dataPath + '/dataloader/train_data.csv'
dataloaderPath_valid = dataPath + '/dataloader/cv_data.csv'
dataloaderPath_test = dataPath + '/dataloader/test_data.csv'
if 'word2vec' not in dir():
    print("Loading the word2vec model, it takes time, don't worry.")
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vecPath, binary=False)num_epochs = 2 # training epoch
learning_rate = 0.1 
batch_size = 32
embedded_size = 300
max_news = 4 # maximum number of news in one data for one stock, pad or truncate if otherwise
hidden_size = 300 # hidden_state size in GRU
seq_len = 10 # window_size N = 10
num_layers = 1 # num_layers in GRU
num_classes = 3 # number of predicted classes
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data loader
train_data = InputDataset(dataPath,
                          dataloaderPath_train,
                          stopwordsPath,
                          embedded_size = 300,
                          max_news_cnt = 4)

valid_data = InputDataset(dataPath,
                          dataloaderPath_valid,
                          stopwordsPath,
                          embedded_size = 300,
                          max_news_cnt = 4)

test_data  = InputDataset(dataPath,
                          dataloaderPath_test,
                          stopwordsPath,
                          embedded_size = 300,
                          max_news_cnt = 4)

train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size, 
                                           shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_data,
                                           batch_size=batch_size, 
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                           batch_size=batch_size, 
                                           shuffle=False)
# model
model = HAN(embedded_size = embedded_size,
            max_news = max_news,
            hidden_size = hidden_size,
            batch_size = batch_size,
            seq_len = seq_len,
            num_layers = num_layers,
            num_classes = num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
# Train the model
import time
total_time = 0
total_step = len(train_loader)
for epoch in range(num_epochs):
    time1 = time.time()
    model.train() # train mode
    print("Start training...")
    for i, (X, labels) in enumerate(train_loader):
        X = X.to(device)
        labels = labels.to(device)
        
        # Forward pass
        y = model(X)
        loss = criterion(y, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    time2 = time.time()
    epoch_time = time2-time1
    total_time+=epoch_time
    print(f'Epoch {epoch} completed, cost time {round(epoch_time,2)} s.')
    
    print("Runing validation")
    
    model.eval() # evaluation mode 
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (X, labels) in enumerate(valid_loader):
            X = X.to(device)
            labels = labels.to(device)
            y = model(X)
            _, predicted = torch.max(y.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Valid Accuracy of the model: {} %'.format(100 * correct / total)) 

############################

############################
# Model Testing 
############################
model.eval() # evaluation mode 
with torch.no_grad():
    correct = 0
    total = 0
    for i, (X, labels) in enumerate(test_loader):
        X = X.to(device)
        labels = labels.to(device)
        y = model(X)
        _, predicted = torch.max(y.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy of the model: {} %'.format(100 * correct / total))
############################
