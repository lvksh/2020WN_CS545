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
    def __init__(self, dataPath, dataloaderFile, embedded_size, max_news_cnt):
        # read in the dataloader file and store it in class
        # make sure some rows that have no news at all are discarded 
        self.df_dataloader = pd.read_csv(dataloaderFile)
        self.embedded_size = embedded_size
        self.max_news_cnt  = max_news_cnt
        self.dataPath = dataPath # prefix of all the files
     
    def __getitem__(self, index):
        # read in one line of data
        # return x, y
        # [['今天', '股市', '又', '降', '了'], sentence2, ..., sentence10], label
        # [vector of corpus1, vector of corpus2, ..., vector of corpus10], label
          
        row = self.df_dataloader.iloc[index, :]
        corpus = []
        for i in range(2, 12): # for each date
            if pd.isna(row[i]): # no news on this date
                # Pads with all 0
                corpus.append([[0]*self.embedded_size]*self.max_news_cnt)
            else:
                sentence = np.genfromtxt(self.dataPath + '/stockNewsVec/' + row[i], delimiter=',')
                sentence = sentence.reshape((-1,300)) # avoid being (300,)
                # pad with 0s
                if len(sentence) > self.max_news_cnt:
                    sentence = sentence[:self.max_news_cnt]
                elif len(sentence) < self.max_news_cnt:
                    sentence = np.append(sentence, np.array([[0]*self.embedded_size]*(self.max_news_cnt-len(sentence))), axis = 0)
                else:
                    pass

                corpus.append(sentence)
        
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
dataPath = "/content/drive/MyDrive/Colab Notebooks/SI671_Final_Project/stockDataFromTushare545" # only change this
dataloaderFile = dataPath + '/dataloader/train_data.csv'

input_data = InputDataset(dataPath,
                          dataloaderFile,  
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
        self.num_layers = num_layers
        ################
        # News-level attention
        ################
        self.Wn = nn.Linear(in_features = self.embedded_size, out_features = self.embedded_size, bias = True) # 1*300
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
                          dropout=0.5) # dropout
        ################
        
        ################
        # Temporal attention
        ################
        self.Wh = nn.Linear(in_features = 2 * self.hidden_size, out_features = 2 * self.hidden_size, bias = True)
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
        Ut = self.sigmoid(self.Wn(X)) # Wn(X)/Ut is [5, 10, 4, 300]
        at = nn.Softmax(dim = 2)(Ut)  # [5, 10, 4, 300]
        dt = torch.mul(at, X) # element-wise multiplication
        dt = torch.sum(dt, 2) # [5, 10, 300]

        # at = nn.Softmax(dim = 3)(Ut.reshape((-1, self.seq_len, 1, self.max_news))) # [5, 10, 1, 4]
        # # [5, 10, 1, 4] * [5, 10, 4, 300] = [5, 10, 1, 300] -> [5, 10, 300]
        # dt = torch.matmul(at, X).reshape((-1, self.seq_len, self.embedded_size))
        # dt = nn.LeakyReLU()(dt)
        ################
        
        ################
        # GRU
        ################
        # input:
        #   x of shape (batch, seq_len, input_size)
        #   h0 of shape (num_layers * num_directions, batch, hidden_size)
        # output:
        #   all ht of shape (batch, seq_len, num_directions * hidden_size)
        #   h_n (the last ht) of shape (batch, num_layers * num_directions,  hidden_size)

        def truncated_normal_(self, tensor,mean=0,std=0.09):
            # returning randomized truncated normal distribution 
            with torch.no_grad():
                size = tensor.shape
                tmp = tensor.new_empty(size+(4,)).normal_()
                valid = (tmp < 2) & (tmp > -2)
                ind = valid.max(-1, keepdim=True)[1]
                tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
                tensor.data.mul_(std).add_(mean)
                return tensor

        h0 = torch.zeros(self.num_layers*2, X.size(0), self.hidden_size).to(device)
        h0 = truncated_normal_(self, h0)
        h, _ = self.gru(dt,h0) 
        ################
        
        ################
        # Temporal attention
        ################
        o_i = self.sigmoid(self.Wh(h)) # [5, 10, 2 * self.hidden_size]
        beta_i = nn.Softmax(dim = 1)(o_i) # [5, 10, 2 * self.hidden_size]
        V = torch.mul(beta_i, h) # [5, 10, 2 * self.hidden_size]
        V = torch.sum(V, 1) # [5, 2 * self.hidden_size]

        ################
        
        ################
        # Discriminative Network
        ################
        #output = self.fc(V)
        output = nn.LeakyReLU()(self.fc(V))
        output = nn.Softmax(dim=1)(output)
        ################
        
        return output


# test code for model module:
############################  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
model = HAN(embedded_size = 300,
                         max_news = 4,
                         hidden_size = 300,
                         batch_size = 128,
                         seq_len = 10,
                         num_layers = 2,
                         num_classes = 3).to(device)
X = X.float().to(device)
print(X.shape)
y = model(X)
print(y.shape)
# torch.Size([5, 10, 4, 300])
# torch.Size([5, 3])
############################

# Notes:  
# 1. matmul automaticlly do batch matrix multiplication: (*, n, m) * (*, m, p) = (*, n, p)  
# 2. Linear layers will keep the dimension in the front and only transform the last dimension: (*, n) -> (*, m)  
# 3. sigmoid is element-wise  
# 4. softmax need to decide which dimension to sum over 
# 5. when specifying shape in forward, try to set -1 for batch_size. Then we don't have to worry about drop_last in dataloader. 
# 6. remember to apply .to(device) on both model and data 
############################

############################
# Model Training 
############################
# Initiation
dataPath = "/content/drive/MyDrive/Colab Notebooks/SI671_Final_Project/stockDataFromTushare545" # change this
dataloaderPath_train = dataPath + '/dataloader/train_data.csv'
dataloaderPath_valid = dataPath + '/dataloader/cv_data.csv'
dataloaderPath_test = dataPath + '/dataloader/test_data.csv'
    
num_epochs = 10 # training epoch
learning_rate = 0.1 
batch_size = 128
embedded_size = 300
max_news = 4 # maximum number of news in one data for one stock, pad or truncate if otherwise
hidden_size = 100 # hidden_state size in GRU
seq_len = 10 # window_size N = 10
num_layers = 2 # num_layers in GRU
num_classes = 3 # number of predicted classes
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data loader
train_data = InputDataset(dataPath,
                          dataloaderPath_train,
                          embedded_size = 300,
                          max_news_cnt = max_news)

valid_data = InputDataset(dataPath,
                          dataloaderPath_valid,
                          embedded_size = 300,
                          max_news_cnt = max_news)

test_data  = InputDataset(dataPath,
                          dataloaderPath_test,
                          embedded_size = 300,
                          max_news_cnt = max_news)

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
model = model.to(device)
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
        X = X.float().to(device)
        labels = labels.to(device)
        
        # Forward pass
        y = model(X) # 0.005s
        loss = criterion(y, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 0.03s 
        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    time2 = time.time()
    epoch_time = time2-time1
    total_time+=epoch_time
    print(f'Epoch {epoch} completed, cost time {round(epoch_time,2)} s.')
    torch.save(model.state_dict(), dataPath + '/HAN_1120_newATT.torch') 
    print("Runing validation")
    
    model.eval() # evaluation mode 
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (X, labels) in enumerate(valid_loader):
            X = X.float().to(device)
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
