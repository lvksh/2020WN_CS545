import pickle, argparse, os, sys
from sklearn.metrics import accuracy_score
import numpy as np
import random
import torch
import torch.nn as nn
import torch.functional as F
import pickle
import re
import pandas as pd

class InputDataset(torch.utils.data.Dataset):
    def __init__(self, dataPath, npyFile, dataloaderFile, embedded_size, max_news_cnt):
        # read in the dataloader file and store it in class
        # make sure some rows that have no news at all are discarded 
        self.npy = np.load(npyFile)
        self.df  = pd.read_csv(dataloaderFile)
        self.embedded_size = embedded_size
        self.max_news_cnt  = max_news_cnt
        self.dataPath = dataPath # prefix of all the files
     
    def __getitem__(self, index):
        # read in one line of data
        # return x, y
        # [['今天', '股市', '又', '降', '了'], sentence2, ..., sentence10], label
        # [vector of corpus1, vector of corpus2, ..., vector of corpus10], label
          
        corpus = self.npy[index, :, :, :]
        
        label = self.df['label'][index]
        if label == 'UP':
            label = 0
        elif label == 'DOWN':
            label = 1
        elif label == 'PRESERVE':
            label = 2
            
        return corpus, label
    def __len__(self):
        return self.df.shape[0]

############################
# Model Building
############################
class HAN(nn.Module):
    def __init__(self, embedded_size, max_news, batch_size, seq_len, hidden_size, num_layers, num_classes, dropout):
        super(HAN, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedded_size = embedded_size
        self.batch_size = batch_size
        self.seq_len = seq_len 
        self.hidden_size = hidden_size
        self.max_news = max_news
        self.num_layers = num_layers
        self.dropout=dropout

        ################
        # News-level attention
        ################
        self.Wn = nn.Linear(in_features = self.embedded_size, out_features = 1, bias = True) # 1*300
        torch.nn.init.xavier_uniform_(self.Wn.weight)
        ################
        
        ################
        # GRU
        ################
        self.gru = nn.GRU(input_size=embedded_size,  # The number of expected features in the input x, which is embedded size 
                          hidden_size=hidden_size, # The number of features in the hidden state h, which is the output dim
                          num_layers=num_layers,  # Number of recurrent layers
                          batch_first=True, # (batch, seq, feature)
                          bidirectional=True, # bidirectional GRU, concatenate two directions
                          dropout=self.dropout) # dropout
        ################
        
        ################
        # Temporal attention
        ################
        self.Wh = nn.Linear(in_features = 2 * self.hidden_size, out_features = 1, bias = True)
        torch.nn.init.xavier_uniform_(self.Wh.weight)
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
        Ut = nn.LeakyReLU()(self.Wn(X.float())) # Wn(X)/Ut is [5, 10, 4, 1]
        Ut = torch.squeeze(Ut).unsqueeze(2) # [5, 10, 4, 1] -> [5, 10, 4] -> [5, 10, 1, 4]
        at = nn.Softmax(dim = 3)(Ut) # [5, 10, 1, 4]
        # [5, 10, 1, 4] * [5, 10, 4, 300] = [5, 10, 1, 300] 
        dt = torch.matmul(at, X) 
        dt = torch.squeeze(dt) # [5, 10, 1, 300] -> [5, 10, 300]
        ################
        
        ################
        # GRU
        ################
        # input:
        #   x of shape (batch, seq_len, input_size)
        #   h0 of shape (num_layers * num_directions, batch, hidden_size)
        # output:
        #   all ht of shape (batch, seq_len, num_directions * hidden_size)
        #   h_n (the last ht) of shape (batch, num_layers * num_directions,  hidden_size
        h0 = torch.zeros(self.num_layers*2, X.size(0), self.hidden_size).to(device)
        nn.init.orthogonal_(h0)
        h, _ = self.gru(dt,h0) 
        ################
        
        ################
        # Temporal attention
        ################
        o_i = nn.LeakyReLU()(self.Wh(h)) # [5, 10, 1]
        beta_i = nn.Softmax(dim = 1)(o_i) # [5, 10, 1]
        V = torch.matmul(beta_i.unsqueeze(3), h.unsqueeze(2)) # [5, 10, 1, 2 * self.hidden_size]
        V = torch.sum(torch.squeeze(V), 1) # [5, 2 * self.hidden_size]
        ################
        
        ################
        # Discriminative Network
        ################
        output = self.fc(V)
        output = nn.Softmax(dim=1)(output)
        ################
        
        return output


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
dataPath = "/Users/lvkunsheng/PycharmProjects/cs545Finals/stockDataFromTushare" # change this
dataloaderPath_train = dataPath + '/dataloader/train_data_full.csv'
dataloaderPath_valid = dataPath + '/dataloader/cv_data_full.csv'
dataloaderPath_test = dataPath + '/dataloader/test_data_full.csv'

npyFile_train = dataPath + '/dataloader/train_bert_np_full.npy'
npyFile_valid = dataPath + '/dataloader/cv_bert_np_full.npy'
npyFile_test = dataPath + '/dataloader/test_bert_np_full.npy'
num_epochs = 30 # training epoch
learning_rate = 0.5
batch_size = 512
embedded_size = 768
max_news = 4 # maximum number of news in one data for one stock, pad or truncate if otherwise
hidden_size = 384 # hidden_state size in GRU
seq_len = 10 # window_size N = 10
num_layers = 2 # num_layers in GRU
num_classes = 3 # number of predicted classes
dropout = 0.5
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data loader
train_data = InputDataset(dataPath,
                          npyFile_train,
                          dataloaderPath_train,
                          embedded_size = embedded_size,
                          max_news_cnt = max_news)

valid_data = InputDataset(dataPath,
                          npyFile_valid,
                          dataloaderPath_valid,
                          embedded_size = embedded_size,
                          max_news_cnt = max_news)

test_data  = InputDataset(dataPath,
                          npyFile_test,
                          dataloaderPath_test,
                          embedded_size = embedded_size,
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
            num_classes = num_classes,
            dropout = dropout)
model = model.to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay = 1e-4)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=3, verbose = True)

import wandb
config = dict(
    epochs=30,
    classes=3,
    batch_size=512,
    learning_rate=0.5)
wandb.init(config=config)
wandb.watch(model,log = 'all')
# Train the model
import time
total_time = 0
elapsed_time = 0
total_step = len(train_loader)
train_loss = []
valid_loss = []
train_acc  = []
valid_acc  = []

for epoch in range(num_epochs):
    time1 = time.time()
    model.train() # train mode
    average_loss = 0
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
        elapsed_time += time.time()-time1
        average_loss += loss.item()

        
        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Elapsed Time: {}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item(), format_time(elapsed_time)))
    average_loss/=len(train_loader)
    train_loss.append(average_loss)
    
    
    
    time2 = time.time()
    epoch_time = time2-time1
    total_time+=epoch_time
    print(f'Epoch {epoch} completed, cost time {format_time(epoch_time)}.')
    torch.save(model.state_dict(), dataPath + '/models/HAN_1210.torch') 
    print("Runing validation")
    
    model.eval() # evaluation mode 
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (X, labels) in enumerate(train_loader):
            X = X.float().to(device)
            labels = labels.to(device)
            y = model(X)
            _, predicted = torch.max(y.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Train Accuracy: {} %, Loss: {}'.format(100 * round(correct / total, 4), round(average_loss,3)))
    train_acc.append(correct / total)   
    
    average_loss = 0
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
            average_loss +=  criterion(y, labels).item()
        average_loss/=len(valid_loader)
        print('Valid Accuracy: {} %, Loss: {}'.format(100 * round(correct / total, 4), round(average_loss,3))) 
        valid_loss.append(average_loss)
        valid_acc.append(correct / total) 

    
    scheduler.step(average_loss)


    wandb.log({'train_loss':train_loss[-1],
               'valid_loss':valid_loss[-1],
               'train_acc':train_acc[-1],
               'valid_acc':valid_acc[-1])
############################
