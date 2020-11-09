import pickle, argparse, os, sys
from sklearn.metrics import accuracy_score
import numpy as np
import random
import torch
import torch.nn as nn
import torch.functional as F

class baseLineRnnModel(nn.Module):
    def __init__(self):
        super(baseLineRnnModel, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # TODO: use Chinese Word2Vec pretrain model to transform the original input data
        # TODO: Model Structure

    def forward(self):
        pass

class inputDataset(torch.utils.data.Dataset):
    def __init__(self, filePath):
        # TODO: In here we do the most job about organizing the news for __getitem__()
        pass
    def __getitem__(self, index):
        # TODO: return data and label for one update process
        # Here it means news corpus from day d-N to d-1 and the result in day d for one stock.
        pass
    def __len__(self):
        # TODO: return the total length of the input
        pass

def trainModel(model, dataloader, optimizer, num_epoch):
    # TODO: main loop of training
    return model

def train(filePath):
    assert os.path.isfile(filePath), 'Training file does not exist'
    # TODO: training wrapper
    dataset = MyDataSet(filePath)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    net = baseLineRnnModel()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    trained_model = train_model(net, dataloader, optimizer, num_epoch)
    return trained_model

def test(modelFile, dataFile, labelFile):
    assert os.path.isfile(model_file), 'Model file does not exist'
    assert os.path.isfile(data_file), 'Data file does not exist'
    assert os.path.isfile(label_file), 'Label file does not exist'

    # TODO: evaluate in the test
    pass

