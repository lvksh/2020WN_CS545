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
import numpy as np
import datetime
import time 
from transformers import BertTokenizer

def berterlize(file):
    # given a txt file having all news 
    f = open(file,encoding='utf-8')
    txt = f.read()
    txt = re.sub(r'(\(image_address="https|image_address="http)?:\/\/(\w|\.|\/|\?|\=|\&|\%\\)*\b', '', txt)
    txt = re.sub(r'\\u[a-zA-Z0-9]*\b', '', txt)
    txt = txt.strip().rstrip()
    txt = txt.split('\n')
    f.close()
    corpus = []
    for sentence in txt:
        inputs = tokenizer(sentence, return_tensors="pt")
        # length greater than 512
        if inputs.input_ids.shape[1] <= 512:
            outputs = model(**inputs)
        else:
            inputs['input_ids'] = inputs['input_ids'][0,:512].reshape((1,-1))
            inputs['token_type_ids'] = inputs['token_type_ids'][0,:512].reshape((1,-1))
            inputs['attention_mask'] = inputs['attention_mask'][0,:512].reshape((1,-1))
            outputs = model(**inputs)

        last_hidden_states = outputs.pooler_output
        corpus.append(last_hidden_states.detach().numpy().reshape(-1))
    return np.array(corpus)

def process(stockFolder):
    print(f"Berterlizing {stockFolder}...\n")
    if stockFolder[:2] != 'sz' and stockFolder[:2] != 'sh':
        return
    for txtFile in os.listdir(dataPath + '/stockNews/' + stockFolder):
        tmp = berterlize(dataPath + '/stockNews/' + stockFolder + '/' + txtFile)
        if not os.path.exists(dataPath + '/stockNewsVecBert/' + stockFolder):
            os.makedirs(dataPath + '/stockNewsVecBert/' + stockFolder)
        np.savetxt(dataPath + '/stockNewsVecBert/' + stockFolder + '/' + txtFile, tmp, delimiter=",")
    pass

from multiprocessing import Pool
from transformers import BertTokenizer, BertModel
import torch

dataPath = 'E:/Jupyter/545stock/stockDataFromTushare/stockDataFromTushare'

tokenizer = BertTokenizer.from_pretrained(dataPath + '/ChineseWord2Vec/FinBERT/')
model = BertModel.from_pretrained(dataPath + '/ChineseWord2Vec/FinBERT/', return_dict=True)

id_list = os.listdir(dataPath + '/stockNews/')
from tqdm import tqdm
for i in tqdm(range(len(id_list))):
    stockFolder = id_list[i]
    process(stockFolder)