import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
 
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def plotCM(classes, matrix, savname):
    """classes: a list of class names"""
 
    # Normalize by row
    matrix = matrix.astype(np.float)
    linesum = matrix.sum(1)
    linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
    matrix /= linesum
 
    # plot
    plt.switch_backend('agg')
    fig = plt.figure()
 
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, cmap = 'Blues', alpha = 0.7)
    fig.colorbar(cax)
 
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
 
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(i, j, str('%.2f' % (matrix[i, j] * 100)), va='center', ha='center')
 
    ax.set_xticklabels([''] + classes)
    ax.set_yticklabels([''] + classes)
 
    # save
    plt.savefig(savname)

# Data Processing for tabular 
import numpy as np
import pandas as pd
dataPath = '/Users/lvkunsheng/PycharmProjects/cs545Finals/stockDataFromTushare'
train_npy = np.load(dataPath + '/dataloader/train_bert_np_full.npy') # [n * seq_len * max_news * embedded_size]
cv_npy = np.load(dataPath + '/dataloader/cv_bert_np_full.npy')
test_npy = np.load(dataPath + '/dataloader/test_bert_np_full.npy')

train = []
for i in range(train_npy.shape[0]):
    train.append(np.mean(train_npy[i, :, :, :], axis = 1).reshape(-1))
train = pd.DataFrame(train)


cv = []
for i in range(cv_npy.shape[0]):
    cv.append(np.mean(cv_npy[i, :, :, :], axis = 1).reshape(-1))
cv = pd.DataFrame(cv)

test = []
for i in range(test_npy.shape[0]):
    test.append(np.mean(test_npy[i, :, :, :], axis = 1).reshape(-1))
test = pd.DataFrame(test)


dataPath = "/Users/lvkunsheng/PycharmProjects/cs545Finals/stockDataFromTushare" # change this
dataloaderPath_train = dataPath + '/dataloader/train_data_full.csv'
dataloaderPath_valid = dataPath + '/dataloader/cv_data_full.csv'
dataloaderPath_test = dataPath + '/dataloader/test_data_full.csv'

df_train = pd.read_csv(dataloaderPath_train)
df_cv    = pd.read_csv(dataloaderPath_valid)
df_test  = pd.read_csv(dataloaderPath_test)

train['label'] = df_train['label']
cv['label'] = df_cv['label']
test['label'] = df_test['label']

# rf
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(200, random_state = 0, n_jobs = 7)
rf.fit(train.iloc[:,:6780], train['label'])
from sklearn.metrics import accuracy_score
pred_train = rf.predict(train.iloc[:,:6780])
print(f"Accuracy in train set: {accuracy_score(train['label'], pred_train)}")
pred_cv = rf.predict(cv.iloc[:,:6780])
print(f"Accuracy in valid set: {accuracy_score(cv['label'], pred_cv)}")
pred_test = rf.predict(test.iloc[:,:6780])
print(f"Accuracy in test set: {accuracy_score(test['label'], pred_test)}")
from sklearn.metrics import confusion_matrix
confusion_mat=confusion_matrix(test['label'],pred_test)
plotCM(['UP', 'DOWN', 'PRESERVE'], confusion_mat, 'rf_confusion_matrix')

col = ['acc','UP_precision','DOWN_precision','PRESERVE_precision','UP_recall','DOWN_recall','PRESERVE_recall']
res = pd.DataFrame()
res = res.append(pd.DataFrame([accuracy_score(test['label'], pred_test)] + \
            list(precision_score(test['label'],pred_test, average = None)) + \
            list(recall_score(test['label'],pred_test, average = None))).transpose())

# lightgbm
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
hist = HistGradientBoostingClassifier(random_state = 0)
hist.fit(train.iloc[:,:6780], train['label'])
from sklearn.metrics import accuracy_score
pred_train = hist.predict(train.iloc[:,:6780])
print(f"Accuracy in train set: {accuracy_score(train['label'], pred_train)}")
pred_cv = hist.predict(cv.iloc[:,:6780])
print(f"Accuracy in valid set: {accuracy_score(cv['label'], pred_cv)}")
pred_test = hist.predict(test.iloc[:,:6780])
print(f"Accuracy in test set: {accuracy_score(test['label'], pred_test)}")
from sklearn.metrics import confusion_matrix
confusion_mat=confusion_matrix(test['label'],pred_test)
plotCM(['UP', 'DOWN', 'PRESERVE'], confusion_mat, 'hist_confusion_matrix')
res = res.append(pd.DataFrame([accuracy_score(test['label'], pred_test)] + \
            list(precision_score(test['label'],pred_test, average = None)) + \
            list(recall_score(test['label'],pred_test, average = None))).transpose())

# mlp
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state = 0)
mlp.fit(train.iloc[:,:6780], train['label'])
from sklearn.metrics import accuracy_score
pred_train = mlp.predict(train.iloc[:,:6780])
print(f"Accuracy in train set: {accuracy_score(train['label'], pred_train)}")
pred_cv = mlp.predict(cv.iloc[:,:6780])
print(f"Accuracy in valid set: {accuracy_score(cv['label'], pred_cv)}")
pred_test = mlp.predict(test.iloc[:,:6780])
print(f"Accuracy in test set: {accuracy_score(test['label'], pred_test)}")
from sklearn.metrics import confusion_matrix
confusion_mat=confusion_matrix(test['label'],pred_test)
plotCM(['UP', 'DOWN', 'PRESERVE'], confusion_mat, 'mlp_confusion_matrix')
res = res.append(pd.DataFrame([accuracy_score(test['label'], pred_test)] + \
            list(precision_score(test['label'],pred_test, average = None)) + \
            list(recall_score(test['label'],pred_test, average = None))).transpose())

res.columns = col
res.index = ['rf','lightgbm','mlp'] 