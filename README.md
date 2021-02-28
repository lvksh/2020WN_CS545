# 2020WN_CS545
Final project repository for umich EECS 545 Machine Learning. 

## Codes

data collection (crawling.py)  
    - Collect news for 300 stocks from 5 major finance websites using TUSHARE from 2019-01-01 to 2020-10-12   
    - Collect daily close price for 300 stocks from 2019-01-01 to 2020-10-12  

data cleansing   
    - remove messy notations like \xa0 and \r\n, patterns that occurs really often, remove stopwords achieved from tsinghua university. (assignNews.py)

data processing   
    - assign news to stocks based on the name of stocks occurs in the news. (assignNews.py)  
    - using chinese FINBERT pretrained model to transform each news to vector. (berterlize.py)  
    - build up the dataframe which includes ten-day news file path and the rise percent in day 11,  determine the label based on the proportion of three labels, delete rows that have no news at all. (dataloader_creation_splitting.py)  
        - To reduce IOs, we read in all the data we need for training, valid, testing in a tensor using numpy and store it ahead of time. (readin_all.py)  
    - split our whole datasets into train, test, valid. (dataloader_creation_splitting.py)  
    	- v2: Because of the sparsity of news, we choose to update the dataloader dataframe by only choosing former 10 days that have news.   (dataloader_creation_splitting.py)  

Model building and validation evaluation (HAN.py)   

Some baseline model and plotting functions (baseline.py)  
    



## MATERIALS
- [Tushare - Stock data scrawling tool](https://tushare.pro/register?reg=395357)
- [Listening to Chaotic Whispers: A Deep Learning Framework for News-oriented Stock Trend Prediction
Ziniu](https://arxiv.org/pdf/1712.02136v3.pdf)
  - kunsheng's note:
    1. This artical only makes use of news data, we can make use of some historical price and the INDEX price too.
    2. The GRU inside the network structure can be replaced by transformers layers.
    3. The discriminative layers is MLP, can we train the models and extract the last layer to get all the features and feed them into something like xgboost?
    4. Using the probability of 'UP' minus the probability of 'DOWN' as measurement to buy topK stocks with the open price, and sell them with the close price, achieving 50% annual return
- **[Coding Repo about HAN in LCW](https://github.com/Pie33000/stock-prediction)**
- **[BERT colab tutorial](https://colab.research.google.com/drive/1Y4o3jh3ZH70tl6mCd76vz_IxX23biCPP#scrollTo=D6TKgyUzPIQc)**
- [Pretrained BERT in huggingface](https://huggingface.co/transformers/v2.2.0/pretrained_models.html)
- [THULAC Chinese word split tool](https://github.com/thunlp/THULAC-Python)
- [中文word2vec预训练](https://github.com/Embedding/Chinese-Word-Vectors)
- [一文看懂 Attention（本质原理+3大优点+5大类型）](https://zhuanlan.zhihu.com/p/91839581)
- [【NLP】Transformer模型原理详解](https://zhuanlan.zhihu.com/p/44121378)  
- [搞懂Transformer结构，看这篇PyTorch实现就够了（上）](https://zhuanlan.zhihu.com/p/48731949)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [BERT中文情感分类](https://github.com/Toyhom/Hei_Dong/tree/master/Project/%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB)


## TODOS
- ~~project proposal~~  
- Things to figure out before everything
  - ~~Common label definitions used in stock prediction~~ 
    - UP, DOWN, PRESERVE - Three class classification problem
  - ~~How can the label related to the actual applications?~~ 
    - The probability of UP can be used to selected stocks. 
  - ~~How's the final data frame looks like?~~ 
    - stock_name, day-10-path, day-9-path, ..., day-1-path, label
    - read in one row at a time and read in corresponding txt files in dataloader.
- Data Retrieval: Using crawling methods to crawl all the data we need for this project.  
  - ~~Historical stock open price, close price, all interesting factors for each stock. (kunsheng)~~ 
    - 2020-10-03 kunsheng: Found an amazing crawling tools [Tushare](https://tushare.pro/register?reg=395357)
  - ~~Historical stock info for INDEX stock like '沪深300' (kunsheng)~~ 
  - ~~Corresponding news for each stock with published date (kunsheng)~~ 
    - 2020-10-14: Use Tushare to crawl all the quick news from 2019-01-01 till now, still need to assign to corresponding stocks with names or ids.
  - ~~Overall news about finance or stock market with published date (kunsheng)~~ 
    - 2020-10-14 kunsheng: Use Tushare to crawl all the general news from 2019-01-01 till now, save some samples in git.
  - ~~Assign News to stocks (kunsheng)~~ 
    - 2020-11-12 kunsheng: if stock name or stock number appear in news, then this news is related to this stock. Use multiprocessing library to accelerate. 
- Data Cleansing: Data from the crawling process may be messy and hard to use, so we need to
preprocess the data first.  
  - ~~Numerical Feature Engineering: It’s hard to predict the stock using solely the text, so we need to~~
~~construct some stock factors like 5-day variance, etc.  (dengrui)~~
  ~~- Text Feature Engineering: word split, stop words, quality control (yindim)~~
  - ~~Dataset splitting: Based on the ratio of 3 classes to split the dataset. (jingxian)~~
    - 2020-11-17 jingxian: we need to compute the threhold to divide UP, DOWN, PRESERVE label later
- Model Building: We will try different machine learning regressors and compare their performance.
  - ~~data loader (kunsheng)~~ 
    - 2020-11-13 kunsheng: finish data loader module, still need to update when preprocessing finish.
  - Short term goal: 
    - ~~Implement News-RNN, HAN from LCW;~~ Build the whole framework of model building, data loading, model evaluation and back-testing.
      - 2020-11-18 kunsheng: Implemented HAN, training time for 1 epoch is about 13h but not converging. Still need to accelerate using pytorch lightning, find better data sources (4 news for one stock in one day for now.), find more powerful servers to run our script, modify the structure and tune parameters.
    - ~~Implement BERT pretraining model by adding classification head.~~
      - 2020-12-05 kunsheng: convert all news to Bert vectors using transformer, then utilize the HAN structure. Note that there are problems using multiprocessing.
  - Mid term goal: Implement HAN-SPL from LWC
  - Long term goal: Replace GRU structures in HAN with multi-head self-attention; Use data other then only news; Utilize pre-trained model like BERT in embedding layers, etc...
- Model Applications: If we have time, we will extract some up-to-date data and validate our model
on real data.  
- Some notes on accelerations:
  - 2020-11-30 kunsheng: read in all the data instead of reading them each time in dataloader, which saves alot by reducing IOs.


## Limitations

In this project, we carefully examine the detail of the Hybrid Attention Networks framework and re-implement it using PyTorch. When applying the original settings in newly crawled stock news, we also try to add some state-of-the-art techniques like replacing Word2Vec with pre-trained BERT. Compared to several baseline classifiers, we found that HAN does have the ability to help model focus on more important news and have a better ability to extract features inside natural languages and achieve relatively great performance in stock prediction task. During the whole project we encounter many obstacles:



**Data Preparation**

At first we decide to crawl the news which are already been assigned to each stocks by the operator of Sina website, but afterwards we found that there were a lot of missing dates for many stocks. We have to buy a crawling program which can only crawl general news that are not necessarily connected to certain stock. Assigning news to stocks are frustrating because it's both time-consuming and not efficient. Some stocks have the name like "MaoTai Cooperation", but news may call it "MaoTai" instead, which will not be assigned to this stock. 

Because of the difficuty in assigning news, we encounter another problem which is news sparsity. Almost 30\% of the samples have no news in the former 10 days. After we delete all of them, almost 50\% of them are missing 7 days out of 10. The quality of the data really impact our training process, many gradients are hence zeroed out and impossible to train. Finally we come up with a somehow brilliant idea, which is to only take the former 10 days that have news into account. It's absolutely not a perfect solution, but it's enough to validate our implementations.

Besides, when using Pytorch and Transformer library to transform news to BERT vectors, we found that the multiprocessing library in Python is not working at all. I have searched all over but only found another library to do sentence2vec task in parallel but cannot use all the pre-trained model in transformer library. Hence it took us 4 days to transform all the news into BERT vectors, which should be able to be reduced to several hours with 16 cores. If you have any advice on it please let me know!

When we are writing dataloader module, at first we organize all the paths to files in a data frame and try to read them in every time we try to get an item. However, it turns out that it takes about 12 hours for only 1 epoch. To reduce IOs, I read in all the data in the format of big tensor and pickle it as npy file. In this project we can read them all in and greatly improve the training speed (3 mins for one epoch), but we also found an useful 
feature called memmap in numpy, which can help you get a slice of the pickled file when you actually need it. In this way, we can read in a batch of data with only one IO then delete them before we read another batch to save memory.

**Model Training**

In the training process, we also found lots of details are important but not explicitly stated in the original paper. For example, instead of using zero initiation in GRU, we use orthogonal initiation and use xavier initiation in fully connected weight as suggested in [12]. 

Also, we found that the validation loss is not stable in the late phase of training, so we utilize learning rate scheduler in PyTorch to reduce the learning rate according to the valid loss, which help to converge and smooth the learning curve.

## Bibliography

[1] Hu, Ziniu, et al. "Listening to chaotic whispers: A deep learning framework for news-oriented stock trend prediction." Proceedings of the eleventh ACM international conference on web search and data mining. 2018.

[2] Malkiel, Burton G. "The efficient market hypothesis and its critics." Journal of economic perspectives 17.1 (2003): 59-82.

[3] Batres-Estrada, Bilberto. "Deep learning for multivariate financial time series." (2015).

[4] Roman, Jovina, and Akhtar Jameel. "Backpropagation and recurrent neural networks in financial analysis of multiple stock market returns." Proceedings of HICSS-29: 29th Hawaii International Conference on System Sciences. Vol. 2. IEEE, 1996.

[5] Jia, Hengjian. "Investigation into the effectiveness of long short term memory networks for stock price prediction." arXiv preprint arXiv:1603.07893 (2016).

[6] Ding, Xiao, et al. "Deep learning for event-driven stock prediction." Twenty-fourth international joint conference on artificial intelligence. 2015.

[7] Ronaghi, Farnoush, et al. "ND-SMPF: A Noisy Deep Neural Network Fusion Framework for Stock Price Movement Prediction." 2020 IEEE 23rd International Conference on Information Fusion (FUSION). IEEE, 2020.

[8] Si, Jianfeng, et al. "Exploiting social relations and sentiment for stock prediction." Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2014.

[9] Li, Xiaodong, et al. "News impact on stock price return via sentiment analysis." Knowledge-Based Systems 69 (2014): 14-23.

[10] Ding, Xiao, et al. "Deep learning for event-driven stock prediction." Twenty-fourth international joint conference on artificial intelligence. 2015.

[11] Zhongguo Li, Maosong Sun. Punctuation as Implicit Annotations for Chinese Word Segmentation. Computational Linguistics, vol. 35, no. 4, pp. 505-512, 2009.

[12] Duan, Y., Schulman, J., Chen, X., Bartlett, P. L., Sutskever, I., & Abbeel, P. (2016). Rl $^ 2$: Fast reinforcement learning via slow reinforcement learning. arXiv preprint arXiv:1611.02779.
