# 2020WN_CS545
Final project repository for umich EECS 545 Machine Learning. 

**Codes**
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
    	- v2: Because of the sparsity of news, we choose to update the dataloader dataframe by only choosing former 10 days that have news. (dataloader_creation_splitting.py)

Model building and validation evaluation (HAN.py) 

Some baseline model and plotting functions (baseline.py)
    



**MATERIALS**
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


**TODOS**
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



