# 2020WN_SI671
Final project repository for umich SI 671 data mining. 

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
- [一文看懂 Attention（本质原理+3大优点+5大类型）](https://zhuanlan.zhihu.com/p/91839581)
- [【NLP】Transformer模型原理详解](https://zhuanlan.zhihu.com/p/44121378)  
- [搞懂Transformer结构，看这篇PyTorch实现就够了（上）](https://zhuanlan.zhihu.com/p/48731949)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [BERT中文情感分类](https://github.com/Toyhom/Hei_Dong/tree/master/Project/%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB)

**TODOS**
- ~~project proposal~~  
- Things to figure out before everything
  - Common label definitions used in stock prediction
  - How can the label related to the actual applications? 
  - How's the final data frame looks like?
- Data Retrieval: Using crawling methods to crawl all the data we need for this project.  
  - Historical stock open price, close price, all interesting factors for each stock. (Zeyuan hu)
    - 2020-10-03 kunsheng: Found an amazing crawling tools [Tushare](https://tushare.pro/register?reg=395357)
  - Historical stock info for INDEX stock like '沪深300' (Zeyuan hu)
  - Corresponding news for each stock with published date (kunsheng)
    - 2020-09-30: Iteratively crawled some news, but the length of content and title are not the same, still needs debugging. 
    - 2020-10-03: Finishing debugging, but still need some preprocessing like word split, cleansing some words like '\n', then store them as files
    - 2020-10-14: Use Tushare to crawl all the quick news from 2019-01-01 till now, still need to assign to corresponding stocks with names or ids.
  - Overall news about finance or stock market with published date (Ivy) 
    - 2020-10-14 kunsheng: Use Tushare to crawl all the general news from 2019-01-01 till now, save some samples in git.
- Data Cleansing: Data from the crawling process may be messy and hard to use, so we need to
preprocess the data first.  
- Numerical Feature Engineering: It’s hard to predict the stock using solely the text, so we need to
construct some stock factors like 5-day variance, etc.  
- Text Feature Engineering: We plan to utilize Deep Learning technique to preprocess the text, so we
need to find a suitable network framework and implement it on Pytorch.  
- Model Building: We will try different machine learning regressors and compare their performance. 
  - Short term goal: Implement News-RNN, HAN from LCW; Build the whole framework of model building, data loading, model evaluation and back-testing.
  - Mid term goal: Implement HAN-SPL from LWC
  - Long term goal: Replace GRU structures in HAN with multi-head self-attention; Use data other then only news; Utilize pre-trained model like BERT in embedding layers, etc...
- Model Applications: If we have time, we will extract some up-to-date data and validate our model
on real data.  

**Messy Thoughts**

- 2020-10-01 kunsheng
  - Using news to predict stock, in a big picture, we want to look at all the information we can get when we want to invest on some stocks. So we can have:
    1. Historical price: many works are done for this, which is a time-series and we can detect the trend or patterns to predict the future trend. But using only this is impossible to predict because stock price doesn't change because of the previous price. However, previous price will give us a great estimate of the variance level or trend because a factor that affect the price will last for a period, that's why time-series makes sense sometimes.
    2. News about this certain stock: this should be a significant features. Let's assumes those stock experts analyzing news and stocks and financial reports are paid to do something really useful, we can use machine learning to imitate their works and analyze the news content.
    3. Besides, the general informations (Like '沪深指数'), which integrate many stocks and can represents the general stock market. We can make use of the historical price and news about this INDEX to gain a overall impression on the whole stock market, which will help too.
    4. Besides news, we can consider some experts reports too. In China, these kind of news may attracts more attentions and hence influence individual investors' decisions more! 
  - I look through some data mining articles about predicting stocks, they always cast it into a classification problems predicting the stock price of the following days to be "UP", "DOWN" or "PRESERVE". In LCW they achieve the accuracy of about 50%.
  - Also in LCW, they did a market trading simulation and conduct a back-testing for a year to validate their models, which achieve 50% accumulated profit. (WHAT???????)

- 2020-10-14
