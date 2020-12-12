'''
数据来源：新浪财经
vip.stock.finance.sina.com.cn
'''

import requests
import re
import pandas as pd
from bs4 import BeautifulSoup
test_stock_id = 'sz002594'
'''
数据来源：新浪财经
vip.stock.finance.sina.com.cn
'''

import requests
import re
import pandas as pd
from bs4 import BeautifulSoup
test_stock_id = 'sz002594'
def request_news_url_title_date(stock_id, page):
    # input stock id
    # output news url list, title_list, date_list
    
    url = 'https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllNewsStock.php?symbol={}&Page={}'.format(str(stock_id), str(page))
    req = requests.get(url)
    req.encoding = req.apparent_encoding
    bs = BeautifulSoup(req.text, 'html.parser')
    datelist = bs.find_all(class_='datelist')[0]
    
    dateText = datelist.prettify()
    date_list = []
    for item in dateText.split('\n'):
        res = re.findall('^\d{4}-\d{2}-\d{2}',item.strip())
        if len(res) == 1:
             date_list += res

    return [
            [item.attrs['href'] for item in datelist.find_all('a')], # urls
            [item.string for item in datelist.find_all('a')], # titles
            date_list # date_list
           ]



def request_news_content(url_list):
    
    content = []
    for url in url_list:
        req = requests.get(url)
        req.encoding = req.apparent_encoding
        bs = BeautifulSoup(req.text, 'html.parser')
        tmp = []
        p = bs.find_all('p')
        for i in range(len(p)):
            if p[i].font:
                tmp.append(p[i].font.text)
            else:
                tmp.append(p[i].text)
        body_content = ''
        for sentence in tmp:
            if not sentence:
                continue
            body_content += sentence.strip('\u3000').strip()
        content.append(body_content)
    return content

def wrapper_for_single_stock(stock_id, theLastDate):
    # input stock id and the last date of news to crawl
    # output news content list
    page = 1
    url_list = []
    title_list = []
    date_list = []
    while True:
        url_list_, title_list_, date_list_ = request_news_url_title_date(stock_id, page)
        if theLastDate in date_list_:
            ind = date_list_.index(theLastDate)
            url_list += url_list_[:ind]
            title_list += title_list_[:ind]
            date_list += date_list_[:ind]
            break
        else:
            url_list += url_list_
            title_list += title_list_
            date_list += date_list_
            page += 1
            
    content = request_news_content(url_list)
    
    return [content, title_list, date_list]

## Testing Examples
## res = wrapper_for_single_stock(test_stock_id, '2020-09-27')
## res[0]
