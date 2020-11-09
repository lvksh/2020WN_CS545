# This script is for data preprocessing
def assignNews(newsPath, mappingPath, outputPath, dailyPath):
    # TODO: assign news from newsPath to certain stock and output to the outputPath
    # outputPath
    #   - stock1
    #     - 20200101
    #        - news1
    #        - news2
    #     - 20200102
    pass


def assessNews(newsPath):
    # TODO: Calculate news count for each stock files
    # TODO: Discard stocks with no or few news or news in a not really continuous range
    # TODO: Find a way to assess the content of the news
    pass

def splitWords(newsPath):
    # TODO: Use jieba to split words and remove stopwords iteratively for each news file
    pass

def getLabel(dailyPath):
    # TODO: Perform a diff operation on the daily close price of all stocks and output label file
    pass

def buildData(dataPath):
    # TODO: Find a way to organize the data for the convinience in future data loader
    pass

def dataPreprocess(dataPath,  # input path
                   mappingPath,  # stockid to name mapping path
                   outputPath,  # output path
                   dailyPath    # daily info path
                   ):
    # the input path has the structure like this
    # dataPath
    #   - sinaNews
    #      - sina-yyyy-mm-dd.csv
    #   - wsNews
    #      - ws-yyyy-mm-dd.csv
    #   - thsNews
    #      - ths-yyyy-mm-dd.csv
    #   - ycjNews
    #      - ycj-yyyy-mm-dd.csv
    #   - eastmoneyNews
    #      - eastmoney-yyyy-mm-dd.csv
    #   - generalNews
    #      - yyyy-mm-dd.csv
    #   - daily_basic (some features about stocks)
    #      - <stock_id>.csv
    #   - daily (some features about stocks)
    #      - <stock_id>.csv

    # Assign news to different stocks
    for path in ['sinaNews', 'wsNews', 'thsNews', 'ycjNews', 'eastmoneyNews']:
        assignNews(newsPath=dataPath + path,
                   mappingPath=mappingPath,
                   outputPath=outputPath,
                   dailyPath=dailyPath)

    # In output path, stockid <== date <== news

    # Assess the news and drop useless stocks
    assessNews(newsPath=outputPath)

    # Split words, remove stop words
    splitWords(newsPath=outputPath)

    # Get label data
    getLabel(dailyPath=dailyPath)

    # Organize the data with the label
    buildData(dataPath=outputPath)
