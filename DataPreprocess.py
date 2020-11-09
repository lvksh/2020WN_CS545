# This script is for data preprocessing
def assignNews(newsPath, mappingPath, outputPath):
    # TODO: assign news from newsPath to certain stock and output to the outputPath
    # TODO: In the output path, each stock have one folder with all the related news
    pass

def assessNews(newsPath):
    # TODO: Calculate news count for each stock files
    # TODO: Discard stocks with no or few news or news in a not really continuous range
    # TODO: Find a way to assess the content of the news
    pass

def dataPreprocess(dataPath,    # input path
                   mappingPath, # stockid to name mapping path
                   outputPath   # output path
                  ):
    # the input path has the sturcture like this
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
        assignNews(newsPath = dataPath + path,
                   mappingPath = mappingPath,
                   outputPath = outputPath)

    # Assess the news
    assessNews(newsPath = outputPath)