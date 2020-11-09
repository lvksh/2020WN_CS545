# Purpose: To do the final project about predicting stocks, we need to do:
#   1. Data Preprocessing
#      - Assign news to different stocks
#      - Discard some stocks with no or few news
#      - Discard news that are useless or having some issues
#      - Split the data into
#        - training set for training
#        - validation set for tuning parameters
#        - test set for performance evaluations and back testing
#      - Write the data loader in PyTorch to organize the input format
#   2. Model Building : We need to test out several models
#      - Baseline: Word2Vec + GRU + FC
#      - HAN
#      - HAN + SPL
#      - HAN + self-attention + bert
#      - ...
#   3. Model Training
#      - Remember to set checkpoints
#      - Utilize tensor board to visualize the process
#      - Store the weights so that we can visualize the attention score
#   4. Model Evaluation
#      - Overall accuracy
#      - Back testing result

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Preprocess the data
    dataPath = 'stockDataFromTushare/'
    dataPreprocess(dataPath,
                   mappingPath,  # stockid to name mapping path
                   outputPath  # output path
                   )



