import pandas as pd
import numpy as np

class DataRetriever():
    '''
    After creating an instance of this class, the user can retrieve data for two keywords (e.g. ":)" and ":(" or
    "happy" and "sad".
    '''
    def __init__(self):
        '''
        Creates empty DataFrames and strings as instance variables.
        '''
        self.pos_key = ""
        self._pos_data = pd.DataFrame()
        self.neg_key = ""
        self._neg_data = pd.DataFrame()
        self.raw_data = pd.DataFrame() # both positive & negative tweets

    def _get_data(self, keyword, positive_sentiment, n):
        '''

        Retrieves data from Twitter according to the keyword.

        :param keyword: string specifying the word or emoticon to retrieve tweets with.
        :param positive_sentiment: boolean. 0 if negative, 1 if positive
        :param n: number of tweets to retrieve
        :return:
        '''

        # get tweets

        # make sure it is English

        # no duplicates

        # assign n tweets to raw_data

        l = list() # should contain only the tweets -> is a list of strings such as ["Today I feel good", "Hey world"]

        # assign retrieved Data to
        if positive_sentiment == 1:
            self.pos_key = keyword
            self._pos_data = pd.DataFrame({"label": np.tile(1, len(l)),
                                         "text": l})
        elif positive_sentiment == 0:
            self.neg_key = keyword
            self._neg_data = pd.DataFrame({"label": np.tile(0, len(l)),
                                         "text": l})

    def _join_data(self):
        self.raw_data = pd.concat([self._neg_data, self._pos_data], ignore_index = True)

    def get_data(self, pos_key, neg_key, N):
        '''
        Called to retrieve the data.

        :param pos_key: string specifying the word or emoticon to retrieve positive tweets with.
        :param neg_key: string specifying the word or emoticon to retrieve negative tweets with.
        :param N: int specifying the total number of tweets. There will be N/2 positive tweets and N/2 negative tweets
        :return:
        '''
        self._get_data(keyword=pos_key, positive_sentiment=1, n=N / 2)
        self._get_data(keyword=neg_key, positive_sentiment=0, n=N / 2)
        self._join_data()

        return self.raw_data

class Analyzer():
    '''
    Performs preprocessing and analysis of tweets, maybe also visualization?
    '''
    def __init__(self, DataRetriever):
        self.raw_df = DataRetriever.raw_data
        self.processed_df = pd.DataFrame()
        self.train_test_df = pd.DataFrame()
        self.final_eval_df = pd.DataFrame()

    def preprocess(self):
        '''
        Preprocesses the data.
        :return:
        '''
        # insert prepocessing steps here
        prep = pd.DataFrame()

        # assign to instance variable
        self.processed_df = prep

        # split processed data into train/test and final evaluation dataset
        self.train_test_df = pd.DataFrame()
        self.final_eval_df = pd.DataFrame()


    def analyze(self):
        '''
        Builds and trains model.
        :return:
        '''

        # use train_test_df here
        pass

    def evaluate(self):
        '''
        Evaluates the model performance on
        :return:
        '''
        # use final_eval_df here
        pass


