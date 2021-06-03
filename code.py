import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import nltk
from sklearn.model_selection import train_test_split

import tensorflow as tf
import matplotlib.pyplot as plt
import re
import collections
import seaborn as sns

from keras import models
from keras.layers import Embedding, Dense, LSTM, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

nltk.download("stopwords")


class DataRetriever:
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
        self.raw_data = pd.DataFrame()  # both positive & negative tweets

    def _retrieve_tweets(self, keyword, positive_sentiment, n):
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

        l = list()  # should contain only the tweets -> is a list of strings such as ["Today I feel good", "Hey world"]

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
        self.raw_data = pd.concat([self._neg_data, self._pos_data], ignore_index=True)

    def get_data(self, pos_key, neg_key, N):
        '''
        Called to retrieve the data.

        :param pos_key: string specifying the word or emoticon to retrieve positive tweets with.
        :param neg_key: string specifying the word or emoticon to retrieve negative tweets with.
        :param N: int specifying the total number of tweets. There will be N/2 positive tweets and N/2 negative tweets
        :return:
        '''
        self._retrieve_tweets(keyword=pos_key, positive_sentiment=1, n=N / 2)
        self._retrieve_tweets(keyword=neg_key, positive_sentiment=0, n=N / 2)
        self._join_data()

        return self.raw_data


class Analyzer:
    '''
    Performs preprocessing and analysis of tweets, maybe also visualization?
    '''

    def __init__(self, DataRetriever):
        self.raw_df = DataRetriever.raw_data
        self.processed_df = pd.DataFrame()
        self.train_test_df = pd.DataFrame()
        self.final_eval_df = pd.DataFrame()

        self._word_index = {}
        self._vocab_size = 0
        self._max_length = 0
        self._x_train = []
        self._x_test = []
        self._y_train = []
        self._y_test = []

    def preprocess_tweets(self):
        '''
        Preprocesses the data, assign the preprocessed DataFrame to self.processed_df and split the data into
        train-test data and final evaluation data (out-of-sample).
        '''
        # insert prepocessing steps here
        prep = self.raw_df.copy(deep=True)
        prep["clean_text"] = prep["text"].progress_apply(
            lambda x: self._clean_tweet(x))  # TODO: adjust colname if necessary
        prep.drop("text", axis=1)

        # TODO: removing empty tweets after preprocessing?

        # assign to instance variable
        self.processed_df = prep

        # split processed data into train/test and final evaluation dataset
        self.train_test_df, self.final_eval_df = train_test_split(prep, test_size=0.1, random_state=7)
        print("Shape of ... Training Data: ", self.train_test_df.shape, " ... Final Evaluation Data: ",
              self.final_eval_df.shape)

    _ACRONYMS = {
        "SRY": "sorry",
        "L8": "late",
        "W8": "wait",
        "M8": "mate",
        "PLZ": "please",
        "PLS": "please",
        "SRSLY": "seriously",
        "OMFG": "Oh my god",
        "OMG": "Oh my god",
        "BROS": "brother",
        "BRO": "brother",
        "KK": "ok",
        "K": "ok",
        "DM": "Direct Message",
        "RT": "Retweet",
        "NSFW": "Not Safe For Work",
        "FTW": "For The Win",
        "FYI": "For Your Information",
        "IRL": "In Real Life",
        "LMK": "Let Me Know",
        "TTYL": "Talk To You Later",
        "AFK": "Away From Keyboard",
        "BTW": "By The Way",
        "GG": "Good Game",
        "GTG": "Got To Go",
        "HT": "Hat Tip",
        "ILY": "I Love You",
        "JK": "Just Kidding",
        "TMI": "Too Much Information",
        "YOLO": "You Only Live Once",
        "AFAIK": "As Far As I Know",
        "AMA": "Ask Me Anything",
        "BRBBe": "Right Back",
        "FB": "Facebook",
        "IMHO": "In My Humble Opinion",
        "IMO": "In My Opinion",
        "LMAO": "Laughing My Ass Off",
        "LOL": "Laughing Out Loud",
        "LULZ": "Laughing Out Loud",
        "NBD": "No Big Deal",
        "ROFL": "Rolling On The Floor Laughing",
        "SMH": "Shaking My Head",
        "TBH": "To Be Honest",
        "TBT": "Throwback Thursday",
        "TIL": "Today I Learned",
        "BFN": "Bye For Now",
        "DAE": "Does Anyone Else",
        "FF": "Follow Friday",
        "FTFY": "Fixed That For You",
        "IA": "Inactive",
        "ICYMI": "In Case You Missed It",
        "IKR": "I Know, Right",
        "JSYK": "Just So You Know",
        "MT": "Modified Tweet",
        "NP": "No Problem",
        "NSFL": "Not Safe For Life",
        "PAW": "Parents Are Watching",
        "PPL": "People",
        "RL": "Real Life",
        "TTYS": "Talk To You Soon",
        "TY": "Thank You",
        "AKA": "Also Known As",
        "ASAP": "As Soon As Possible",
        "ATM": "At The Moment",
        "B/C": "Because",
        "B4": "Before",
        "BAE": "Before Anyone Else",
        "BBL": "Be Back Later",
        "BBS": "Be Back Soon",
        "BF": "Boyfriend",
        "BFF": "Best Friend Forever",
        "DIY": "Do It Yourself",
        "BRB": "Be Right Back",
        "FAQ": "Frequently Asked Questions",
        "FWIW": "For What It's Worth",
        "FYEO": "For Your Eyes Only",
        "G2G": "Got To Go",
        "GF": "Girlfriend",
        "GR8": "Great"
    }
    _INSULTS = [
        "arse",
        "ass",
        "asshole",
        "bastard",
        "bitch",
        "bitchfucks",
        "bollocks",
        "brotherfucker",
        "bugger",
        "bullshit",
        "child-fucker",
        "crap",
        "cunt",
        "damn",
        "dick",
        "dickhead",
        "dumbass",
        "effing",
        "faggot",
        "fatass",
        "fatherfucker",
        "frigger",
        "fuck",
        "fucking",
        "goddamn",
        "godsdamn",
        "hell",
        "holy shit",
        "horseshit",
        "idiot",
        "motherfucker",
        "motherfucking",
        "nigga",
        "piss",
        "prick",
        "pussy",
        "scumbag",
        "shit",
        "shit ass",
        "shitass",
        "sisterfucker",
        "slut",
        "son of a bitch",
        "son of a whore",
        "sucker",
        "twat",
        "whore"]
    _NEGATIONS = [
        "isn't",
        "isnt",
        "aren't",
        "arent",
        "ain't",
        "aint",
        "don't",
        "dont",
        "didn't",
        "didnt",
        "doesn't",
        "doesnt",
        "haven't",
        "havent",
        "hasn't",
        "hasnt",
        "wasn't",
        "wasnt",
        "weren't",
        "werent",
        "won't",
        "wont",
        "never",
        "can't",
        "cant",
        "cannot",
        "couldn't",
        "couldnt",
        "wouldn't",
        "wouldnt",
        "shouldn't",
        "shouldnt"]

    def _get_stopwords(self):
        """
        Get stopwords from nltk package and modify them appropriately.
        :return: list of stopwords
        """
        stopWords = stopwords.words("english")
        stopWords.extend(["i'm", "i'll", "u", "amp", "quot", "lt"])
        stopWords.remove("not")
        return stopWords

    def _clean_tweet(self, tweet):
        """
        remove urls, hashtag, quotes, RT, punctuation. Then tokenize and replace acronyms by their meaning,
        negations by "not" and swear words by "bad_word". Remove stopwords and convert to lower.

        :param tweet: a string containing to tweet to be cleaned.
        :return: a string of the clean tweet.
        """
        clean_tweet = re.sub(r'"', "'", tweet)

        # remove urls
        clean_tweet = re.sub(r"(http|https):\S+", " ", clean_tweet)
        # remove the url of twitter's pic
        clean_tweet = re.sub(r"pic.twitter.com\S+", " ", clean_tweet)
        # remove mentions
        clean_tweet = re.sub(r"@\S+", " ", clean_tweet)

        # remove the punctuation
        clean_tweet = re.sub(r"(\\|\.|,|:|;|\?|!|\)|\(|\-|\[|\]|\{|\}|\*|\||\<|\>|%|&|/|$|\+|@|\$|£|=|\^|~)", " ",
                             clean_tweet)

        # remove the hashtag
        clean_tweet = re.sub(r"#", "", clean_tweet)
        # remove the RT
        clean_tweet = re.sub(r"(RT )", " ", clean_tweet)

        clean_tweet = clean_tweet.lower()
        clean_tweet = " " + clean_tweet + " "
        t = TweetTokenizer(reduce_len=True)  # deletes @mentions and reduces coooooool to coool
        tokens = t.tokenize(clean_tweet)

        for token in tokens:
            if self._ACRONYMS.get(token.upper()) is not None:  # replace acronym with meaning and tokenize again
                acr = self._ACRONYMS.get(token.upper())
                textNew = re.sub(token, " " + acr + " ", clean_tweet)
                tokens = t.tokenize(textNew)

        out = []
        stopWords = self._get_stopwords()

        for token in tokens:
            token = token.lower()
            if token in self._NEGATIONS:  # replace negations by not
                out.append("not")
            elif token in self._INSULTS:  # replace swear words by bad_word
                out.append("bad_word")
            elif token in stopWords:  # do not append stopwords or empty tokens
                pass
            else:
                token = re.sub("[^a-z0-9<>]", "", token)  # delete cryptic stuff
                if token != "":
                    out.append(token)  # append non-problematic words

        newTweet = " ".join(out).lower()
        return newTweet

    def train_model(self):
        '''
        Builds and trains model based on 70% training and 30% testing data.
        :return:
        '''

        train_data, test_data = train_test_split(self.train_test_df, test_size=0.3)
        self._prepare_model_input(train_data=train_data, test_data=test_data, chatty=True)
        self._create_model()
        # TODO evaluate

    def _prepare_model_input(self, train_data, test_data, chatty=False):
        # Tokenization
        tokenizer = Tokenizer(oov_token='UNK')
        tokenizer.fit_on_texts(train_data["clean_text"])  # Updates internal vocabulary based on a list of texts.
        self._word_index = tokenizer.word_index  # maps words in our vocabulary to their numeric representation

        # Encode training data sentences into sequences: "My name is Matthew," to something like "6 8 2 19,"
        train_seq = tokenizer.texts_to_sequences(train_data["clean_text"])
        test_seq = tokenizer.texts_to_sequences(test_data["clean_text"])
        self._vocab_size = len(tokenizer.word_index) + 1
        if chatty: print("There were " + str(self._vocab_size) + " unique words found.")

        # Padding sequences to same length
        self._max_length = max([len(x) for x in train_seq])
        if chatty: print('Max length of tweet (number of words): ' + str(self._max_length))
        self._x_train = pad_sequences(train_seq, maxlen=self._max_length)
        self._x_test = pad_sequences(test_seq, maxlen=self._max_length)

        # target variables
        self._y_train = np.array(train_data["label"].to_list())
        self._y_test = np.array(test_data["label"].to_list())

    def _create_model(self, glove_dim=50, lstm_size=64, dropout_rate=0.5, epochs=5, batch_size=128):

        # get the pretrained word embedding
        emb_dict = {}
        glove = open("glove.twitter.27B.50d.txt")
        for line in glove:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            emb_dict[word] = vector
        glove.close()

        # build embedding matrix to set weights for embedding layer
        emb_matrix = np.zeros((self._vocab_size, glove_dim))
        for w, i in self._word_index.items():
            # if chatty: print(w)
            if i < self._vocab_size:
                vect = emb_dict.get(w)
                if vect is not None:
                    emb_matrix[i] = vect
            else:
                break

        m = models.Sequential()
        m.add(Embedding(self._vocab_size, glove_dim, input_length=self._max_length))
        m.add(LSTM(lstm_size, return_sequences=True))
        m.add(Dropout(rate=dropout_rate))
        m.add(LSTM(lstm_size))
        m.add(Dropout(rate=dropout_rate))
        m.add(Dense(1, activation='sigmoid'))

        # adjust embedding layer
        m.layers[0].set_weights([emb_matrix])
        m.layers[0].trainable = False

        m.compile(optimizer='Adam', loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=tf.keras.metrics.BinaryAccuracy())

        self._model = m
        self._model_history = self._model.fit(self._x_train, self._y_train, epochs=epochs, batch_size=batch_size,
                                              verbose=1, validation_data=(self._x_test, self._y_test))

    def evaluate_out_of_sample(self):
        '''
        Evaluates the model performance on
        :return:
        '''
        # use final_eval_df here
        pass
