import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import nltk
from sklearn.model_selection import train_test_split

import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import re
import collections
import seaborn as sns
import io
import json

from keras import models
from keras.layers import Embedding, Dense, LSTM, Dropout
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.metrics import BinaryAccuracy, TrueNegatives, TruePositives, FalseNegatives, FalsePositives

import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler, Stream, Cursor
import time

import os
#from dotenv import load_dotenv

#load_dotenv('.env.txt')


nltk.download("stopwords")


class DataRetriever:
    def __init__(self):
        '''
        After creating an instance of this class, the user can retrieve data for two keywords (e.g. ":)" and ":(" or
        "happy" and "sad" with the function get_data().
        '''
        self.pos_key = ""
        self.neg_key = ""
        self.training_data = pd.DataFrame()  # half positive & half negative tweets
        self.realistic_data = pd.DataFrame()  # tweets that contain the topic_key regardless of the sentiment

    # we need to assign access token, access token secret, consumer key, consumer secret to this function
    def _retrieve_tweets(self, topic_key, positive_sentiment, n,
                         access_token, access_token_secret,
                         consumer_key, consumer_secret, additional_key=None):
        '''
        Internal function to retrieve data from Twitter according to the keyword argument.

        :param keyword: string specifying the word or emoticon to retrieve tweets with.
        :param positive_sentiment: boolean. 0 if negative, 1 if positive
        :param n: number of tweets to retrieve
        :param access_token: # TODO Marina: insert descriptions for these 4 inputs
        :param access_token_secret:
        :param consumer_key:
        :param consumer_secret:
        :return: pd.DataFrame with information on time, id, text and sentiment of tweets retrieved.
        '''

        if additional_key is not None:
            keyword=
            # 2 kewords
        else:
            keyword=
            # 1 keyword

        # get tweets
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        # api for further tweets download
        api = tweepy.API(auth)

        tweets = tweepy.Cursor(api.search, q=keyword, wait_on_rate_limit = True, wait_on_rate_limit_notify = True).items(int(n))

        tweets_list = [
            [tweet.created_at, tweet.id, tweet.text]
            for tweet in tweets]

        tweets_df = pd.DataFrame(tweets_list)

        if tweets_df is not None:
            # assign retrieved Data to class fields
            if positive_sentiment == 1:
                self.pos_key = keyword
                label = np.tile(1, tweets_df.shape[0])
            else:
                self.neg_key = keyword
                label = np.tile(0, tweets_df.shape[0])

            tweets_df["label"] = label

            return tweets_df

    def _drop_keyword_from_text(self, tweet, word_to_drop):
        """
        Internal function to drop a keyword from a string.
        :param tweet: string the keyword should be dropped from.
        :param word_to_drop: string wf the keyword to drop
        :return: string of tweet without keyword.
        """
        clean_tweet = re.sub(word_to_drop, " ", tweet)
        return clean_tweet

    def get_training_data(self, access_token, access_token_secret, consumer_key, consumer_secret,
                          topic_key, pos_key="\:\)", neg_key="\:\(", N=100000, save_to_csv=True, file_path=None):
        '''
        Retrieve tweets according to the specified keywords. For training purposes it is important to have evenly
         distributed classes and therefore half positive and half negative tweets will be retrieved.
         Optionally save the data as csv to prespecified path.

        :param access_token:# TODO Marina: insert descriptions for these 4 inputs
        :param access_token_secret:
        :param consumer_key:
        :param consumer_secret:
        :param topic_key: string specifying the word to retrieve tweets for.
        :param pos_key: string specifying the word or emoticon to retrieve positive tweets with. Make sure to escape
        characters if necessary, e.g. instead of pos_key = ":)" put pos_key = "\:\)".
        :param neg_key: string specifying the word or emoticon to retrieve negative tweets with. Make sure to escape
        characters if necessary, e.g. instead of neg_key = ":(" put neg_key = "\:\(".
        :param N: int specifying the total number of tweets. There will be N/2 positive tweets and N/2 negative tweets
        :param save_to_csv: if True data is saved to specified path (file_path).
        :param file_path: file path to save the retrieved data to.
        :return: retrieved data with information on time, id, text and sentiment of tweets retrieved.
        '''
        if save_to_csv and not file_path:
            raise TypeError("Please provide file_path if save_to_csv=True!")

        # call _retrieve_tweets fpr positive and negative sentiment, retrieving half of the desired number of tweets in each case
        positive_tweets = self._retrieve_tweets(topic_key=topic_key, positive_sentiment=1, n=1.5 * N / 2,
                                                access_token=access_token, access_token_secret=access_token_secret,
                                                consumer_key=consumer_key, consumer_secret=consumer_secret,
                                                additional_key=pos_key)
        negative_tweets = self._retrieve_tweets(topic_key=topic_key, positive_sentiment=0, n=1.5 * N / 2,
                                                access_token=access_token, access_token_secret=access_token_secret,
                                                consumer_key=consumer_key, consumer_secret=consumer_secret,
                                                additional_key=neg_key)

        # only get N tweets evenly distributed over the two classes
        negative_tweets = negative_tweets.iloc[:np.min([N / 2,negative_tweets.shape[0]])]
        positive_tweets = positive_tweets.iloc[:np.min([N / 2,positive_tweets.shape[0]])]

        # merge retrieved data
        temp = pd.concat([negative_tweets, positive_tweets], ignore_index=True)
        temp.columns = ["time", "id", "text", "label"]

        # remove duplicates
        temp = temp.drop_duplicates(subset="text")
        # drop keywords from tweets because the classifier would easily detect that the pos_key is the safe indicator
        # for positive tweets and vice versa. In the realistic data these positive/negative keys are not necessarily
        # included.
        temp["text"] = temp["text"].apply(self._drop_keyword_from_text, args=(topic_key,))
        temp["text"] = temp["text"].apply(self._drop_keyword_from_text, args=(pos_key,))
        temp["text"] = temp["text"].apply(self._drop_keyword_from_text, args=(neg_key,))

        self.training_data = temp
        if save_to_csv:
            self.training_data.to_csv(file_path + "/training_data.csv")

        return self.training_data

    def get_realistic_data(self, access_token, access_token_secret, consumer_key, consumer_secret,
                           topic_key, N, save_to_csv=True, file_path=None):
        '''
        Retrieve tweets according to the specified keyword. There will be no attention paid to the sentiment that might
        be included in the tweet. Thus, this function retrieves a realistic distribution of tweets for the specified
        keyword to be further analyzed. This helps to do a market analysis regarding the overall sentiment Twitter
        users have for the specified topic. Optionally save the data as csv to prespecified path.

        :param access_token:# TODO Marina: insert descriptions for these 4 inputs
        :param access_token_secret:
        :param consumer_key:
        :param consumer_secret:
        :param topic_key: string specifying the word to retrieve tweets for.
        :param N: int specifying the total number of tweets.
        :param save_to_csv: if True data is saved to specified path (file_path).
        :param file_path: file path to save the retrieved data to.
        :return: retrieved data with information on time, id, text and sentiment of tweets retrieved.
        '''
        if save_to_csv and not file_path:
            raise TypeError("Please provide file_path if save_to_csv=True!")

        # get tweets according to keyword
        realistic_tweets = self._retrieve_tweets(topic_key=topic_key, positive_sentiment=1, n=1.5*N, # retrieve more tweets that necessary bc of duplicates that will be deleted
                                                 access_token=access_token, access_token_secret=access_token_secret,
                                                 consumer_key=consumer_key, consumer_secret=consumer_secret)
        realistic_tweets.columns = ["time", "id", "text", "label"]

        # remove duplicates
        realistic_tweets = realistic_tweets.drop_duplicates(subset="text")

        # drop keyword from tweets
        realistic_tweets["text"] = realistic_tweets["text"].apply(self._drop_keyword_from_text, args=(topic_key,))

        # only get N tweets
        realistic_tweets = realistic_tweets.iloc[:np.min([N,realistic_tweets.shape[0]])]

        self.realistic_data = realistic_tweets
        if save_to_csv:
            self.realistic_data.to_csv(file_path + "/realistic_data.csv")

        return self.realistic_data


class Models:
    # acronyms to be replaced by their meaning in _clean_tweets
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
    # insults to be replaced by "bad_word" in _clean_tweets
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
    # negations to be replaced by "not" in _clean_tweets
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

    def __init__(self, raw_data, model_folder_path, colname_tweets):
        """
        Parent class for ModelTrainer and Modelapplier.

        :param raw_data: pd.DataFrame containing the raw data.
        :param model_folder_path: path the model should be saved to or loaded from.
        :param colname_tweets: string specifying the column name of the column with the tweets.
        """
        self._model_folder_path = model_folder_path
        self.raw_df = raw_data
        self._colname_tweets = colname_tweets

        self.preprocessed_df = None
        self.predicted_df = None
        self._y_test = None
        self._x_test = None
        self._model = None
        self.model_history = None
        self.evaluation_results = None

    def _preprocess_tweets(self):
        """
        Internal function to apply preprocessing to the raw data.
        """
        prep = self.raw_df.copy(deep=True)
        prep["clean_text"] = prep[self._colname_tweets].apply(lambda x: self._clean_tweet(x))

        # assign to instance variable
        self.preprocessed_df = prep

    def _clean_tweet(self, tweet):
        """
        Internal function to by applied to a single tweet. Removes urls, hashtag, quotes, RT, punctuation.
        Then tokenizes and replaces acronyms by their meaning, negations by "not" and swear words by "bad_word".
        Removes stopwords and converts to lower.

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

        # disassemble tweet into words
        t = TweetTokenizer(reduce_len=True)
        tokens = t.tokenize(clean_tweet)

        # replace acronyms by meaning
        for token in tokens:
            if self._ACRONYMS.get(token.upper()) is not None:  # replace acronym with meaning and tokenize again
                acr = self._ACRONYMS.get(token.upper())
                textNew = re.sub(token, " " + acr + " ", clean_tweet)
                tokens = t.tokenize(textNew)

        # get and modify stopwords
        stopWords = stopwords.words("english")
        stopWords.extend(["i'm", "i'll", "u", "amp", "quot", "lt"])
        stopWords.remove("not")

        out = []
        # apply remaining preprocessing steps
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

    def _predict_new_data(self, return_predictions=True, confusion_matrix=True, predictions_histogram=True):
        """
        Internal function to predict the sentiment of data (either validation data or new data)

        :param return_predictions: if True it returns a DataFrame containing the tweets, the assigned label and
        their probability for a positive sentiment.
        :param confusion_matrix: if True print a confusion matrix
        :param predictions_histogram: if True print a histogram showing the distribution of predicted probabilities.
        :return: DataFrame containing the tweets, the assigned label and
        their probability for a positive sentiment (if return_predictions=True)
        """
        y_prob = self._model.predict(self._x_test)
        y_pred = (y_prob > 0.5).astype("int32")

        self.evaluation_results = self._model.evaluate(x=self._x_test, y=self._y_test)
        # match original tweet with predicted label
        self.predicted_df = pd.DataFrame({self._colname_tweets: self.predicted_df[self._colname_tweets],
                                          "predicted label": y_pred.flatten(),
                                          "probability for positive label": y_prob.flatten().round(decimals=5)})

        if confusion_matrix:
            self._confusion_matrix_plot()

        if predictions_histogram:
            self._predictions_histogram()

        if return_predictions:
            return self.predicted_df

    def _confusion_matrix_plot(self):
        """
        Internal function creating a confusion matrix based on self.evaluation_results.
        """
        loss, binary_accuracy, tn, tp, fn, fp = self.evaluation_results

        total_sum = tp + fp + tn + fn
        cm_perc = [[tp / total_sum, fp / total_sum], [fn / total_sum, tn / total_sum]]

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
        sns.heatmap(cm_perc, annot=True, vmax=1, vmin=0)
        ax.set_xlabel('Actual Label')
        ax.set_ylabel('Predicted Label')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(['Positive', 'Negative'])
        ax.yaxis.set_ticklabels(['Positive', 'Negative'])

    def _predictions_histogram(self):
        """
        Internal function creating a histogram of predicted probabilites for positive label based on self.predicted_df.
        """
        fig_basic, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
        sns.set_style('white')
        sns.histplot(data=self.predicted_df,
                     x='probability for positive label',
                     ax=ax, color='palegoldenrod',
                     stat='probability', bins=101)
        ax.set_ylabel('Relative Frequency')
        ax.set_xlabel('Predicted Probability for Positive Sentiment')


class ModelTrainer(Models):
    def __init__(self, raw_data, model_folder_path, colname_tweets="text", colname_label="label"):
        """
        ModelTrainer allows to train a binary classifier to predict the sentiment of tweets (positive/negative).
        The model is trained on 90% of the data and can later be evaluated out-of-sample using the remaining 10%.
        The trained LSTM model can be saved to use it later.

        :param raw_data: pd.DataFrame containing the tweets and sentiment to train the model on.
        :param model_folder_path: path the model should be saved to.
        :param colname_tweets: string specifying the column name of the column with the tweets.
        :param colname_label: string specifying the column name of the column with the sentiment.
        """
        super().__init__(raw_data=raw_data, model_folder_path=model_folder_path,
                         colname_tweets=colname_tweets)
        self._x_train = None
        self._y_train = None
        self._colname_label = colname_label
        self._word_index = None
        self._vocab_size = None
        self._max_length = None
        self._n_epochs = None

    def train_model_on_data(self, glove_path, overfitting_plot=True, save_model=True,
                            glove_dim=50, lstm_size=64, dropout_rate=0.5, n_epochs=5, batch_size=64):
        '''
        Builds and trains deep LSTM model with embedding layer based on pretrained glove embedding. It consists of
        embedding layer, LSTm layer, dropout layer, LSTM layer, dropout layer and dense layer.

        :param glove_path: path to the glove data.
        :param overfitting_plot: if True an overfitting plot it returned.
        :param save_model: if True the model is saved to the prespecified model_folder_path.
        :param glove_dim: dimension of the pretrained glove embedding, default 50.
        :param lstm_size: size of the LSTM layers, default 64.
        :param dropout_rate: droupout rate of the dsopout layers, default 0.5.
        :param n_epochs: number of epochs to train the model, default 5.
        :param batch_size: size of batches for model training, default 64.

        :return the trained model.
        '''

        super()._preprocess_tweets()
        self._prepare_model_input(save_model=save_model)
        self._create_and_train_model(glove_path=glove_path, glove_dim=glove_dim, lstm_size=lstm_size,
                                     dropout_rate=dropout_rate, n_epochs=n_epochs, batch_size=batch_size)

        # show training
        if overfitting_plot:
            self._n_epochs = n_epochs
            self._overfitting_plot()

        # save model
        if save_model:
            self._model.save(self._model_folder_path + "/model")

        return self._model

    def evaluate_out_of_sample(self, return_predictions=False, confusion_matrix=True, predictions_histogram=True):
        """
        Evaluates the trained model on the 10% out-of-sample data and provides helpful graphics to assess the model's
        perfomrance on data that it has not seen before.

        :param return_predictions: if True it returns a DataFrame containing the tweets, the assigned label and
        their probability for a positive sentiment.
        :param confusion_matrix: if True print a confusion matrix
        :param predictions_histogram: if True print a histogram showing the distribution of predicted probabilities.
        :return: DataFrame containing the tweets, the assigned label and
        their probability for a positive sentiment (if return_predictions=True)
        """
        super()._predict_new_data(return_predictions=return_predictions, confusion_matrix=confusion_matrix,
                                  predictions_histogram=predictions_histogram)

    def _prepare_model_input(self, save_model):
        """
        Internal function preparing the model input. The prepocessed data is tokenized, transformed to sequences
        :param save_model: if True, Tokenizer is saved to be applied to new data later. This is necessary if new
        data should be classified with a saved model.
        """
        train_test_df, final_eval_df = train_test_split(self.preprocessed_df, test_size=0.1, random_state=7)
        print("Shape of ... Training Data: ", train_test_df.shape, " ... Final Evaluation Data: ",
              final_eval_df.shape)

        # save final eval df for predictions
        self.predicted_df = final_eval_df

        # Tokenization
        tokenizer = Tokenizer(oov_token='UNK')
        tokenizer.fit_on_texts(train_test_df["clean_text"])  # Updates internal vocabulary based on training data
        self._word_index = tokenizer.word_index  # maps words in our vocabulary to their numeric representation

        # Encode training data sentences into sequences: "My name is Matthew," to something like "6 8 2 19,"
        train_seq = tokenizer.texts_to_sequences(train_test_df["clean_text"])
        test_seq = tokenizer.texts_to_sequences(final_eval_df["clean_text"])
        self._vocab_size = len(tokenizer.word_index) + 1
        print("There were " + str(self._vocab_size) + " unique words found.")

        # Padding sequences to same length
        self._max_length = max([len(x) for x in train_seq])

        # apply padding and save training and testing data
        self._x_train = pad_sequences(train_seq, maxlen=self._max_length)
        self._x_test = pad_sequences(test_seq, maxlen=self._max_length)
        self._y_train = np.array(train_test_df[self._colname_label].to_list())
        self._y_test = np.array(final_eval_df[self._colname_label].to_list())

        # save tokenizer to use later if model should be saved
        if save_model:
            tokenizer_json = tokenizer.to_json()
            with io.open(self._model_folder_path + '/tokenizer.json', 'w', encoding='utf-8') as f:
                f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    def _create_and_train_model(self, glove_path, glove_dim, lstm_size, dropout_rate, n_epochs, batch_size):
        """
        Internal function to create and train the LSTM model.
        :param glove_path: path specifying where the pretrained embedding is saved.
        :param glove_dim: dimension of the pretrained glove embedding.
        :param lstm_size: size of the LSTM layers.
        :param dropout_rate: droupout rate of the dsopout layers.
        :param n_epochs: number of epochs to train the model.
        :param batch_size: size of batches for model training.
        :return:
        """
        # get the pretrained word embedding
        emb_dict = {}
        glove = open(glove_path)
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

        # build model
        m = models.Sequential()
        m.add(Embedding(self._vocab_size, glove_dim, input_length=self._max_length))
        m.add(LSTM(lstm_size, return_sequences=True))
        m.add(Dropout(rate=dropout_rate))
        m.add(LSTM(lstm_size))
        m.add(Dropout(rate=dropout_rate))
        m.add(Dense(1, activation='sigmoid'))

        # adjust embedding layer
        m.layers[0].set_weights([emb_matrix])
        m.layers[0].trainable = False  # prevent overfitting

        # compile and train
        m.compile(optimizer='Adam', loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[BinaryAccuracy(), TrueNegatives(), TruePositives(), FalseNegatives(), FalsePositives()])
        hist = m.fit(self._x_train, self._y_train, epochs=n_epochs,
                     batch_size=batch_size,
                     verbose=1, validation_split=0.2)
        self._model = m
        self.model_history = pd.DataFrame(hist.history)

    def _overfitting_plot(self):
        """
        Internal function to create an overfitting plot. This allows to monitor overfitting in the training process.
        Overfitting is indicated by either an increasing loss or an accuracy that is higher in the training data
        than in the test data.
        """
        # create df for plots
        df = pd.DataFrame(data=np.repeat(['Training', 'Validation'], repeats=self._n_epochs),
                          columns=['trainval'])
        df['xaxis'] = np.array([range(1, self._n_epochs + 1)] * 2).flatten()
        df['binary_accuracy'] = np.array(
            [self.model_history['binary_accuracy'], self.model_history['val_binary_accuracy']]).flatten()
        df['loss'] = np.array([self.model_history['loss'], self.model_history['val_loss']]).flatten()

        # initialize plot
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        # accuracy plot
        sns.scatterplot(ax=axes[0], x=df['xaxis'], y=df['binary_accuracy'], hue=df['trainval'], palette='Dark2',
                        legend=False)
        sns.lineplot(ax=axes[0], x=df['xaxis'], y=df['binary_accuracy'], hue=df['trainval'], palette='Dark2', alpha=0.3,
                     estimator=None, legend=None)
        axes[0].legend().remove()
        axes[0].set_xlabel('Epoch', size=20)
        axes[0].set_ylabel('Accuracy', size=20)
        axes[0].tick_params(axis='both', which='major', labelsize=15)
        axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # loss plot
        sns.scatterplot(ax=axes[1], x=df['xaxis'], y=df['loss'], hue=df['trainval'], palette='Dark2')
        sns.lineplot(ax=axes[1], x=df['xaxis'], y=df['loss'], hue=df['trainval'], palette='Dark2', alpha=0.3,
                     estimator=None, legend=None)
        axes[1].legend(fontsize=15).set_title('')
        axes[1].set_xlabel('Epoch', size=20)
        axes[1].set_ylabel('Loss', size=20)
        axes[1].tick_params(axis='both', which='major', labelsize=15)
        axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.tight_layout()


class ModelApplier(Models):
    def __init__(self, raw_data, model_folder_path, colname_tweets="text"):
        """
        Loads a pretrained model and predicts new data.
        :param raw_data: a pd.DataFrame containing
        :param model_folder_path:
        :param colname_tweets:
        """
        super().__init__(raw_data=raw_data, model_folder_path=model_folder_path,
                         colname_tweets=colname_tweets)
        self._model = models.load_model(model_folder_path + "/model")
        self._padding_length = self._model.input_shape[1]  # length if inputs required for trained model

    def _prepare_model_input(self):
        """
        Internal function to prepare the model input. The saved tokenizer is loaded and the the preprocessed data
        is tokenized, padded and saved.
        """
        # load tokenizer adjusted to training data
        tokenizer_path = self._model_folder_path + '/tokenizer.json'
        with open(tokenizer_path) as f:
            json_data = json.load(f)
            tokenizer = tokenizer_from_json(json_data)

        # encode tweets with tokenizer trained on training data
        self.preprocessed_df["clean_text"] = self.preprocessed_df["clean_text"].astype(str)
        self.predicted_df = self.preprocessed_df  # save for model evaluation
        data_seq = tokenizer.texts_to_sequences(self.preprocessed_df["clean_text"])

        # Padding sequences to the length matching the model input
        self._x_test = pad_sequences(data_seq, maxlen=self._padding_length)

    def predict_new_data(self, return_predictions=True, predictions_histogram=True):
        """
        Applies preprocessing to tweets, prepares the model input and predicts sentiment for raw data

        :param return_predictions: if True it returns a DataFrame containing the tweets, the assigned label and
        their probability for a positive sentiment.
        :param predictions_histogram: if True print a histogram showing the distribution of predicted probabilities.

        :return: DataFrame containing the tweets, the assigned label and
        their probability for a positive sentiment (if return_predictions=True)
        """
        super()._preprocess_tweets()
        self._prepare_model_input()
        super()._predict_new_data(return_predictions=return_predictions, confusion_matrix=False,
                                  predictions_histogram=predictions_histogram)


if __name__ == '__main__':
    # api access code

    access_token = os.getenv('access_token')
    access_token_secret = os.getenv('access_token_secret')
    consumer_key = os.getenv('consumer_key')
    consumer_secret = os.getenv('consumer_secret')

    # streamList = StdOutListener()
    dataRetr = DataRetriever()
    print(dataRetr._retrieve_tweets(keyword=["glad"], positive_sentiment=1, n=400,
                                    access_token=access_token, access_token_secret=access_token_secret,
                                    consumer_key=consumer_key, consumer_secret=consumer_secret))

    # analyzyer = Analyzer(DataRetriever=dataRetr)

    exit()
