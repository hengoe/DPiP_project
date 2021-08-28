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

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler, Stream, Cursor
import time

import os

nltk.download("stopwords")


class StdOutListener(StreamListener):

    def on_data(self, data):
        print(data)
        return True

    def on_error(self, status):
        print(status)


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
        self.neg_key = ""
        self.raw_data = pd.DataFrame()  # both positive & negative tweets

    def _retrieve_tweets(self, keyword, positive_sentiment, n):
        '''

        Retrieves data from Twitter according to the keyword argument.

        :param keyword: string specifying the word or emoticon to retrieve tweets with.
        :param positive_sentiment: boolean. 0 if negative, 1 if positive
        :param n: number of tweets to retrieve
        :return:
        '''

        # get tweets
        l = StdOutListener()
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        twitterStream = Stream(auth, l, wait_on_rate_limit=True,
                               wait_on_rate_limit_notify=True)
        # twitterStream.filter(track=["happy"], languages=["en"])

        try:
            tweets = Cursor(auth.search, q=keyword).items(int(n))
            tweets_list = [[tweet.created_at, tweet.id, tweet.text] for tweet in tweets]
            tweets_df = pd.DataFrame(tweets_list)
        except BaseException as b:
            print('failed', str(b))
            time.sleep(3)

        # assign retrieved Data to class fields
        if positive_sentiment == 1:
            self.pos_key = keyword
            label = np.tile(1, tweets_df.shape[0])

        elif positive_sentiment == 0:
            self.neg_key = keyword
            label = np.tile(0, tweets_df.shape[0])

        tweets_df["label"] = label

        return tweets_df

    def get_data(self, pos_key, neg_key, N, save_to_csv=True, file_path=None):
        '''
        Called to retrieve the data.

        :param pos_key: string specifying the word or emoticon to retrieve positive tweets with.
        :param neg_key: string specifying the word or emoticon to retrieve negative tweets with.
        :param N: int specifying the total number of tweets. There will be N/2 positive tweets and N/2 negative tweets
        :return:
        '''
        if save_to_csv and not file_path:
            raise TypeError("Please provide file_path if save_to_csv=True!")

        # call _retrieve_tweets fpr positive and negative sentiment, retrieving half of the desired number of tweets in each case
        positive_tweets = self._retrieve_tweets(keyword=pos_key, positive_sentiment=1, n=N / 2)
        negative_tweets = self._retrieve_tweets(keyword=neg_key, positive_sentiment=0, n=N / 2)

        # merge retrieved data
        self.raw_data = pd.concat([negative_tweets, positive_tweets], ignore_index=True)

        if save_to_csv:
            self.raw_data.to_csv(file_path)

        return self.raw_data

    testdata = pd.DataFrame({"label": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                             "text": ["I'm so happy today!",
                                      "This is the best week of my life!",
                                      "YAY I graduated",
                                      "Today my sister married the love of her life!",
                                      "going off to Canada today! looking so forward",
                                      "It's raining and I have an appointment",
                                      "My boss fired me today",
                                      "I broke my leg",
                                      "I'll never be as happy as I want to be!",
                                      "My life sucks"]})


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
    # insults to be replaced by bad_word in _clean_tweets
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
    # negations to be replaced by not in _clean_tweets
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

    def __init__(self, raw_data, model_folder_path, colname_tweets, colname_label):
        self._model_folder_path = model_folder_path
        self.raw_df = raw_data  # todo make sure correct format
        self._colname_tweets = colname_tweets
        self._colname_label = colname_label
        self.preprocessed_df = None
        self.predicted_df = None
        self._y_test = None
        self._x_test = None
        self._model = None
        self.model_history = None
        self.evaluation_results = None

    def _preprocess_tweets(self):
        prep = self.raw_df.copy(deep=True)
        prep["clean_text"] = prep[self._colname_tweets].apply(
            lambda x: self._clean_tweet(x))  # TODO: adjust colname if necessary
        # prep.drop(self._colname_tweets, axis=1)
        # TODO: removing empty tweets after preprocessing?

        # assign to instance variable
        self.preprocessed_df = prep

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
        super().__init__(raw_data=raw_data, model_folder_path=model_folder_path,
                         colname_tweets=colname_tweets, colname_label=colname_label)
        self._x_train = None
        self._y_train = None
        self._word_index = None
        self._vocab_size = None
        self._max_length = None
        self._training_specs = {"lstm_size": 64, "dropout_rate": 0.5, "n_epochs": 5, "batch_size": 128}

    def train_model_on_data(self, glove_path, overfitting_plot=True, save_model=True,
                            training_specs=None):
        '''
        Builds and trains model based on 70% training and 30% testing data.
        :return:
        '''
        if training_specs is not None:
            self._training_specs = training_specs

        super()._preprocess_tweets()
        self._prepare_model_input(chatty=True)
        self._create_and_train_model(glove_path=glove_path)

        # show training
        if overfitting_plot:
            self._overfitting_plot()

        # save model
        if save_model:
            self._model.save(self._model_folder_path + "/model")

    def evaluate_out_of_sample(self, return_predictions=False, confusion_matrix=True, predictions_histogram=True):
        super()._predict_new_data(return_predictions=return_predictions, confusion_matrix=confusion_matrix,
                                  predictions_histogram=predictions_histogram)

    def _prepare_model_input(self, chatty=False):
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
        if chatty: print("There were " + str(self._vocab_size) + " unique words found.")

        # Padding sequences to same length
        self._max_length = max([len(x) for x in train_seq])
        if chatty: print('Max length of tweet (number of words): ' + str(self._max_length))

        # apply padding and save training and testing data
        self._x_train = pad_sequences(train_seq, maxlen=self._max_length)
        self._x_test = pad_sequences(test_seq, maxlen=self._max_length)
        self._y_train = np.array(train_test_df[self._colname_label].to_list())
        self._y_test = np.array(final_eval_df[self._colname_label].to_list())

        # save tokenizer to use later:
        tokenizer_json = tokenizer.to_json()
        with io.open(self._model_folder_path + '/tokenizer.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    def _create_and_train_model(self, glove_path):
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
        emb_matrix = np.zeros((self._vocab_size, self._training_specs.get("glove_dim")))
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
        m.add(Embedding(self._vocab_size, 50, input_length=self._max_length))
        m.add(LSTM(self._training_specs.get("lstm_size"), return_sequences=True))
        m.add(Dropout(rate=self._training_specs.get("dropout_rate")))
        m.add(LSTM(self._training_specs.get("lstm_size")))
        m.add(Dropout(rate=self._training_specs.get("dropout_rate")))
        m.add(Dense(1, activation='sigmoid'))

        # adjust embedding layer
        m.layers[0].set_weights([emb_matrix])
        m.layers[0].trainable = False # prevent overfitting

        # compile and train
        m.compile(optimizer='Adam', loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[BinaryAccuracy(), TrueNegatives(), TruePositives(), FalseNegatives(), FalsePositives()])
        hist = m.fit(self._x_train, self._y_train, epochs=self._training_specs.get("n_epochs"),
                     batch_size=self._training_specs.get("batch_size"),
                     verbose=1, validation_split=0.2)
        self._model = m
        self.model_history = pd.DataFrame(hist.history)

        return m

    def _overfitting_plot(self):
        df = pd.DataFrame(data=np.repeat(['Training', 'Validation'], repeats=self._training_specs.get("n_epochs")),
                          columns=['trainval'])
        df['xaxis'] = np.array([range(1, self._training_specs.get("n_epochs") + 1)] * 2).flatten()
        df['binary_accuracy'] = np.array(
            [self.model_history['binary_accuracy'], self.model_history['val_binary_accuracy']]).flatten()
        df['loss'] = np.array([self.model_history['loss'], self.model_history['val_loss']]).flatten()

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        # fig.subplots_adjust(wspace=.4, hspace=0.4)
        # fig.suptitle('Performance on training and validation dataset', fontsize=24)

        sns.scatterplot(ax=axes[0], x=df['xaxis'], y=df['binary_accuracy'], hue=df['trainval'], palette='Dark2',
                        legend=False)
        sns.lineplot(ax=axes[0], x=df['xaxis'], y=df['binary_accuracy'], hue=df['trainval'], palette='Dark2', alpha=0.3,
                     estimator=None, legend=None)
        axes[0].legend().remove()
        axes[0].set_xlabel('Epoch', size=20)
        axes[0].set_ylabel('Loss', size=20)
        axes[0].tick_params(axis='both', which='major', labelsize=15)
        axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

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
    def __init__(self, raw_data, model_folder_path, colname_tweets="text", colname_label="label"):
        super().__init__(raw_data=raw_data, model_folder_path=model_folder_path,
                         colname_tweets=colname_tweets, colname_label=colname_label)
        self._model = models.load_model(model_folder_path + "/model")
        self._padding_length = self._model.input_shape[1]  # length if inputs required for trained model

    def _prepare_model_input(self):
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
        super()._preprocess_tweets()
        self._prepare_model_input()
        super()._predict_new_data(return_predictions=return_predictions, confusion_matrix=False,
                                  predictions_histogram=predictions_histogram)


if __name__ == '__main__':
    # api access code

    access_token = os.environ.get('access_token')
    access_token_secret = os.environ.get('access_token_secret')
    consumer_key = os.environ.get('consumer_key')
    consumer_secret = os.environ.get('consumer_secret')

    print(access_token)

    #streamList = StdOutListener()
    #dataRetr = DataRetriever()
    #analyzyer = Analyzer(DataRetriever=dataRetr)

    exit()
