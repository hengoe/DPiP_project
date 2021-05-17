import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import nltk
from sklearn.model_selection import train_test_split

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

    def preprocess_tweets(self):
        '''
        Preprocesses the data, assign the preprocessed DataFrame to self.processed_df and split the data into
        train-test data and final evaluation data (out-of-sample).
        '''
        # insert prepocessing steps here
        prep = self.raw_df.copy(deep = True)
        prep["clean_text"] = prep["text"].progress_apply(lambda x: self._clean_tweet(x)) # TODO: adjust colname if necessary
        prep.drop("text", axis = 1)

        # TODO: empty tweets after preprocessing?

        # assign to instance variable
        self.processed_df = prep

        # split processed data into train/test and final evaluation dataset
        self.train_test_df, self.final_eval_df = train_test_split(prep, test_size=0.1, random_state=7)
        print("Shape of ... Training Data: ", self.train_test_df.shape, " ... Final Evaluation Data: ", self.final_eval_df.shape)


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
