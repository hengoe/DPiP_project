from code import DataRetriever
import unittest


class TestPreprocessing(unittest.TestCase):
    def test_clean_tweet(self):
        # initialize DataRetriever
        retr = DataRetriever()
        retr.training_data = retr.testdata

        # initialize Analyzer
        analy = Analyzer(retr)

        # preprocess tweets
        result = analy._clean_tweet("I was lost")

        self.assertEqual(result, "lost")


if __name__ == '__main__':
    unittest.main()
