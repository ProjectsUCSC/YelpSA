from io import StringIO
import unittest
import numpy as np
import pandas as pd
from pre_processing import *

class TestMap (unittest.TestCase):
    def test_tokenize_and_stopwords (self):
        test_sample = ["this is a test sample test test that has been written to understand and test the feauture engineering used in the contact.  It should remove numbers such as 1, 2,3 and special characters like ! . Also we should not see stop words such as a, an , the. "
                        , "adding another test string", "test should be the most occuring word"]  
        #list(test_sample)
        test_sample = pd.Series(data = test_sample)
        #print test_sample.shape
        test_preprocessing_sample = tokenize_and_stopwords(test_sample)
        pred = test_preprocessing_sample[2] == ['test', 'occuring', 'word']
        print pred
        self.assertEqual(pred, True)

    def test_stemmer (self):
        test_sample = ["this is a test sample that has been written to understand and test the feauture engineering used in the contact.  It should remove numbers such as 1, 2,3 and special characters like ! . Also we should not see stop words such as a, an , the. "
                        , "adding another test string", "test should be the most occuring word. these sentences will join"]  
        #list(test_sample)
        test_sample = pd.Series(data = test_sample)
        test_token_sample = tokenize_and_stopwords(test_sample)
        test_stem_sample = stemmer(test_token_sample)
        self.assertEqual( test_stem_sample[2],'test occuring word sentences join')

suite = unittest.TestLoader().loadTestsFromTestCase( TestMap )
unittest.TextTestRunner(verbosity=1).run( suite )