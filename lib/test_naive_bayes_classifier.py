
# coding: utf-8

# In[ ]:

import naive_bayes_classifier as nb
import unittest
class TestMap (unittest.TestCase):
    def test_naive (self):
        x_train = [[1],[2],[3],[4],[5],[6],[7],[8]]
        y_train= [1,1,1,1,1,1,1,1]
        x_test=[[1],[2],[3]]
        model =nb.train(x_train, y_train ,5)
        prior=[1,0,0,0,0]
        [pred, prob] = nb.predict(x_test,model,prior)
        pred = pred == [1,1,1]
        self.assertEqual(pred, True)
        
    
    def test_naive_bin (self):
        x_train = [[1],[2],[3],[4],[5],[6],[7],[8]]
        y_train= [1,1,1,1,1,1,2,2]
        x_test=[[1],[2],[3]]
        model =nb.train(x_train, y_train ,5)
        prior=[0.75,0.25,0,0,0]
        [pred, prob] = nb.pred_binary(x_test,model,prior)
        pred = pred == [0,0,0]
        self.assertEqual(pred, True)
        
    def test_naive_ternary (self):
        x_train = [[1],[2],[3],[4],[5],[6],[7],[8]]
        y_train= [1,1,1,1,1,1,1,1]
        x_test=[[1],[2],[3]]
        model =nb.train(x_train, y_train ,5)
        prior=[1,0,0,0,0]
        [pred, prob] = nb.pred_ternary(x_test,model,prior)
        pred = pred == [0,0,0]
        print pred
        self.assertEqual(pred, True)
        
        
suite = unittest.TestLoader().loadTestsFromTestCase( TestMap )
unittest.TextTestRunner(verbosity=1).run( suite )

