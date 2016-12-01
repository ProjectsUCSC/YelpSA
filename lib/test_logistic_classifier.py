import logistic_classifier as clf
import unittest
class TestMap (unittest.TestCase):
    def test_logistic (self):
        x_train = [[1],[2],[3],[4],[5],[6],[7],[8]]
        y_train= [1,1,1,1,1,1,1,1]
        x_test=[[1],[2],[3]]
        model =clf.train_logistic_regression(x_train, y_train ,2)
        [pred, prob] = clf.predict(model, x_test)
        pred = pred == [1,1,1]
        self.assertEqual(pred[1 ], True)
        self.assertEqual(pred[0 ], True)
        self.assertEqual(pred[2 ], True)
    
    def test_logistic_bin (self):
        x_train = [[1],[2],[3],[4],[5],[6],[7],[8]]
        y_train= [1,1,1,1,1,1,0,2]
        x_test=[[1],[2],[3]]
        model =clf.train_logistic_regression(x_train, y_train ,5)
        [pred, prob] = clf.predict_binary(model, x_test)
        pred = pred == [0,0,0]
        self.assertEqual(pred[1 ], True)
        self.assertEqual(pred[0 ], True)
        self.assertEqual(pred[2 ], True)
        
    def test_logistic_ternary (self):
        x_train = [[1],[2],[3],[4],[5],[6],[7],[8]]
        y_train= [1,1,1,1,1,1,1,1]
        x_test=[[1],[2],[3]]
        model =clf.train_logistic_regression(x_train, y_train ,5)
        [pred, prob] = clf.predict_ternary(model, x_test)
        pred = pred == [0,0,0]
        self.assertEqual(pred[1 ], True)
        self.assertEqual(pred[0 ], True)
        self.assertEqual(pred[2 ], True)
        
        
suite = unittest.TestLoader().loadTestsFromTestCase( TestMap )
unittest.TextTestRunner(verbosity=1).run( suite )