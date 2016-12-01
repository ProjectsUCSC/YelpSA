import logistic_classifier as clf_log
import naive_bayes_classifier as clf_nb
import numpy as np
import copy

def train(X_train, y_train, no_classes):
    w_nb, prior = clf_nb.train(X_train, y_train, no_classes)
    print w_nb
    w_lr = copy.deepcopy(w_nb)
    w_lr = clf_log.train_logistic_regression(X_train, y_train, no_classes, w_lr)
    return w_lr, prior, w_nb

#def predict_ternary(w_lr, X_test) lr  
#def predict(X_test, w_nb, prior) nb
#return pred, prob
def predict_ternary(X_test, w_lr, w_nb, prior):
    pred_nb, prob_nb = clf_nb.pred_ternary(X_test, w_nb, prior)
    pred_lr, prob_lr = clf_log.predict_ternary(w_lr, X_test)
    prob_nb = [i[0] for i in prob_nb]
    pred = np.zeros((len(pred_nb)))
    pred[prob_nb > prob_lr] = pred_nb[prob_nb > prob_lr]
    pred[prob_nb <= prob_lr] = pred_lr[prob_nb <= prob_lr]
    
    return pred

def predict_binary(X_test, w_lr, w_nb, prior):
    pred_nb, prob_nb = clf_nb.pred_binary(X_test, w_nb, prior)
    pred_lr, prob_lr = clf_log.predict_binary(w_lr, X_test)
    prob_nb = [i[0] for i in prob_nb]
    pred = np.zeros((len(pred_nb)))
    pred[prob_nb > prob_lr] = pred_nb[prob_nb > prob_lr]
    pred[prob_nb <= prob_lr] = pred_lr[prob_nb <= prob_lr]
    
    return pred

def predict(X_test, w_lr, w_nb, prior):
    pred_nb, prob_nb = clf_nb.predict(X_test, w_nb, prior)
    pred_lr, prob_lr = clf_log.predict(w_lr, X_test)
    prob_nb = [i[0] for i in prob_nb]
    pred = np.zeros((len(pred_nb)))
    pred[prob_nb > prob_lr] = pred_nb[prob_nb > prob_lr]
    pred[prob_nb <= prob_lr] = pred_lr[prob_nb <= prob_lr]
    
    return pred
    
    