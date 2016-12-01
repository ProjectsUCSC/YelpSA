
# coding: utf-8

# In[1]:

import pandas as pd
import nltk
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import copy
import re
import pickle
from nltk.stem.porter import *
import sys
from sklearn.feature_extraction.text import CountVectorizer
import pickle


# In[ ]:

def stemmer(preprocessed_data_sample):
    print "stemming "
    #Create a new Porter stemmer.
    stemmer = PorterStemmer()
    #try:
    for i in range(len(preprocessed_data_sample)):
        #Stemming
        #preprocessed_data_sample[i] = " ".join([stemmer.stem(str(word)) for word in preprocessed_data_sample[i]])
        #No stemming
        preprocessed_data_sample[i] = " ".join([str(word) for word in preprocessed_data_sample[i]])
    return preprocessed_data_sample

def vectorize(preprocessed_data_sample):
   

    file = open("features.obj",'rb')
    all_features = pickle.load(file)
    file.close()


    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    no_features = 200#500#806#150#800#600#350
    #ngram_range=(1, 1)
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, vocabulary = all_features, ngram_range=(1,2))#, #, max_features = no_features, ngram_range=(2,2)) 
    #vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(2,2), max_features = no_features)
    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of 
    # strings.
    train_data_features = vectorizer.fit_transform(preprocessed_data_sample)

    # Numpy arrays are easy to work with, so convert the result to an 
    # array
    train_data_features = train_data_features.toarray()
    return [train_data_features, vectorizer, no_features]

def tokenize_and_stopwords(data_sample):
    #data_sample = list(data_sample)
    #Get all english stopwords
    stop = stopwords.words('english')# + list(string.punctuation) + ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    #Use only characters from reviews
    data_sample = data_sample.str.replace("[^a-zA-Z ]", " ")#, " ")
    #print data_sample
    #tokenize and remove stop words
    return [[i for i in word_tokenize(sentence) if i not in stop] for sentence in data_sample]

def get_data_label(data):

    #remove stop words and punctuations.
    samples = 40000#55000
    
    #data=data.apply(np.random.permutation)
    data_labels = copy.deepcopy(data["stars"])

    DATA_SAMPLE = data[samples:samples+2500]['text'].str.lower()
    Y_TEST = data[samples:samples+2500]['stars']
    
    #important
    data_sample = data[0:samples]['text'].str.lower()#.split('\n')
    labels_sample = data_labels[0:samples]

    #data_bad = data_sample[(labels_sample==1) | (labels_sample==2)]
    #labels_bad = labels_sample[(labels_sample==1) | (labels_sample==2)]

    #data_average = data_sample[(labels_sample==3)]
    #labels_average = labels_sample[(labels_sample==3)]

    #data_good = data_sample[(labels_sample==4) | (labels_sample==5)]
    #labels_good = labels_sample[(labels_sample==4) | (labels_sample==5)]

    #print len(data_bad), len(data_average), len(data_good)
    #train_length = 4000
    #test_length = 1000
    #Appending training samples
    #data_sample = (data_bad[0:train_length][:].append(data_average[0:train_length][:])).append(data_good[0:train_length][:])
    #labels_sample = (labels_bad[0:train_length][:].append(labels_average[0:train_length][:])).append(labels_good[0:train_length][:])
    #Appending testing samples
    #data_sample = data_sample.append((data_bad[train_length:(train_length+test_length)][:].append(data_average[train_length:(train_length+test_length)][:])).append(data_good[train_length:(train_length+test_length)][:]))
    #labels_sample = labels_sample.append((labels_bad[train_length:(train_length+test_length)][:].append(labels_average[train_length:(train_length+test_length)][:])).append(labels_good[train_length:(train_length+test_length)][:]))

    #print data_sample
    #print (data_sample)
    #print len(data_sample)
    #print (labels_sample[0:3])
    #print (labels_sample[4000:4003])
    #print (labels_sample[8000:8003])
    #print (labels_sample[12000:12003])
    #print (labels_sample[13000:13003])
    #print (labels_sample[14000:14003])
    print  len(data_sample)
    #print data_sample[2992]
    return [data_sample, labels_sample, samples]


# In[ ]:

def preprocess():
    
    #Read datasets
	#Add your path here
    data = pd.read_csv("C:/Users/sanja/Desktop/Sanjana/books/UCSC/YelpSA/Data/yelp_academic_dataset_review.csv")
    business_data = pd.read_csv("C:/Users/sanja/Desktop/Sanjana/books/UCSC/YelpSA/Data/yelp_academic_dataset_business.csv")
    business_id = business_data[:][['business_id','categories','review_count']]
    
    #Merge datasets on key
    data = pd.merge(data, business_id, on='business_id')
    
    #obtain only restaurant reviews
    rest_exist=[]
    for i in data[:]['categories']:
        if "Restaurants" in i:
            rest_exist.append(True)
        else:
            rest_exist.append(False)
    #Add a column, True if restaurant, False for any other business
    data['rest_exist']=rest_exist
    #Selecting particular columns
    data= data[:][[0,2,3,5,6,9,11,12]]
    #Get only restaurant reviews
    data=data[data['rest_exist']][:]
    data=data.drop('rest_exist',1)
    
    pd.set_option('display.max_colwidth',-1)
    [data_sample, labels_sample, samples] = get_data_label(data)
    
    #Tokenize and remove stopwords
    preprocessed_data_sample = tokenize_and_stopwords(data_sample)
    #PREPROCESSED_DATA_SAMPLE = tokenize_and_stopwords(DATA_SAMPLE)
    
    file = open("features.obj",'rb')
    all_features = pickle.load(file)
    file.close()
    
    #Fake stemming
    preprocessed_data_sample = stemmer(preprocessed_data_sample)
    #PREPROCESSED_DATA_SAMPLE = stemmer(PREPROCESSED_DATA_SAMPLE)
    data_sample_copy = copy.deepcopy(preprocessed_data_sample)
    
    #Vectorize
    [vectorized_preprocessed_data_sample, vectorizer, no_features] = vectorize(preprocessed_data_sample)
    #[PREPROCESSED_DATA_SAMPLE, v, no_f] = vectorize(PREPROCESSED_DATA_SAMPLE)
    
    vocab = vectorizer.get_feature_names()
    print "length of vocabulary", len(vocab)
    return [vectorized_preprocessed_data_sample, labels_sample, vectorizer, len(vocab), samples, data_sample_copy]

