
# coding: utf-8

# In[ ]:

import collections
import re
from itertools import islice, izip
import numpy as np
import pre_processing
def feature_selection(labels, data_sample_copy):  
    new_vocab = []
    new_vocab = set(new_vocab)
    #vocab_list_count = [len(word_here.split()) for word_here in vocab]
    
    #counter = collections.Counter(vocab_list_count)
    #print counter
    
    #Obtain unigram count
    #print len(vocab)
    #print type(vocab)
    #print type(data_sample_copy)
    bad = np.array(((labels == 1) | (labels == 2)))
    average = np.array((labels == 3))
    good = np.array(((labels == 4) | (labels == 5)))
    #print type(good)
    data_sample_string_good = " ".join(data_sample_copy[good])
    data_sample_string_average = " ".join(data_sample_copy[average])
    data_sample_string_bad = " ".join(data_sample_copy[bad])
    len_data = [len(data_sample_copy[bad]), len(data_sample_copy[average]), len(data_sample_copy[good])]
    #data_sample_string[1:15000]
    from collections import Counter
    j =0
    for data_sample_string in [data_sample_string_bad, data_sample_string_average, data_sample_string_good]:
        temp_words = data_sample_string.split()
        wordCount = Counter(temp_words)
        #print len(wordCount)
        for key in wordCount.keys() :
             wordCount[key] /= float(len_data[j])
        j = j+1
        wordCount = wordCount.most_common(700)
        #print "u",type(wordCount)
        for k in range(len(wordCount)):
            wordCount[k] = wordCount[k][0]
            if(wordCount[k] in new_vocab):
                new_vocab.remove(wordCount[k])
            else:
                new_vocab.add(wordCount[k])
        #new_vocab += wordCount
        print "unigrams", len(new_vocab)
        #Obtain bigram count and choose top 150 bigrams
        words = re.findall("\w+", data_sample_string)
        req_bigrams = Counter(izip(words, islice(words, 1, None))).most_common(150)
        #print "b", req_bigrams
        for k in range(len(req_bigrams)):
            req_bigrams[k] = req_bigrams[k][0][0] + " " + req_bigrams[k][0][1]
            if(req_bigrams[k] in new_vocab):
                new_vocab.remove(req_bigrams[k])
            else:
                new_vocab.add(req_bigrams[k])
        print "after addign bigrams", len(new_vocab)

    all_features = {}
    new_vocab = list(new_vocab)
    for k in range(len(new_vocab)):
        new_vocab[k]
    print "overall", len(new_vocab)
    print new_vocab
    #Saving features
    import pickle
    i = 0
    for bw in new_vocab:#required_features:
        all_features[bw] = i
        i += 1
    
    filehandler = open("features.obj","wb")
    pickle.dump(all_features,filehandler)
    filehandler.close()


# In[ ]:

#To be called in main
[preprocessed_data_sample, labels_sample, vectorizer, no_features, samples] = preprocess()
data_sample_copy = (preprocessed_data_sample)
feature_selection(labels_sample, np.array(data_sample_copy))

