
# coding: utf-8

# In[644]:

import pandas as pd
data = pd.read_csv("Data/yelp_academic_dataset_review.csv")
len(data)


# In[645]:

list(data)


# In[646]:

business_data = pd.read_csv("Data/yelp_academic_dataset_business.csv")
list(business_data)


# In[647]:

list(business_data)


# In[648]:

business_id = business_data[:][['business_id','categories','review_count']]


# In[649]:

data = pd.merge(data, business_id, on='business_id')
len(data)


# In[650]:

list(data)


# In[651]:

rest_exist=[]
for i in data[:]['categories']:
    if "Restaurants" in i:
        rest_exist.append(True)
    else:
        rest_exist.append(False)
        
    


# In[652]:

data['rest_exist']=rest_exist
list(data)


# In[653]:


print list(data)
data= data[:][[0,2,3,5,6,9,11,12]]
print list(data)


# In[654]:

len(data)


# In[655]:

data=data[data['rest_exist']][:]


# In[656]:

data=data.drop('rest_exist',1)


# In[657]:

pd.set_option('display.max_colwidth',-1)


# In[658]:

data[0:1]['text']


# filtering for city

# In[91]:

data = data[data[city] == 'New York']


# end of city filtering

# Filtering data

# In[21]:

import collections
counter = collections.Counter(data["user_id"])


# In[22]:

freq = counter.values()
temp = counter.keys()
print counter["Qh5A5NlP4UVvddSasOYR4A"]


# In[24]:

import numpy as np
boolean_array = np.array([False] * len(data))
users = np.array(data["user_id"])
#print users[0:10]
for i in range(len(users)):
    if counter[users[i]] >= 100:
        boolean_array[i] = True


# In[25]:

print len(data)
print len(boolean_array)
print sum(boolean_array == True)


# In[26]:

data = data[boolean_array][:]


# End of filtering

# In[305]:

len(data)


# In[163]:

import matplotlib.pyplot as plt
import math
axes = plt.gca()
#axes.set_xlim([xmin,xmax])
axes.set_ylim([0,100])
plt.hist(freq, bins=int(max(freq)))
#ylim(0,5000)
plt.show()


# Adding additional feature

# In[226]:

print len(data)
import pandas as pd
data['user_review_count']=np.zeros(len(data))
print len(data["user_id"])
#print counter[data['user_id'][0]]
for i in range(len(data)):
    try:
        print data['user_id'][i]
    except:
        print data['user_id'][i]
        break


# In[25]:




# In[26]:

print len(data["text"])
#print (data["text"])[2992]
#print type(data["text"][0])


#  Text Preprocessing
#  1. Remove stop words and number and punctuations

# In[659]:

import nltk
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords

import string
#remove stop words and punctuations.
samples = 20000#55000
#Shuffling data
#data=data.apply(np.random.permutation)

import copy
data_labels = copy.deepcopy(data["stars"])
#labels[0:5]
#print len(data_labels)
#print (data_labels)
#len(data)
#data_labels = data["stars"]
#print len(data_labels)
#print len(data_labels), len(data)

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
train_length = 4000
test_length = 1000
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


# In[661]:

print data_sample[0]


# Function to tokenize and remove stopwords

# In[662]:

import re
def tokenize_and_stopwords(data_sample):
    #data_sample = list(data_sample)
    #Get all english stopwords
    stop = stopwords.words('english')# + list(string.punctuation) + ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    #Use only characters from reviews
    data_sample = data_sample.str.replace("[^a-zA-Z ]", " ")#, " ")
    #print data_sample
    #tokenize and remove stop words
    return [[i for i in word_tokenize(sentence) if i not in stop] for sentence in data_sample]


# In[664]:

len(data_sample)
print data_sample[0]
print labels_sample[0]


# Remove stopwords

# In[665]:

preprocessed_data_sample = tokenize_and_stopwords(data_sample)
PREPROCESSED_DATA_SAMPLE = tokenize_and_stopwords(DATA_SAMPLE)


# In[666]:

print preprocessed_data_sample[0]
print "hello"
print data_sample[0]


# Load vocabulary

# In[673]:

import pickle

#filehandler = open("features.obj","wb")
#pickle.dump(all_features,filehandler)
#filehandler.close()

file = open("features.obj",'rb')
all_features = pickle.load(file)
file.close()


# In[674]:

def vectorize(preprocessed_data_sample):
    from sklearn.feature_extraction.text import CountVectorizer

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


# In[675]:

from nltk.stem.porter import *
import sys

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


# In[670]:

import copy
preprocessed_data_sample = stemmer(preprocessed_data_sample)
PREPROCESSED_DATA_SAMPLE = stemmer(PREPROCESSED_DATA_SAMPLE)
data_sample_copy = (preprocessed_data_sample)


# In[676]:

[preprocessed_data_sample, vectorizer, no_features] = vectorize(preprocessed_data_sample)
[PREPROCESSED_DATA_SAMPLE, v, no_f] = vectorize(PREPROCESSED_DATA_SAMPLE)


# In[678]:

print (data_sample_copy[0])


# In[679]:

#Trial for wordcount without stemming
#for i in range(len(preprocessed_data_sample)):
#    preprocessed_data_sample[i] = " ".join(preprocessed_data_sample[i])
#preprocessed_data_sample = vectorize(preprocessed_data_sample)


# In[680]:

vocab = vectorizer.get_feature_names()
#counts = vectorizer.get_params()
#print counts
print len(vocab)
print type(vocab)
print vocab


# Feature Selection - One time

# In[671]:

import collections
import re
from itertools import islice, izip
import numpy as np
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
        #new_vocab += req_bigrams
    #combine unigrams and bigrams
    #top 807 unigrams
    #required_features = {u'food': 11288, u'good': 10313, u'place': 9703, u'order': 7331, u'great': 6734, u'like': 6499, u'go': 6076, u'get': 6010, u'time': 5996, u'one': 5363, u'servic': 5239, u'restaur': 4473, u'would': 4369, u'realli': 4268, u'back': 4167, u'tri': 3893, u'pizza': 3758, u'also': 3464, u'love': 3415, u'best': 3399, u'even': 3295, u'nice': 3237, u'come': 3174, u'eat': 3161, u'us': 3148, u'chees': 3147, u'dont': 3136, u'fri': 3103, u'bar': 3100, u'got': 3079, u'ive': 3077, u'menu': 3070, u'tabl': 3043, u'alway': 2920, u'wait': 2906, u'chicken': 2903, u'well': 2846, u'salad': 2787, u'im': 2783, u'make': 2757, u'littl': 2736, u'sauc': 2726, u'want': 2709, u'delici': 2708, u'drink': 2694, u'meal': 2690, u'look': 2592, u'tast': 2529, u'came': 2509, u'beer': 2505, u'pittsburgh': 2499, u'price': 2475, u'pretti': 2456, u'sandwich': 2433, u'night': 2414, u'much': 2297, u'never': 2288, u'lunch': 2242, u'went': 2230, u'didnt': 2221, u'dinner': 2174, u'could': 2134, u'definit': 2106, u'peopl': 2093, u'first': 2079, u'think': 2075, u'say': 2073, u'thing': 2029, u'better': 2029, u'ask': 2025, u'serv': 2002, u'dish': 1992, u'friend': 1966, u'experi': 1951, u'two': 1934, u'know': 1933, u'take': 1924, u'friendli': 1910, u'enjoy': 1908, u'made': 1900, u'flavor': 1871, u'side': 1822, u'staff': 1814, u'server': 1807, u'special': 1804, u'seat': 1793, u'fresh': 1785, u'bread': 1784, u'burger': 1755, u'ever': 1753, u'cook': 1742, u'way': 1699, u'top': 1694, u'recommend': 1689, u'minut': 1677, u'day': 1667, u'visit': 1648, u'lot': 1634, u'right': 1628, u'bit': 1600, u'favorit': 1589, u'still': 1587, u'around': 1582, u'star': 1580, u'give': 1546, u'wasnt': 1540, u'though': 1525, u'amaz': 1521, u'bad': 1514, u'seem': 1510, u'said': 1498, u'waitress': 1498, u'hot': 1496, u'sure': 1488, u'year': 1468, u'busi': 1450, u'sinc': 1435, u'review': 1435, u'someth': 1433, u'atmospher': 1416, u'roll': 1415, u'small': 1396, u'expect': 1363, u'area': 1356, u'cant': 1353, u'excel': 1343, u'feel': 1339, u'steak': 1335, u'took': 1326, u'walk': 1323, u'pasta': 1314, u'last': 1305, u'locat': 1305, u'mani': 1300, u'need': 1300, u'appet': 1298, u'select': 1291, u'everi': 1285, u'everyth': 1279, u'stop': 1271, u'use': 1271, u'disappoint': 1270, u'find': 1269, u'work': 1252, u'soup': 1251, u'potato': 1250, u'hour': 1244, u'dine': 1244, u'meat': 1242, u'anoth': 1218, u'wine': 1207, u'tasti': 1206, u'noth': 1198, u'see': 1197, u'perfect': 1188, u'sweet': 1179, u'dessert': 1176, u'next': 1173, u'reason': 1173, u'portion': 1171, u'your': 1164, u'home': 1163, u'italian': 1154, u'big': 1148, u'worth': 1147, u'long': 1140, u'enough': 1140, u'start': 1117, u'sushi': 1107, u'awesom': 1090, u'usual': 1087, u'new': 1075, u'larg': 1068, u'plate': 1066, u'differ': 1059, u'entre': 1052, u'check': 1051, u'end': 1048, u'fish': 1044, u'actual': 1039, u'grill': 1034, u'qualiti': 1033, u'id': 1024, u'that': 1021, u'happi': 1009, u'howev': 1006, u'quit': 1003, u'charlott': 1003, u'sit': 1002, u'huge': 994, u'decid': 982, u'offer': 976, u'breakfast': 970, u'shrimp': 965, u'probabl': 964, u'ok': 963, u'spot': 956, u'option': 944, u'kind': 942, u'live': 918, u'decent': 916, u'park': 910, u'manag': 906, u'anyth': 906, u'thought': 901, u'told': 897, u'cake': 889, u'away': 880, u'call': 877, u'egg': 869, u'close': 866, u'wonder': 863, u'famili': 852, u'half': 851, u'beef': 847, u'green': 841, u'person': 832, u'full': 827, u'crowd': 824, u'mayb': 824, u'dog': 814, u'els': 812, u'old': 808, u'choic': 808, u'room': 807, u'wing': 804, u'tomato': 804, u'spici': 801, u'cours': 799, u'left': 798, u'outsid': 796, u'game': 795, u'slice': 789, u'open': 787, u'without': 780, u'let': 779, u'overal': 777, u'cold': 775, u'arriv': 770, u'waiter': 761, u'put': 758, u'attent': 757, u'reserv': 754, u'almost': 754, u'rice': 752, u'everyon': 752, u'parti': 749, u'week': 746, u'item': 746, u'found': 745, u'custom': 745, u'ill': 742, u'return': 732, u'three': 729, u'husband': 724, u'brunch': 721, u'insid': 718, u'point': 708, u'absolut': 707, u'gener': 706, u'pancak': 704, u'super': 703, u'local': 696, u'fantast': 693, u'group': 687, u'onion': 687, u'coupl': 685, u'help': 685, u'cream': 679, u'town': 677, u'quick': 675, u'least': 667, u'sever': 665, u'fill': 665, u'isnt': 663, u'part': 662, u'leav': 659, u'high': 646, u'bartend': 646, u'lobster': 645, u'must': 645, u'hous': 643, u'salmon': 643, u'final': 643, u'mac': 642, u'bacon': 641, u'fan': 638, u'may': 637, u'perfectli': 629, u'hard': 627, u'includ': 626, u'bring': 626, u'crust': 625, u'especi': 624, u'list': 623, u'decor': 622, u'season': 622, u'fast': 619, u'street': 618, u'far': 612, u'size': 610, u'sat': 609, u'hand': 608, u'bite': 607, u'ate': 606, u'chang': 605, u'impress': 603, u'brought': 603, u'glass': 603, u'second': 601, u'piec': 595, u'citi': 595, u'pay': 594, u'guy': 593, u'red': 592, u'dress': 591, u'pork': 588, u'light': 588, u'fun': 587, u'coffe': 587, u'tell': 584, u'fine': 583, u'prepar': 582, u'wife': 579, u'averag': 578, u'french': 578, u'kitchen': 577, u'mix': 577, u'doesnt': 576, u'chines': 573, u'either': 573, u'wont': 570, u'thai': 570, u'water': 568, u'warm': 566, u'care': 563, u'wrong': 562, u'whole': 562, u'veggi': 560, u'done': 559, u'line': 558, u'saturday': 557, u'dri': 557, u'although': 555, u'crab': 552, u'less': 549, u'bean': 547, u'watch': 544, u'show': 543, u'pick': 541, u'finish': 538, u'gave': 537, u'mean': 536, u'couldnt': 533, u'regular': 533, u'cheap': 532, u'real': 532, u'move': 531, u'often': 531, u'oliv': 529, u'pack': 528, u'keep': 526, u'rememb': 525, u'someon': 525, u'free': 525, u'okay': 524, u'might': 524, u'pm': 523, u'extra': 521, u'name': 521, u'sausag': 517, u'pepper': 516, u'eaten': 512, u'amount': 510, u'surpris': 509, u'instead': 509, u'miss': 508, u'sunday': 507, u'chocol': 506, u'highli': 506, u'felt': 504, u'clean': 502, u'wish': 500, u'ye': 500, u'white': 499, u'owner': 498, u'style': 498, u'yet': 496, u'past': 494, u'wouldnt': 494, u'there': 493, u'consist': 493, u'butter': 492, u'hope': 491, u'crispi': 491, u'neighborhood': 491, u'share': 490, u'door': 488, u'extrem': 488, u'cool': 485, u'mushroom': 484, u'ice': 481, u'slow': 480, u'friday': 478, u'kid': 478, u'weekend': 476, u'diner': 474, u'pleas': 474, u'oh': 473, u'date': 473, u'thank': 468, u'abl': 468, u'quickli': 467, u'complet': 466, u'toast': 465, u'cut': 464, u'hit': 462, u'birthday': 461, u'bland': 459, u'run': 458, u'later': 456, u'total': 456, u'rib': 456, u'bottl': 455, u'fact': 454, u'recent': 453, u'boyfriend': 452, u'front': 452, u'pho': 451, u'sometim': 449, u'talk': 448, u'tea': 446, u'suggest': 446, u'guess': 444, u'rather': 443, u'trip': 441, u'deal': 437, u'head': 434, u'late': 433, u'money': 433, u'plu': 433, u'main': 433, u'chef': 429, u'german': 429, u'seafood': 428, u'mention': 427, u'buffet': 426, u'type': 426, u'theyr': 422, u'typic': 422, u'given': 421, u'corn': 417, u'chip': 416, u'problem': 414, u'terribl': 413, u'bill': 411, u'lack': 411, u'shop': 410, u'stand': 410, u'dip': 410, u'havent': 407, u'authent': 405, u'tip': 404, u'stuf': 401, u'taco': 400, u'except': 400, u'vegetarian': 400, u'rare': 398, u'hoagi': 398, u'ago': 398, u'cocktail': 398, u'turn': 396, u'downtown': 396, u'expens': 395, u'heard': 394, u'blue': 393, u'spinach': 393, u'bbq': 393, u'stay': 392, u'plenti': 392, u'today': 388, u'uptown': 387, u'youll': 386, u'incred': 385, u'yummi': 385, u'deliveri': 384, u'receiv': 384, u'ingredi': 382, u'empti': 381, u'tender': 381, u'read': 380, u'veget': 380, u'deliv': 379, u'mind': 379, u'comfort': 378, u'worst': 378, u'ad': 377, u'garlic': 377, u'other': 376, u'set': 375, u'happen': 375, u'entir': 375, u'grab': 374, u'short': 373, u'tuna': 372, u'lettuc': 371, u'pie': 370, u'four': 369, u'base': 367, u'rate': 367, u'noodl': 366, u'interest': 365, u'chain': 362, u'varieti': 362, u'hostess': 362, u'add': 360, u'homemad': 360, u'ambianc': 358, u'uniqu': 358, u'greasi': 358, u'martini': 358, u'soon': 358, u'yelp': 358, u'standard': 356, u'normal': 356, u'medium': 354, u'hungri': 351, u'ravioli': 350, u'n': 349, u'drive': 349, u'believ': 349, u'complaint': 348, u'consid': 344, u'werent': 344, u'anyon': 344, u'beauti': 344, u'bottom': 343, u'filet': 343, u'patio': 343, u'gone': 342, u'alreadi': 338, u'saw': 338, u'solid': 336, u'basic': 335, u'rest': 335, u'wall': 334, u'understand': 333, u'notic': 333, u'calamari': 332, u'chop': 331, u'rock': 331, u'cash': 328, u'rude': 327, u'simpl': 326, u'twice': 326, u'mert': 326, u'cover': 325, u'scallop': 325, u'strip': 325, u'treat': 325, u'wrap': 324, u'prefer': 323, u'stuff': 321, u'readi': 321, u'joint': 321, u'month': 319, u'split': 317, u'roast': 315, u'bowl': 315, u'black': 314, u'salt': 314, u'stick': 314, u'cornbread': 311, u'near': 310, u'pricey': 310, u'soul': 309, u'card': 308, u'thin': 307, u'charg': 306, u'thick': 306, u'pleasant': 305, u'across': 305, u'etc': 304, u'choos': 303, u'lamb': 302, u'excit': 302, u'oil': 302, u'earli': 302, u'play': 301, u'togeth': 301, u'immedi': 300, u'mouth': 299, u'sort': 299, u'serious': 298, u'space': 295, u'five': 295, u'refil': 293, u'establish': 293, u'greet': 292, u'spring': 291, u'slaw': 289, u'spice': 288, u'store': 288, u'curri': 288, u'horribl': 287, u'kept': 287, u'world': 284, u'salti': 284, u'girl': 284, u'unfortun': 284, u'mediocr': 283, u'along': 282, u'certainli': 281, u'arent': 280, u'job': 280, u'crave': 280, u'seen': 279, u'complain': 279, u'write': 279, u'cup': 278, u'slightli': 278, u'meatbal': 278, u'tofu': 277, u'satisfi': 277, u'crepe': 277, u'chili': 276, u'loud': 276, u'spend': 276, u'sound': 275, u'primanti': 274, u'floor': 274, u'tradit': 273, u'counter': 273, u'chanc': 273, u'brown': 273, u'easi': 272, u'soft': 272, u'dark': 271, u'liter': 271, u'glad': 271, u'addit': 269, u'music': 268, u'life': 268, u'issu': 268, u'plan': 267, u'note': 267, u'wed': 266, u'hate': 266, u'southern': 266, u'exactli': 266, u'sub': 265, u'cuisin': 265, u'melt': 263, u'frozen': 262, u'man': 262, u'simpli': 262, u'casbah': 261, u'occas': 261, u'knew': 261, u'wow': 260, u'hotel': 260, u'event': 259, u'request': 259, u'mom': 259, u'app': 258, u'bake': 258, u'sign': 258, u'present': 258, u'die': 258, u'turkey': 257, u'burrito': 257, u'outdoor': 256, u'gyro': 256, u'american': 255, u'smoke': 255, u'buffalo': 254, u'sport': 254, u'ton': 253, u'rush': 252, u'aw': 252, u'anyway': 252, u'hear': 252, u'idea': 251, u'pass': 250, u'ladi': 250, u'soba': 250, u'tv': 250, u'min': 250, u'upstair': 250, u'quiet': 249, u'casual': 248, u'behind': 248, u'sour': 248, u'pepperoni': 248, u'weve': 248, u'cost': 247, u'tini': 247, u'girlfriend': 247, u'appl': 247, u'chose': 246, u'avail': 246, u'overpr': 246, u'mine': 246, u'poor': 245, u'welcom': 245, u'lidia': 244, u'opinion': 244, u'smell': 243, u'suppos': 242, u'appreci': 240, u'mash': 239, u'provid': 239, u'pub': 237, u'fairli': 237, u'taken': 237, u'collard': 237, u'outstand': 237, u'possibl': 236, u'draft': 236, u'compar': 235, u'honestli': 235, u'employe': 234, u'tap': 234, u'low': 233, u'meet': 232, u'yum': 232, u'due': 232, u'number': 231, u'mostli': 231, u'anywher': 231, u'cozi': 231, u'morn': 230, u'patron': 230, u'takeout': 228, u'banana': 226, u'textur': 226, u'eye': 225, u'combin': 224, u'within': 224, u'creami': 224, u'realiz': 223, u'booth': 223, u'pull': 221, u'bun': 221, u'build': 221, u'plain': 221, u'save': 220, u'tonight': 219, u'frequent': 219, u'paid': 219, u'truli': 219, u'sorri': 219, u'pita': 219, u'easili': 219, u'agre': 218, u'trio': 218, u'dirti': 217, u'matter': 217, u'heart': 217, u'rich': 216, u'forward': 216, u'broccoli': 216, u'despit': 216, u'classic': 215, u'accommod': 214, u'box': 213, u'celebr': 213, u'true': 213, u'forget': 213, u'appar': 212, u'gotten': 212, u'somewher': 211, u'figur': 211, u'crisp': 211, u'apolog': 211, u'fanci': 211, u'origin': 210, u'middl': 210, u'afternoon': 209, u'duck': 209, u'pickl': 209, u'crazi': 208, u'burgh': 208, u'explain': 208, u'dollar': 207, u'convers': 206, u'avoid': 206, u'juici': 206, u'harri': 205, u'level': 205, u'ds': 205, u'upon': 205, u'word': 205, u'shadysid': 204, u'sampl': 204, u'fair': 204, u'buy': 204, u'summer': 204, u'soggi': 203, u'school': 202, u'buck': 202, u'mood': 201, u'longer': 201, u'gnocchi': 200, u'weird': 200, u'market': 200}
    #longer
    #required_features = {u'good': 20624, u'food': 17687, u'place': 17212, u'like': 14958, u'order': 14055, u'time': 11385, u'get': 11230, u'go': 11066, u'one': 10594, u'great': 9093, u'realli': 8709, u'servic': 8361, u'tri': 7686, u'chicken': 7556, u'back': 7295, u'would': 7039, u'restaur': 6513, u'come': 6247, u'also': 6075, u'love': 6066, u'menu': 6044, u'sauc': 5797, u'lunch': 5745, u'salad': 5741, u'eat': 5732, u'fri': 5714, u'littl': 5688, u'got': 5636, u'even': 5635, u'nice': 5556, u'chees': 5481, u'well': 5477, u'pretti': 5356, u'want': 5338, u'alway': 5194, u'look': 5054, u'make': 4997, u'tabl': 4820, u'came': 4791, u'tast': 4784, u'much': 4656, u'pizza': 4614, u'us': 4553, u'think': 4494, u'meal': 4446, u'bar': 4403, u'wait': 4382, u'know': 4376, u'drink': 4343, u'price': 4337, u'flavor': 4256, u'thing': 4253, u'best': 4222, u'sandwich': 4191, u'dish': 4141, u'say': 3952, u'side': 3931, u'better': 3931, u'serv': 3752, u'star': 3743, u'fresh': 3733, u'night': 3718, u'locat': 3717, u'dinner': 3700, u'could': 3673, u'take': 3670, u'delici': 3602, u'went': 3574, u'two': 3562, u'first': 3559, u'special': 3541, u'never': 3541, u'lot': 3505, u'enjoy': 3476, u'way': 3467, u'peopl': 3447, u'day': 3443, u'friend': 3440, u'sinc': 3423, u'right': 3400, u'bread': 3338, u'still': 3329, u'ask': 3299, u'hot': 3264, u'made': 3207, u'bit': 3192, u'friendli': 3134, u'soup': 3091, u'rice': 3039, u'year': 3016, u'sure': 2991, u'seem': 2983, u'bad': 2941, u'top': 2936, u'meat': 2919, u'though': 2912, u'give': 2902, u'someth': 2873, u'server': 2870, u'definit': 2859, u'review': 2844, u'burger': 2838, u'cook': 2813, u'see': 2789, u'around': 2780, u'use': 2774, u'roll': 2765, u'beef': 2647, u'said': 2645, u'small': 2643, u'last': 2635, u'favorit': 2629, u'visit': 2606, u'area': 2580, u'big': 2579, u'tasti': 2571, u'seat': 2558, u'need': 2546, u'mani': 2518, u'feel': 2510, u'sweet': 2494, u'breakfast': 2487, u'work': 2481, u'hour': 2455, u'egg': 2444, u'everyth': 2442, u'experi': 2435, u'next': 2427, u'plate': 2385, u'ok': 2377, u'minut': 2338, u'happi': 2307, u'potato': 2307, u'usual': 2306, u'busi': 2290, u'find': 2284, u'anoth': 2274, u'noth': 2264, u'long': 2256, u'ever': 2251, u'enough': 2242, u'taco': 2238, u'dine': 2222, u'staff': 2220, u'decid': 2191, u'took': 2183, u'steak': 2177, u'home': 2171, u'shrimp': 2167, u'beer': 2145, u'stop': 2116, u'larg': 2105, u'start': 2097, u'everi': 2095, u'walk': 2075, u'expect': 2048, u'new': 2044, u'bean': 2029, u'differ': 2024, u'grill': 2002, u'chip': 1996, u'sushi': 1994, u'check': 1985, u'reason': 1973, u'perfect': 1970, u'spici': 1962, u'item': 1960, u'wine': 1957, u'kind': 1951, u'offer': 1939, u'thought': 1934, u'actual': 1932, u'portion': 1928, u'probabl': 1917, u'mayb': 1914, u'spot': 1894, u'huge': 1892, u'old': 1884, u'fish': 1881, u'decent': 1878, u'end': 1866, u'cream': 1865, u'salsa': 1845, u'onion': 1844, u'sit': 1830, u'full': 1829, u'amaz': 1807, u'fast': 1807, u'howev': 1799, u'dessert': 1797, u'select': 1784, u'quit': 1777, u'qualiti': 1774, u'hous': 1744, u'half': 1726, u'option': 1724, u'insid': 1718, u'park': 1698, u'awesom': 1689, u'green': 1674, u'call': 1667, u'mexican': 1663, u'close': 1631, u'choic': 1617, u'pork': 1611, u'open': 1593, u'red': 1589, u'slice': 1566, u'super': 1564, u'appet': 1560, u'away': 1558, u'recommend': 1551, u'waitress': 1545, u'disappoint': 1544, u'fan': 1533, u'tomato': 1523, u'excel': 1516, u'put': 1511, u'chines': 1506, u'let': 1498, u'anyth': 1491, u'fill': 1490, u'coupl': 1488, u'entre': 1487, u'overal': 1478, u'pasta': 1478, u'ice': 1458, u'famili': 1458, u'almost': 1451, u'thai': 1451, u'worth': 1450, u'clean': 1446, u'els': 1433, u'style': 1427, u'town': 1421, u'size': 1409, u'italian': 1395, u'arriv': 1394, u'veggi': 1391, u'live': 1390, u'drive': 1387, u'quick': 1381, u'week': 1368, u'atmospher': 1357, u'oh': 1352, u'bite': 1348, u'room': 1339, u'hard': 1333, u'left': 1329, u'person': 1328, u'found': 1327, u'cours': 1323, u'burrito': 1320, u'three': 1306, u'yelp': 1304, u'least': 1298, u'wing': 1293, u'pick': 1289, u'outsid': 1289, u'part': 1287, u'noodl': 1277, u'final': 1277, u'husband': 1259, u'light': 1257, u'wonder': 1256, u'rememb': 1255, u'return': 1254, u'tea': 1249, u'either': 1248, u'help': 1246, u'water': 1244, u'bring': 1240, u'kid': 1238, u'piec': 1232, u'gener': 1232, u'includ': 1231, u'may': 1226, u'bacon': 1223, u'pepper': 1221, u'ye': 1218, u'mean': 1217, u'cheap': 1216, u'without': 1215, u'line': 1198, u'free': 1198, u'impress': 1191, u'bowl': 1190, u'tell': 1183, u'white': 1175, u'high': 1173, u'told': 1172, u'sever': 1167, u'keep': 1164, u'custom': 1161, u'chang': 1156, u'everyon': 1156, u'today': 1150, u'point': 1138, u'crispi': 1137, u'far': 1136, u'especi': 1133, u'brought': 1128, u'whole': 1125, u'dress': 1118, u'might': 1117, u'deal': 1110, u'name': 1110, u'regular': 1101, u'share': 1100, u'guy': 1096, u'less': 1089, u'late': 1085, u'mix': 1084, u'cool': 1084, u'dip': 1084, u'coffe': 1083, u'waiter': 1076, u'hand': 1074, u'extra': 1074, u'ate': 1073, u'guess': 1072, u'must': 1068, u'okay': 1067, u'manag': 1067, u'garlic': 1067, u'decor': 1065, u'surpris': 1057, u'averag': 1055, u'instead': 1053, u'real': 1050, u'group': 1050, u'pay': 1049, u'bbq': 1043, u'crowd': 1041, u'fine': 1041, u'fun': 1039, u'parti': 1037, u'cold': 1033, u'dri': 1033, u'leav': 1033, u'season': 1030, u'cake': 1024, u'chain': 1022, u'although': 1019, u'chocol': 1013, u'glass': 1013, u'door': 1006, u'plenti': 1006, u'run': 1003, u'warm': 1002, u'done': 999, u'care': 989, u'attent': 987, u'sometim': 983, u'local': 982, u'chili': 976, u'wrong': 975, u'pie': 973, u'wall': 969, u'pack': 964, u'amount': 964, u'mushroom': 960, u'type': 959, u'dog': 957, u'game': 955, u'sat': 955, u'pm': 955, u'past': 954, u'phoenix': 954, u'crust': 953, u'shop': 950, u'consist': 947, u'stuff': 946, u'watch': 946, u'gave': 945, u'second': 945, u'hit': 945, u'total': 943, u'finish': 941, u'head': 938, u'yummi': 936, u'sunday': 934, u'patio': 934, u'perfectli': 932, u'miss': 931, u'wife': 927, u'plu': 927, u'pancak': 926, u'street': 924, u'french': 924, u'notic': 923, u'someon': 922, u'quickli': 922, u'yet': 919, u'list': 919, u'ago': 917, u'fact': 916, u'sausag': 910, u'thank': 905, u'rib': 897, u'crave': 896, u'buffet': 894, u'cut': 892, u'crab': 889, u'toast': 884, u'show': 880, u'prepar': 863, u'w': 861, u'anyway': 857, u'recent': 854, u'talk': 854, u'tender': 849, u'wish': 849, u'butter': 846, u'owner': 838, u'corn': 830, u'often': 828, u'tortilla': 822, u'hope': 821, u'add': 820, u'counter': 814, u'front': 811, u'mind': 810, u'ad': 807, u'turn': 805, u'kitchen': 804, u'saturday': 804, u'later': 803, u'slow': 802, u'fantast': 801, u'typic': 796, u'lettuc': 795, u'bake': 792, u'lack': 791, u'authent': 787, u'wrap': 786, u'move': 784, u'pho': 783, u'soon': 780, u'felt': 779, u'four': 769, u'tip': 769, u'bagel': 764, u'happen': 762, u'strip': 756, u'rather': 751, u'pleas': 750, u'basic': 749, u'date': 749, u'seafood': 749, u'grab': 745, u'eaten': 743, u'set': 741, u'mom': 741, u'lobster': 739, u'hungri': 739, u'friday': 737, u'mention': 735, u'saw': 735, u'suggest': 734, u'stuf': 730, u'greasi': 723, u'normal': 723, u'joint': 722, u'veget': 722, u'bill': 721, u'soft': 719, u'diner': 717, u'curri': 717, u'varieti': 716, u'abl': 715, u'refil': 714, u'read': 714, u'spice': 712, u'bland': 712, u'thin': 712, u'roast': 711, u'problem': 710, u'standard': 710, u'combo': 708, u'bottl': 706, u'sign': 703, u'believ': 703, u'complet': 702, u'along': 702, u'choos': 699, u'alreadi': 698, u'enchilada': 697, u'solid': 696, u'base': 695, u'turkey': 694, u'vega': 694, u'absolut': 691, u'readi': 691, u'salmon': 691, u'brown': 691, u'sour': 689, u'birthday': 687, u'rate': 686, u'ingredi': 682, u'prefer': 682, u'play': 679, u'la': 678, u'stick': 677, u'except': 674, u'chef': 674, u'morn': 671, u'sub': 670, u'salt': 667, u'booth': 665, u'cover': 664, u'main': 663, u'etc': 661, u'cup': 660, u'store': 654, u'simpl': 651, u'medium': 647, u'note': 644, u'consid': 639, u'near': 639, u'valley': 637, u'thick': 636, u'comfort': 635, u'interest': 632, u'reserv': 630, u'girl': 629, u'serious': 628, u'brunch': 626, u'bartend': 625, u'oliv': 624, u'ton': 621, u'rest': 619, u'deli': 617, u'stand': 616, u'entir': 616, u'stay': 614, u'weekend': 611, u'card': 610, u'rare': 609, u'given': 607, u'money': 601, u'mall': 601, u'earli': 601, u'sound': 600, u'black': 599, u'gone': 598, u'month': 595, u'easi': 595, u'heard': 594, u'sort': 593, u'tuna': 592, u'mac': 591, u'man': 591, u'other': 586, u'greet': 585, u'american': 583, u'knew': 583, u'neighborhood': 583, u'short': 582, u'scottsdal': 579, u'glad': 579, u'smell': 579, u'mouth': 577, u'spinach': 577, u'music': 573, u'melt': 573, u'yum': 572, u'trip': 572, u'charg': 571, u'coupon': 569, u'salti': 566, u'wow': 565, u'ambianc': 561, u'meet': 558, u'across': 558, u'mine': 558, u'empti': 555, u'kept': 553, u'box': 550, u'pass': 549, u'oil': 549, u'school': 548, u'crisp': 547, u'ladi': 547, u'creami': 547, u'togeth': 542, u'satisfi': 541, u'gravi': 536, u'chop': 536, u'unfortun': 532, u'treat': 529, u'five': 528, u'n': 526, u'heat': 525, u'vegetarian': 525, u'pull': 524, u'waffl': 524, u'bun': 523, u'eye': 523, u'remind': 523, u'extrem': 521, u'behind': 520, u'anyon': 520, u'spring': 519, u'gyro': 518, u'lo': 517, u'seen': 517, u'combin': 516, u'tofu': 515, u'build': 513, u'job': 512, u'expens': 510, u'deliv': 510, u'case': 509, u'margarita': 509, u'rush': 508, u'kinda': 507, u'car': 507, u'buck': 507, u'split': 506, u'smoke': 506, u'low': 506, u'chose': 505, u'cafe': 503, u'homemad': 503, u'twice': 503, u'photo': 503, u'orang': 500, u'mash': 499, u'immedi': 498, u'honey': 497, u'asian': 496, u'dark': 494, u'figur': 493, u'compar': 492, u'excit': 489, u'hate': 486, u'yeah': 485, u'blue': 485, u'receiv': 483, u'rock': 482, u'number': 481, u'weird': 481, u'event': 481, u'exactli': 481, u'soda': 480, u'tradit': 480, u'idea': 475, u'charlott': 474, u'issu': 470, u'write': 467, u'drop': 466, u'space': 465, u'slightli': 465, u'establish': 464, u'textur': 463, u'incred': 462, u'whatev': 462, u'mostli': 461, u'ham': 460, u'origin': 459, u'addit': 457, u'mood': 456, u'hear': 456, u'biscuit': 456, u'buy': 455, u'life': 454, u'tv': 452, u'kick': 452, u'sport': 450, u'pita': 450, u'world': 447, u'omelet': 447, u'middl': 446, u'cooki': 445, u'duck': 445, u'chile': 445, u'plan': 445, u'crazi': 444, u'understand': 444, u'corner': 444, u'club': 444, u'realiz': 443, u'com': 441, u'uniqu': 441, u'hostess': 440, u'opt': 437, u'liter': 437, u'employe': 437, u'hubbi': 434, u'fairli': 433, u'matter': 433, u'b': 433, u'filet': 433, u'avail': 432, u'citi': 431, u'th': 431, u'pickl': 431, u'healthi': 430, u'complaint': 430, u'juici': 429, u'goe': 429, u'appl': 428, u'possibl': 428, u'word': 427, u'ranch': 426, u'boyfriend': 425, u'appreci': 424, u'fruit': 423, u'hang': 422, u'deliveri': 422, u'honestli': 422, u'casual': 422, u'taken': 421, u'valu': 420, u'due': 420, u'deep': 420, u'request': 420, u'beauti': 420, u'save': 419, u'window': 419, u'perhap': 418, u'updat': 417, u'upon': 417, u'scallop': 415, u'face': 415, u'provid': 414, u'cost': 413, u'dirti': 412, u'chanc': 412, u'non': 411, u'tini': 411, u'ring': 410, u'certainli': 410, u'forget': 409, u'crunchi': 408, u'app': 406, u'center': 406, u'lemon': 406, u'market': 406, u'fanci': 406, u'limit': 403, u'fix': 402, u'spend': 402, u'view': 402, u'bell': 399, u'pot': 398, u'meatbal': 398, u'easili': 397, u'avocado': 396, u'suppos': 395, u'classic': 395, u'folk': 394, u'pad': 393, u'broth': 392, u'pleasant': 392, u'highli': 392, u'present': 391, u'downtown': 391, u'tonight': 390, u'chicago': 390, u'smile': 389, u'level': 386, u'korean': 385, u'despit': 384, u'pictur': 383, u'fair': 383, u'heavi': 381, u'calamari': 381, u'c': 381, u'sampl': 379, u'e': 376, u'unless': 376, u'sorri': 375, u'step': 375, u'touch': 374, u'frozen': 374, u'loud': 373, u'forward': 373, u'cocktail': 373, u'thru': 371, u'pop': 371, u'opinion': 371, u'hash': 369, u'hole': 369, u'bag': 369, u'damn': 368, u'terribl': 367, u'longer': 367, u'min': 366, u'dive': 366, u'broccoli': 365, u'rich': 364, u'cheesecak': 364, u'doubl': 364, u'speak': 364, u'pricey': 363, u'lol': 363, u'cash': 362, u'plain': 362, u'daughter': 361, u'patron': 361, u'appar': 361, u'load': 360, u'south': 359, u'dollar': 359, u'moist': 359, u'frequent': 358, u'true': 358, u'afternoon': 356, u'cute': 356, u'strawberri': 356, u'within': 356, u'offic': 355, u'bf': 354, u'occas': 354, u'book': 354, u'floor': 353, u'agre': 353, u'somewher': 352, u'conveni': 352, u'mediocr': 351, u'welcom': 350, u'teriyaki': 350, u'color': 350, u'http': 349, u'version': 348, u'ball': 348, u'imagin': 347, u'shot': 347, u'bottom': 346, u'simpli': 346, u'appear': 345, u'jalapeno': 345, u'beat': 345, u'gotten': 344, u'hold': 343, u'suck': 343, u'similar': 342, u'carn': 342, u'carrot': 341, u'complain': 340, u'mustard': 340, u'sad': 340, u'greek': 339, u'prime': 339, u'round': 338, u'platter': 338, u'alon': 338, u'www': 337, u'truli': 337, u'p': 336, u'contain': 336, u'japanes': 335, u'leftov': 335, u'de': 334, u'paid': 334, u'bare': 334, u'babi': 333, u'ahead': 333, u'worker': 333, u'nacho': 332, u'banana': 332, u'basil': 330, u'follow': 330, u'countri': 330, u'steam': 330, u'continu': 329, u'worst': 327, u'road': 327, u'outstand': 327, u'enter': 326, u'overli': 326, u'singl': 324, u'chipotl': 323, u'fat': 323, u'yelper': 322, u'asada': 322, u'anywher': 322, u'bathroom': 322, u'boy': 321, u'question': 321, u'awhil': 321, u'smaller': 319, u'explain': 319, u'guacamol': 319, u'heart': 318, u'mayo': 318, u'eggplant': 317, u'station': 316, u'meh': 314, u'odd': 314, u'peanut': 313, u'st': 313, u'batter': 312, u'chewi': 310, u'sugar': 309, u'monday': 308, u'outdoor': 307, u'state': 307, u'biz': 307, u'tuesday': 306, u'chair': 305, u'strong': 305, u'forgot': 304, u'mess': 304, u'cozi': 304, u'poor': 302, u'becom': 302, u'met': 302, u'na': 302, u'shred': 302, u'pepperoni': 302, u'cuisin': 301, u'break': 301, u'desert': 301, u'mozzarella': 299, u'particular': 299, u'interior': 298, u'popular': 297, u'bunch': 297, u'shell': 297, u'forev': 296, u'hell': 295, u'tough': 295, u'delish': 294, u'pittsburgh': 291, u'mini': 291, u'garden': 290, u'otherwis': 290, u'coke': 289, u'admit': 289, u'fare': 288, u'fall': 288, u'somewhat': 288, u'phone': 288, u'buffalo': 287, u'lamb': 287, u'soggi': 287, u'futur': 287, u'effici': 286, u'coconut': 286, u'aw': 286, u'summer': 286, u'daili': 286, u'horribl': 285, u'gross': 284, u'delight': 284, u'known': 284, u'yesterday': 284, u'east': 283, u'young': 282, u'section': 281, u'co': 281, u'fajita': 281, u'hey': 280, u'convers': 279, u'celebr': 279, u'chunk': 279, u'hummu': 277, u'arizona': 277, u'fire': 276, u'avoid': 276, u'caesar': 276, u'ravioli': 276, u'mcdonald': 275, u'none': 274, u'bone': 274, u'die': 273, u'surprisingli': 272, u'throw': 272, u'nearli': 272, u'spaghetti': 271, u'describ': 271, u'paper': 270, u'begin': 270, u'heaven': 269, u'nearbi': 267, u'flat': 267, u'skip': 267, u'per': 267, u'juic': 267, u'credit': 267, u'indian': 267, u'compani': 266, u'complimentari': 266, u'specialti': 266, u'cheesi': 266, u'parmesan': 265, u'clear': 265, u'cucumb': 264, u'relax': 263, u'pub': 263, u'son': 263, u'gem': 261, u'girlfriend': 261, u'throughout': 260, u'slaw': 260, u'own': 259, u'jack': 258, u'vietnames': 258, u'quesadilla': 258, u'mild': 257, u'dim': 257, u'lime': 257, u'pre': 256, u'allow': 255, u'brew': 254, u'overcook': 254, u'breast': 254, u'bother': 254, u'tap': 254, u'sum': 253, u'funni': 253, u'ten': 253, u'york': 253, u'chow': 252, u'afford': 252, u'tamal': 252, u'post': 252, u'alright': 252, u'accommod': 251, u'oz': 251, u'shake': 251, u'basket': 250, u'muffin': 250, u'featur': 249, u'accompani': 249, u'cowork': 248, u'beverag': 248, u'tend': 247, u'belli': 247, u'syrup': 247, u'fit': 246, u'mother': 246, u'separ': 245, u'notch': 245, u'cheddar': 243, u'pud': 243, u'diet': 243, u'martini': 243, u'overpr': 243, u'rave': 243, u'patti': 242, u'rude': 242, u'anymor': 242, u'north': 241, u'fabul': 241, u'wonton': 241, u'import': 240, u'sister': 240, u'caus': 240, u'movi': 240, u'takeout': 240, u'cinnamon': 239, u'older': 239, u'pile': 239, u'dad': 239, u'elsewher': 238, u'gift': 238, u'rang': 237, u'calori': 236, u'skin': 236, u'moment': 236, u'ta': 236, u'improv': 235, u'inform': 235, u'obvious': 235, u'stori': 234, u'ground': 234, u'promptli': 234, u'mistak': 233, u'larger': 233, u'clearli': 233, u'central': 233, u'fork': 233, u'particularli': 233, u'tempura': 233, u'annoy': 233, u'entertain': 232, u'hidden': 232, u'doubt': 232, u'straight': 232, u'pour': 231, u'parent': 231, u'difficult': 230, u'cabbag': 230, u'comment': 230, u'mein': 230, u'sun': 229, u'slider': 229, u'greas': 229, u'clam': 228, u'guest': 228, u'bore': 227, u'ridicul': 227, u'handl': 227, u'age': 227, u'confus': 226, u'wors': 226, u'mile': 226, u'memori': 226, u'english': 226, u'posit': 225, u'sesam': 225, u'sashimi': 225, u'vanilla': 225, u'onlin': 224, u'previou': 224, u'learn': 223, u'stomach': 223, u'product': 223, u'vibe': 223, u'california': 223, u'bruschetta': 222, u'cherri': 222, u'al': 222, u'brisket': 222, u'depend': 222, u'oven': 222, u'dozen': 221, u'finger': 221, u'sens': 221, u'ny': 221, u'wood': 221, u'apart': 220, u'board': 220, u'spectacular': 220, u'denni': 219, u'strang': 219, u'artichok': 219, u'closer': 219, u'alcohol': 218, u'giant': 218, u'refresh': 218, u'island': 217, u'fluffi': 217, u'balanc': 217, u'apolog': 217, u'charm': 216, u'marinara': 216, u'host': 216, u'count': 215, u'pineappl': 215, u'pecan': 215, u'macaroni': 215, u'west': 214, u'burn': 214, u'god': 214, u'warn': 214, u'express': 213, u'groupon': 213, u'major': 213, u'pesto': 212, u'savori': 212, u'invit': 212, u'variou': 211, u'bud': 211, u'everywher': 210, u'higher': 210, u'subway': 210, u'spent': 210, u'whip': 210, u'golden': 209, u'worri': 209, u'par': 209, u'pleasantli': 208, u'flour': 208, u'darn': 207, u'pan': 207, u'regist': 207, u'extens': 206, u'weather': 206, u'az': 206, u'deserv': 206, u'theme': 205, u'carri': 205, u'bigger': 205, u'peak': 205, u'oyster': 205, u'hawaiian': 204, u'snack': 204, u'pair': 204, u'goat': 203, u'polit': 203, u'sell': 202, u'toward': 202, u'thursday': 202, u'hamburg': 202, u'answer': 202, u'raw': 202, u'factor': 201, u'band': 201, u'inexpens': 201, u'pastrami': 201, u'lover': 201, u'assum': 200, u'steakhous': 200, u'overwhelm': 199, u'tart': 199, u'fountain': 199, u'cashier': 199, u'mr': 198, u'gon': 198, u'whenev': 198, u'six': 198, u'chill': 197, u'colleg': 197, u'freshli': 197, u'ass': 196, u'rel': 196, u'pool': 195, u'sake': 195, u'luckili': 195, u'layer': 195, u'win': 195, u'quiet': 194, u'accept': 194, u'pound': 194, u'groceri': 194, u'asparagu': 194, u'lost': 193, u'stuck': 193, u'omelett': 193, u'milk': 193, u'cater': 192, u'machin': 192, u'woman': 192, u'wear': 192, u'vinegar': 192, u'ayc': 192, u'ginger': 191, u'pastri': 191, u'besid': 191, u'yellow': 191, u'nut': 191, u'dont': 191, u'beyond': 191, u'process': 190, u'g': 190, u'napkin': 190, u'wheat': 189, u'correct': 189, u'soy': 188, u'bomb': 188, u'wild': 188, u'sick': 188, u'john': 187, u'certain': 187, u'sprout': 187, u'fatti': 186, u'experienc': 186, u'crap': 186}
    #smaller
    #required_features = {u'good': 20624, u'food': 17687, u'place': 17212, u'like': 14958, u'order': 14055, u'time': 11385, u'get': 11230, u'go': 11066, u'one': 10594, u'great': 9093, u'realli': 8709, u'servic': 8361, u'tri': 7686, u'chicken': 7556, u'back': 7295, u'would': 7039, u'restaur': 6513, u'come': 6247, u'also': 6075, u'love': 6066, u'menu': 6044, u'sauc': 5797, u'lunch': 5745, u'salad': 5741, u'eat': 5732, u'fri': 5714, u'littl': 5688, u'got': 5636, u'even': 5635, u'nice': 5556, u'chees': 5481, u'well': 5477, u'pretti': 5356, u'want': 5338, u'alway': 5194, u'look': 5054, u'make': 4997, u'tabl': 4820, u'came': 4791, u'tast': 4784, u'much': 4656, u'pizza': 4614, u'us': 4553, u'think': 4494, u'meal': 4446, u'bar': 4403, u'wait': 4382, u'know': 4376, u'drink': 4343, u'price': 4337, u'flavor': 4256, u'thing': 4253, u'best': 4222, u'sandwich': 4191, u'dish': 4141, u'say': 3952, u'side': 3931, u'better': 3931, u'serv': 3752, u'star': 3743, u'fresh': 3733, u'night': 3718, u'locat': 3717, u'dinner': 3700, u'could': 3673, u'take': 3670, u'delici': 3602, u'went': 3574, u'two': 3562, u'first': 3559, u'special': 3541, u'never': 3541, u'lot': 3505, u'enjoy': 3476, u'way': 3467, u'peopl': 3447, u'day': 3443, u'friend': 3440, u'sinc': 3423, u'right': 3400, u'bread': 3338, u'still': 3329, u'ask': 3299, u'hot': 3264, u'made': 3207, u'bit': 3192, u'friendli': 3134, u'soup': 3091, u'rice': 3039, u'year': 3016, u'sure': 2991, u'seem': 2983, u'bad': 2941, u'top': 2936, u'meat': 2919, u'though': 2912, u'give': 2902, u'someth': 2873, u'server': 2870, u'definit': 2859, u'review': 2844, u'burger': 2838, u'cook': 2813, u'see': 2789, u'around': 2780, u'use': 2774, u'roll': 2765, u'beef': 2647, u'said': 2645, u'small': 2643, u'last': 2635, u'favorit': 2629, u'visit': 2606, u'area': 2580, u'big': 2579, u'tasti': 2571, u'seat': 2558, u'need': 2546, u'mani': 2518, u'feel': 2510, u'sweet': 2494, u'breakfast': 2487, u'work': 2481, u'hour': 2455, u'egg': 2444, u'everyth': 2442, u'experi': 2435, u'next': 2427, u'plate': 2385, u'ok': 2377, u'minut': 2338, u'happi': 2307, u'potato': 2307, u'usual': 2306, u'busi': 2290, u'find': 2284, u'anoth': 2274, u'noth': 2264, u'long': 2256, u'ever': 2251, u'enough': 2242, u'taco': 2238, u'dine': 2222, u'staff': 2220, u'decid': 2191, u'took': 2183, u'steak': 2177, u'home': 2171, u'shrimp': 2167, u'beer': 2145, u'stop': 2116, u'larg': 2105, u'start': 2097, u'everi': 2095, u'walk': 2075, u'expect': 2048, u'new': 2044, u'bean': 2029, u'differ': 2024, u'grill': 2002, u'chip': 1996, u'sushi': 1994, u'check': 1985, u'reason': 1973, u'perfect': 1970, u'spici': 1962, u'item': 1960, u'wine': 1957, u'kind': 1951, u'offer': 1939, u'thought': 1934, u'actual': 1932, u'portion': 1928, u'probabl': 1917, u'mayb': 1914, u'spot': 1894, u'huge': 1892, u'old': 1884, u'fish': 1881, u'decent': 1878, u'end': 1866, u'cream': 1865, u'salsa': 1845, u'onion': 1844, u'sit': 1830, u'full': 1829, u'amaz': 1807, u'fast': 1807, u'howev': 1799, u'dessert': 1797, u'select': 1784, u'quit': 1777, u'qualiti': 1774, u'hous': 1744, u'half': 1726, u'option': 1724, u'insid': 1718, u'park': 1698, u'awesom': 1689, u'green': 1674, u'call': 1667, u'mexican': 1663, u'close': 1631, u'choic': 1617, u'pork': 1611, u'open': 1593, u'red': 1589, u'slice': 1566, u'super': 1564, u'appet': 1560, u'away': 1558, u'recommend': 1551, u'waitress': 1545, u'disappoint': 1544, u'fan': 1533, u'tomato': 1523, u'excel': 1516, u'put': 1511, u'chines': 1506, u'let': 1498, u'anyth': 1491, u'fill': 1490, u'coupl': 1488, u'entre': 1487, u'overal': 1478, u'pasta': 1478, u'ice': 1458, u'famili': 1458, u'almost': 1451, u'thai': 1451, u'worth': 1450, u'clean': 1446, u'els': 1433, u'style': 1427, u'town': 1421, u'size': 1409, u'italian': 1395, u'arriv': 1394, u'veggi': 1391, u'live': 1390, u'drive': 1387, u'quick': 1381, u'week': 1368, u'atmospher': 1357, u'oh': 1352, u'bite': 1348, u'room': 1339, u'hard': 1333, u'left': 1329, u'person': 1328, u'found': 1327, u'cours': 1323, u'burrito': 1320, u'three': 1306, u'yelp': 1304, u'least': 1298, u'wing': 1293, u'pick': 1289, u'outsid': 1289, u'part': 1287, u'noodl': 1277, u'final': 1277, u'husband': 1259, u'light': 1257, u'wonder': 1256, u'rememb': 1255, u'return': 1254, u'tea': 1249, u'either': 1248, u'help': 1246, u'water': 1244, u'bring': 1240, u'kid': 1238, u'piec': 1232, u'gener': 1232, u'includ': 1231, u'may': 1226, u'bacon': 1223, u'pepper': 1221, u'ye': 1218, u'mean': 1217, u'cheap': 1216, u'without': 1215, u'line': 1198, u'free': 1198, u'impress': 1191, u'bowl': 1190, u'tell': 1183, u'white': 1175, u'high': 1173, u'told': 1172, u'sever': 1167, u'keep': 1164, u'custom': 1161, u'chang': 1156, u'everyon': 1156, u'today': 1150, u'point': 1138, u'crispi': 1137, u'far': 1136, u'especi': 1133, u'brought': 1128, u'whole': 1125, u'dress': 1118, u'might': 1117, u'deal': 1110, u'name': 1110, u'regular': 1101, u'share': 1100, u'guy': 1096, u'less': 1089, u'late': 1085, u'mix': 1084, u'cool': 1084, u'dip': 1084, u'coffe': 1083, u'waiter': 1076, u'hand': 1074, u'extra': 1074, u'ate': 1073, u'guess': 1072, u'must': 1068, u'okay': 1067, u'manag': 1067, u'garlic': 1067, u'decor': 1065, u'surpris': 1057, u'averag': 1055, u'instead': 1053, u'real': 1050, u'group': 1050, u'pay': 1049, u'bbq': 1043, u'crowd': 1041, u'fine': 1041, u'fun': 1039, u'parti': 1037, u'cold': 1033, u'dri': 1033, u'leav': 1033, u'season': 1030, u'cake': 1024, u'chain': 1022, u'although': 1019, u'chocol': 1013, u'glass': 1013, u'door': 1006, u'plenti': 1006, u'run': 1003, u'warm': 1002, u'done': 999, u'care': 989, u'attent': 987, u'sometim': 983, u'local': 982, u'chili': 976, u'wrong': 975, u'pie': 973, u'wall': 969, u'pack': 964, u'amount': 964, u'mushroom': 960, u'type': 959, u'dog': 957, u'game': 955, u'sat': 955, u'pm': 955, u'past': 954, u'phoenix': 954, u'crust': 953, u'shop': 950, u'consist': 947, u'stuff': 946, u'watch': 946, u'gave': 945, u'second': 945, u'hit': 945, u'total': 943, u'finish': 941, u'head': 938, u'yummi': 936, u'sunday': 934, u'patio': 934, u'perfectli': 932, u'miss': 931, u'wife': 927, u'plu': 927, u'pancak': 926, u'street': 924, u'french': 924, u'notic': 923, u'someon': 922, u'quickli': 922, u'yet': 919, u'list': 919, u'ago': 917, u'fact': 916, u'sausag': 910, u'thank': 905, u'rib': 897, u'crave': 896, u'buffet': 894, u'cut': 892, u'crab': 889, u'toast': 884, u'show': 880, u'prepar': 863, u'w': 861, u'anyway': 857, u'recent': 854, u'talk': 854, u'tender': 849, u'wish': 849, u'butter': 846, u'owner': 838, u'corn': 830, u'often': 828, u'tortilla': 822, u'hope': 821, u'add': 820, u'counter': 814, u'front': 811, u'mind': 810, u'ad': 807, u'turn': 805, u'kitchen': 804, u'saturday': 804, u'later': 803, u'slow': 802, u'fantast': 801, u'typic': 796, u'lettuc': 795, u'bake': 792, u'lack': 791, u'authent': 787, u'wrap': 786, u'move': 784, u'pho': 783, u'soon': 780, u'felt': 779, u'four': 769, u'tip': 769, u'bagel': 764, u'happen': 762, u'strip': 756, u'rather': 751, u'pleas': 750, u'basic': 749, u'date': 749, u'seafood': 749, u'grab': 745, u'eaten': 743, u'set': 741, u'mom': 741, u'lobster': 739, u'hungri': 739, u'friday': 737, u'mention': 735, u'saw': 735, u'suggest': 734, u'stuf': 730, u'greasi': 723, u'normal': 723, u'joint': 722, u'veget': 722, u'bill': 721, u'soft': 719, u'diner': 717, u'curri': 717, u'varieti': 716, u'abl': 715, u'refil': 714, u'read': 714, u'spice': 712, u'bland': 712, u'thin': 712, u'roast': 711, u'problem': 710, u'standard': 710, u'combo': 708, u'bottl': 706, u'sign': 703, u'believ': 703, u'complet': 702, u'along': 702, u'choos': 699, u'alreadi': 698, u'enchilada': 697, u'solid': 696, u'base': 695, u'turkey': 694, u'vega': 694, u'absolut': 691, u'readi': 691, u'salmon': 691, u'brown': 691, u'sour': 689, u'birthday': 687, u'rate': 686, u'ingredi': 682, u'prefer': 682, u'play': 679, u'la': 678, u'stick': 677, u'except': 674, u'chef': 674, u'morn': 671, u'sub': 670, u'salt': 667, u'booth': 665, u'cover': 664, u'main': 663, u'etc': 661, u'cup': 660, u'store': 654, u'simpl': 651, u'medium': 647, u'note': 644, u'consid': 639, u'near': 639, u'valley': 637, u'thick': 636, u'comfort': 635, u'interest': 632, u'reserv': 630, u'girl': 629, u'serious': 628, u'brunch': 626, u'bartend': 625, u'oliv': 624, u'ton': 621, u'rest': 619, u'deli': 617, u'stand': 616, u'entir': 616, u'stay': 614, u'weekend': 611, u'card': 610, u'rare': 609, u'given': 607, u'money': 601, u'mall': 601, u'earli': 601, u'sound': 600, u'black': 599, u'gone': 598, u'month': 595, u'easi': 595, u'heard': 594, u'sort': 593, u'tuna': 592, u'mac': 591, u'man': 591, u'other': 586, u'greet': 585, u'american': 583, u'knew': 583, u'neighborhood': 583, u'short': 582, u'scottsdal': 579, u'glad': 579, u'smell': 579, u'mouth': 577, u'spinach': 577, u'music': 573, u'melt': 573, u'yum': 572, u'trip': 572, u'charg': 571, u'coupon': 569, u'salti': 566, u'wow': 565, u'ambianc': 561, u'meet': 558, u'across': 558, u'mine': 558, u'empti': 555, u'kept': 553, u'box': 550, u'pass': 549, u'oil': 549, u'school': 548, u'crisp': 547, u'ladi': 547, u'creami': 547, u'togeth': 542, u'satisfi': 541, u'gravi': 536, u'chop': 536, u'unfortun': 532, u'treat': 529, u'five': 528, u'n': 526, u'heat': 525, u'vegetarian': 525, u'pull': 524, u'waffl': 524, u'bun': 523, u'eye': 523, u'remind': 523, u'extrem': 521, u'behind': 520, u'anyon': 520, u'spring': 519, u'gyro': 518, u'lo': 517, u'seen': 517, u'combin': 516, u'tofu': 515, u'build': 513, u'job': 512, u'expens': 510, u'deliv': 510, u'case': 509, u'margarita': 509, u'rush': 508, u'kinda': 507, u'car': 507, u'buck': 507, u'split': 506, u'smoke': 506, u'low': 506, u'chose': 505, u'cafe': 503, u'homemad': 503, u'twice': 503, u'photo': 503, u'orang': 500, u'mash': 499, u'immedi': 498, u'honey': 497, u'asian': 496, u'dark': 494, u'figur': 493, u'compar': 492, u'excit': 489, u'hate': 486, u'yeah': 485, u'blue': 485, u'receiv': 483, u'rock': 482, u'number': 481, u'weird': 481, u'event': 481, u'exactli': 481, u'soda': 480, u'tradit': 480, u'idea': 475, u'charlott': 474, u'issu': 470, u'write': 467, u'drop': 466, u'space': 465, u'slightli': 465, u'establish': 464, u'textur': 463, u'incred': 462, u'whatev': 462, u'mostli': 461, u'ham': 460, u'origin': 459, u'addit': 457, u'mood': 456, u'hear': 456, u'biscuit': 456, u'buy': 455, u'life': 454, u'tv': 452, u'kick': 452, u'sport': 450, u'pita': 450, u'world': 447, u'omelet': 447, u'middl': 446, u'cooki': 445, u'duck': 445, u'chile': 445, u'plan': 445, u'crazi': 444, u'understand': 444, u'corner': 444, u'club': 444, u'realiz': 443, u'com': 441, u'uniqu': 441, u'hostess': 440, u'opt': 437, u'liter': 437, u'employe': 437, u'hubbi': 434, u'fairli': 433, u'matter': 433, u'b': 433, u'filet': 433, u'avail': 432, u'citi': 431, u'th': 431, u'pickl': 431, u'healthi': 430, u'complaint': 430, u'juici': 429, u'goe': 429, u'appl': 428, u'possibl': 428, u'word': 427, u'ranch': 426, u'boyfriend': 425, u'appreci': 424, u'fruit': 423, u'hang': 422, u'deliveri': 422, u'honestli': 422, u'casual': 422, u'taken': 421, u'valu': 420, u'due': 420, u'deep': 420, u'request': 420, u'beauti': 420, u'save': 419, u'window': 419, u'perhap': 418, u'updat': 417, u'upon': 417, u'scallop': 415, u'face': 415, u'provid': 414, u'cost': 413, u'dirti': 412, u'chanc': 412, u'non': 411, u'tini': 411, u'ring': 410, u'certainli': 410, u'forget': 409, u'crunchi': 408, u'app': 406, u'center': 406, u'lemon': 406, u'market': 406, u'fanci': 406, u'limit': 403, u'fix': 402, u'spend': 402, u'view': 402, u'bell': 399, u'pot': 398, u'meatbal': 398, u'easili': 397, u'avocado': 396, u'suppos': 395, u'classic': 395, u'folk': 394, u'pad': 393, u'broth': 392, u'pleasant': 392, u'highli': 392}#, u'present': 391, u'downtown': 391, u'tonight': 390, u'chicago': 390, u'smile': 389, u'level': 386, u'korean': 385, u'despit': 384, u'pictur': 383, u'fair': 383, u'heavi': 381, u'calamari': 381, u'c': 381, u'sampl': 379, u'e': 376, u'unless': 376, u'sorri': 375, u'step': 375, u'touch': 374, u'frozen': 374, u'loud': 373, u'forward': 373, u'cocktail': 373, u'thru': 371, u'pop': 371, u'opinion': 371, u'hash': 369, u'hole': 369, u'bag': 369, u'damn': 368, u'terribl': 367, u'longer': 367, u'min': 366, u'dive': 366, u'broccoli': 365, u'rich': 364, u'cheesecak': 364, u'doubl': 364, u'speak': 364, u'pricey': 363, u'lol': 363, u'cash': 362, u'plain': 362, u'daughter': 361, u'patron': 361, u'appar': 361, u'load': 360, u'south': 359, u'dollar': 359, u'moist': 359, u'frequent': 358, u'true': 358, u'afternoon': 356, u'cute': 356, u'strawberri': 356, u'within': 356, u'offic': 355, u'bf': 354, u'occas': 354, u'book': 354, u'floor': 353, u'agre': 353, u'somewher': 352, u'conveni': 352, u'mediocr': 351, u'welcom': 350, u'teriyaki': 350, u'color': 350, u'http': 349, u'version': 348, u'ball': 348, u'imagin': 347, u'shot': 347, u'bottom': 346, u'simpli': 346, u'appear': 345, u'jalapeno': 345, u'beat': 345, u'gotten': 344, u'hold': 343, u'suck': 343, u'similar': 342, u'carn': 342, u'carrot': 341, u'complain': 340, u'mustard': 340, u'sad': 340, u'greek': 339, u'prime': 339, u'round': 338, u'platter': 338, u'alon': 338, u'www': 337, u'truli': 337, u'p': 336, u'contain': 336, u'japanes': 335, u'leftov': 335, u'de': 334, u'paid': 334, u'bare': 334, u'babi': 333, u'ahead': 333, u'worker': 333, u'nacho': 332, u'banana': 332, u'basil': 330, u'follow': 330, u'countri': 330, u'steam': 330, u'continu': 329, u'worst': 327, u'road': 327, u'outstand': 327, u'enter': 326, u'overli': 326, u'singl': 324, u'chipotl': 323, u'fat': 323, u'yelper': 322, u'asada': 322, u'anywher': 322, u'bathroom': 322, u'boy': 321, u'question': 321, u'awhil': 321, u'smaller': 319, u'explain': 319, u'guacamol': 319, u'heart': 318, u'mayo': 318, u'eggplant': 317, u'station': 316, u'meh': 314, u'odd': 314, u'peanut': 313, u'st': 313, u'batter': 312, u'chewi': 310, u'sugar': 309, u'monday': 308, u'outdoor': 307, u'state': 307, u'biz': 307, u'tuesday': 306, u'chair': 305, u'strong': 305, u'forgot': 304, u'mess': 304, u'cozi': 304, u'poor': 302, u'becom': 302, u'met': 302, u'na': 302, u'shred': 302, u'pepperoni': 302, u'cuisin': 301, u'break': 301, u'desert': 301, u'mozzarella': 299, u'particular': 299, u'interior': 298, u'popular': 297, u'bunch': 297, u'shell': 297, u'forev': 296, u'hell': 295, u'tough': 295, u'delish': 294, u'pittsburgh': 291, u'mini': 291, u'garden': 290, u'otherwis': 290, u'coke': 289, u'admit': 289, u'fare': 288, u'fall': 288, u'somewhat': 288, u'phone': 288, u'buffalo': 287, u'lamb': 287, u'soggi': 287, u'futur': 287, u'effici': 286, u'coconut': 286, u'aw': 286, u'summer': 286, u'daili': 286, u'horribl': 285, u'gross': 284, u'delight': 284, u'known': 284, u'yesterday': 284, u'east': 283, u'young': 282, u'section': 281, u'co': 281, u'fajita': 281, u'hey': 280, u'convers': 279, u'celebr': 279, u'chunk': 279, u'hummu': 277, u'arizona': 277, u'fire': 276, u'avoid': 276, u'caesar': 276, u'ravioli': 276, u'mcdonald': 275, u'none': 274, u'bone': 274, u'die': 273, u'surprisingli': 272, u'throw': 272, u'nearli': 272, u'spaghetti': 271, u'describ': 271, u'paper': 270, u'begin': 270, u'heaven': 269, u'nearbi': 267, u'flat': 267, u'skip': 267, u'per': 267, u'juic': 267, u'credit': 267, u'indian': 267, u'compani': 266, u'complimentari': 266, u'specialti': 266, u'cheesi': 266, u'parmesan': 265, u'clear': 265, u'cucumb': 264, u'relax': 263, u'pub': 263, u'son': 263, u'gem': 261, u'girlfriend': 261, u'throughout': 260, u'slaw': 260, u'own': 259, u'jack': 258, u'vietnames': 258, u'quesadilla': 258, u'mild': 257, u'dim': 257, u'lime': 257, u'pre': 256, u'allow': 255, u'brew': 254, u'overcook': 254, u'breast': 254, u'bother': 254, u'tap': 254, u'sum': 253, u'funni': 253, u'ten': 253, u'york': 253, u'chow': 252, u'afford': 252, u'tamal': 252, u'post': 252, u'alright': 252, u'accommod': 251, u'oz': 251, u'shake': 251, u'basket': 250, u'muffin': 250, u'featur': 249, u'accompani': 249, u'cowork': 248, u'beverag': 248, u'tend': 247, u'belli': 247, u'syrup': 247, u'fit': 246, u'mother': 246, u'separ': 245, u'notch': 245, u'cheddar': 243, u'pud': 243, u'diet': 243, u'martini': 243, u'overpr': 243, u'rave': 243, u'patti': 242, u'rude': 242, u'anymor': 242, u'north': 241, u'fabul': 241, u'wonton': 241, u'import': 240, u'sister': 240, u'caus': 240, u'movi': 240, u'takeout': 240, u'cinnamon': 239, u'older': 239, u'pile': 239, u'dad': 239, u'elsewher': 238, u'gift': 238, u'rang': 237, u'calori': 236, u'skin': 236, u'moment': 236, u'ta': 236, u'improv': 235, u'inform': 235, u'obvious': 235, u'stori': 234, u'ground': 234, u'promptli': 234, u'mistak': 233, u'larger': 233, u'clearli': 233, u'central': 233, u'fork': 233, u'particularli': 233, u'tempura': 233, u'annoy': 233, u'entertain': 232, u'hidden': 232, u'doubt': 232, u'straight': 232, u'pour': 231, u'parent': 231, u'difficult': 230, u'cabbag': 230, u'comment': 230, u'mein': 230, u'sun': 229, u'slider': 229, u'greas': 229, u'clam': 228, u'guest': 228, u'bore': 227, u'ridicul': 227, u'handl': 227, u'age': 227, u'confus': 226, u'wors': 226, u'mile': 226, u'memori': 226, u'english': 226, u'posit': 225, u'sesam': 225, u'sashimi': 225, u'vanilla': 225, u'onlin': 224, u'previou': 224, u'learn': 223, u'stomach': 223, u'product': 223, u'vibe': 223, u'california': 223, u'bruschetta': 222, u'cherri': 222, u'al': 222, u'brisket': 222, u'depend': 222, u'oven': 222, u'dozen': 221, u'finger': 221, u'sens': 221, u'ny': 221, u'wood': 221, u'apart': 220, u'board': 220, u'spectacular': 220, u'denni': 219, u'strang': 219, u'artichok': 219, u'closer': 219, u'alcohol': 218, u'giant': 218, u'refresh': 218, u'island': 217, u'fluffi': 217, u'balanc': 217, u'apolog': 217, u'charm': 216, u'marinara': 216, u'host': 216, u'count': 215, u'pineappl': 215, u'pecan': 215, u'macaroni': 215, u'west': 214, u'burn': 214, u'god': 214, u'warn': 214, u'express': 213, u'groupon': 213, u'major': 213, u'pesto': 212, u'savori': 212, u'invit': 212, u'variou': 211, u'bud': 211, u'everywher': 210, u'higher': 210, u'subway': 210, u'spent': 210, u'whip': 210, u'golden': 209, u'worri': 209, u'par': 209, u'pleasantli': 208, u'flour': 208, u'darn': 207, u'pan': 207, u'regist': 207, u'extens': 206, u'weather': 206, u'az': 206, u'deserv': 206, u'theme': 205, u'carri': 205, u'bigger': 205, u'peak': 205, u'oyster': 205, u'hawaiian': 204, u'snack': 204, u'pair': 204, u'goat': 203, u'polit': 203, u'sell': 202, u'toward': 202, u'thursday': 202, u'hamburg': 202, u'answer': 202, u'raw': 202, u'factor': 201, u'band': 201, u'inexpens': 201, u'pastrami': 201, u'lover': 201, u'assum': 200, u'steakhous': 200, u'overwhelm': 199, u'tart': 199, u'fountain': 199, u'cashier': 199, u'mr': 198, u'gon': 198, u'whenev': 198, u'six': 198, u'chill': 197, u'colleg': 197, u'freshli': 197, u'ass': 196, u'rel': 196, u'pool': 195, u'sake': 195, u'luckili': 195, u'layer': 195, u'win': 195, u'quiet': 194, u'accept': 194, u'pound': 194, u'groceri': 194, u'asparagu': 194, u'lost': 193, u'stuck': 193, u'omelett': 193, u'milk': 193, u'cater': 192, u'machin': 192, u'woman': 192, u'wear': 192, u'vinegar': 192, u'ayc': 192, u'ginger': 191, u'pastri': 191, u'besid': 191, u'yellow': 191, u'nut': 191, u'dont': 191, u'beyond': 191, u'process': 190, u'g': 190, u'napkin': 190, u'wheat': 189, u'correct': 189, u'soy': 188, u'bomb': 188, u'wild': 188, u'sick': 188, u'john': 187, u'certain': 187, u'sprout': 187, u'fatti': 186, u'experienc': 186, u'crap': 186}
    #required
    #required_features = list(required_features.keys())#  = dict(required_features, **vocab)
    #Adding top 150 bigrams
    #required_features += vocab #Here vocab is obtained from the vectorizer when run only for top 150 bigrams. Can be obtaind from req_bigrams as well
    #print len(required_features)
    #all_features.update(bigrams_dict)
    #print len(all_features)
    #all_features= Counter(all_features)
    #best_words = all_features.keys()
    all_features = {}
    
    
    #print all_features#.values()
    
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


# In[672]:

feature_selection(labels_sample, np.array(data_sample_copy))


# End

# In[681]:

preprocessed_data_sample.shape


# In[682]:

data_labels


# In[ ]:




# Choosing 20000 samples from the entire data - Splitting into train and test

# In[706]:

import copy

train_plus_test = samples#20000#(train_length + test_length) * 3#20000
ratio = int(train_plus_test * 0.7)
#ratio = train_length * 6
#labels = labels_sample[0:train_plus_test]

X_train = preprocessed_data_sample[0:ratio]
y_train = labels_sample[0:ratio]
X_test = preprocessed_data_sample[ratio:train_plus_test]
y_test = labels_sample[ratio:train_plus_test]
X_TEST = PREPROCESSED_DATA_SAMPLE
print len(X_test), len(y_train)


# In[684]:

sum( X_train[1] == 1)


# In[685]:

print y_test, "\n\n\n",y_train


# In[686]:

def sigmoid(value):
    #, value.shape()
    return (1.0 / (1.0 + np.exp(-1.0 * value)))


# In[687]:

def cost_function(w, X, labels):
    #print "w func", w, w.shape
    #w.shape = (1, len(w))
    #pred = sigmoid(np.dot(w, np.transpose(X)))
    pred = np.dot(w, np.transpose(X))
    #print pred.shape, labels.shape
    #print "pred", pred
    #error = -(np.dot(labels,np.transpose(np.log(pred))) + (np.dot((1 - labels) , np.transpose(np.log(1 - pred))))) / len(labels)
    #multivariate
    error = -np.dot(labels,np.transpose(np.log(pred))) / len(labels)
    #binary
    #error = (-np.dot(labels,np.transpose(np.log(sigmoid(pred)))) - np.dot((1 - labels), np.log(1 - np.transpose(sigmoid(pred))))) / len(labels)
    #print "error =", error
    grad = np.dot((pred - labels), X) / len(labels)
    #print "grad =", grad
    return [error, grad]


# In[688]:

import numpy as np
from scipy.optimize import minimize
def train_logistic_regression(X_train, y_train, no_classes):
    no_features = len(X_train[0])
    w  = np.zeros((no_classes, no_features))
    print "init",w[:,:]
    print len(w[0])
    for i in range(len(w)):
        print "\n\nclass", i+1, "\n\n\n"
        #multivariate
        w[i] = minimize(cost_function, w[i].reshape((1,no_features)), args=(X_train, np.equal(y_train,(i+1))), method="Newton-CG", jac=True).x
        #binary
        #w[i] = minimize(cost_function, w[i].reshape((1,no_features)), args=(X_train, y_train_thresholded), method="Newton-CG", jac=True).x
    #temp = np.matrix(temp)
    #w = np.matrix(temp)
    #print "this is temp", temp
    print w
    return w


# In[689]:

no_classes = 5
w = train_logistic_regression(X_train, y_train, no_classes)


# Thresholding our labels. ratings on 0 and 1 map to 0, and  3, 4 and 5 map to 1

# In[690]:

import copy
print y_test[0:5]
y_test_thresholded = np.array(copy.deepcopy(y_test))
#print type(y_test_thresholded)
#print y_test_thresholded
y_test_thresholded[y_test_thresholded < 3] = 0
y_test_thresholded[y_test_thresholded == 3] = 1
y_test_thresholded[y_test_thresholded > 3] = 2
#print  y_test_thresholded
y_train_thresholded = np.array(copy.deepcopy(y_train))
y_train_thresholded[y_train_thresholded < 3] = 0
y_train_thresholded[y_train_thresholded == 3] = 1
y_train_thresholded[y_train_thresholded > 3] = 2
print id(y_test)
print (y_test_thresholded)


# In[691]:

Y_TEST_THRESHOLDED = np.array(copy.deepcopy(Y_TEST))
#print type(y_test_thresholded)
#print y_test_thresholded
Y_TEST_THRESHOLDED[Y_TEST_THRESHOLDED < 3] = 0
Y_TEST_THRESHOLDED[Y_TEST_THRESHOLDED == 3] = 1
Y_TEST_THRESHOLDED[Y_TEST_THRESHOLDED > 3] = 2


# In[692]:

print len(X_train)
print len(y_train)


# In[693]:

print (w[0][0:10])


# In[694]:

min(y_train)


# In[695]:

#Predicts on all 5 classes - 1, 2, 3, 4, 5
def predict(w, X_test):
    prob = np.transpose(np.dot(w, np.transpose(X_test)))
    return (np.argmax(prob, axis=1)+1)


# In[696]:

#To be used when logistic regression is trained only on two classes
def predict_binary(w, X_test):
    prob = np.transpose(np.dot(w, np.transpose(X_test)))
    prob[prob >= 0.5] = 1
    prob[prob < 0.5] = 0
    return prob


# In[697]:

pred = predict(w, X_test)
PRED = predict(w, X_TEST)


# In[698]:

max(y_test)
#pred = predict_binary(w, X_test)
#pred.shape = (len(pred), 1)


# Test accuracies

# In[699]:

print sum(pred==y_test) * 100.0 / len(y_test)
print sum(PRED==Y_TEST) * 100.0 / len(Y_TEST)
from sklearn.metrics import confusion_matrix
print confusion_matrix(y_test, pred)
#print confusion_matrix(PRED, Y_TEST_THRESHOLDED)


# In[700]:

#print pred, "\n", y_test


# In[701]:

pred[pred < 3] = 0
pred[pred == 3] = 1
pred[pred > 3] = 2
PRED[PRED < 3] = 0
PRED[PRED == 3] = 1
PRED[PRED > 3] = 2
print pred, "\n", y_test_thresholded


# In[702]:

#pred[predicted==1] = 1


# In[703]:

print sum(pred==y_test_thresholded) * 100.0 / len(y_test_thresholded)
print sum(PRED==Y_TEST_THRESHOLDED) * 100.0 / len(Y_TEST_THRESHOLDED)


# In[705]:

from sklearn.metrics import confusion_matrix
print confusion_matrix(y_test_thresholded, pred)
print confusion_matrix(Y_TEST_THRESHOLDED, PRED)


# In[643]:

print sum(y_train_thresholded == 0)
print sum(y_train_thresholded == 1)
print sum(y_train_thresholded == 2)


# In[657]:

from sklearn.metrics import confusion_matrix
confusion_matrix(pred, y_test_thresholded)


# In[658]:

len(y_test_thresholded)


# In[659]:

print pred[((pred!=1) & (y_test_thresholded==1))]
#pred[]


# Train accuracies

# In[387]:

pred = predict(w, X_train)
print sum(pred==y_train) * 100.0 / len(y_train)


# In[388]:

#pred = predict_binary(w, X_train)
#pred.shape = (len(pred,))


# In[389]:

pred[pred < 3] = 0
pred[pred == 3] = 1
pred[pred > 3] = 2
print sum(pred==y_train_thresholded) * 100.0 / len(y_train_thresholded)


# In[390]:

confusion_matrix(pred, y_train_thresholded)


# In[190]:

y_train_bayes = np.array(copy.deepcopy(y_train))
y_test_bayes = np.array(copy.deepcopy(y_test))
y_train_bayes[y_train_bayes != 3] = 0
y_train_bayes[y_train_bayes == 3] = 1
y_test_bayes[y_test_bayes != 3] = 0
y_test_bayes[y_test_bayes == 3] = 1


# In[191]:

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train, y_train_bayes)


# In[192]:

clf


# In[193]:

predicted = clf.predict(X_test)


# In[194]:

predicted


# In[195]:

sum(predicted==y_test_bayes) * 100.0 / len(y_test_bayes)


# In[221]:

from sklearn.metrics import confusion_matrix
confusion_matrix(predicted, y_test_bayes)


# In[197]:

sum(predicted==0)


# In[ ]:

y_train_bayes = 


# In[ ]:

y_test_thresholded


# In[ ]:

type(required_features)


# In[ ]:

required_features


# In[ ]:

X_train[0][806:]


# Linear Regression

# In[165]:

from sklearn import datasets, linear_model
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# The coefficients
#print('Coefficients: \n', regr.coef_)
# The mean squared error
pred = regr.predict(X_test)


# In[ ]:

type(y_test)


# In[166]:

pred = pred + 0.5
pred = pred.astype('int32')


# In[167]:

pred


# In[168]:

print sum(pred==y_test) * 100.0 / len(y_test)


# In[169]:

pred[pred < 3] = 0
pred[pred == 3] = 1
pred[pred > 3] = 2
print sum(pred==y_test_thresholded) * 100.0 / len(y_test_thresholded)


# In[170]:

from sklearn.metrics import confusion_matrix
confusion_matrix(pred, y_test_thresholded)


# Poly reg

# In[ ]:

from sklearn.preprocessing import PolynomialFeatures

#poly = PolynomialFeatures(degree=2)
X_polytrain = poly.fit_transform(X_train)
X_polytest = poly.fit_transform(X_test)


# In[97]:

for i in range(5):
    print max(w[i]), min(w[i])


# In[138]:

print sum(y_train_thresholded == 2)
print sum(y_train_thresholded == 0)
print sum(y_train_thresholded == 1)


# In[392]:

required_features = {u'good': 20624, u'food': 17687, u'place': 17212, u'like': 14958, u'order': 14055, u'time': 11385, u'get': 11230, u'go': 11066, u'one': 10594, u'great': 9093, u'realli': 8709, u'servic': 8361, u'tri': 7686, u'chicken': 7556, u'back': 7295, u'would': 7039, u'restaur': 6513, u'come': 6247, u'also': 6075, u'love': 6066, u'menu': 6044, u'sauc': 5797, u'lunch': 5745, u'salad': 5741, u'eat': 5732, u'fri': 5714, u'littl': 5688, u'got': 5636, u'even': 5635, u'nice': 5556, u'chees': 5481, u'well': 5477, u'pretti': 5356, u'want': 5338, u'alway': 5194, u'look': 5054, u'make': 4997, u'tabl': 4820, u'came': 4791, u'tast': 4784, u'much': 4656, u'pizza': 4614, u'us': 4553, u'think': 4494, u'meal': 4446, u'bar': 4403, u'wait': 4382, u'know': 4376, u'drink': 4343, u'price': 4337, u'flavor': 4256, u'thing': 4253, u'best': 4222, u'sandwich': 4191, u'dish': 4141, u'say': 3952, u'side': 3931, u'better': 3931, u'serv': 3752, u'star': 3743, u'fresh': 3733, u'night': 3718, u'locat': 3717, u'dinner': 3700, u'could': 3673, u'take': 3670, u'delici': 3602, u'went': 3574, u'two': 3562, u'first': 3559, u'special': 3541, u'never': 3541, u'lot': 3505, u'enjoy': 3476, u'way': 3467, u'peopl': 3447, u'day': 3443, u'friend': 3440, u'sinc': 3423, u'right': 3400, u'bread': 3338, u'still': 3329, u'ask': 3299, u'hot': 3264, u'made': 3207, u'bit': 3192, u'friendli': 3134, u'soup': 3091, u'rice': 3039, u'year': 3016, u'sure': 2991, u'seem': 2983, u'bad': 2941, u'top': 2936, u'meat': 2919, u'though': 2912, u'give': 2902, u'someth': 2873, u'server': 2870, u'definit': 2859, u'review': 2844, u'burger': 2838, u'cook': 2813, u'see': 2789, u'around': 2780, u'use': 2774, u'roll': 2765, u'beef': 2647, u'said': 2645, u'small': 2643, u'last': 2635, u'favorit': 2629, u'visit': 2606, u'area': 2580, u'big': 2579, u'tasti': 2571, u'seat': 2558, u'need': 2546, u'mani': 2518, u'feel': 2510, u'sweet': 2494, u'breakfast': 2487, u'work': 2481, u'hour': 2455, u'egg': 2444, u'everyth': 2442, u'experi': 2435, u'next': 2427, u'plate': 2385, u'ok': 2377, u'minut': 2338, u'happi': 2307, u'potato': 2307, u'usual': 2306, u'busi': 2290, u'find': 2284, u'anoth': 2274, u'noth': 2264, u'long': 2256, u'ever': 2251, u'enough': 2242, u'taco': 2238, u'dine': 2222, u'staff': 2220, u'decid': 2191, u'took': 2183, u'steak': 2177, u'home': 2171, u'shrimp': 2167, u'beer': 2145, u'stop': 2116, u'larg': 2105, u'start': 2097, u'everi': 2095, u'walk': 2075, u'expect': 2048, u'new': 2044, u'bean': 2029, u'differ': 2024, u'grill': 2002, u'chip': 1996, u'sushi': 1994, u'check': 1985, u'reason': 1973, u'perfect': 1970, u'spici': 1962, u'item': 1960, u'wine': 1957, u'kind': 1951, u'offer': 1939, u'thought': 1934, u'actual': 1932, u'portion': 1928, u'probabl': 1917, u'mayb': 1914, u'spot': 1894, u'huge': 1892, u'old': 1884, u'fish': 1881, u'decent': 1878, u'end': 1866, u'cream': 1865, u'salsa': 1845, u'onion': 1844, u'sit': 1830, u'full': 1829, u'amaz': 1807, u'fast': 1807, u'howev': 1799, u'dessert': 1797, u'select': 1784, u'quit': 1777, u'qualiti': 1774, u'hous': 1744, u'half': 1726, u'option': 1724, u'insid': 1718, u'park': 1698, u'awesom': 1689, u'green': 1674, u'call': 1667, u'mexican': 1663, u'close': 1631, u'choic': 1617, u'pork': 1611, u'open': 1593, u'red': 1589, u'slice': 1566, u'super': 1564, u'appet': 1560, u'away': 1558, u'recommend': 1551, u'waitress': 1545, u'disappoint': 1544, u'fan': 1533, u'tomato': 1523, u'excel': 1516, u'put': 1511, u'chines': 1506, u'let': 1498, u'anyth': 1491, u'fill': 1490, u'coupl': 1488, u'entre': 1487, u'overal': 1478, u'pasta': 1478, u'ice': 1458, u'famili': 1458, u'almost': 1451, u'thai': 1451, u'worth': 1450, u'clean': 1446, u'els': 1433, u'style': 1427, u'town': 1421, u'size': 1409, u'italian': 1395, u'arriv': 1394, u'veggi': 1391, u'live': 1390, u'drive': 1387, u'quick': 1381, u'week': 1368, u'atmospher': 1357, u'oh': 1352, u'bite': 1348, u'room': 1339, u'hard': 1333, u'left': 1329, u'person': 1328, u'found': 1327, u'cours': 1323, u'burrito': 1320, u'three': 1306, u'yelp': 1304, u'least': 1298, u'wing': 1293, u'pick': 1289, u'outsid': 1289, u'part': 1287, u'noodl': 1277, u'final': 1277, u'husband': 1259, u'light': 1257, u'wonder': 1256, u'rememb': 1255, u'return': 1254, u'tea': 1249, u'either': 1248, u'help': 1246, u'water': 1244, u'bring': 1240, u'kid': 1238, u'piec': 1232, u'gener': 1232, u'includ': 1231, u'may': 1226, u'bacon': 1223, u'pepper': 1221, u'ye': 1218, u'mean': 1217, u'cheap': 1216, u'without': 1215, u'line': 1198, u'free': 1198, u'impress': 1191, u'bowl': 1190, u'tell': 1183, u'white': 1175, u'high': 1173, u'told': 1172, u'sever': 1167, u'keep': 1164, u'custom': 1161, u'chang': 1156, u'everyon': 1156, u'today': 1150, u'point': 1138, u'crispi': 1137, u'far': 1136, u'especi': 1133, u'brought': 1128, u'whole': 1125, u'dress': 1118, u'might': 1117, u'deal': 1110, u'name': 1110, u'regular': 1101, u'share': 1100, u'guy': 1096, u'less': 1089, u'late': 1085, u'mix': 1084, u'cool': 1084, u'dip': 1084, u'coffe': 1083, u'waiter': 1076, u'hand': 1074, u'extra': 1074, u'ate': 1073, u'guess': 1072, u'must': 1068, u'okay': 1067, u'manag': 1067, u'garlic': 1067, u'decor': 1065, u'surpris': 1057, u'averag': 1055, u'instead': 1053, u'real': 1050, u'group': 1050, u'pay': 1049, u'bbq': 1043, u'crowd': 1041, u'fine': 1041, u'fun': 1039, u'parti': 1037, u'cold': 1033, u'dri': 1033, u'leav': 1033, u'season': 1030, u'cake': 1024, u'chain': 1022, u'although': 1019, u'chocol': 1013, u'glass': 1013, u'door': 1006, u'plenti': 1006, u'run': 1003, u'warm': 1002, u'done': 999, u'care': 989, u'attent': 987, u'sometim': 983, u'local': 982, u'chili': 976, u'wrong': 975, u'pie': 973, u'wall': 969, u'pack': 964, u'amount': 964, u'mushroom': 960, u'type': 959, u'dog': 957, u'game': 955, u'sat': 955, u'pm': 955, u'past': 954, u'phoenix': 954, u'crust': 953, u'shop': 950, u'consist': 947, u'stuff': 946, u'watch': 946, u'gave': 945, u'second': 945, u'hit': 945, u'total': 943, u'finish': 941, u'head': 938, u'yummi': 936, u'sunday': 934, u'patio': 934, u'perfectli': 932, u'miss': 931, u'wife': 927, u'plu': 927, u'pancak': 926, u'street': 924, u'french': 924, u'notic': 923, u'someon': 922, u'quickli': 922, u'yet': 919, u'list': 919, u'ago': 917, u'fact': 916, u'sausag': 910, u'thank': 905, u'rib': 897, u'crave': 896, u'buffet': 894, u'cut': 892, u'crab': 889, u'toast': 884, u'show': 880, u'prepar': 863, u'w': 861, u'anyway': 857, u'recent': 854, u'talk': 854, u'tender': 849, u'wish': 849, u'butter': 846, u'owner': 838, u'corn': 830, u'often': 828, u'tortilla': 822, u'hope': 821, u'add': 820, u'counter': 814, u'front': 811, u'mind': 810, u'ad': 807, u'turn': 805, u'kitchen': 804, u'saturday': 804, u'later': 803, u'slow': 802, u'fantast': 801, u'typic': 796, u'lettuc': 795, u'bake': 792, u'lack': 791, u'authent': 787, u'wrap': 786, u'move': 784, u'pho': 783, u'soon': 780, u'felt': 779, u'four': 769, u'tip': 769, u'bagel': 764, u'happen': 762, u'strip': 756, u'rather': 751, u'pleas': 750, u'basic': 749, u'date': 749, u'seafood': 749, u'grab': 745, u'eaten': 743, u'set': 741, u'mom': 741, u'lobster': 739, u'hungri': 739, u'friday': 737, u'mention': 735, u'saw': 735, u'suggest': 734, u'stuf': 730, u'greasi': 723, u'normal': 723, u'joint': 722, u'veget': 722, u'bill': 721, u'soft': 719, u'diner': 717, u'curri': 717, u'varieti': 716, u'abl': 715, u'refil': 714, u'read': 714, u'spice': 712, u'bland': 712, u'thin': 712, u'roast': 711, u'problem': 710, u'standard': 710, u'combo': 708, u'bottl': 706, u'sign': 703, u'believ': 703, u'complet': 702, u'along': 702, u'choos': 699, u'alreadi': 698, u'enchilada': 697, u'solid': 696, u'base': 695, u'turkey': 694, u'vega': 694, u'absolut': 691, u'readi': 691, u'salmon': 691, u'brown': 691, u'sour': 689, u'birthday': 687, u'rate': 686, u'ingredi': 682, u'prefer': 682, u'play': 679, u'la': 678, u'stick': 677, u'except': 674, u'chef': 674, u'morn': 671, u'sub': 670, u'salt': 667, u'booth': 665, u'cover': 664, u'main': 663, u'etc': 661, u'cup': 660, u'store': 654, u'simpl': 651, u'medium': 647, u'note': 644, u'consid': 639, u'near': 639, u'valley': 637, u'thick': 636, u'comfort': 635, u'interest': 632, u'reserv': 630, u'girl': 629, u'serious': 628, u'brunch': 626, u'bartend': 625, u'oliv': 624, u'ton': 621, u'rest': 619, u'deli': 617, u'stand': 616, u'entir': 616, u'stay': 614, u'weekend': 611, u'card': 610, u'rare': 609, u'given': 607, u'money': 601, u'mall': 601, u'earli': 601, u'sound': 600, u'black': 599, u'gone': 598, u'month': 595, u'easi': 595, u'heard': 594, u'sort': 593, u'tuna': 592, u'mac': 591, u'man': 591, u'other': 586, u'greet': 585, u'american': 583, u'knew': 583, u'neighborhood': 583, u'short': 582, u'scottsdal': 579, u'glad': 579, u'smell': 579, u'mouth': 577, u'spinach': 577, u'music': 573, u'melt': 573, u'yum': 572, u'trip': 572, u'charg': 571, u'coupon': 569, u'salti': 566, u'wow': 565, u'ambianc': 561, u'meet': 558, u'across': 558, u'mine': 558, u'empti': 555, u'kept': 553, u'box': 550, u'pass': 549, u'oil': 549, u'school': 548, u'crisp': 547, u'ladi': 547, u'creami': 547, u'togeth': 542, u'satisfi': 541, u'gravi': 536, u'chop': 536, u'unfortun': 532, u'treat': 529, u'five': 528, u'n': 526, u'heat': 525, u'vegetarian': 525, u'pull': 524, u'waffl': 524, u'bun': 523, u'eye': 523, u'remind': 523, u'extrem': 521, u'behind': 520, u'anyon': 520, u'spring': 519, u'gyro': 518, u'lo': 517, u'seen': 517, u'combin': 516, u'tofu': 515, u'build': 513, u'job': 512, u'expens': 510, u'deliv': 510, u'case': 509, u'margarita': 509, u'rush': 508, u'kinda': 507, u'car': 507, u'buck': 507, u'split': 506, u'smoke': 506, u'low': 506, u'chose': 505, u'cafe': 503, u'homemad': 503, u'twice': 503, u'photo': 503, u'orang': 500, u'mash': 499, u'immedi': 498, u'honey': 497, u'asian': 496, u'dark': 494, u'figur': 493, u'compar': 492, u'excit': 489, u'hate': 486, u'yeah': 485, u'blue': 485, u'receiv': 483, u'rock': 482, u'number': 481, u'weird': 481, u'event': 481, u'exactli': 481, u'soda': 480, u'tradit': 480, u'idea': 475, u'charlott': 474, u'issu': 470, u'write': 467, u'drop': 466, u'space': 465, u'slightli': 465, u'establish': 464, u'textur': 463, u'incred': 462, u'whatev': 462, u'mostli': 461, u'ham': 460, u'origin': 459, u'addit': 457, u'mood': 456, u'hear': 456, u'biscuit': 456, u'buy': 455, u'life': 454, u'tv': 452, u'kick': 452, u'sport': 450, u'pita': 450, u'world': 447, u'omelet': 447, u'middl': 446, u'cooki': 445, u'duck': 445, u'chile': 445, u'plan': 445, u'crazi': 444, u'understand': 444, u'corner': 444, u'club': 444, u'realiz': 443, u'com': 441, u'uniqu': 441, u'hostess': 440, u'opt': 437, u'liter': 437, u'employe': 437, u'hubbi': 434, u'fairli': 433, u'matter': 433, u'b': 433, u'filet': 433, u'avail': 432, u'citi': 431, u'th': 431, u'pickl': 431, u'healthi': 430, u'complaint': 430, u'juici': 429, u'goe': 429, u'appl': 428, u'possibl': 428, u'word': 427, u'ranch': 426, u'boyfriend': 425, u'appreci': 424, u'fruit': 423, u'hang': 422, u'deliveri': 422, u'honestli': 422, u'casual': 422, u'taken': 421, u'valu': 420, u'due': 420, u'deep': 420, u'request': 420, u'beauti': 420, u'save': 419, u'window': 419, u'perhap': 418, u'updat': 417, u'upon': 417, u'scallop': 415, u'face': 415, u'provid': 414, u'cost': 413, u'dirti': 412, u'chanc': 412, u'non': 411, u'tini': 411, u'ring': 410, u'certainli': 410, u'forget': 409, u'crunchi': 408, u'app': 406, u'center': 406, u'lemon': 406, u'market': 406, u'fanci': 406, u'limit': 403, u'fix': 402, u'spend': 402, u'view': 402, u'bell': 399, u'pot': 398, u'meatbal': 398, u'easili': 397, u'avocado': 396, u'suppos': 395, u'classic': 395, u'folk': 394, u'pad': 393, u'broth': 392, u'pleasant': 392, u'highli': 392}#, u'present': 391, u'downtown': 391, u'tonight': 390, u'chicago': 390, u'smile': 389, u'level': 386, u'korean': 385, u'despit': 384, u'pictur': 383, u'fair': 383, u'heavi': 381, u'calamari': 381, u'c': 381, u'sampl': 379, u'e': 376, u'unless': 376, u'sorri': 375, u'step': 375, u'touch': 374, u'frozen': 374, u'loud': 373, u'forward': 373, u'cocktail': 373, u'thru': 371, u'pop': 371, u'opinion': 371, u'hash': 369, u'hole': 369, u'bag': 369, u'damn': 368, u'terribl': 367, u'longer': 367, u'min': 366, u'dive': 366, u'broccoli': 365, u'rich': 364, u'cheesecak': 364, u'doubl': 364, u'speak': 364, u'pricey': 363, u'lol': 363, u'cash': 362, u'plain': 362, u'daughter': 361, u'patron': 361, u'appar': 361, u'load': 360, u'south': 359, u'dollar': 359, u'moist': 359, u'frequent': 358, u'true': 358, u'afternoon': 356, u'cute': 356, u'strawberri': 356, u'within': 356, u'offic': 355, u'bf': 354, u'occas': 354, u'book': 354, u'floor': 353, u'agre': 353, u'somewher': 352, u'conveni': 352, u'mediocr': 351, u'welcom': 350, u'teriyaki': 350, u'color': 350, u'http': 349, u'version': 348, u'ball': 348, u'imagin': 347, u'shot': 347, u'bottom': 346, u'simpli': 346, u'appear': 345, u'jalapeno': 345, u'beat': 345, u'gotten': 344, u'hold': 343, u'suck': 343, u'similar': 342, u'carn': 342, u'carrot': 341, u'complain': 340, u'mustard': 340, u'sad': 340, u'greek': 339, u'prime': 339, u'round': 338, u'platter': 338, u'alon': 338, u'www': 337, u'truli': 337, u'p': 336, u'contain': 336, u'japanes': 335, u'leftov': 335, u'de': 334, u'paid': 334, u'bare': 334, u'babi': 333, u'ahead': 333, u'worker': 333, u'nacho': 332, u'banana': 332, u'basil': 330, u'follow': 330, u'countri': 330, u'steam': 330, u'continu': 329, u'worst': 327, u'road': 327, u'outstand': 327, u'enter': 326, u'overli': 326, u'singl': 324, u'chipotl': 323, u'fat': 323, u'yelper': 322, u'asada': 322, u'anywher': 322, u'bathroom': 322, u'boy': 321, u'question': 321, u'awhil': 321, u'smaller': 319, u'explain': 319, u'guacamol': 319, u'heart': 318, u'mayo': 318, u'eggplant': 317, u'station': 316, u'meh': 314, u'odd': 314, u'peanut': 313, u'st': 313, u'batter': 312, u'chewi': 310, u'sugar': 309, u'monday': 308, u'outdoor': 307, u'state': 307, u'biz': 307, u'tuesday': 306, u'chair': 305, u'strong': 305, u'forgot': 304, u'mess': 304, u'cozi': 304, u'poor': 302, u'becom': 302, u'met': 302, u'na': 302, u'shred': 302, u'pepperoni': 302, u'cuisin': 301, u'break': 301, u'desert': 301, u'mozzarella': 299, u'particular': 299, u'interior': 298, u'popular': 297, u'bunch': 297, u'shell': 297, u'forev': 296, u'hell': 295, u'tough': 295, u'delish': 294, u'pittsburgh': 291, u'mini': 291, u'garden': 290, u'otherwis': 290, u'coke': 289, u'admit': 289, u'fare': 288, u'fall': 288, u'somewhat': 288, u'phone': 288, u'buffalo': 287, u'lamb': 287, u'soggi': 287, u'futur': 287, u'effici': 286, u'coconut': 286, u'aw': 286, u'summer': 286, u'daili': 286, u'horribl': 285, u'gross': 284, u'delight': 284, u'known': 284, u'yesterday': 284, u'east': 283, u'young': 282, u'section': 281, u'co': 281, u'fajita': 281, u'hey': 280, u'convers': 279, u'celebr': 279, u'chunk': 279, u'hummu': 277, u'arizona': 277, u'fire': 276, u'avoid': 276, u'caesar': 276, u'ravioli': 276, u'mcdonald': 275, u'none': 274, u'bone': 274, u'die': 273, u'surprisingli': 272, u'throw': 272, u'nearli': 272, u'spaghetti': 271, u'describ': 271, u'paper': 270, u'begin': 270, u'heaven': 269, u'nearbi': 267, u'flat': 267, u'skip': 267, u'per': 267, u'juic': 267, u'credit': 267, u'indian': 267, u'compani': 266, u'complimentari': 266, u'specialti': 266, u'cheesi': 266, u'parmesan': 265, u'clear': 265, u'cucumb': 264, u'relax': 263, u'pub': 263, u'son': 263, u'gem': 261, u'girlfriend': 261, u'throughout': 260, u'slaw': 260, u'own': 259, u'jack': 258, u'vietnames': 258, u'quesadilla': 258, u'mild': 257, u'dim': 257, u'lime': 257, u'pre': 256, u'allow': 255, u'brew': 254, u'overcook': 254, u'breast': 254, u'bother': 254, u'tap': 254, u'sum': 253, u'funni': 253, u'ten': 253, u'york': 253, u'chow': 252, u'afford': 252, u'tamal': 252, u'post': 252, u'alright': 252, u'accommod': 251, u'oz': 251, u'shake': 251, u'basket': 250, u'muffin': 250, u'featur': 249, u'accompani': 249, u'cowork': 248, u'beverag': 248, u'tend': 247, u'belli': 247, u'syrup': 247, u'fit': 246, u'mother': 246, u'separ': 245, u'notch': 245, u'cheddar': 243, u'pud': 243, u'diet': 243, u'martini': 243, u'overpr': 243, u'rave': 243, u'patti': 242, u'rude': 242, u'anymor': 242, u'north': 241, u'fabul': 241, u'wonton': 241, u'import': 240, u'sister': 240, u'caus': 240, u'movi': 240, u'takeout': 240, u'cinnamon': 239, u'older': 239, u'pile': 239, u'dad': 239, u'elsewher': 238, u'gift': 238, u'rang': 237, u'calori': 236, u'skin': 236, u'moment': 236, u'ta': 236, u'improv': 235, u'inform': 235, u'obvious': 235, u'stori': 234, u'ground': 234, u'promptli': 234, u'mistak': 233, u'larger': 233, u'clearli': 233, u'central': 233, u'fork': 233, u'particularli': 233, u'tempura': 233, u'annoy': 233, u'entertain': 232, u'hidden': 232, u'doubt': 232, u'straight': 232, u'pour': 231, u'parent': 231, u'difficult': 230, u'cabbag': 230, u'comment': 230, u'mein': 230, u'sun': 229, u'slider': 229, u'greas': 229, u'clam': 228, u'guest': 228, u'bore': 227, u'ridicul': 227, u'handl': 227, u'age': 227, u'confus': 226, u'wors': 226, u'mile': 226, u'memori': 226, u'english': 226, u'posit': 225, u'sesam': 225, u'sashimi': 225, u'vanilla': 225, u'onlin': 224, u'previou': 224, u'learn': 223, u'stomach': 223, u'product': 223, u'vibe': 223, u'california': 223, u'bruschetta': 222, u'cherri': 222, u'al': 222, u'brisket': 222, u'depend': 222, u'oven': 222, u'dozen': 221, u'finger': 221, u'sens': 221, u'ny': 221, u'wood': 221, u'apart': 220, u'board': 220, u'spectacular': 220, u'denni': 219, u'strang': 219, u'artichok': 219, u'closer': 219, u'alcohol': 218, u'giant': 218, u'refresh': 218, u'island': 217, u'fluffi': 217, u'balanc': 217, u'apolog': 217, u'charm': 216, u'marinara': 216, u'host': 216, u'count': 215, u'pineappl': 215, u'pecan': 215, u'macaroni': 215, u'west': 214, u'burn': 214, u'god': 214, u'warn': 214, u'express': 213, u'groupon': 213, u'major': 213, u'pesto': 212, u'savori': 212, u'invit': 212, u'variou': 211, u'bud': 211, u'everywher': 210, u'higher': 210, u'subway': 210, u'spent': 210, u'whip': 210, u'golden': 209, u'worri': 209, u'par': 209, u'pleasantli': 208, u'flour': 208, u'darn': 207, u'pan': 207, u'regist': 207, u'extens': 206, u'weather': 206, u'az': 206, u'deserv': 206, u'theme': 205, u'carri': 205, u'bigger': 205, u'peak': 205, u'oyster': 205, u'hawaiian': 204, u'snack': 204, u'pair': 204, u'goat': 203, u'polit': 203, u'sell': 202, u'toward': 202, u'thursday': 202, u'hamburg': 202, u'answer': 202, u'raw': 202, u'factor': 201, u'band': 201, u'inexpens': 201, u'pastrami': 201, u'lover': 201, u'assum': 200, u'steakhous': 200, u'overwhelm': 199, u'tart': 199, u'fountain': 199, u'cashier': 199, u'mr': 198, u'gon': 198, u'whenev': 198, u'six': 198, u'chill': 197, u'colleg': 197, u'freshli': 197, u'ass': 196, u'rel': 196, u'pool': 195, u'sake': 195, u'luckili': 195, u'layer': 195, u'win': 195, u'quiet': 194, u'accept': 194, u'pound': 194, u'groceri': 194, u'asparagu': 194, u'lost': 193, u'stuck': 193, u'omelett': 193, u'milk': 193, u'cater': 192, u'machin': 192, u'woman': 192, u'wear': 192, u'vinegar': 192, u'ayc': 192, u'ginger': 191, u'pastri': 191, u'besid': 191, u'yellow': 191, u'nut': 191, u'dont': 191, u'beyond': 191, u'process': 190, u'g': 190, u'napkin': 190, u'wheat': 189, u'correct': 189, u'soy': 188, u'bomb': 188, u'wild': 188, u'sick': 188, u'john': 187, u'certain': 187, u'sprout': 187, u'fatti': 186, u'experienc': 186, u'crap': 186}
len(required_features)


# In[3]:

print all_features


# In[67]:

from nltk.corpus import wordnet as wn


# In[84]:

wn.synsets('delay')
#cat = wn.synset('amazing')
#dog.path_similarity(cat)


# In[156]:

all_features


# In[ ]:



