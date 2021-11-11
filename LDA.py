# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 23:54:05 2019

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 22:54:13 2019

@author: user
"""
from sklearn.datasets import fetch_20newsgroups
from gensim.models import Word2Vec
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation
from nltk.stem import PorterStemmer
newsgroups_train = fetch_20newsgroups(subset='train',shuffle=True,remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test',shuffle=True,remove=('headers', 'footers', 'quotes'))
ps = PorterStemmer()
sentences = []
docu = []
wordvec=[]
k=0
for i in newsgroups_train.data:
    train = i.lower()
    for c in string.punctuation:
        train = train.replace(c,' ')
    train = ''.join([i for i in train if not i.isdigit()])
    train = ''.join([ps.stem(i) for i in train])
    docu.append(train)
    wordvec.append(train.split())
for i in newsgroups_test.data:
    test = i.lower()
    for c in string.punctuation:
        test = test.replace(c,' ')
    test = ''.join([i for i in test if not i.isdigit()])
    test = ''.join([ps.stem(i) for i in test])
    docu.append(test)
    wordvec.append(test.split())
vectorizer = CountVectorizer(max_df=0.01, max_features=2000, binary = True, stop_words='english')
tf = vectorizer.fit_transform(docu)
print('======================chart-write==========================')
embed_model = Word2Vec(wordvec, window = 5, size = 100,min_count=1)
print('====================word2vec-finish====================')

n_topics = 10
n_iter = 10

lda = LatentDirichletAllocation(n_components=n_topics,
                                max_iter = n_iter,
                                learning_method='batch')
lda.fit(tf)
def print_top_words(model, feature_names, n_top_words):
    topic_list=[]
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        for i in topic.argsort()[:-n_top_words - 1:-1]:
            topic_list.append(feature_names[i])
    return topic_list

n_top_words=5
tf_feature_names = vectorizer.get_feature_names()
word_list = print_top_words(lda, tf_feature_names, n_top_words)
print('=========================LDA-FINISH====================')
def display_word_vec(model, words):
    col = 0
    color_list=['green','hotpink','yellow','black','red','blue','silver','purple','lightgreen',
                'plum','white','pink','gray','magenta','indigo','skyblue','aqua','orangered','peru','brown']
    pca = PCA(n_components=2)
    plt.figure(figsize=(10,10))

    word_vectors = np.array([model[w] for w in words])
    result = pca.fit_transform(word_vectors)
    
    lenn = int(len(result)/5)
    for i in range (lenn):
        start = int(i*5)
        end = start+5
        plt.scatter(result[start:end,0], result[start:end,1], edgecolors='k', c=color_list[i])
    
    for word, (x,y) in zip(words, result):
        plt.text(x+0.05, y+0.05, word)
    col = col + 1
display_word_vec(embed_model, word_list)



