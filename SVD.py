# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 22:54:13 2019

@author: user
"""
from sklearn.datasets import fetch_20newsgroups
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
import string
from string import digits
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from nltk.stem import PorterStemmer

newsgroups_train = fetch_20newsgroups(subset='train',shuffle=True,remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test',shuffle=True,remove=('headers', 'footers', 'quotes'))
ps = PorterStemmer()
sentences = []
docu = []
for i in newsgroups_train.data:
    train = i.lower()
    for c in string.punctuation:
        train = train.replace(c,' ')
    train = ''.join([i for i in train if not i.isdigit()])
    train = ''.join([ps.stem(i) for i in train])
    docu.append(train)
for i in newsgroups_test.data:
    test = i.lower()
    for c in string.punctuation:
        test = test.replace(c,' ')
    test = ''.join([i for i in test if not i.isdigit()])
    test = ''.join([ps.stem(i) for i in test])
    docu.append(test)
vectorizer = CountVectorizer(max_df=0.1, min_df=2, binary = True,max_features=2000, stop_words='english')
tf = vectorizer.fit_transform(docu)
tf = tf.T
print('=================================chart-write====================================')
TD_matrix = np.matrix(tf.toarray())
print(TD_matrix.shape)
print('=================================matrix-create====================================')
U, S, VT = np.linalg.svd(tf.toarray(),full_matrices=False)
K=2

print('=================================USVT-create====================================')
U2 = U[:, 0:K]
S2 = np.diag(S[0:K])
VT2 = VT[0:K,:]

term_vectors = np.dot(U2, S2)
doc_vectors = np.dot(S2, VT2)

print(len(term_vectors))
print('Sigma Vectors: \n', S2)
print('Document Vectors: \n', doc_vectors) 
print('Term Vectors: \n', term_vectors)
print('=================================================')
count = 0
dname = []
for i in range(len(docu)):
    dname.append('d'+str(i))
word_vectors = np.array(term_vectors)
docu_vectors = np.array(doc_vectors.T)
pca = PCA(n_components=2)
plt.figure(figsize=(10,10))

result = pca.fit_transform(word_vectors)
plt.scatter(result[:,0], result[:,1],marker='>', c='r')
for word, (x,y) in zip(vectorizer.get_feature_names(), result):
    plt.text(x+0.05, y+0.05, word)
result2 = pca.fit_transform(docu_vectors)
plt.scatter(result2[:,0], result2[:,1], c = 'b')
for d, (x,y) in zip(dname, result2):
    plt.text(x+0.05, y+0.05, d)
        