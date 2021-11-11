# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 21:55:22 2019

@author: user
"""

from gensim.models import KeyedVectors

filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True, limit=100000) 
color = ['blue','yellow','red']
country = ['Spain', 'Italy', 'France', 'Poland', 'Greece', 'Turkey', 'Russia', 'Korea', 'China', 'Taiwan' ]
course = ['biology','ecology','sciences','geology','mathematics','arithmetic','topography','physics','philosophy','geographic']
fruit = []
city = []
lesson = []
similar_word = model.most_similar(positive=['apple'],topn=5)
not_similar_word = model.most_similar(negative=['apple'],topn=5)
for a in similar_word:
    fruit.append(a[0])
for b in not_similar_word:
    fruit.append(b[0])    
print(similar_word)
for c in country:
    result = model.most_similar(positive=['Tokyo', c], negative=['Japan'], topn=1)
    print(str(c)+str(result))
    city.append(c)
    city.append(result[0][0])
for d in course:
    result2 = model.most_similar(positive=['math', d], negative=['mathematics'], topn=1) 
    print(str(d)+str(result2))
    lesson.append(d)
    lesson.append(result2[0][0])
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

print("======================display-apple===================")
fruit_vectors = np.array([model[w] for w in fruit])
pca = PCA(n_components=2)
result = pca.fit_transform(fruit_vectors)
plt.figure(figsize=(10,10))
plt.scatter(result[:5,0], result[:5,1],marker='D', c='green')
plt.scatter(result[5:,0], result[5:,1], c='red')
for word, (x,y) in zip(fruit, result):
    plt.text(x+0.05, y+0.05, word)
print("======================display-country&capital===================")
city_vectors = np.array([model[w] for w in city])
pca = PCA(n_components=2)
result2 = pca.fit_transform(city_vectors)
plt.figure(figsize=(10,10))
xx , yy = np.split(result2.T,2,axis=0)
xx = xx.reshape(10,2)
yy = yy.reshape(10,2)
for i in range(len(xx.T)):
    if i%2==0:
        plt.scatter(xx.T[i],yy.T[i], edgecolors='k', c='blue')
    else:
        plt.scatter(xx.T[i],yy.T[i], edgecolors='k',marker='D', c='red')
for j in range(len(xx)):
    plt.plot(xx[j],yy[j], color='black')
for word, (x,y) in zip(city, result2):
    plt.text(x+0.05, y+0.05, word)
    
    plt.text(x+0.05, y+0.05, word)
print("======================display-course===================")
course_vectors = np.array([model[w] for w in lesson])
pca = PCA(n_components=2)
result3 = pca.fit_transform(course_vectors)
plt.figure(figsize=(10,10))
xx , yy = np.split(result3.T,2,axis=0)
xx = xx.reshape(10,2)
yy = yy.reshape(10,2)
for i in range(len(xx.T)):
    if i%2==0:
        plt.scatter(xx.T[i],yy.T[i], edgecolors='k', c='blue')
    else:
        plt.scatter(xx.T[i],yy.T[i], edgecolors='k',marker='D', c='red')
for j in range(len(xx)):
    plt.plot(xx[j],yy[j], color='black')
for word, (x,y) in zip(lesson, result3):
    plt.text(x+0.05, y+0.05, word)