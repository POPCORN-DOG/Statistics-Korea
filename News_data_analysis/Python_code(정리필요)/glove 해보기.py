from urllib.request import urlopen
import urllib.request as req
from bs4 import BeautifulSoup
import pandas as pd
import pandas as pd
import numpy as np
from urllib.request import urlopen #url의 html 을 가져 오기 위한 패키지
from bs4 import BeautifulSoup  #크롤링 필수 패키지 설치하려면 cmd창에서 pip install bs4
import os
import re
from selenium import webdriver
from bs4 import BeautifulSoup #크롤링 도구
from selenium.webdriver.common.keys import Keys
import time
from tqdm import tqdm
import os
import re
import time
import json
import datetime
from konlpy.tag import Okt
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pylab as pl
import statsmodels.api as sm
from sklearn.cluster import DBSCAN
import sklearn as skl
import sklearn.model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import random
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from ckonlpy.tag import Twitter
import string
import glob
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim
import warnings
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib import font_manager, rc
from ckonlpy.tag import Postprocessor

#glove 패키지 다운
from glove import Corpus, Glove

#데이터 불러오기
data =  pd.read_excel('C:/Users/USER/Desktop/labelingdata2018_word.xlsx', index_col=0)#데이터 불러오기
data['words'] = [i.replace("'", "").replace('[', '').replace(']', '').replace(' ', '').split(',') for i in
          list(data['words'])]  # words가 텍스트 형식으로 되어 있을 경우
data = data.drop_duplicates(['text'])
data = data.reset_index()

corpus = Corpus()
corpus.fit(data['words'], window=5)

glove = Glove(no_components=100, learning_rate= 0.05)
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True) #쓰레드 갯수는 4, 에포크 20
glove.add_dictionary(corpus.dictionary)

glove.most_similar('일자리', number = 10)
help(glove.most_similar)
#w2v해보기
words = data['words']
embedding_model = Word2Vec(words, size=100, window = 6, min_count=100, workers=4, iter=100)

embedding_model.wv.save_word2vec_format('C:/Users/USER/Desktop/백업/news_w2v_2019') # 모델 저장
embedding_model = KeyedVectors.load_word2vec_format('C:/Users/USER/Desktop/백업/news_w2v_2019') # 모델 로드

embedding_model.most_similar('일자리')


#lstm 해보기

data =  pd.read_excel('C:/Users/USER/Desktop/labelingdata2018_w2v.xlsx', index_col=0)#데이터 불러오기
data['words'] = [i.replace("'", "").replace('[', '').replace(']', '').replace(' ', '').split(',') for i in
          list(data['words'])]  # words가 텍스트 형식으로 되어 있을 경우
data = data.drop_duplicates(['text'])
data = data.reset_index()

from keras.models import Sequential
from keras.layers import Dense,LSTM

x = data.iloc[:,14:114]
y = data.iloc[:,12]


x1 = np.array(x)
x2 = x1.reshape((x1.shape[0],x1.shape[1],1))
y1 = np.array(y)

model = Sequential()
model.add(LSTM(10, activation='relu'))
model.add(Dense(5))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss',patience=100, mode='auto')

model.fit(x2,y1,epochs=1000, batch_size=1, verbose=2, callbacks=[early_stopping])

from numpy import array
