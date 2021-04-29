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

data = pd.read_excel('C:/Users/USER/Desktop/1~8newsurl/covid_url38_word50_2.xlsx', index_col=0)
embedding_model = KeyedVectors.load_word2vec_format('C:/Users/USER/Desktop/백업/covid_model38_2') # 모델 로드

match_index = pd.DataFrame(embedding_model.wv.vocab.keys(), columns=['name']) #벡터화된 단어들
embedding_model.get_vector('소상공인') #벡터값 찾기
word_vector = pd.DataFrame(embedding_model.wv.vectors) #백터화된 단어들의 벡터값

data_v = pd.concat([match_index, word_vector], axis=1)

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

model = KMeans(20)
model.fit(data_v.iloc[:,1:])
print(model.score(data_v.iloc[:,1:]))

def elbow(n_cluster, data):
    sse =[]
    for i in range(1,n_cluster):
        km = KMeans(n_clusters=i, init='k-means++', random_state=100)
        km.fit(data)
        sse.append(km.inertia_)
    plt.plot(range(1,n_cluster), sse, marker = 'o')
    plt.xlabel('클러스터 수')
    plt.ylabel('sse')
    plt.show()

elbow(10,data_v.iloc[:,1:])

plt.title('p')
plt.plot([1,2,3,4])
plt.show()


#나온 단어들로 계층적 군집 하기


words = ['비대면','의료진','소상공인','취약계층','재택근무','원격수업','자영업자','배달','화상회의','고용','경제활동','간호사',
         '물류센터','근로자','실업','금융지원','해고','기본소득','무급휴직','택배','고용유지지원금','노동자','실업자','출퇴근',
         '실업률','특고','구직','일자리','유연근무제','유통업계','특수고용직','비정규직','프랜차이즈','고용노동부','스타트업','유통',
         '취업','채용','청년']
data_w = pd.DataFrame(columns=list(data_v.columns))
for i in words:
    data_w = data_w.append(pd.DataFrame(data_v[data_v['name'] == i]))

# 계층적 군집 model
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib import font_manager, rc
#글씨체
font_name = font_manager.FontProperties(fname='C:/Windows/Fonts/H2HDRM.TTF').get_name()
rc('font', family=font_name)
data_w.reset_index()

cluster = linkage(y=data_w.iloc[:,1:],method='complete',metric='euclidean')
cluster.shape
plt.figure( figsize = (15, 8) )
dendrogram(cluster, leaf_rotation=60, leaf_font_size=12,labels=list(data_w['name']))
# leaf_rotation=90 : 글자 각도
# leaf_font_size=20 : 글자 사이즈
list(data_w['name'])


#####박스플랏 함수 구하기
#위에 패키지들 임포트
data = pd.read_excel('C:/Users/USER/Desktop/1~8newsurl/covid_url38_word50_2.xlsx', index_col=0)
embedding_model = KeyedVectors.load_word2vec_format('C:/Users/USER/Desktop/백업/covid_model38_2') # 모델 로드
words2 = ['비대면','의료진','소상공인','취약계층','재택근무','원격수업','자영업자','배달','화상회의','고용','경제활동','간호사',
         '물류센터','근로자','실업','금융지원','해고','기본소득','무급휴직','택배','고용유지지원금','노동자','실업자','출퇴근',
         '실업률','특고','구직','일자리','유연근무제','유통업계','특수고용직','비정규직','프랜차이즈','고용노동부','스타트업','유통',
         '취업','채용','청년']
data = data.reset_index()

def word_score(word):
    a =[]
    for j in range(len(data)):
        if word in data['words'][j]:
            a.append(data.iloc[j])
    a = pd.DataFrame(a).reset_index(drop=True)
    a['cw'] = word
    a = a[['cw','date','title','text','category','url','words']]
    aa = a['words']
    a2 = [i.replace("'","").replace('[','').replace(']','').replace(' ','').split(',') for i in aa]
    #score구하기
    avg_dist = []
    dist = []
    dist_dist = []
    for i in tqdm(range(len(a2))):
        for k in a2[i]:
            try:
                 # 비교하여 similarity 구하기
                dist_dist.append(embedding_model.similarity(word, k))
            except:
                dist_dist.append(0)
            dist2 = np.array(dist_dist)
            dist.append(sum(dist2))
            dist_dist = []
        if len(dist) > 1000:
            dist = sorted(dist, reverse=True)[:1000]
        avg_dist.append(np.mean(dist))
        dist = []
    avg_dist
    a['score'] = avg_dist
    #이상치 경계 구하기
    q1 = np.percentile(a['score'], 25)
    q3 = np.percentile(a['score'],75)
    iqr = q3-q1
    outlier = q1 -  1.5 * iqr
    print(outlier, ' 값 이하 제거 필요')
    j=0
    for k in a['score']:
        if k < outlier:
            j +=1
    print('이상치 ', j, '개 있음')
    sns.boxplot( data = a['score'])
    plt.show()
    return a
a = word_score('재택근무') #시험용

data_cw = pd.DataFrame(columns=['cw','date','title','text','category','url','words','score'])
for i in words2:
    try:
        a2 = word_score(i)
    except:
        a = []
        for j in range(len(data)):
            if i in data['text'][j]:
                a.append(data.iloc[j])
        a2 = pd.DataFrame(a).reset_index(drop=True)
        a2['cw'] = i
        a2['score'] = 0
        a2 = a2[['cw','date','title','text','category','url','words','score']]
    data_cw = pd.concat([data_cw,a2])
