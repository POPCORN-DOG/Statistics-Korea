from urllib.request import urlopen
import urllib.request as req
from bs4 import BeautifulSoup
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
import matplotlib

#데이터 불러오기
try:
    data = pd.read_excel('C:/Users/USER/Desktop/통계용어사업data.xlsx')
    data['words'] = [i.replace("'", "").replace('[', '').replace(']', '').replace(' ', '').split(',') for i in
                       list(data['words'])]  # words가 텍스트 형식으로 되어 있을 경우
except:
    #2018, 19, 20 일부 뉴스 가져와서 합치기
    #2018년 뉴스
    data2018= pd.read_excel('C:/Users/USER/Desktop/2018newsdata1105.xlsx', index_col=0)

    #2019년 뉴스
    data2019 = pd.read_excel('C:/Users/USER/Desktop/data1015.xlsx', index_col=0)

    #2020년 뉴스
    data2020 = pd.read_excel('C:/Users/USER/Desktop/1~8newsurl/data0928.xlsx', index_col=0)

    #통합 데이터 만들기
    colnames = ['date','title','text','category','words']
    data2018['date'] = [datetime.datetime.strptime(str(i), '%Y%m%d') for i in data2018['date']] #date변수 date타입으로 바꿔주기
    data = pd.concat([data2018[colnames],data2019[colnames],data2020[colnames]])
    data.to_excel('C:/Users/USER/Desktop/통계용어사업data.xlsx', index= False)
    del data2018; del data2019; del data2020;

#단어 데이터 불러오기
work_word_data = pd.read_excel('C:/Users/USER/Desktop/고용관련단어.xlsx')
#단어 리스트 만들기
work_word_list = [i for i in work_word_data['선정용어']]
work_word_list


#딘어별 거리 보기 word2vec사용해보기
try:
    embedding_model = Word2Vec.load('C:/Users/USER/Desktop/고용단어분류_w2v_min5.model')  # 모델 로드
except:
    embedding_model = Word2Vec(data['words'], size=100, window=5, min_count=50, workers=4, iter=100)
    embedding_model.save('C:/Users/USER/Desktop/고용단어분류_w2v')  # 모델 저장

embedding_model = Word2Vec(data['words'], size=100, window=5, min_count=50, workers=4, iter=1)
embedding_model.save('C:/Users/USER/Desktop/고용단어분류_w2v_test.model')

embedding_model.most_similar('국민')
embedding_model.most_similar(positive=["청년"], topn=10)
print(embedding_model.similarity('일자리','고용'))
match_index = pd.DataFrame(embedding_model.wv.vocab.keys(), columns=['name']) #벡터화된 단어들
word_vector = pd.DataFrame(embedding_model.wv.vectors) #백터화된 단어들의 벡터값

vector_work_word = []; m =0; vector_name_list = [j for j in match_index['name']];
for i in work_word_list:
    if i in vector_name_list:
        m += 1
        print(i)
        vector_work_word.append(i)
print(m)
embedding_model.wv.distance('고용보험','최저임금')
embedding_model.similarity('고용보험','최저임금')

embedding_model.most_similar(positive=["부가가치"], topn=10)


##############################고용단어 형태소 사전에 넣어서 다시 돌리기

#단어 데이터 불러오기
work_word_data = pd.read_excel('C:/Users/USER/Desktop/고용관련단어.xlsx')
#단어 리스트 만들기
work_word_list = [i for i in work_word_data['선정용어']]
work_word_list

#밑줄 단어 만들기
work_word_list2 = []
for i in  work_word_list:
    work_word_list2.append(i.replace(' ','_'))

#띄어쓰기 없는단어 만들기
work_word_list3 = []
for i in  work_word_list:
    work_word_list3.append(i.replace(' ',''))

#단어 리스트 형태소 분석기에 넣기
twi = Twitter()
for i in range(len(work_word_list2)):
    twi.add_dictionary(work_word_list2[i],'Noun')
    twi.add_dictionary(work_word_list[i],'Noun')
    twi.add_dictionary(work_word_list3[i],'Noun')

#통합어 딕셔너리 만들기
replace = {}
for i in range(len(work_word_list2)):
    replace[work_word_list2[i]] = work_word_list[i]
    replace[work_word_list3[i]] = work_word_list[i]
#기타 통합어 추가
replace['특고'] = '특수형태근로종사자'
replace['특고종사자'] = '특수형태근로종사자'
replace['특수고용직'] = '특수형태근로종사자'
replace['맞벌이'] = '맞벌이 가구'

#분석할 데이터 셋 만들기 2018년도만 가져오기
data2018 = data.loc[0:23929,:]

postprocessor = Postprocessor(base_tagger=twi, replace=replace, passtags={'Noun'})
under_bar_text = [m.replace(' ','_') for m in data2018['text']] #본문에도 띄어쓰기를 _로 바꿔야 하는데 못바꿈 2.0버젼에 적용
#형태소 분석 하기
words = [[j[i][0] for i in range(len(j))] for j in [postprocessor.pos(i) for i in tqdm(under_bar_text)]]

#배제어 등록하기
Prohibit_words = ['기자','연합뉴스','뉴시스','시사저널','신문','뉴스','사진','헤럴드경제','노컷뉴스','파이낸셜뉴스','특파원',
                  '라며','대해','지난','위해','오전','오후','무단','배포','이데일리','머니투데이','앵커','지금','때문','이번',
                  '통해','정도','경우','관련','이미지','출처','일보','바로가기','까지','여개','도록','이나','재배포','처럼','면서',
                  '거나','이제','지난달','어요','재배']

#배제어 제거, 한 글자 제거하기
j = 0
for i in tqdm(words):
    for k in Prohibit_words:
        while k in i:
            i.remove(k)
    words[j] = i
    j += 1 #불용어 제외

for k in range(len(words)):
    words[k] = [i for i in words[k] if len(i) > 1]  # 한글자 제외

data2018['words2'] = words
data2018.to_excel('C:/Users/USER/Desktop/통계용어사업data2018_2.0.xlsx', index=False)

##############################################
data = pd.read_excel('C:/Users/USER/Desktop/통계용어사업data2018.xlsx')
data['words'] = [i.replace("'", "").replace('[', '').replace(']', '').replace(' ', '').split(',') for i in
                 list(data['words'])]  # words가 텍스트 형식으로 되어 있을 경우
data['words2'] = [i.replace("'", "").replace('[', '').replace(']', '').replace(' ', '').split(',') for i in
                 list(data['words2'])]  # words가 텍스트 형식으로 되어 있을 경우
a = []
for i in work_word_list:
    k= 0
    for j in data['words2']:
        if i in j:
            k += 1
    a.append(k)

plt.boxplot(a)
pd.DataFrame(a).to_excel('C:/Users/USER/Desktop/8.xlsx', index=False)


##############단어 연관어 분석 해보기
data = pd.read_excel('C:/Users/USER/Desktop/통계용어사업data2018_2.0.xlsx')
data['words'] = [i.replace("'", "").replace('[', '').replace(']', '').replace(' ', '').split(',') for i in
                 list(data['words'])]  # words가 텍스트 형식으로 되어 있을 경우
data['words2'] = [i.replace("'", "").replace('[', '').replace(']', '').replace(' ', '').split(',') for i in
                 list(data['words2'])]  # words가 텍스트 형식으로 되어 있을 경우

# 추가 배제어
words2 = []
for i in data['words2']:
    while '재배' in i:
        i.remove('재배')
    words2.append(i)
data['words2'] = words2

words2 = []
for i in data['words2']:
    while '금지' in i:
        i.remove('금지')
    words2.append(i)
data['words2'] = words2

#apriori로 연관성 분석하기
from apyori import apriori
import networkx as nx
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
dataset = list(data['words2'])

#신뢰도 보기
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(dataset, min_support=0.10, use_colnames = True)
a = pd.DataFrame(frequent_itemsets, columns=['itemsets','support','ordered_statistics'])
confidence = association_rules(a, metric='confidence',min_threshold=0.3)
#지지도

result = list(apriori(dataset, min_support=0.10))
df = pd.DataFrame(result)
df['length'] = df['items'].apply(lambda x: len(x))
df = df[(df['length'] == 2) & (df['support'] >= 0.10)].sort_values(by = 'support', ascending= False)

#그래프 정의
g = nx.Graph()
ar = (df['items'])
g.add_edges_from(ar)

#페이지 랭크
pr = nx.pagerank(g)
nsize = np.array([v for v in pr.values()])
nsize = 2000 * (nsize - min(nsize)) / (max(nsize) - min(nsize))

#레이아웃
#pos = nx.planar_layout(g)
pos = nx.shell_layout(g)
pos = nx.spring_layout(g)
pos = nx.kamada_kawai_layout(g)


#글씨체
font_name = font_manager.FontProperties(fname='C:/Windows/Fonts/H2HDRM.TTF').get_name()
rc('font', family=font_name)
matplotlib.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(16,12)); plt.axis('off')
nx.draw_networkx(g, pos=pos, font_family = font_name, node_size = nsize,
                 alpha = 0.7, edge_color = '.3') #, cmap = plt.cm.rainbow ,node_color = list(pr.values())

##통계용어사전 해보기
from apyori import apriori
import networkx as nx

data = pd.read_excel('C:/Users/USER/Desktop/통계용어사업data2018_2.0.xlsx')
data['words'] = [i.replace("'", "").replace('[', '').replace(']', '').replace(' ', '').split(',') for i in
                 list(data['words'])]  # words가 텍스트 형식으로 되어 있을 경우
data['words2'] = [i.replace("'", "").replace('[', '').replace(']', '').replace(' ', '').split(',') for i in
                 list(data['words2'])]  # words가 텍스트 형식으로 되어 있을 경우

#단어 데이터 불러오기
work_word_data = pd.read_excel('C:/Users/USER/Desktop/고용관련단어.xlsx')

data['words2'][0]

j = 0
w =[]
words2 = []
for i in tqdm(data['words2']):
    for k in work_word_data['선정용어']:
        if k in i:
            words2.append(k)
    w.append(words2)
    words2 = []
    j += 1 #불용어 제외
w2 = [i for i in w if len(i) > 1]


#지지도
result = list(apriori(w2, min_support=0.005))
df = pd.DataFrame(result)
df['length'] = df['items'].apply(lambda x: len(x))
df = df[(df['length'] == 2) & (df['support'] >= 0.005)].sort_values(by = 'support', ascending= False)

#그래프 정의
g = nx.Graph()
ar = (df['items'])
g.add_edges_from(ar)

#페이지 랭크
pr = nx.pagerank(g)
nsize = np.array([v for v in pr.values()])
nsize = 4000 * (nsize - min(nsize)) / (max(nsize) - min(nsize))

#레이아웃
#pos = nx.planar_layout(g)
pos = nx.shell_layout(g)
pos = nx.spring_layout(g)
pos = nx.kamada_kawai_layout(g)


#글씨체
font_name = font_manager.FontProperties(fname='C:/Windows/Fonts/H2HDRM.TTF').get_name()
rc('font', family=font_name)
matplotlib.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(16,12)); plt.axis('off')
nx.draw_networkx(g, pos=pos, font_family = font_name, node_size = nsize,
                 alpha = 0.7, edge_color = '.3',node_color = list(pr.values())) #, cmap = plt.cm.rainbow ,node_color = list(pr.values())








