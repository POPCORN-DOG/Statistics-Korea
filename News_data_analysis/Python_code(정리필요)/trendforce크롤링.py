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
#import 색 보이기 Unresolved references
requrl = req.Request('https://www.trendforce.com/price', headers={'User-Agent': 'Mozilla/5.0'})
html = urlopen(requrl)
bsob = BeautifulSoup(html, "html.parser")

a=bsob.select('table')
#테이블 가져오기 참고
aa = pd.read_html(str(a[1]))
dt = aa[0]
dt.to_excel('C:/Users/USER/Desktop/test.xlsx')

# 다 가져 오기
urllist = ['https://www.trendforce.com/price','https://www.trendforce.com/price/flash','https://www.trendforce.com/price/ssd','https://www.trendforce.com/price/lcd',
           'https://www.trendforce.com/price/pv']
for i in urllist:
    p = re.compile('\d')
    requrl = req.Request(i, headers={'User-Agent': 'Mozilla/5.0'})
    html = urlopen(requrl)
    bsob = BeautifulSoup(html, "html.parser")
    a = bsob.select('table')
    b = bsob.select('.text-right')
    if i.split('/')[-1] == 'price':
        name = 'DRAM'
    else:
        name = i.split('/')[-1]
    for j in range(len(a)):
        aa = pd.read_html(str(a[j]))
        dt = aa[0]
        dt['update'] = ''.join(p.findall(b[j].text))
        dt.to_excel('C:/Users/USER/Desktop/' + name + str(j+1) + '.xlsx')


'''#날짜 가져오기
p = re.compile('\d')
b=bsob.select('.text-right')
b[0].text
ba[0].text.

p.findall(a[0].text)
''.join(p.findall(a[0].text))
'''

topicnum1007 = pd.read_excel('C:/Users/USER/Desktop/topicnum1007.xlsx')#데이터 불러오기



data50 = pd.read_excel('C:/Users/USER/Desktop/1~8newsurl/data0923.xlsx', index_col=0)#데이터 불러오기
embedding_model = KeyedVectors.load_word2vec_format('C:/Users/USER/Desktop/1~8newsurl/w2v0923') # 모델 로드
data50['words'] = [i.replace("'", "").replace('[', '').replace(']', '').replace(' ', '').split(',') for i in
          list(data50['words'])]  # words가 텍스트 형식으로 되어 있을 경우

#한글자 제외
data = data50.reset_index()
words = list(data['words'])
for k in range(len(words)):
    words[k] = [i for i in words[k] if len(i) > 2 or i in ['배달','택배','고용','취업','실업','채용','구직','청년','유통','해고']]  # 두글자 이하 제외
data['words'] = words


news =  data['words']#words가 텍스트 형식으로 되어 있을 경우
#사전 만들기
id2word = corpora.Dictionary(news)

texts = news

corpus = [id2word.doc2bow(text) for text in texts]

#이상치 자르기
def cut_row(word):
    # word단어가 포함된 기사만 가져오기
    a =[]
    for j in range(len(data)):
        if word in data['words'][j]:
            a.append(data.iloc[j])
    a = pd.DataFrame(a).reset_index(drop=True)
    a['cw'] = word
    a = a[['cw','date','title','text','category','url','words']]
    a2 = a['words']
    #score 구하기
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
    a['score'] = avg_dist

    #이상치 구하기
    q1 = np.percentile(a['score'], 25)
    q3 = np.percentile(a['score'],75)
    iqr = q3-q1
    outlier = q1 -  1.5 * iqr
    print(outlier, ' 값 이하 제거 필요')
    j=0
    for k in a['score']:
        if k < outlier:
            j +=1
    print('이상치 ', j, '개 제거')
    return_data = a[a['score'] > outlier]
    return return_data
#토픽모델링 함수
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
      print(num_topics)
      model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                              id2word=id2word,
                                              num_topics=num_topics,
                                              random_state=100,
                                              update_every=1,
                                              chunksize=100,
                                              passes=10,
                                              alpha='auto',
                                              per_word_topics=True)
      model_list.append(model)
      coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
      coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values
def topic2(word,num):
    bb2 = cut_row(word)
    #news = [i.replace("'", "").replace('[', '').replace(']', '').replace(' ', '').split(',') for i in
    #        list(bb2['words'])]  # words가 텍스트 형식으로 되어 있을 경우
    news = bb2['words']
    id2word = corpora.Dictionary(news)
    texts = news
    corpus = [id2word.doc2bow(text) for text in texts]

    def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
        coherence_values = []
        model_list = []
        for num_topics in tqdm(range(start, limit, step)):

            model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=num_topics,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=10,
                                                    alpha='auto',
                                                    per_word_topics=True)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())
        return model_list, coherence_values
    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=texts, start=num,
                                                            limit=num+2, step=1)

    # Show graph
    limit = num+2; start = num; step = 1;
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    print(coherence_values.index(max(coherence_values)) + 2,'개의 주제가 이상적')
    print(word)
    #주제 dataframe화 하기
    coherence_values.index(max(coherence_values))
    optimal_model = model_list[coherence_values.index(max(coherence_values))]
    topic_dic = {}
    for i in range(num):
        words2 = optimal_model.show_topic(i, topn=50)
        topic_dic['topic ' + '{:02d}'.format(i + 1)] = [i[0] for i in words2]
    da = pd.DataFrame(topic_dic)
    return da
a123 = topic2('언택트',13)
for i,j in zip(topicnum1007['name'],topicnum1007['topic_num']):
    da2 = topic2(i,j)
    da2.to_excel('C:/Users/USER/Desktop/1~8newsurl/topicmodeling/'+str(i)+'2글자제거50.xlsx', header=True)

