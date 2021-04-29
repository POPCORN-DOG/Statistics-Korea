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




twi = Twitter()
okt = Okt()
twi.nouns('확진자와 비대면으로 거리두기를 사랑제일교회에서부터 전광훈이 사전조사 의료기관과 의료진이 집단감염이 걸려 집단감염, 재택근무하고 '
          '유연근무와 유연근무제를 하고 지역경제와 한국판뉴딜과 그린뉴딜과 디지털뉴딜을 휴면뉴딜하고 취약계층의 긴급재난지원금을 특수고용직에게'
          '고용보험을 진료거부하며 택배기사는 온라인수업으로 선별진료소로 간다 소상공인 갭투자를 계액갱신공매공매도공시가격과징금과학기술괴리율국가경쟁력국내총생산'
          '국민건강보험국민연금국제원유가격규제지역그린뉴딜그린벨트금리금리인하금융기관금융세제금융위기금융지원배달서비스배당금배터리백신보유세보이스피싱부동산부동산시장'
          '분양가분양가상한제불법사금융블록체인비상경제비상경제회의빅데이터사모펀드사이드카사회적거리두기산유국상장주식상장지수증권상장지수펀드생필품서킷브레이커선물세트선물환세금'
          '세입자셧다운소비심리소상공인손소독제수소차수출규제스마트스토어스마트팜스타트업시장심리지수신용대출신용등급신용카드신재생에너지실물경제실업자안전자산액화천연가스양도세양도소득양도소득세'
          '양도차익양적완화에너지저장시스템오픈뱅킹온라인쇼핑몰외환시장외환위기용적률우선주운전자보험원격수업원격의료원달러환율위안화') #확인용
okt.nouns('확진자와 비대면으로 거리두기를 사랑제일교회에서부터 전광훈이 사전조사 의료기관과 의료진이 집단감염이 걸려 집단감염, 재택근무하고 '
          '유연근무와 유연근무제를 하고 지역경제와 한국판뉴딜과 그린뉴딜과 디지털뉴딜을 휴면뉴딜하고 취약계층의 긴급재난지원금을 특수고용직에게'
          '고용보험을 진료거부하며 택배기사는 온라인수업으로 선별진료소로 간다 소상공인 갭투자를 계액갱신공매공매도공시가격과징금과학기술괴리율국가경쟁력국내총생산'
          '국민건강보험국민연금국제원유가격규제지역그린뉴딜그린벨트금리금리인하금융기관금융세제금융위기금융지원배달서비스배당금배터리백신보유세보이스피싱부동산부동산시장'
          '분양가분양가상한제불법사금융블록체인비상경제비상경제회의빅데이터사모펀드사이드카사회적거리두기산유국상장주식상장지수증권상장지수펀드생필품서킷브레이커선물세트선물환세금'
          '세입자셧다운소비심리소상공인손소독제수소차수출규제스마트스토어스마트팜스타트업시장심리지수신용대출신용등급신용카드신재생에너지실물경제실업자안전자산액화천연가스양도세양도소득양도소득세'
          '양도차익양적완화에너지저장시스템오픈뱅킹온라인쇼핑몰외환시장외환위기용적률우선주운전자보험원격수업원격의료원달러환율위안화') #확인용
#단어 추가 하기      '서구 문학'일 경우는 서구 와 문학으로 나뉨 띄어쓰기가 안됨
twi.add_dictionary('비대면','Noun')
twi.add_dictionary('확진자','Noun')
twi.add_dictionary('거리두기','Noun')
twi.add_dictionary('사랑제일교회','Noun')
twi.add_dictionary('의료기관','Noun')
twi.add_dictionary('의료진','Noun')
twi.add_dictionary('집단감염','Noun')
twi.add_dictionary('유연근무','Noun')
twi.add_dictionary('한국판뉴딜','Noun')
twi.add_dictionary('그린뉴딜','Noun')
twi.add_dictionary('디지털뉴딜','Noun')
twi.add_dictionary('휴면뉴딜','Noun')
twi.add_dictionary('취약계층','Noun')
twi.add_dictionary('긴급재난지원금','Noun')
twi.add_dictionary('고용보험','Noun')
twi.add_dictionary('진료거부','Noun')
twi.add_dictionary('택배기사','Noun')
twi.add_dictionary('선별진료소','Noun')
twi.add_dictionary('고용보험','Noun')
twi.add_dictionary('특수고용직','Noun')


data = pd.read_excel('C:/Users/USER/Desktop/1~8newsurl/covid_url38_word50.xlsx', index_col=0)
data = data[0:200]

list_text = data['text']
words = [twi.nouns(i) for i in tqdm(list(list_text))] #시간 걸림

Prohibit_words = ['기자','연합뉴스','뉴시스','시사저널','신문','뉴스','사진','헤럴드경제','노컷뉴스','파이낸셜뉴스','특파원',
                  '라며','대해','지난','위해','오전','오후','무단','배포','이데일리','머니투데이','앵커','지금','때문','이번',
                  '통해','정도','경우','관련','이미지','출처','일보']

j = 0
for i in tqdm(words):
    for k in Prohibit_words:
        while k in i:
            i.remove(k)
    words[j] = i
    j += 1 #불용어 제외

for k in range(len(words)):
    words[k] = [i for i in words[k] if len(i) > 1]  # 한글자 제외

data['words'] = words

data['length_word'] = [len(i) for i in tqdm(data['words'])]

data50 = data[data['length_word'] >= 50]  #단어 갯수 50개 이하 제거

embedding_model = Word2Vec(list(data50['words']), size=100, window = 6, min_count=150, workers=4, iter=100)
embedding_model.wv.save_word2vec_format('C:/Users/USER/Desktop/백업/covid_model38_2')# 모델 저장

news = data50['words'].tolist()
#news = [sent.split(',') for sent in news]
#type(str(news[0]).split(','))
news = [i.replace("'","").replace('[','').replace(']','').replace(' ','').split(',') for i in list(a['words'])] #words가 텍스트 형식으로 되어 있을 경우
#사전 만들기
id2word = corpora.Dictionary(news)

texts = news

corpus = [id2word.doc2bow(text) for text in texts]

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


model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=texts, start=2, limit=20, step=2)

# Show graph
limit=20; start=2; step=2;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# Print the coherence scores
for m, cv in zip(x, coherence_values):
  print("Num Topics =", m, " has Coherence Value of", round(cv, 4))




# Select the model and print the topics
optimal_model = model_list[10]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=15))

#
# Visualize the topics
'''pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(optimal_model, corpus, id2word)
vis #주피터에서 됨'''


