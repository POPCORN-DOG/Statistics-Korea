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
#글씨체
font_name = font_manager.FontProperties(fname='C:/Windows/Fonts/H2HDRM.TTF').get_name()
rc('font', family=font_name)

###url 데이터 불러오고 파싱하여 기사의 본문 가져오기
'''data= pd.read_excel('C:/Users/USER/Desktop/1~8newsurl/covid_url38_word.xlsx', index_col=0)
data = data.dropna(axis=0)
data= data[['date','title','text','category','url']]
#테스트용
#data = data[34700:35000]#본문 가져오기'''


###배제어 포함어를 추가한 okt를 사용해서 형태소 분석하기
'''twi = Twitter()
add_noun = ['비대면','확진자','거리두기','사랑제일교회','의료기관','의료진','집단감염','유연근무','유연근무제','유연근로제','한국판뉴딜','디지털뉴딜',
            '그린뉴딜','휴먼뉴딜','취약계층','긴급재난지원금','고용보험','진료거부','택배기사','선별진료소','고용보험','특수고용직','특고','재배포',
            '고용안정지원금','공유오피스','공용오피스','저소득층','실업급여','유통업계','물류센터','소상공인','유연근로','특수형태근로자','특수형태근로종사자']
eco_word = pd.read_excel('C:/Users/USER/Desktop/★주요키워드 및 배제어.xlsx')
eco_keyword = list(eco_word['경제키워드'])
for i in eco_keyword:
    add_noun.append(i)

#새로운 단어 등록 하기
for i in add_noun:
    twi.add_dictionary(i,'Noun')

#통합어 만들기
replace = {'유연근로제':'유연근무제','유연근무':'유연근무제','유연근로':'유연근무제','특고':'특수고용직','특수형태근로자':'특수고용직','특수형태근로종사자':'특수고용직'}
postprocessor = Postprocessor(base_tagger=twi, replace=replace, passtags={'Noun'})

#단어 등록 확인하기
postprocessor.pos('확진자와 비대면으로 거리두기를 사랑제일교회에서부터 전광훈이 사전조사 의료기관과 의료진이 집단감염이 걸려 집단감염, 재택근무하고 '
          '유연근무와 유연근무제를 하고 지역경제와 한국판뉴딜과 그린뉴딜과 디지털뉴딜을 휴면뉴딜하고 취약계층의 긴급재난지원금을 특수고용직에게'
          '고용보험을 진료거부하며 택배기사는 온라인수업으로 선별진료소로 간다 소상공인 갭투자를 계액갱신공매공매도공시가격과징금과학기술괴리율국가경쟁력국내총생산'
          '국민건강보험국민연금국제원유가격규제지역그린뉴딜그린벨트금리금리인하금융기관금융세제금융위기금융지원배달서비스배당금배터리백신보유세보이스피싱부동산부동산시장'
          '분양가분양가상한제불법사금융블록체인비상경제비상경제회의빅데이터사모펀드사이드카사회적거리두기산유국상장주식상장지수증권상장지수펀드생필품서킷브레이커선물세트선물환세금'
          '세입자셧다운소비심리소상공인손소독제수소차수출규제스마트스토어스마트팜스타트업시장심리지수신용대출신용등급신용카드신재생에너지실물경제실업자안전자산액화천연가스양도세양도소득양도소득세'
          '양도차익양적완화에너지저장시스템오픈뱅킹온라인쇼핑몰외환시장외환위기용적률우선주운전자보험원격수업원격의료원달러환율위안화 AI에서 특수형태근로종사자와 특수형태근로자와 특고')

#형태소 분석 하기
words = [[j[i][0] for i in range(len(j))] for j in [postprocessor.pos(i) for i in tqdm(list(data['text']))]]

#배제어 등록하기
Prohibit_words = ['기자','연합뉴스','뉴시스','시사저널','신문','뉴스','사진','헤럴드경제','노컷뉴스','파이낸셜뉴스','특파원',
                  '라며','대해','지난','위해','오전','오후','무단','배포','이데일리','머니투데이','앵커','지금','때문','이번',
                  '통해','정도','경우','관련','이미지','출처','일보','바로가기','까지','여개','도록','이나','재배포','처럼','면서',
                  '거나','이제','지난달','어요']
for i in list(eco_word['배제어']):
    Prohibit_words.append(i)
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

data['words'] = words

data['length_word'] = [len(i) for i in tqdm(data['words'])]

data50 = data[data['length_word'] >= 50]  #단어 갯수 50개 이하 제거
data50.to_excel('C:/Users/USER/Desktop/1~8newsurl/data0921.xlsx',header=True) #저장하기

#3글자 부터 밎 주요 2단어 포함 토픽모델링용
data = data50.reset_index(drop=True)
words = list(data['words'])
for k in range(len(words)):
    words[k] = [i for i in words[k] if len(i) > 2 or i in ['배달','택배','고용','취업','실업','채용','구직','청년','유통','해고']]  # 두글자 이하 제외
data['words'] = words
data['words'][:5]
del words; del k;'''
data50 = pd.read_excel('C:/Users/USER/Desktop/1~8newsurl/data0928.xlsx', index_col=0)
data50['words'] = [i.replace("'", "").replace('[', '').replace(']', '').replace(' ', '').split(',') for i in
          list(data50['words'])]  # words가 텍스트 형식으로 되어 있을 경우


###형태소 분석한것을 이용하여 기사 단위로 word2vec 분석하기
embedding_model = Word2Vec(data50['words'], size=100, window = 6, min_count=100, workers=4, iter=100)
embedding_model.wv.save_word2vec_format('C:/Users/USER/Desktop/1~8newsurl/w2v')# 모델 저장'''
#data50 = pd.read_excel('C:/Users/USER/Desktop/1~8newsurl/data0921.xlsx', index_col=0)
embedding_model = KeyedVectors.load_word2vec_format('C:/Users/USER/Desktop/1~8newsurl/w2v') # 모델 로드

#포스팅용
sample_news_list = data50['words'][18063]
for i in sample_news_list:
    try:
        print(embedding_model.similarity('고용',i))
    except:
        print("0")

def scoring(word):
    avg_dist = []
    dist = []
    dist_dist = []
    for i in tqdm(range(len(word))):
        for k in word[i]:
            try:
                 # 비교하여 similarity 구하기
                dist_dist.append(embedding_model.similarity('고용', k))
            except:
                dist_dist.append(0)
            dist.append(np.array(dist_dist))
            dist_dist = []
        avg_dist.append(np.mean(dist))
        dist = []
    return avg_dist
a = scoring(data50['words'])
data50['score'] = a
data50_sorted = data50.sort_values(by="score", ascending=False)

news = data50['words']
id2word = corpora.Dictionary(news)
corpus = [id2word.doc2bow(text) for text in news]

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


model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=news, start=2,
                                                        limit=20, step=2)
# Show graph
limit=20; start=2; step=2;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


###경제키워드 상위 단어들의 벡터값을 이용해서 계층적 군집분석 하기 (일, 고용, 근로형태 관련 단어가 묶여야함)
'''#나온 단어들로 계층적 군집 하기
match_index = pd.DataFrame(embedding_model.vocab.keys(), columns=['name']) #벡터화된 단어들
word_vector = pd.DataFrame(embedding_model.vectors) #백터화된 단어들의 벡터값
data_v = pd.concat([match_index, word_vector], axis=1)

#경제키워드 상위랑 우리가 정한 키워드 합치기 경제쪽만 키위드 나와서 임의로 사회 관련 키워드 적용해 보았음
word = ['비대면','의료진','소상공인','취약계층','재택근무','원격수업','자영업자','배달','화상회의','고용','경제활동','간호사',
         '물류센터','근로자','실업','금융지원','해고','기본소득','무급휴직','택배','고용유지지원금','노동자','실업자','출퇴근',
         '실업률','특고','구직','일자리','유연근무제','유통업계','특수고용직','비정규직','프랜차이즈','고용노동부','스타트업','유통',
         '취업','채용','청년']
keyword = eco_keyword
for i in words:
    keyword.append(i)

data_w = pd.DataFrame(columns=list(data_v.columns))
for i in keyword:
    data_w = data_w.append(pd.DataFrame(data_v[data_v['name'] == i]))
#중복 제거
data_w = data_w.drop_duplicates(['name'])

# 계층적 군집 model

#글씨체
font_name = font_manager.FontProperties(fname='C:/Windows/Fonts/H2HDRM.TTF').get_name()
rc('font', family=font_name)
data_w.reset_index()

cluster = linkage(y=data_w.iloc[:,1:],method='complete',metric='euclidean')
cluster.shape
plt.figure( figsize = (15, 8) )
dendrogram(cluster, leaf_rotation=90, leaf_font_size=5,labels=list(data_w['name']))

###군집을 보고 일 관련된 키워드들 선정하기
#딱히 뭐 없음 잘 안나옴 비계층적으로 해야 하는데 그럼 또 시각화가 어려움'''
match_index = pd.DataFrame(embedding_model.wv.vocab.keys(), columns=['name']) #벡터화된 단어들
word_vector = pd.DataFrame(embedding_model.vectors) #백터화된 단어들의 벡터값
data_v = pd.concat([match_index, word_vector], axis=1)
embedding_model.most_similar('고용')
embedding_model.similarity('청년','일자리')
###작년대비 언급량이 많아진것으로 걸러내기 (엑셀에서 작업 함)

###키워드 선정 후 키워드 별로 점수화 하여 이상치 걸러내기 (박스플랏 시각화 추가)

#원래 단어
'''옛날 꺼word = ['비대면','의료진','소상공인','취약계층','재택근무','원격수업','자영업자','배달','화상회의','고용','경제활동','간호사',
         '근로자','실업','금융지원','해고','기본소득','무급휴직','택배','고용유지지원금','노동자','실업자','출퇴근',
         '실업률','특고','구직','일자리','유연근무제','특수고용직','비정규직','프랜차이즈','고용노동부','스타트업','유통',
         '취업','채용','청년']  #물류센터,유통업계 불가 나중에 단어 추가 해서 마지막으로 돌려 봐야 할듯'''
word = ['비대면','의료진','소상공인','취약계층','재택근무','원격수업','자영업자','배달','화상회의','고용','경제활동','간호사',
         '근로자','실업','금융지원','해고','기본소득','무급휴직','택배','고용유지지원금','노동자','실업자','출퇴근',
         '실업률','구직','일자리','유연근무제','특수고용직','비정규직','프랜차이즈','고용노동부','스타트업','유통',
         '취업','채용','청년','저소득층','실업급여','유통업계','고용안정지원금','공유오피스']

data = data50.reset_index()
#박스 플랏 시각화
def word_score2(word,allnum,num):
    a =[]
    for j in range(len(data)):
        if word in data['words'][j]:
            a.append(data.iloc[j])
    a = pd.DataFrame(a).reset_index(drop=True)
    a['cw'] = word
    a = a[['cw','date','title','text','category','url','words']]
    aa = a['words']
    a2 = [i.replace("'","").replace('[','').replace(']','').replace(' ','').split(',') for i in aa] #words가 텍스트 형식으로 되어 있을 경우
    #a2 = a['words'] #words가 list로 되어 있을 경우
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
    plt.subplot(int('1'+ str(allnum) + str(num))) #박스플랏 바꿔야함
    sns.boxplot( data = a['score'])
    plt.title(word)
    plt.axis('off')
    plt.show()
    return a
#a = word_score2('의료진',5,1) #시험용
'''data_cw = pd.DataFrame(columns=['cw','date','title','text','category','url','words','score'])
k=1
for i in word:
    try:
        a2 = word_score2(i,len(word),k)
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
    k += 1
#현재 이거 안됨 박스플랏 시각화 하는거'''
###정제된 키워드별 기사들로 각각 토픽모델링 해보기

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
    aa = a['words']
    a2 = [i.replace("'","").replace('[','').replace(']','').replace(' ','').split(',') for i in aa] #words가 텍스트 형식으로 되어 있을 경우
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
#b = cut_row('의료진')
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
def topic(word):
    def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
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
    bb2 = cut_row(word)
    news = [i.replace("'", "").replace('[', '').replace(']', '').replace(' ', '').split(',') for i in
            list(bb2['words'])]  # words가 텍스트 형식으로 되어 있을 경우
    id2word = corpora.Dictionary(news)
    texts = news
    corpus = [id2word.doc2bow(text) for text in texts]

    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=texts, start=2,
                                                            limit=15, step=1)
    # Show graph
    limit = 15; start = 2; step = 1;
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    print(coherence_values.index(max(coherence_values)) + 2,'개의 주제가 이상적')

    #주제 dataframe화 하기
    coherence_values.index(max(coherence_values))
    optimal_model = model_list[coherence_values.index(max(coherence_values))]
    topic_dic = {}
    for i in range(coherence_values.index(max(coherence_values)) + 2):
        words2 = optimal_model.show_topic(i, topn=20)
        topic_dic['topic ' + '{:02d}'.format(i + 1)] = [i[0] for i in words2]
    da = pd.DataFrame(topic_dic)
    return da
word22 = ['비대면','의료진','소상공인','취약계층','재택근무','원격수업','자영업자','배달','화상회의','고용','경제활동','간호사',
         '근로자','실업','금융지원','해고','기본소득','무급휴직','택배','고용유지지원금','노동자','실업자','출퇴근',
         '실업률','구직','일자리','유연근무제','특수고용직','비정규직','프랜차이즈','고용노동부','스타트업','유통',
         '취업','채용','청년','저소득층','실업급여','고용안정지원금','공유오피스']
for kkk in word22:
    topic11 = topic(kkk)
    topic11.to_excel('C:/Users/USER/Desktop/1~8newsurl/topicmodeling/'+str(kkk)+'.xlsx', header=True)  # 저장하기





