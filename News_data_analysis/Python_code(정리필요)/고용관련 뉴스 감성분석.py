import pandas as pd
import glob
import os
import numpy as np
import re
import time
from tqdm import tqdm
from konlpy.tag import Okt
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
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
from collections import Counter
import matplotlib
import tomotopy as tp
from pyvis.network import Network
#글씨체
font_name = font_manager.FontProperties(fname='C:/Windows/Fonts/H2HDRM.TTF').get_name()
rc('font', family=font_name)
matplotlib.rcParams['axes.unicode_minus'] = False
#파일 한번에 불러와서 저장하기
flist = os.path.join('C:/Users/USER/Desktop/FW_ 고용기사 감성지수 Labeling 자료 공유', '*합본.csv') #끝에 합본.csv로 끝나는 것들 불러오기
colnames = ['date','a1','a2','title','a3','a4','text','user1','user2','user3','user4','user5','user6']
flist2 = os.path.join('C:/Users/USER/Desktop/FW_ 고용기사 감성지수 Labeling 자료 공유', '*.xlsx') #끝에 합본.csv로 끝나는 것들 불러오기
glob.glob(flist2)[0]  #파일 list 불러오기
pd.read_excel(glob.glob(flist2)[0], encoding='CP949',usecols=range(14))#,header=0, usecols=colnames)
def get_merge_csv(filelist):
    return pd.concat([pd.read_excel(i,encoding='CP949',usecols=range(13)) for i in filelist], axis=0)
def get_merge_csv2(filelist):
    return pd.concat([pd.read_csv(i,encoding='CP949',usecols=range(13)) for i in filelist], axis=0)
d = get_merge_csv(glob.glob(flist2))
d2 = get_merge_csv2(glob.glob(flist))
d.columns = colnames
d2.columns = colnames
d = pd.concat([d,d2])
d = d.reset_index(drop=True)
d.iloc[:,0:10]
d.columns = colnames

usecols=['date','title','text','user1','user2','user3','user5','user6']
d = d[usecols] #필요한 컬럼만 받기
#기본 데이터 셋 완성
#데이터 저장 하기
d = d.drop_duplicates(['text'])
d.to_excel('C:/Users/USER/Desktop/news_sentiment.xlsx', header=True)#'''

#고용기사만 분류 해보기



######형태소 분석하기###
'''data = pd.read_excel('C:/Users/USER/Desktop/news_sentiment.xlsx', index_col= 0)
twi = Twitter()
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
                  '거나','이제','지난달','어요', '부터']
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
data.to_excel('C:/Users/USER/Desktop/news_sentiment_words.xlsx', header=True)# '''

#########################################
data = pd.read_excel('C:/Users/USER/Desktop/news_sentiment_words.xlsx', index_col=0)
data['words'] = [i.replace("'", "").replace('[', '').replace(']', '').replace(' ', '').split(',') for i in
          list(data['words'])]  # words가 텍스트 형식으로 되어 있을 경우
data = data.reset_index(drop=True)

news = data['words']
id2word = corpora.Dictionary(news)
texts = news
corpus = [id2word.doc2bow(text) for text in texts]
##일반 토픽모델링 해보기
'''def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
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

model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=texts, start=10,
                                                        limit=15, step=1)
# Show graph
limit = 15;
start = 10;
step = 1;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()                                         #12 주제가 이상적
# 주제 dataframe화 하기
coherence_values.index(max(coherence_values))
optimal_model = model_list[coherence_values.index(max(coherence_values))]
topic_dic = {}
for i in range(12):
    words2 = optimal_model.show_topic(i, topn=20)
    topic_dic['topic ' + '{:02d}'.format(i + 1)] = [i[0] for i in words2]
da = pd.DataFrame(topic_dic)
da.to_excel('C:/Users/USER/Desktop/18년고용관련기사lda토픽모델링 결과.xlsx')#'''

#tf idf 사용 결과 별로 안좋음
'''from gensim import models
tf_ko = [id2word.doc2bow(text) for text in texts]
tfidf_model_ko = models.TfidfModel(tf_ko)
tfidf_ko = tfidf_model_ko[tf_ko]

model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=tfidf_ko, texts=texts, start=2,
                                                        limit=20, step=2)
# Show graph
limit = 20; start = 2; step = 2;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
# 주제 dataframe화 하기
coherence_values.index(max(coherence_values))
optimal_model = model_list[coherence_values.index(max(coherence_values))]
topic_dic = {}
for i in range(4):
    words2 = optimal_model.show_topic(i, topn=20)
    topic_dic['topic ' + '{:02d}'.format(i + 1)] = [i[0] for i in words2]
da2 = pd.DataFrame(topic_dic) #'''

## hdp토픽모델링 해보기
'''from gensim import models
hdp_ko = models.hdpmodel.HdpModel(corpus, id2word=id2word)
hdp_ko.print_topics(30) #비슷한 토픽이 많이 나옴'''

## tomotopy 사용해보기
import tomotopy as tp
from pyvis.network import Network

def tomotopy_lda(news_word_list,min_num, max_num):
    model_l = []; perplexity_l = []; likeliwood_l = [];
    for j in tqdm(range(min_num,max_num)):
        mdl = tp.LDAModel(k=j) #일반 lda
        for line in news_word_list:
            mdl.add_doc(line)
        for i in range(100):
            mdl.train()
        model_l.append(mdl)
        perplexity_l.append(mdl.perplexity)
        likeliwood_l.append(mdl.ll_per_word)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('topic_num')
    ax1.set_ylabel('likeliwood', color='red')
    ax1.plot([len(i.get_count_by_topics()) for i in model_l], likeliwood_l, color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('perplexity', color='blue')
    ax2.plot([len(i.get_count_by_topics()) for i in model_l], perplexity_l, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    return model_l,perplexity_l,likeliwood_l

model,per,likeli = tomotopy_lda(news,10,30)
likeli.index(max(likeli))
len(model[likeli.index(max(likeli))].get_count_by_topics())
model[0].get_topic_words(1,top_n = 20)
mdl.perplexity #최적 k값 정하는 방법 2가지 perplexity 값이 최소가 되는 지점
mdl.ll_per_word # 로그 가능도가 최대가 되는 지점

###########class만들기
class kyunpic_modeling:
    def setdata(self,word_list):
        self.word_list = word_list
    def lda_model(self,min_num=3,max_num=20,train_num=100,tw = tp.TermWeight.ONE):
        model_l = []; perplexity_l = []; likeliwood_l = [];
        news_word_list = self.word_list
        for j in tqdm(range(min_num,max_num+1)):
            mdl = tp.LDAModel(tw=tw,k=j, seed = 123) #일반 lda
            for line in news_word_list:
                mdl.add_doc(line)
            for i in range(train_num):
                mdl.train()
            model_l.append(mdl)
            perplexity_l.append(mdl.perplexity)
            likeliwood_l.append(mdl.ll_per_word)
        best_topic_model = model_l[likeliwood_l.index(max(likeliwood_l))]
        print('최적 토픽 수 {}개'.format(len(best_topic_model.get_count_by_topics())))
        self.model_l = model_l
        self.perplexity_l = perplexity_l
        self.likeliwood_l = likeliwood_l
        self.best_topic_model = best_topic_model
        self.min_num = min_num
        self.max_num = max_num
    def graph(self):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('topic_num')
        ax1.set_ylabel('likeliwood', color='red')
        ax1.plot([len(i.get_count_by_topics()) for i in self.model_l], self.likeliwood_l, color='red')
        ax1.tick_params(axis='y', labelcolor='red')

        #ax2 = ax1.twinx()
        #ax2.set_ylabel('perplexity', color='blue')
        #ax2.plot([len(i.get_count_by_topics()) for i in self.model_l],  self.perplexity_l, color='blue')
        #ax2.tick_params(axis='y', labelcolor='blue')
    def best_topic_data(self):
        topic_dic={}
        for mmm in range(len(self.best_topic_model.get_count_by_topics())):
            words2 = self.best_topic_model.get_topic_words(mmm, top_n=20)
            topic_dic['topic ' + '{:02d}'.format(mmm + 1)] = [i[0] for i in words2]
        da = pd.DataFrame(topic_dic)
        return da
    def topic_name(self, count_topic, mu= 0.1): #mu의 기본값은 0.25 높일수록 특이한거 나옴
        extractor = tp.label.PMIExtractor()
        cands = extractor.extract(self.model_l[count_topic-self.min_num])
        labeler = tp.label.FoRelevance(self.model_l[count_topic-self.min_num], cands, mu=mu, min_df=30)
        for i in range(len(self.model_l[count_topic-self.min_num].get_count_by_topics())):
            print(labeler.get_topic_labels(i, top_n=3))
            print(self.model_l[count_topic-self.min_num].get_topic_words(i, top_n=10))
            print('')
        topic_dic={}
        for mmm in range(len(self.model_l[count_topic-self.min_num].get_count_by_topics())):
            words2 = self.model_l[count_topic-self.min_num].get_topic_words(mmm, top_n=20)
            topic_dic['{},{},{}'.format(labeler.get_topic_labels(mmm)[0][0],
                                        labeler.get_topic_labels(mmm)[1][0],
                                        labeler.get_topic_labels(mmm)[2][0])] = [i[0] for i in words2]
        da = pd.DataFrame(topic_dic)
        return da
    #다이나믹 모델링 함수 추가
    def make_timelist(self,my_list):
        new_list = []
        for v in my_list:
            if v not in new_list:
                new_list.append(v)
        num_list = [i for i in range(len(new_list))]
        time_dict = dict(zip(new_list, num_list))
        self.time_dict=time_dict
        my_timelist = [time_dict.get(i) for i in my_list]
        self.my_timelist = my_timelist
        return my_timelist
    def setdata_time(self,wordlist_dtm,timelist = 0):
        self.word_list_dtm = wordlist_dtm
        try:
            self.timelist = self.my_timelist
        except:
            self.timelist = timelist
    def dynamic_model(self,min_num=3,max_num=20,train_num=100,tw = tp.TermWeight.ONE):
        model_l = []; perplexity_l = []; likeliwood_l = [];
        for j in tqdm(range(min_num, max_num+1)):
            mdl2 = tp.DTModel(tw = tw,k=j, t = len(Counter(self.timelist)),seed=123)  # 일반 lda
            for line, time2 in zip(self.word_list_dtm, self.timelist):
                mdl2.add_doc(line, timepoint=time2)
            for i in range(train_num):
                mdl2.train()
            model_l.append(mdl2)
            perplexity_l.append(mdl2.perplexity)
            likeliwood_l.append(mdl2.ll_per_word)
        best_topic_model = model_l[likeliwood_l.index(max(likeliwood_l))]
        print('최적 토픽 수 {}개'.format(len(best_topic_model.get_count_by_topics()[0])))
        self.model_l_dtm = model_l
        self.perplexity_l_dtm = perplexity_l
        self.likeliwood_l_dtm = likeliwood_l
        self.best_topic_model_dtm = best_topic_model
        self.min_num_dtm = min_num
        self.max_num_dtm = max_num
    def graph_dtm(self):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('topic_num')
        ax1.set_ylabel('likeliwood', color='red')
        ax1.plot([len(i.get_count_by_topics()[0]) for i in self.model_l_dtm], self.likeliwood_l_dtm, color='red')
        ax1.tick_params(axis='y', labelcolor='red')

        #ax2 = ax1.twinx()
        #ax2.set_ylabel('perplexity', color='blue')
        #ax2.plot([len(i.get_count_by_topics()[0]) for i in self.model_l_dtm],  self.perplexity_l_dtm, color='blue')
        #ax2.tick_params(axis='y', labelcolor='blue')
    def topic_name_dtm(self,count_topic, mu = 0.1, timepoint = 0):
        extractor = tp.label.PMIExtractor()
        cands = extractor.extract(self.model_l_dtm[count_topic-self.min_num_dtm])
        labeler = tp.label.FoRelevance(self.model_l_dtm[count_topic-self.min_num_dtm], cands, mu=mu, min_df=30)
        for i in range(len(self.model_l_dtm[count_topic-self.min_num_dtm].get_count_by_topics()[0])):
            print(labeler.get_topic_labels(i, top_n=3))
            print(self.model_l_dtm[count_topic-self.min_num_dtm].get_topic_words(i, top_n=10, timepoint=timepoint))
            print('')
        topic_dic = {}
        for mmm in range(len(self.model_l_dtm[count_topic - self.min_num_dtm].get_count_by_topics()[0])):
            words2 = self.model_l_dtm[count_topic - self.min_num_dtm].get_topic_words(mmm, top_n=20,timepoint=timepoint)
            topic_dic['{},{},{}'.format(labeler.get_topic_labels(mmm)[0][0],
                                        labeler.get_topic_labels(mmm)[1][0],
                                        labeler.get_topic_labels(mmm)[2][0])] = [i[0] for i in words2]
        da = pd.DataFrame(topic_dic)
        self.labeler = labeler
        return da
    def dtm_topic_distribution(self,count_topic):
        asd = []; name = []
        for i in range(len(Counter(self.timelist))):
            asd.append(self.model_l_dtm[count_topic - self.min_num_dtm].get_alpha(timepoint=i))
        for i in range(count_topic):
            try:
                name.append('{},{},{}'.format(self.labeler.get_topic_labels(i)[0][0],
                                              self.labeler.get_topic_labels(i)[1][0],
                                              self.labeler.get_topic_labels(i)[2][0]))
            except:
                name.append(i)
        asd2 = pd.DataFrame(asd, columns=name)
        asd2.plot.line()
        return asd2

#lda 예시코드
a = kyunpic_modeling()
a.setdata(new)
a.word_list
a.lda_model(max_num=10)
asd = a.best_topic_data()
asd2 = a.topic_name(7, mu = 0.1)
a.graph()
#dtm예시코드 정수형태의 타임포인트가 있어야함
timepoint = [int(str(i)[4:6]) for i in data['date']]
timeserise = []
for i in timepoint:
    if i <= 3:
        timeserise.append(0)
    else:
        if i <= 6:
            timeserise.append(1)
        else:
            if i <= 9:
                timeserise.append(2)
            else:
                timeserise.append(3)
data['timepoint'] = timepoint
data['timeserise'] = timeserise

a2 = kyunpic_modeling()
a2.make_timelist(data['timepoint'])
a2.setdata_time(data['words'])
a2.dynamic_model(min_num=7,max_num=10)
a2.perplexity_l_dtm
a2.graph_dtm()
a2.best_topic_model_dtm.summary()
ex = a2.topic_name_dtm(7,timepoint=1,mu=0.1)
a2.min_num_dtm
a2.dtm_topic_distribution(7)
a2.my_timelist

len(Counter(a2.timelist))
#다른 데이터 시험
'''data = pd.read_excel('C:/Users/USER/Desktop/data1015.xlsx', index_col=0)
data['words'] = [i.replace("'", "").replace('[', '').replace(']', '').replace(' ', '').split(',') for i in
          list(data['words'])]  # words가 텍스트 형식으로 되어 있을 경우
data = data.reset_index(drop=True)

a = kyunpic_modeling()
a.setdata(data['words'])
a.lda_model(max_num=15)
a.graph()
qqq = a.topic_name(7)

data['date'][0].month
data['month'] = [ i.month for i in data['date']]

a2 = kyunpic_modeling()
a2.make_timelist(data['month'])
a2.my_timelist
len(Counter(a2.my_timelist))
a2.setdata_time(data['words'])
a2.dynamic_model(max_num=12,train_num=60)
a2.best_topic_model_dtm.summary()
zxc2 = a2.topic_name_dtm(4, mu=0.25)
a2.graph_dtm()
a2.dtm_topic_distribution(5)

a2.model_l_dtm[7 - a2.min_num_dtm].get_alpha(timepoint=11)
a2.model_l_dtm[7 - a2.min_num_dtm].num_docs_by_timepoint
for i in range(12):
    print(i)'''
#클래스 만들때 연습장
'''for k in range(mdl.k):
    print('top 10 words of topic {}'.format(k))
    print(mdl.get_topic_words(k,top_n = 10))
print(mdl.k)
mdl.summary() #잘 나오는데 coherence 계산 못함

#이름 후보 나오게 하기
extractor = tp.label.PMIExtractor()
cands = extractor.extract(mdl)
labeler = tp.label.FoRelevance(mdl,cands, mu = 0.1, min_df = 30)
for i in range(k):
    print(labeler.get_topic_labels(i, top_n = 5))
    print(mdl.get_topic_words(i,top_n = 10))
    print('')
labeler.get_topic_labels(1)[3][0]

#다이나믹 토픽모델링 토픽별 비중 알파?
mdl2 = tp.DTModel(k = 5,t = 4)



for line, time2 in zip(news, timeserise):
    mdl2.add_doc(line, timepoint = time2)
for i in range(100):
    mdl2.train()
    #print(i, mdl.ll_per_word)
for k in range(5):
    print('top 10 words of topic {}'.format(k))
    print(mdl2.get_topic_words(k,top_n = 10, timepoint = 1))
mdl2.get_topic_words(2,3)
mdl2.get_topic_word_dist(2,3)
mdl2.get_count_by_topics()[0] #

a2.best_topic_model_dtm.get_alpha(timepoint = 0)
a2.model_l_dtm[5].get_alpha(timepoint = 3)
asd = []; name = []
for i in range(len(a2.model_l_dtm[4].get_count_by_topics())):
    asd.append(a2.model_l_dtm[1].get_alpha(timepoint = i))
    name.append('{},{},{}'.format(labeler.get_topic_labels(i)[0][0],
                                labeler.get_topic_labels(i)[1][0],
                                labeler.get_topic_labels(i)[2][0]))
asd2 = pd.DataFrame(asd, columns=name)
asd2.plot.line()
a2.model_l_dtm[5].summary()


extractor = tp.label.PMIExtractor()
cands = extractor.extract(a2.model_l_dtm[5])
labeler = tp.label.FoRelevance(a2.model_l_dtm[5], cands, mu=0.1, min_df=30)
for i in range(len(a2.model_l_dtm[5].get_count_by_topics()[0])):
    print(labeler.get_topic_labels(i, top_n=3))
    print(a2.model_l_dtm[5].get_topic_words(i, top_n=10, timepoint = 0))
    print('')

my_list = ['A', 'B', 'C', 'D', 'B', 'D', 'E']
new_list = []
for v in my_list:
    if v not in new_list:
        new_list.append(v)
print(new_list)

def make_timelist(my_list):
    new_list = []
    for v in my_list:
        if v not in new_list:
            new_list.append(v)
    num_list = [i for i in range(len(new_list))]
    time_dict = dict(zip(new_list,num_list))
    my_timelist = [time_dict.get(i) for i in my_list]
    return my_timelist


qwe2 = make_timelist(data['date'])
qwe.get(data['date'][19])#'''

#2020 3월부터 8월 까지 데이터 돌려보기
data0928 = pd.read_excel('C:/Users/USER/Desktop/1~8newsurl/data0928.xlsx', index_col=0)
data0928['words'] = [i.replace("'", "").replace('[', '').replace(']', '').replace(' ', '').split(',') for i in
          list(data0928['words'])]  # words가 텍스트 형식으로 되어 있을 경우
data0928 = data0928.reset_index(drop=True)

a = kyunpic_modeling()
a.setdata(data0928['words'])
a.lda_model(max_num=10,train_num=75, tw=tp.TermWeight.IDF)
a.graph()
a.best_topic_model.summary()
qwe2 = a.topic_name(10)
a.model_l[2].summary()
a.model_l[2].alpha
a.model_l[2].used_vocab_freq


#월별 다이나믹
data0928['month'] = [i.month for i in data0928['date']]
data0928 = data0928[data0928['month'] != 2]
data0928 = data0928.reset_index(drop=True)
a2 = kyunpic_modeling()
a2.make_timelist(data0928['month'])
a2.setdata_time(data0928['words'])
a2.dynamic_model(max_num=6,train_num=80, tw=tp.TermWeight.IDF)
a2.best_topic_model_dtm.summary()
a2.graph_dtm()
a2.model_l_dtm[1].summary()
zxc2 = a2.topic_name_dtm(5, mu=0.25)
a2.graph_dtm()
a2.dtm_topic_distribution(4)

a2.model_l_dtm[1].alpha
a2.model_l_dtm[1].get_alpha(0)
a3 = kyunpic_modeling()
a3.make_timelist(data0928['month'])
a3.setdata_time(data0928['words'])
a3.dynamic_model(max_num=7,train_num=80, tw=tp.TermWeight.PMI)
zxc3 = a3.topic_name_dtm(5, mu=0.25)



##예측
'''# 1 ) 합으로 해서 예측하기
d['sum'] = d[usecols[3:8]].sum(axis=1)#라벨링 합계 구하기
d['y'] = [1 if i > 3 else 0 for i in d['sum']] #1 과 0
d.describe()

#벡터 형식 넣기
vec_data = pd.read_excel('C:/Users/USER/Desktop/labelingdata2018_w2v.xlsx')
data = pd.merge(d,vec_data, on = 'text', how = 'left') #text가 키고, 왼쪽을 기준으로 데이터 결합'''

##word2vec과 lstm으로 감성예측하기

#데이터 가져오기 감성점수와 형태소 분석이 완료된 데이터 필요
data = pd.read_excel('C:/Users/USER/Desktop/news_sentiment_words.xlsx', index_col=0)
data['words'] = [i.replace("'", "").replace('[', '').replace(']', '').replace(' ', '').split(',') for i in
          list(data['words'])]  # words가 텍스트 형식으로 되어 있을 경우
data = data.reset_index(drop=True) #인덱스 초기화

#word2vec 사용하기
embedding_model = Word2Vec(data['words'], size=100, window = 4, min_count=30, workers=4, iter=200)
match_index = pd.DataFrame(embedding_model.wv.vocab.keys(), columns=['name']) #벡터화된 단어들
word_vector = pd.DataFrame(embedding_model.wv.vectors) #백터화된 단어들의 벡터값
data_v = pd.concat([match_index, word_vector], axis=1) #백터화된 단어들과 그 벡터값들을 하나로 합침
embedding_model.most_similar('취업') #벡터화 결과 살펴보기

#뉴스기사에서 벡터화된 단어만 남기고 나머지 단어들 제거
data['vec_words'] = ['na' for i in range(len(data))]
match_index['name']
vec_words = []
for num,i in enumerate(data['words']):
    for j in match_index['name']:
        if j in i:
            vec_words.append(j)
    data['vec_words'][num] = vec_words
    vec_words = []

#tokenization 토큰화 하기 형태소분석으로 일단 다 나눠놈
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['vec_words']) #형태소 분석에서 벡터화된 단어list를 tokenizer dictionary에 추가
print(len(tokenizer.word_index)) #벡터단어 list의 길이와 같음

text_sequence = tokenizer.texts_to_sequences(data['vec_words']) # 만들어진 dict을 기준으로 텍스트를 숫자로 변경

tokenizer.word_index


#길이 맞추기 (기사 단어 수 맞추기 , 짧은 기사는 뒤에 0 채우기)
from keras.preprocessing import sequence
max_length = max([len(i) for i in data['vec_words']]) #기사의 최대길이
pad_text = sequence.pad_sequences(text_sequence,
                                  maxlen=max_length,
                                  padding='post', value= 0) #padding = "post" 뒤에 0 채우기, 기본값은 앞에 0 채우기
pad_text[:5]

#x,y 데이터 만들기
from sklearn.model_selection import train_test_split
#y data 생성
y = [0 if i < 15 else 1 for i in data['sum'] ]
Y_data = np.array(y)

x_train, x_valid, y_train, y_valid = train_test_split(pad_text, Y_data, test_size=0.1, random_state=30)


#숫자화된 기사 속 단어와 벡터 맞춰주기
VOCAB_SIZE = len(tokenizer.index_word) + 1
embedding_dim = 100
embedding_matrix = np.zeros((VOCAB_SIZE, embedding_dim))

# tokenizer에 있는 단어 사전을 순회하면서 word2vec의 300차원 vector를 가져옵니다
for word, idx in tokenizer.word_index.items():
    embedding_vector = embedding_model[word] if word in embedding_model else None
    if embedding_vector is not None:
        embedding_matrix[idx] = embedding_vector

embedding_matrix.shape



'''
#각 기사의 단어에 맞는 벡터값을 가져오기
arr = []; arr2 = []
for i in data['vec_words']:
    for j in range(len(i)):
        arr2.append(list(data_v[data_v['name'] == i[j]].iloc[0, 1:101]))
    arr.append(arr2)
    arr2 = []
X = np.array(arr)


#최대 길이 구하기
max([len(i) for i in data['vec_words']])

#길이 맞추기
from keras.preprocessing import sequence
from keras import backend
dtype = backend.floatx()
X_data = sequence.pad_sequences(X, padding='post', value= 0.0, dtype= dtype) #x data 생성'''

#모델 만들기
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional #패키지 다운
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D
from keras import optimizers

deep_model = Sequential() #기본 토대
deep_model.add(Embedding(VOCAB_SIZE,embedding_dim,input_length=max_length,
                         weights=[embedding_matrix],trainable=False)) # embedding layer에 대한 train은 꼭 false로 지정
deep_model.add(Bidirectional(LSTM(128, return_sequences=False)))
deep_model.add(Dropout(0.2))
deep_model.add(Dense(64))
deep_model.add(Dropout(0.2))
deep_model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4) # 손실 최소화에서 4번째 까지 에포크 돌리기
checkpoint_path='C:/Users/USER/Desktop/news_sentiment_test.h5'
mc = ModelCheckpoint(checkpoint_path, monitor='val_acc', mode='max', verbose=1, save_best_only=True) #정확도 제일 높을때 모델저장
deep_model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['acc'])

with tf.device('/cpu:0'):
    history = deep_model.fit(x_train, y_train,validation_data=(x_valid, y_valid), epochs=20, callbacks=[es, mc], batch_size=32, validation_split=0.1)
    #배치 사이즈, 한번에 32개를 넣어 돌리겠다는 뜻, validation_split 검증데이터 크기

del deep_model

embedding_model.wv.vectors


