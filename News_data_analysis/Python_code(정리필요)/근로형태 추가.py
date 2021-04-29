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
#모델 데이터 불러오기
data50 = pd.read_excel('C:/Users/USER/Desktop/1~8newsurl/data0923.xlsx', index_col=0)#데이터 불러오기
embedding_model = KeyedVectors.load_word2vec_format('C:/Users/USER/Desktop/1~8newsurl/w2v0923') # 모델 로드
data50['words'] = [i.replace("'", "").replace('[', '').replace(']', '').replace(' ', '').split(',') for i in
          list(data50['words'])]  # words가 텍스트 형식으로 되어 있을 경우

#3글자 부터 밎 주요 2단어 포함
data = data50.reset_index(drop=True)
words = list(data['words'])
for k in range(len(words)):
    words[k] = [i for i in words[k] if len(i) > 2 or i in ['배달','택배','고용','취업','실업','채용','구직','청년','유통','해고']]  # 두글자 이하 제외
data['words'] = words
data['words'][:5]
del words; del k; del data50
#data.to_excel('C:/Users/USER/Desktop/1~8newsurl/data0928.xlsx')
def word_score2(word,allnum,num):
    a =[]
    for j in range(len(data)):
        if word in data['words'][j]:
            a.append(data.iloc[j])
    a = pd.DataFrame(a).reset_index(drop=True)
    a['cw'] = word
    a = a[['cw','date','title','text','category','url','words']]
    aa = a['words']
    #a2 = [i.replace("'","").replace('[','').replace(']','').replace(' ','').split(',') for i in aa] #words가 텍스트 형식으로 되어 있을 경우
    a2 = a['words'] #words가 list로 되어 있을 경우
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
    plt.text(0.43,q1,round(q1, 2),fontsize = 9)
    plt.text(0.43,np.median(a['score']),round(np.median(a['score']),2), fontsize = 9)
    plt.text(0.43,q3,round(q3,2),fontsize = 9)
    plt.text(0.4,outlier,round(outlier,2),fontsize = 9)
    plt.show()
    return a
word_score2('소상공인',5,1)
word_score2('취약계층',5,2)
word_score2('의료진',5,3)
word_score2('화상회의',5,4)
word_score2('실업',5,5)
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
    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=texts, start=2,
                                                            limit=20, step=1)

    # Show graph
    limit = 20; start = 2; step = 1;
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
    for i in range(coherence_values.index(max(coherence_values)) + 2):
        words2 = optimal_model.show_topic(i, topn=20)
        topic_dic['topic ' + '{:02d}'.format(i + 1)] = [i[0] for i in words2]
    da = pd.DataFrame(topic_dic)
    return da

###### 이상치 기사 뽑아내기 (박스플랏 연계)
def cut_row2(word):
    # word단어가 포함된 기사만 가져오기
    a =[]
    for j in range(len(data)):
        if word in data['words'][j]:
            a.append(data.iloc[j])
    a = pd.DataFrame(a).reset_index(drop=True)
    a['cw'] = word
    a = a[['cw','date','title','text','category','url','words']]
    aa = a['words']
    #a2 = [i.replace("'","").replace('[','').replace(']','').replace(' ','').split(',') for i in aa] #words가 텍스트 형식으로 되어 있을 경우
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
    print('이상치 ', j, '개 제거 해야함')
    return_data = a.sort_values(by='score')
    return_data.loc[return_data['score'] > outlier,'dummy'] = 1
    return_data['dummy'] = return_data['dummy'].fillna(value=0)
    for i in range(len(return_data)):
        if return_data['dummy'][i] == 0:
            print(return_data['title'][i])
    return return_data
#a = cut_row2('재택근무')

###### 토픽 모델링 심화
'''print('모델 난이도 : ', optimal_model.log_perplexity(corpus)) #모델 난이도 뽑기
coherence_values[1]
print(optimal_model.print_topics(20))


lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=20,
                                        random_state=100,
                                        update_every=1,
                                        chunksize=100,
                                        passes=10,
                                        alpha='auto',
                                        per_word_topics=True)'''

def make_topictable_per_doc(ldamodel, corpus):
    topic_table = pd.DataFrame()

    # 몇 번째 문서인지를 의미하는 문서 번호와 해당 문서의 토픽 비중을 한 줄씩 꺼내온다.
    for i, topic_list in enumerate(ldamodel[corpus]):
        doc = topic_list[0] if ldamodel.per_word_topics else topic_list
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
        # 각 문서에 대해서 비중이 높은 토픽순으로 토픽을 정렬한다.
        # EX) 정렬 전 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (10번 토픽, 5%), (12번 토픽, 21.5%),
        # Ex) 정렬 후 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (12번 토픽, 21.5%), (10번 토픽, 5%)
        # 48 > 25 > 21 > 5 순으로 정렬이 된 것.

        # 모든 문서에 대해서 각각 아래를 수행
        for j, (topic_num, prop_topic) in enumerate(doc): #  몇 번 토픽인지와 비중을 나눠서 저장한다.
            if j == 0:  # 정렬을 한 상태이므로 가장 앞에 있는 것이 가장 비중이 높은 토픽
                topic_table = topic_table.append(pd.Series([int(topic_num), round(prop_topic,4), topic_list]), ignore_index=True)
                # 가장 비중이 높은 토픽과, 가장 비중이 높은 토픽의 비중과, 전체 토픽의 비중을 저장한다.
            else:
                break
    return(topic_table)

word = ['비대면','의료진','소상공인','취약계층','재택근무','원격수업','자영업자','배달','화상회의','고용','경제활동','간호사',
         '근로자','실업','금융지원','해고','기본소득','무급휴직','택배','고용유지지원금','노동자','실업자','출퇴근',
         '실업률','구직','일자리','유연근무제','특수고용직','비정규직','프랜차이즈','고용노동부','스타트업','유통',
         '취업','채용','청년','저소득층','실업급여','유통업계','고용안정지원금','공유오피스']

def most_topic(word2):
    def cut_row(word):
        # word단어가 포함된 기사만 가져오기
        a = []
        for j in range(len(data)):
            if word in data['words'][j]:
                a.append(data.iloc[j])
        a = pd.DataFrame(a).reset_index(drop=True)
        a['cw'] = word
        a = a[['cw', 'date', 'title', 'text', 'category', 'url', 'words']]
        a2 = a['words']
        # score 구하기
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

        # 이상치 구하기
        q1 = np.percentile(a['score'], 25)
        q3 = np.percentile(a['score'], 75)
        iqr = q3 - q1
        outlier = q1 - 1.5 * iqr
        print(outlier, ' 값 이하 제거 필요')
        j = 0
        for k in a['score']:
            if k < outlier:
                j += 1
        print('이상치 ', j, '개 제거')
        return_data = a[a['score'] > outlier]
        return return_data
    bb = cut_row(word2)
    news = bb['words']
    time.sleep(0.5)
    # 사전 만들기
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
    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=texts, start=2,
                                                            limit=10, step=1)
    coherence_values.index(max(coherence_values))
    optimal_model = model_list[coherence_values.index(max(coherence_values))]

    def make_topictable_per_doc(ldamodel, corpus):
        topic_table = pd.DataFrame()

        # 몇 번째 문서인지를 의미하는 문서 번호와 해당 문서의 토픽 비중을 한 줄씩 꺼내온다.
        for i, topic_list in enumerate(ldamodel[corpus]):
            doc = topic_list[0] if ldamodel.per_word_topics else topic_list
            doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
            # 각 문서에 대해서 비중이 높은 토픽순으로 토픽을 정렬한다.
            # EX) 정렬 전 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (10번 토픽, 5%), (12번 토픽, 21.5%),
            # Ex) 정렬 후 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (12번 토픽, 21.5%), (10번 토픽, 5%)
            # 48 > 25 > 21 > 5 순으로 정렬이 된 것.

            # 모든 문서에 대해서 각각 아래를 수행
            for j, (topic_num, prop_topic) in enumerate(doc):  # 몇 번 토픽인지와 비중을 나눠서 저장한다.
                if j == 0:  # 정렬을 한 상태이므로 가장 앞에 있는 것이 가장 비중이 높은 토픽
                    topic_table = topic_table.append(pd.Series([int(topic_num), round(prop_topic, 4), topic_list]),
                                                     ignore_index=True)
                    # 가장 비중이 높은 토픽과, 가장 비중이 높은 토픽의 비중과, 전체 토픽의 비중을 저장한다.
                else:
                    break
        return (topic_table)
    topictable = make_topictable_per_doc(optimal_model, corpus)
    topictable = topictable.reset_index() # 문서 번호을 의미하는 열(column)로 사용하기 위해서 인덱스 열을 하나 더 만든다.
    topictable.columns = ['문서 번호', '가장 비중이 높은 토픽', '가장 높은 토픽의 비중', '각 토픽의 비중']
    as2 = pd.concat([bb,topictable], axis= 1)
    return as2

a= most_topic('재택근무')

a.to_excel('C:/Users/USER/Desktop/1~8newsurl/test.xlsx')



asd1 = cut_row2('취약계층')
asd2 = cut_row2('화상회의')
asd3 = cut_row2('언택트')
asd4 = cut_row2('특수고용직')
asd5 = cut_row2('실업')
asd = pd.concat([asd1,asd2,asd3,asd4,asd5])
sns.boxplot(x = 'cw', y = 'score', data = asd)
sns.swarmplot(x = 'cw', y = 'score',data = asd, color='0.25')
plt.show()


#word2vec 시각화
word22 = ['비대면','의료진','소상공인','취약계층','재택근무','원격수업','자영업자','배달','화상회의','고용','경제활동','간호사',
         '근로자','실업','금융지원','해고','기본소득','무급휴직','택배','고용유지지원금','노동자','실업자','출퇴근',
         '실업률','구직','일자리','유연근무제','특수고용직','비정규직','프랜차이즈','고용노동부','스타트업','유통',
         '취업','채용','청년','저소득층','실업급여','고용안정지원금','공유오피스']


match_index = pd.DataFrame(embedding_model.wv.vocab.keys(), columns=['name']) #벡터화된 단어들
word_vector = pd.DataFrame(embedding_model.wv.vectors) #백터화된 단어들의 벡터값
data_v = pd.concat([match_index, word_vector], axis=1)
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(data_v.iloc[:,1:101])
pca = PCA(n_components=2)
pc = pca.fit_transform(x)
pc_data = pd.DataFrame(pc)
pc_data = pd.concat([pc_data,match_index], axis=1)

zx = pd.DataFrame()
for i in pc_data['name']:
    if i in word22:
        zx = pd.concat([zx,pc_data[pc_data['name'] == i]], axis = 0)
plt.scatter(zx.iloc[:,0],zx.iloc[:,1])
zx = zx.reset_index(drop=True)
for i in range(len(zx)):
    plt.text(zx.iloc[:,0][i],zx.iloc[:,1][i],zx.iloc[:,2][i],size = 9)

embedding_model.similarity('택배','모바일')
embedding_model.most_similar('택배')


for i in word:
    date_set = cut_row(i)
    news = data_set['words']
    id2word = corpora.Dictionary(news)
    texts = news
    corpus = [id2word.doc2bow(text) for text in texts]
    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=texts, start=2,
                                                            limit=15, step=1)