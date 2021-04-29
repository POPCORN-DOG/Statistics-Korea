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

#파일 한번에 불러와서 저장하기
'''flist = os.path.join('C:/Users/USER/Desktop/1~8newsurl', '*합본.csv') #끝에 합본.csv로 끝나는 것들 불러오기
colnames = ['date','a1','a2','title','a3','a4','text','user1','user2','user3','user4','user5','user6']
usecols=['date','title','text','user1','user2','user3','user4','user5','user6']
glob.glob(flist)  #파일 list 불러오기

def get_merge_csv(filelist):
    return pd.concat([pd.read_csv(i, encoding='CP949',header=0, usecols=range(13)) for i in filelist], axis=0)

d = get_merge_csv(glob.glob(flist))
d = d.reset_index(drop=True)
d.iloc[:,0:10]
d.columns = colnames
d = d[usecols] #필요한 컬럼만 받기
for i in range(1,7): #라벨링한거 전처리 하기
    d._set_value(d['user'+str(i)] == 'X','user'+str(i),0)._set_value(d['user'+str(i)] == 'x','user'+str(i),0)._set_value(d['user'+str(i)] != 0,'user'+str(i),1)

d['sum'] = d[usecols[3:9]].sum(axis=1)#라벨링 합계 구하기
d['y'] = [1 if i > 3 else 0 for i in d['sum']] #1 과 0

#데이터 저장 하기
d.to_excel('C:/Users/USER/Desktop/labelingdata2018.xlsx', header=True)
data = d
del d#'''

#통합 및 정리본 불러오기
data = pd.read_excel('C:/Users/USER/Desktop/labelingdata2018.xlsx', index_col=0)



####적은 갯수 다시 불러오기
flist = os.path.join('C:/Users/USER/Desktop/새홀리기', '*.csv') #끝에 합본.csv로 끝나는 것들 불러오기
filelist = glob.glob(flist)  #파일 list 불러오기
filename = [i[i.find('\\')+1 : i.find('.csv')] for i in filelist]
a = pd.read_csv(filelist[0], header=0, usecols=range(25))
a = a.reset_index(drop=True)
a['cate'] = filename[0]
a2 = a[['cate','8']]

for i in range(len(filename)):
    globals()['x{}'.format(i)] = pd.read_csv(filelist[i], usecols=range(23), index_col=False, header=None)
    globals()['x{}'.format(i)]['cate'] = filename[i]

urldata = pd.concat([x0,x1,x2,x3,x4,x5], axis=0)
for i in range(6):
    del globals()['x{}'.format(i)]

urllist = urldata[11]
################

data = pd.DataFrame(columns=['date', 'title', 'text', 'category','url'])
j = 0
for i in tqdm(urllist):
    try:
        requrl = req.Request(i, headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(requrl)
        soup = BeautifulSoup(webpage, "html.parser")
        #webpage = urlopen(i)
        #soup = BeautifulSoup(webpage, 'html.parser')

        aa = soup.select('.article_info  h3#articleTitle')
        test_name = aa[0].text  # 제목 가져오기

        aa = soup.select('.article_info .t11')
        aa[0].text[0:10]
        test_date = datetime.datetime.strptime(aa[0].text[0:10], '%Y.%m.%d')  # 날짜 가져오기

        # 본문가져오고 정제하기
        aa = soup.select('#articleBodyContents')
        te = aa[0].text.replace(soup.select('#articleBodyContents script')[0].text, "")
        trash = [i.text for i in soup.select('#articleBodyContents a')]
        for k in trash:
            te = te.replace(k, "")
        text = te.replace('\n', "").replace("  ", "").replace('\t', "").replace('무단 전재 및 재배포 금지', "")

        cate = '경제' if len(re.findall('sid1=101',i)) == 1 else '사회'
        tlist = [test_date, test_name, text, cate,i]
        data.loc[j, :] = tlist
        j = j + 1
    except:
        tlist = [0,0,0,0,0]
        data.loc[j, :] = tlist
        j = j + 1
del i; del j;  #del soup; del test_date; del test_name; del text; del tlist; del trash; del k; del cate;
#본문 전처리
data = data.drop_duplicates() #중복제거
data = data[data['text'] != 0] #0인 값 제거


twi = Twitter()
add_noun = ['비대면','확진자','거리두기','사랑제일교회','의료기관','의료진','집단감염','유연근무','유연근무제','유연근로제','한국판뉴딜','디지털뉴딜',
            '그린뉴딜','휴먼뉴딜','취약계층','긴급재난지원금','고용보험','진료거부','택배기사','선별진료소','고용보험','특수고용직','특고','재배포','고용유지지원금','언택트'
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



urldata['url'] = urldata[11]
urldata = urldata[['url','cate']]


words = list(data['words'])
for k in range(len(words)):
    words[k] = [i for i in words[k] if len(i) > 2 or i in ['배달','택배','고용','취업','실업','채용','구직','청년','유통','해고']]  # 두글자 이하 제외
data['words'] = words

data['length_word'] = [len(i) for i in tqdm(data['words'])]
data50 = data[data['length_word'] >= 50]  #단어 갯수 50개 이하 제거
data50 = pd.merge(data50, urldata, on = 'url')

def topic(word):
    bb2 = data50[data50['cate'] == word]
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

asd = topic('재택근무')

for kkk in filename:
    topic11 = topic(kkk)
    topic11.to_excel('C:/Users/USER/Desktop/1~8newsurl/topicmodeling/'+str(kkk)+'2글자제거.xlsx', header=True)  # 저장하기



