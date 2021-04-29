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
import matplotlib.pyplot as pp
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
okt = Okt()
#driver = webdriver.Chrome('C:/r-selenium/chromedriver.exe')  # 크롬 드라이버 연결

data_url = pd.read_excel('C:/Users/USER/Desktop/1~8newsurl/covid_url38.xlsx')

def sampling_func(data777, sample_pct):
    np.random.seed(123)
    N = len(data777)
    sample_n = int(len(data777)*sample_pct) # integer
    sample = data777.take(np.random.permutation(N)[:sample_n])
    return sample
data_url = data_url.groupby('RGSDE',group_keys=False).apply(sampling_func,sample_pct=0.2) #일자별 샘플링

data_url.groupby(data_url['RGSDE']).size()


urllist = data_url['DTA_URL']
################

data = pd.DataFrame(columns=['date', 'title', 'text', 'category','url'])
j = 0
for i in tqdm(urllist):
    try:
        webpage = urlopen(i)
        soup = BeautifulSoup(webpage, 'html.parser')

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
del i; del j; del te; del soup; del test_date; del test_name; del text; del tlist; del trash; del k; del cate;
#본문 전처리
data = data.drop_duplicates() #중복제거
data = data[data['text'] != 0] #0인 값 제거

#형태소 분석
list_text = data['text']

words = [okt.nouns(i) for i in tqdm(list(list_text))] #시간 걸림

Prohibit_words = ['기자','연합뉴스','뉴시스','시사저널','신문','뉴스','사진','헤럴드경제','노컷뉴스','파이낸셜뉴스','특파원',
                  '라며','대해','지난','위해','오전','오후','무단','배포','이데일리','머니투데이','앵커',]

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
data.to_excel('C:/Users/USER/Desktop/1~8newsurl/covid_url38_word.xlsx',header=True)

data50 = data[data['length_word'] >= 50]  #단어 갯수 50개 이하 제거
data50 = pd.read_excel('C:/Users/USER/Desktop/1~8newsurl/covid_url38_word50.xlsx', index_col=0)

embedding_model = Word2Vec(list(data50['words']), size=100, window = 6, min_count=150, workers=4, iter=100)
embedding_model.wv.save_word2vec_format('C:/Users/USER/Desktop/백업/covid_model38')# 모델 저장
embedding_model = KeyedVectors.load_word2vec_format('C:/Users/USER/Desktop/백업/covid_model38') # 모델 로드
del aa; del i; del j; del k;

#단어 빈도 검색

index1 = [i for i in range(len(data))]
data.index = index1
def top30(keyword):
    x=[]
    for i in range(len(words)):
        x.append([words[i],data['category'][i]])
    x1 = [i[0] for i in x]
    x2 = []
    for i in x1:
        for k in i:
            x2.append(k)
    print(pd.Series(x2).value_counts().head(100))
    x3 = pd.Series(x2).value_counts().head(3000)
    x4 = pd.DataFrame(x3)
    x4['ca'] = keyword
    x4['wor'] = x4.index
    x4['score'] = x4[0] / x4[0].sum(axis=0)
    return x4
bin = top30('경제')
bin.to_excel('C:/Users/USER/Desktop/1~8newsurl/bindo.xlsx',header=True)


print(embedding_model.most_similar(positive=["스타벅스"], topn=100)) #관련단어 검색
print(embedding_model.similarity('일자리','시장'))

# 형태소 분석 수정
'''okt.nouns('소상공인')
okt.nouns('확진자')
okt.nouns('유연근무제, 유연근무,')#'''

#점수내기
words = list(data50['words'])

word_list = ['고용','일자리','채용','청년','취업','배달','유통','근로자','재택근무','노동자','뉴딜','제조업','기본소득','노동','간호사','최저임금','실업','택배','제조','택시',
             '생계','스타트업','근로','구직','파업','고용노동부','해고','폐지','인턴','제조업체','노동조합','실업률','비정규직','실업자','생활비','소매','출퇴근']

def word2vec_kkh2(word_list):
    xx_n = word_list
    avg_dist = []
    dist = []
    dist_dist = []
    for i in tqdm(range(len(words))):
        for k in words[i]:
            try:
                for m in xx_n:
                    # 비교하여 similarity 구하기
                    dist_dist.append(embedding_model.similarity(m, k))
            except:
                dist_dist.append(0)
            dist2 = np.array(dist_dist)
            dist.append(sum(dist2))
            dist_dist = []
        if len(dist) > 1000:
            dist = sorted(dist, reverse=True)[:1000]
        avg_dist.append(np.mean(dist))
        dist = []
    return avg_dist

data50['score'] = word2vec_kkh2(word_list)
data50.to_excel('C:/Users/USER/Desktop/1~8newsurl/covid_url38_word50.xlsx',header=True)

#점수 상위 2천개 일자리 관련 기사라고 라벨링 2천등 점수 1.51505
#data50 = data50.sort_values('score', ascending=False)
#data50['rank'] = [i+1 for i in range(len(data50))]
#data50['y'] = [1 if i <= 2000 else 0 for i in data50['rank']]


#기사 벡터화 하기
match_index = list(embedding_model.wv.vocab.keys()) #벡터화된 단어들
embedding_model.get_vector('연합') #벡터값 찾기
word_vector = list(embedding_model.wv.vectors) #백터화된 단어들의 벡터값
vecdata = pd.DataFrame(columns=['x{}'.format(i) for i in range(100)])
for i in tqdm(range(len(words))):
    vec = pd.DataFrame(columns = ['x{}'.format(i) for i in range(100)])
    j = 0
    for w in match_index:
        if w in words[i]:
            vec.loc[j] = list(word_vector[match_index.index(w)])
            j += 1
        score_vec = vec.mean()
    vecdata.loc[i] = score_vec
vecdata.to_excel('C:/Users/USER/Desktop/1~8newsurl/covid_url38_word_vecdata.xlsx',header=True)
###########################랭크 다시 하고 다시 정렬

data50.index = [i for i in range(len(data50))]
data50_all = pd.concat([data50,vecdata],axis=1)
data50_all['rank'] = data50_all['score'].rank(ascending=False)
data50_all['y'] = [1 if i <= 2000 else 0 for i in data50_all['rank']]
#data50_all.to_excel('C:/Users/USER/Desktop/1~8newsurl/covid_url38_word50_all.xlsx',header=True)
data50_all = pd.read_excel('C:/Users/USER/Desktop/1~8newsurl/covid_url38_word50_all.xlsx', index_col=0)
#data50_all[data50_all['y']== 1].to_excel('C:/Users/USER/Desktop/1~8newsurl/covid_url38_word50_all_job.xlsx',header=True)
###knn써보기
x = data50_all.iloc[:,8:108]
y = data50_all.iloc[:,109]
x_train, x_test, y_train, y_test = skl.model_selection.train_test_split(x, y,random_state=100,stratify=y) #stratify = y y를 동일한 비율로 나누어 준다.
#정규화시키는 함수
scaler = skl.preprocessing.StandardScaler()

#x_train 정규화
scaler.fit(x_train) #x_train 데이터에 대해서 평균과 표준편차 계산
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
###########knn사용해서 분류해보기

classifier = KNeighborsClassifier(n_neighbors= 2)
classifier.fit(x_train,y_train)
knn_pre = classifier.predict(x_test)

#성능 평가
print(confusion_matrix(y_test,knn_pre))

#평가결과 확인
print(classification_report(y_test,knn_pre)) #'''
print(classifier.score(x_test, y_test))

##smote 사용해보기
sm = SMOTE()
x_smote, y_smote = sm.fit_sample(x_train,y_train)
scaler = skl.preprocessing.StandardScaler()

#x_train 정규화
scaler.fit(x_smote) #x_train 데이터에 대해서 평균과 표준편차 계산
x_smote = scaler.transform(x_smote)
x_test = scaler.transform(x_test)

classifier = KNeighborsClassifier(n_neighbors= 2)
classifier.fit(x_smote,y_smote)
knn_pre = classifier.predict(x_test)

#성능 평가
print(confusion_matrix(y_test,knn_pre))

#평가결과 확인
print(classification_report(y_test,knn_pre))
print(classifier.score(x_test, y_test)) #'''

y_test2 = list(y_test)
ckeck = pd.DataFrame(y_test)
ckeck['a'] = list(ckeck.index); ckeck['prey'] = list(knn_pre)
t_l = [data50_all['title'][i] for i in ckeck['a']]
ckeck['title'] = t_l
ckeck2 = ckeck.loc[(ckeck['prey'] == 0) & (ckeck['y'] == 1),:]

data = data[0:10]





