#필요패키지
import pandas as pd
import numpy as np
from urllib.request import urlopen #url의 html 을 가져 오기 위한 패키지
from bs4 import BeautifulSoup  #크롤링 필수 패키지 설치하려면 cmd창에서 pip install bs4
import os
import re
from selenium import webdriver #셀레니움 도구
from bs4 import BeautifulSoup #크롤링 도구
from selenium.webdriver.common.keys import Keys #셀레니움 도구
import time #sleep 관련 패키지
from tqdm import tqdm #진행상황

driver = webdriver.Chrome('C:/r-selenium/chromedriver.exe')  # 크롬 드라이버 연결

#크롤링 함수 만들기
def news_kkh(url):
    urllist = []
    for i in tqdm(range(1,1000)):
        try:
            driver.get( url + '#&date=%2000:00:00&page=' + str(i))#정치 url 이동
            time.sleep(0.4)
            for j in range(1,5):
                for k in range(1,6):
                    try :
                        urllist.append(driver.find_element_by_xpath(
                            '//*[@id="section_body"]/ul[' + str(j) + ']/li[' + str(k) + ']/dl/dt[2]/a').get_attribute('href'))
                    except :
                        urllist.append(driver.find_element_by_xpath(
                                '//*[@id="section_body"]/ul[' + str(j) + ']/li[' + str(k) + ']/dl/dt/a').get_attribute('href'))
        except:
            return urllist
            break

#통합 url 만들기
#정치
j_urllist = news_kkh('https://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1=100')
#경제
e_urllist = news_kkh('https://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1=101')
#사회
s_urllist = news_kkh('https://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1=102')
#생활/문화
l_urllist = news_kkh('https://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1=103')
#세계
g_urllist = news_kkh('https://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1=104')

ju = [[i] + ["정치"] for i in j_urllist]
eu = [[i] + ["경제"] for i in e_urllist]
su = [[i] + ["사회"] for i in s_urllist]
lu = [[i] + ["생활/문화"] for i in l_urllist]
gu = [[i] + ["세계"] for i in g_urllist]

urllist = ju + eu + su + lu + gu

# url 저장 url 2020-08-04 부터 2020-08-11 까지 정치 및 5분야에서 뉴스 url 수집 24667개 수집
import json
with open('C:/Users/USER/Desktop/백업/newsurllist.json', 'w') as make_file:
    json.dump(urllist, make_file)
with open('C:/Users/USER/Desktop/백업/newsurllist.json') as f:
    urllist = json.load(f)

#네이버 랭킹뉴스 파싱하기
import datetime
#url list에서 가져오기  2시간 정도 걸림
data = pd.DataFrame(columns=['date', 'title', 'text', 'category','url'])
j = 0
for i in tqdm(urllist):
    try:
        webpage = urlopen(i[0])
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

        tlist = [test_date, test_name, text, i[1],i[0]]
        data.loc[j, :] = tlist
        j = j + 1
    except:
        tlist = [0,0,0,0,0]
        data.loc[j, :] = tlist
        j = j + 1


#본문 전처리
data = data.drop_duplicates() #중복제거
for i in range(len(data)):
    qw = data.iloc[i,2]
    data.iloc[i,2] = qw.replace('무단전재', "").replace('재배포', "")
del qw
data.to_csv('C:/Users/USER/Desktop/백업/news_url1.csv',header=False, encoding='utf-8-sig') #엑셀에서 데이터 없는건 지움
data = pd.read_csv('C:/Users/USER/Desktop/백업/news_url.csv', encoding='cp949')


###형태소 분석
from konlpy.tag import Okt
okt = Okt()

wor = data.loc[:,'text']
word = list(wor)
words = [okt.nouns(i) for i in tqdm(word)] #시간 걸림

#배제어 만들고 한글자 제거
Prohibit_words = ['기자','연합뉴스','뉴시스','시사저널','신문','뉴스','사진','헤럴드경제','노컷뉴스','파이낸셜뉴스','특파원']
j = 0
for i in tqdm(words):
    for k in Prohibit_words:
        while k in i:
            i.remove(k)
    words[j] = i
    j += 1

for k in range(len(words)):
    words[k] = [i for i in words[k] if len(i) > 1]  # 한글자 제외

# 본문 형태소 분석 저장
import json
with open('C:/Users/USER/Desktop/백업/text.json', 'w') as make_file:
    json.dump(words, make_file)

#검색 함수 만들기
def search_kkh(word):
    j = 0
    for i in range(len(data)):
        if word in data.iloc[i,2]:
            j += 1
    print('뉴스제목에 ' + str(word) + "(이)가 포함된 기사는 " + str(j) +"개입니다.")
    k = 0
    for i in range(len(data)):
        if word in data.iloc[i,3]:
            k += 1
    print('뉴스본문에 ' + str(word) + "(이)가 포함된 기사는 " + str(k) +"개입니다.")
    m = 0
    for i in range(len(data)):
        if word in data.iloc[i,2]:
            if word in data.iloc[i,3]:
                m += 1
    print('뉴스 본문과 제목에 ' + str(word) + "(이)가 포함된 기사는 " + str(m) +"개입니다.")
search_kkh('유연근무제')
data.iloc[2,2]
# Word2Vec embedding
from gensim.models import Word2Vec
embedding_model = Word2Vec(words, size=100, window = 6, min_count=100, workers=4, iter=100)

# check embedding result
print(embedding_model.most_similar(positive=["경제"], topn=15))

#모델 저장 및 불러오기
from gensim.models import KeyedVectors
embedding_model.wv.save_word2vec_format('C:/Users/USER/Desktop/백업/news_w2v') # 모델 저장
embedding_model = KeyedVectors.load_word2vec_format('C:/Users/USER/Desktop/백업/news_w2v') # 모델 로드

#참조 코드
'''#기사 제목 만 가져오기
title_word = list(data.loc[:,'title'])

avg_dist = []
dist = []
for i in tqdm(range(len(words))):
    for k in words[i]:
        try:
            dist.append(embedding_model.similarity('일자리',k)) #연기력과 비교하여 similarity 구하기
        except:
            continue
    if len(dist) > 30:
        dist = sorted(dist, reverse=True)[:30] #게시물이 길어질수록 낮게 나오는 경향을 없어기 위함
    np.mean(dist)
    avg_dist.append([title_word[i],np.mean(dist)])
    dist = []


score = [avg_dist[i][1] for i in range(len(avg_dist))]
np.max(score)
#분포보기
import matplotlib.pyplot as plt
plt.hist(score)
#일자리에 관련된 리뷰 뽑기
score_20 = [i for i in score if i > 0.5]
avg_dist[score.index(score_20[0])][0]
rev = [avg_dist[score.index(score_20[i])][0] for i in range(len(score_20))]'''

####################################word2vec 검증 하기 분류를 잘 나눌 수 있는지
#분야별 빈도 알아보기


#정치
print(embedding_model.most_similar(positive=["정당"], topn=15))
category = list(data.loc[:,'category'])
avg_dist = []
dist = []
for i in tqdm(range(len(words))):
    for k in words[i]:
        try:
            dist.append(embedding_model.similarity('정당',k)) #연기력과 비교하여 similarity 구하기
        except:
            continue
    if len(dist) > 30:
        dist = sorted(dist, reverse=True)[:30] #게시물이 길어질수록 낮게 나오는 경향을 없어기 위함
    np.mean(dist)
    avg_dist.append([category[i],np.mean(dist)])
    dist = []

score = [avg_dist[i][1] for i in range(len(avg_dist))]
np.max(score)
plt.hist(score)
#정치에 관련된 리뷰 뽑기
score_20 = [i for i in score if i > 0.35]
avg_dist[score.index(score_20[0])][0]
rev = [avg_dist[score.index(score_20[i])][0] for i in range(len(score_20))]

pd.Series(rev).value_counts()



#내가 만든 분류기
import matplotlib.pyplot as pp
import sklearn.metrics as metrics
import pylab as pl
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

category = list(data.loc[:,'category'])
xx = embedding_model.most_similar(positive=["정당"], topn=4)
xx_0 = [i[1] for i in xx]
xx_0.append(1)
xx_l = np.array(xx_0)
xx_n = [i[0] for i in xx]
xx_n.append('정당')
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
        sc = dist2 * xx_l
        dist.append(sum(sc))
        dist_dist = []
    if len(dist) > 30:
        dist = sorted(dist, reverse=True)[:30]
    np.mean(dist)
    avg_dist.append([category[i],np.mean(dist)])
    dist = []

hist = [i[1] for i in avg_dist]
plt.hist(hist)
np.max(hist)
name = [ 1 if i[0] == "정치" else 0 for i in avg_dist]
hist01 = [1 if i > 0.78 else 0 for i in hist]
Value = np.array(name)
Probability = np.array(hist)
fpr, tpr,thresholds = metrics.roc_curve(Value, Probability)


modelAROCAUC = metrics.auc(fpr, tpr) #정확도
pp.xlabel("False Positive Rate(1 - Specificity)")
pp.ylabel("True Positive Rate(Sensitivity)")
pp.plot(fpr, tpr, "b", label="Model A (AUC = %0.2f)" % modelAROCAUC)
pp.plot([0, 1], [1, 1], "y--")
pp.plot([0, 1], [0, 1], "r--")
pp.legend(loc="lower right")
pp.show()

i = np.arange(len(tpr))
fpr, tpr,thresholds = metrics.roc_curve(Value, Probability)
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i),
                    'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
a=roc.iloc[(roc.tf-0).abs().argsort()[:1]].iloc[0,4] #최적 점 1.855868
a

#최적점으로 분류 결과 보기
hist01 = [1 if i > a else 0 for i in hist]
confusion_matrix(name, hist01,labels=[1,0])
print(classification_report(name,hist01))

# 한줄요약
# 장점 : 정치관련된 기사를 잘 뽑아내고 특히 정치와 관련없는 기사를 구별하는 성능이 뛰어나다
# 단점 : 사회, 경제도 정치와 직·간접적인 영향이 있어 사회, 경제 기사도 많이 포함한다.





####################### 최적 분류 단어 찾아내기
def top30(keyword):
    x=[]
    for i in range(len(words)):
        x.append([words[i],data['category'][i]])
    x1 = [i[0] for i in x if i[1] == keyword]
    x2 = []
    for i in x1:
        for k in i:
            x2.append(k)
    print(pd.Series(x2).value_counts().head(30))
    x3 = pd.Series(x2).value_counts().head(30)
    x4 = pd.DataFrame(x3)
    x4['ca'] = keyword
    x4['wor'] = x4.index
    x4['score'] = x4[0] / x4[0].sum(axis=0)
    return x4
j_top = top30('정치')
e_top = top30('경제')
s_top = top30('사회')
l_top = top30('생활/문화')
g_top = top30('세계')

ttt = pd.concat([j_top,e_top,s_top,l_top,g_top],axis=0)

ttt.loc[ttt['ca'] == "정치",['wor','score']].iloc[0,0]
ttt.loc[ttt['ca'] != "정치",['wor','score']].iloc[119,1]
def cor(keyword):
    val_list = []; val_l = []
    for j in tqdm(range(30)):
        val = 0
        for i in range(120):
            a = embedding_model.similarity(ttt.loc[ttt['ca'] == keyword,['wor','score']].iloc[j,0],
                                           ttt.loc[ttt['ca'] != keyword,['wor','score']].iloc[i,0])*ttt.loc[ttt['ca'] != keyword,['wor','score']].iloc[i,1]
            val += a
        val_l.append(val)
        val_list.append([ttt.loc[ttt['ca'] == keyword,['wor','score']].iloc[j,0],val])
    val_list[val_l.index(min(val_l))]
    print(val_list[val_l.index(min(val_l))])
    return val_list
cor('사회')


####################신경망에 필요한 통합 데이터셋 만들기
#기사 벡터화 시키기
word_vector = embedding_model.wv.vectors
match_index = embedding_model.wv.index2word
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
vecdata['score'] = data['score']

# 원하는 등수 만큼 자르기
cut_off = vecdata.loc[vecdata['score'].rank(ascending=False) == 301,:]['score'].values[0]
vecdata['y'] = [1 if i > cut_off  else 0 for i in vecdata['score']]
vecdata.loc[vecdata["y"] == 1,:]

#스코어링 데이터셋 만들기(내가 직접 1,0 나누기한거 가져오기 )
sc = pd.read_excel('C:/Users/USER/Desktop/scoring.xlsx')
sc = sc[["a","y"]]
scoring = pd.merge(data,sc,on = "a",how='left') #왼쪽을 기준으로 데이터 합치기
scoring = scoring.fillna(value=0) #nan 값을 0으로 만들어 준다
scoring.loc[scoring['y'] == 1, :]
del sc
vecdata['y'] = scoring['y']

###scoring 수정
scoring.to_excel('C:/Users/USER/Desktop/scoreing수정.xlsx', index=False)



#변수 이름붙이기
names = ["a"]; x_list = ['x{}'.format(i) for i in range(100)]
for i in range(100):
    names.append(x_list[i])
names.append('score'); names.append('y')
vecdata.to_csv('C:/Users/USER/Desktop/vector_data.csv', header=True, index=False)
vecdata = pd.read_csv("C:/Users/USER/Desktop/vector_data.csv") #내가 라벨링 한거
del names; del x_list

#데이터 합치기
data_all = pd.merge(data,vecdata,on = "a",how='left') #왼쪽을 기준으로 데이터 합치기
data_all.to_csv('C:/Users/USER/Desktop/data_all.csv', header=True, index=False)
data_all = pd.read_csv('C:/Users/USER/Desktop/data_all.csv')

#knn 분류
###knn써보기
x = data_all.iloc[:,6:106]
y = data_all.iloc[:,107]
x_train, x_test, y_train, y_test = skl.model_selection.train_test_split(x, y,random_state=100,stratify=y,test_size = 0) #stratify = y y를 동일한 비율로 나누어 준다.
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



