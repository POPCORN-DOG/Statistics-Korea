#필요패키지
import pandas as pd
import numpy as np
import os
import re
import time
from tqdm import tqdm
import json
import datetime
from konlpy.tag import Okt
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import matplotlib.pyplot as pp
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as pp
import numpy as np
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
d_all = pd.read_csv('C:/Users/USER/Desktop/data_all.csv')
okt = Okt()
data = pd.read_excel('C:/Users/USER/Desktop/news_url.xlsx', header = None,names=["a",'date','title','text','category','url'])
#파일 모델 가져오기
#data = pd.read_csv('C:/Users/USER/Desktop/백업/news_url.csv', encoding='euc-kr', header = None,names=['date','title','text','category','url']) #오류남
with open('C:/Users/USER/Desktop/백업/text.json') as f:
    words = json.load(f) #형태소 분석 결과 로드
del f
embedding_model = KeyedVectors.load_word2vec_format('C:/Users/USER/Desktop/백업/news_w2v') # 모델 로드
#배제어를 추가하여 다시 모델링을 할경우
'''Prohibit_words = ['기자','연합뉴스','뉴시스','시사저널','신문','뉴스','사진','헤럴드경제','노컷뉴스','파이낸셜뉴스','특파원',
                  '라며','대해','지난','위해','오전','오후','포토','동아일보','한겨레','오마이','조선일보','경향신문','머니투데이',
                  '한경닷컴',]
j = 0
for i in tqdm(words):
    for k in Prohibit_words:
        while k in i:
            i.remove(k)
    words[j] = i
    j += 1
# 본문 형태소 분석 저장
with open('C:/Users/USER/Desktop/백업/text.json', 'w') as make_file:
    json.dump(words, make_file)
    
embedding_model = Word2Vec(words, size=200, window = 6, min_count=100, workers=4, iter=100)
embedding_model.wv.save_word2vec_format('C:/Users/USER/Desktop/백업/news_w2v') # 모델 저장'''
#스코어링 데이터셋 만들기
def word2vec_kkh(cate,key_word):
    category = list(data.loc[:, 'category'])
    xx = embedding_model.most_similar(positive=[key_word], topn=2)
    xx_0 = [i[1] for i in xx]
    xx_0.append(1)
    xx_l = np.array(xx_0)
    xx_n = [i[0] for i in xx]
    xx_n.append(key_word)
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
        if len(dist) > 300:
            dist = sorted(dist, reverse=True)[:300]
        np.mean(dist)
        avg_dist.append([category[i], np.mean(dist)])
        dist = []

    hist = [i[1] for i in avg_dist]
    name = [1 if i[0] == cate else 0 for i in avg_dist]
    Value = np.array(name)
    Probability = np.array(hist)
    fpr, tpr, thresholds = metrics.roc_curve(Value, Probability)

    modelAROCAUC = metrics.auc(fpr, tpr)  # 정확도
    pp.xlabel("False Positive Rate(1 - Specificity)")
    pp.ylabel("True Positive Rate(Sensitivity)")
    pp.plot(fpr, tpr, "r", label="word2vec (AUC = %0.2f)" % modelAROCAUC)
    pp.plot([0, 1], [1, 1], "y--")
    pp.plot([0, 1], [0, 1], "r--")
    pp.legend(loc="lower right")
    pp.show()

    i = np.arange(len(tpr))
    roc = pd.DataFrame(
        {'fpr': pd.Series(fpr, index=i), 'tpr': pd.Series(tpr, index=i), '1-fpr': pd.Series(1 - fpr, index=i),
         'tf': pd.Series(tpr - (1 - fpr), index=i), 'thresholds': pd.Series(thresholds, index=i)})
    a = roc.iloc[(roc.tf - 0).abs().argsort()[:1]].iloc[0, 4]  # 최적 점

    # 최적점으로 분류 결과 보기
    hist01 = [1 if i > a else 0 for i in hist]
    print('정확도는 {0:.2f} 입니다.'.format(modelAROCAUC))
    print(confusion_matrix(name, hist01,labels=[1,0]))
    print(classification_report(name, hist01,labels=[1,0],target_names=[str(cate),str(cate) + "외"]))
    print(modelAROCAUC)
    return modelAROCAUC
def search_kkh(cate,key_word):
    category = list(data.loc[:, 'category'])
    name = [1 if i == cate else 0 for i in category]
    title_name = [1 if key_word in data.iloc[i,2] else 0 for i in range(len(data))]
    text_name = [1 if key_word in data.iloc[i,3] else 0 for i in range(len(data))]
    Value = np.array(name)
    Probability = np.array(title_name)
    fpr, tpr,thresholds = metrics.roc_curve(Value, Probability)
    modelAROCAUC = metrics.auc(fpr, tpr)  # 정확도
    print('제목에 {0}을(를) 검색했을시 정확도는 {1:.2f} 입니다.'.format(key_word,modelAROCAUC))
    print(confusion_matrix(name, title_name,labels=[1,0]))
    print(classification_report(name, title_name,labels=[1,0],target_names=[str(cate),str(cate) + "외"]))

    Value = np.array(name)
    Probability = np.array(text_name)
    fpr, tpr,thresholds = metrics.roc_curve(Value, Probability)
    modelAROCAUC = metrics.auc(fpr, tpr)  # 정확도
    pp.xlabel("False Positive Rate(1 - Specificity)")
    pp.ylabel("True Positive Rate(Sensitivity)")
    pp.plot(fpr, tpr, "b", label="wordsearch (AUC = %0.2f)" % modelAROCAUC)
    pp.plot([0, 1], [1, 1], "y--")
    pp.plot([0, 1], [0, 1], "r--")
    pp.legend(loc="lower right")
    pp.show()
    print('본문에 {0}을(를) 검색했을시 정확도는 {1:.2f} 입니다.'.format(key_word,modelAROCAUC))
    print(confusion_matrix(name, text_name,labels=[1,0]))
    print(classification_report(name, text_name,labels=[1,0],target_names=[str(cate),str(cate) + "외"]))
    aa12 =  confusion_matrix(name, text_name,labels=[1,0])
    print('본문에 {0}을(를) 검색했을시 민감도는 {1:.2f} 입니다.'.format(key_word,aa12[0,0] / (aa12[0,0] + aa12[0,1])))
    return modelAROCAUC

search_kkh('세계','미국')
word2vec_kkh('세계','미국')
print(embedding_model.most_similar(positive=["악순환"], topn=10)) #관련단어 검색
print(embedding_model.similarity('근무','재택근무'))
okt.nouns('유연근무제')
##################일자리 골라 내는거 보기
'''print(embedding_model.most_similar(positive=["일자리"], topn=15))
#기사 제목 만 가져오기
xx = embedding_model.most_similar(positive=['일자리'], topn=2)
xx_0 = [i[1] for i in xx]
xx_0.append(1)
xx_l = np.array(xx_0)
xx_n = [i[0] for i in xx]
xx_n.append('일자리')
avg_dist = []; dist = []; dist_dist = []
for i in tqdm(range(len(words))):
    for k in words[i]:
        for m in xx_n:
            try:
                # 비교하여 similarity 구하기
                dist_dist.append(embedding_model.similarity(m, k))
            except:
                dist_dist.append(0)
        dist2 = np.array(dist_dist)
        sc = dist2 * xx_l
        dist.append(sum(sc))
        dist_dist = []
    if len(dist) > 300:
        dist = sorted(dist, reverse=True)[:300]
    np.mean(dist)
    avg_dist.append(np.mean(dist))
    dist = []
score = [avg_dist[i] for i in range(len(avg_dist))]
del xx; del xx_n; del xx_l; del xx_0; del dist; del avg_dist; del dist_dist; del dist2; del sc; del i; del k ; del m
np.max(score)
#분포보기
#plt.hist(score)
#일자리에 관련된 리뷰 뽑기
data1 = data
data1['score'] = score
data1['rank'] = data1['score'].rank(ascending=False)
data1 = data1.loc[data1['rank'] < 1000,:]
data1.to_csv('C:/Users/USER/Desktop/백업/tes33.csv',header=False, encoding='utf-8-sig')

####일반검색법 일자리 골라내기
sea = pd.DataFrame([data.iloc[i] for i in range(len(data)) if '일자리' in data['text'][i]])
sea.to_csv('C:/Users/USER/Desktop/백업/sea.csv',header=False, encoding='utf-8-sig') #'''


#스코어링 데이터셋 만들기
'''sc = pd.read_excel('C:/Users/USER/Desktop/scoring.xlsx')
sc = sc[["a","y"]]
scoring = pd.merge(data,sc,on = "a",how='left') #왼쪽을 기준으로 데이터 합치기
scoring = scoring.fillna(value=0) #nan 값을 0으로 만들어 준다
scoring.loc[scoring['y'] == 1, :]
del sc
#
match_index = list(embedding_model.wv.vocab.keys()) #벡터화된 단어들
embedding_model.get_vector('연합') #벡터값 찾기
word_vector = list(embedding_model.wv.vectors) #백터화된 단어들의 벡터값
vecdata = pd.DataFrame(columns=['x{}'.format(i) for i in range(100)])  ######y 를 score로 바꾸고 y2를 y로 변환해야함
for i in tqdm(range(len(words))):
    vec = pd.DataFrame(columns = ['x{}'.format(i) for i in range(100)])
    j = 0
    for w in match_index:
        if w in words[i]:
            vec.loc[j] = list(word_vector[match_index.index(w)])
            j += 1
        score_vec = vec.mean()
    vecdata.loc[i] = score_vec
vecdata['score'] = data1['score']
cut_off = vecdata.loc[vecdata['score'].rank(ascending=False) == 301,:]['score'].values[0]
vecdata['y'] = [1 if i > cut_off  else 0 for i in vecdata['score']]
vecdata.loc[vecdata["y"] == 1,:]
#변수 이름붙이기
names = ["a"]; x_list = ['x{}'.format(i) for i in range(100)]
for i in range(100):
    names.append(x_list[i])
names.append('score'); names.append('y')
vecdata = pd.read_csv("C:/Users/USER/Desktop/백업/vector_data.csv", header= None, names=names)
del names; del x_list #'''

#scoring data 수정
scoring_2 = pd.read_excel('C:/Users/USER/Desktop/scoreing수정.xlsx')

'''#벡터데이터 가져오기
vecdata = pd.read_csv("C:/Users/USER/Desktop/vector_data.csv")

############내가 직접 라벨링한 결과 보기
vecdata['y'] = scoring['y']
vecdata['y'] = scoring_2['y']'''

#test train set 나누기
x = d_all.iloc[:,6:106]
y = d_all.iloc[:,107]
x_train, x_test, y_train, y_test = skl.model_selection.train_test_split(x, y,random_state=20)
#정규화시키는 함수
scaler = skl.preprocessing.StandardScaler()

#x_train 정규화
scaler.fit(x_train) #x_train 데이터에 대해서 평균과 표준편차 계산
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

##################인공신경망 사용
'''
# 알고리즘 불러오기 및 hidden layer할당

mlp = MLPClassifier(hidden_layer_sizes=(10,10,10,10,10)) #3개의 은닉층을 만들고 10개의 노드씩 할당
mlp.fit(x_train,y_train) #mlp를 이용해 train data 학습

#예측한 x_test를 저장
prediction = mlp.predict(x_test)

#성능 평가

print(confusion_matrix(y_test,prediction))

#평가결과 확인
print(classification_report(y_test,prediction))

 #잘못선택된거 확인
y_test2 = list(y_test)
ckeck = pd.DataFrame(y_test)
ckeck['a'] = list(ckeck.index)
ckeck['prey'] = list(prediction)
t_l = [data['title'][i] for i in ckeck['a']]
ckeck['title'] = t_l
ckeck2 = ckeck.loc[ckeck['y'] != ckeck['prey'],:]
##########비율 맞춰서 해보기
cut_off = vecdata.loc[vecdata["y"].rank(ascending=True) == 10001,:]["y"].values[0]
vecdata['y2'] = [2 if vecdata['y'][i] < cut_off  else vecdata['y2'][i] for i in range(len(vecdata['y']))]
vecdata['y2'].value_counts()
baldata = vecdata.loc[vecdata['y2'] >= 1,:]
baldata['y2'] = [0 if i == 2 else 1 for i in baldata['y2']]
baldata['y2'].value_counts()


x = baldata.iloc[:,0:100]
y = baldata.iloc[:,101]
x_train, x_test, y_train, y_test = skl.model_selection.train_test_split(x, y)
#정규화시키는 함수
scaler = skl.preprocessing.StandardScaler()

#x_train 정규화
scaler.fit(x_train) #x_train 데이터에 대해서 평균과 표준편차 계산
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 알고리즘 불러오기 및 hidden layer할당
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10,10,10)) #3개의 은닉층을 만들고 10개의 노드씩 할당
mlp.fit(x_train,y_train) #mlp를 이용해 train data 학습

#예측한 x_test를 저장
prediction = mlp.predict(x_test)

#성능 평가
print(confusion_matrix(y_test,prediction))

#평가결과 확인
print(classification_report(y_test,prediction)) #'''

###########knn사용해서 분류해보기

classifier = KNeighborsClassifier(n_neighbors= 2)
classifier.fit(x_train,y_train)
knn_pre = classifier.predict(x_test)

#성능 평가
print(confusion_matrix(y_test,knn_pre))

#평가결과 확인
print(classification_report(y_test,knn_pre)) #'''
print(classifier.score(x_test, y_test))


#잘못선택된거 확인
y_test2 = list(y_test)
ckeck = pd.DataFrame(y_test)
ckeck['a'] = list(ckeck.index); ckeck['prey'] = list(knn_pre)
t_l = [d_all['title'][i] for i in ckeck['a']]
ckeck['title'] = t_l
ckeck['score'] = [d_all['score'][i] for i in ckeck['a']]
ckeck2 = ckeck.loc[ckeck['prey'] == 1,:]
ckeck3 = ckeck.loc[ckeck['prey'] == 0,:]
ckeck4 = ckeck3.loc[ckeck3['y'] == 1,:]
# 15000개로 줄여서 해보기
e = vecdata.loc[vecdata['y'] == 1,:]
sam = random.sample(list(vecdata.loc[vecdata['y'] == 0,'a']),14785)
vecdata.loc[2,:]
v = pd.DataFrame([vecdata.loc[i,:] for i in sam])
ve = pd.concat([v,e], axis=0)
#test train set 나누기
x = ve.iloc[:,1:101]
y = ve.iloc[:,102]
x_train, x_test, y_train, y_test = skl.model_selection.train_test_split(x, y,random_state=100)
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

##여러 분석방법 해보기
'''
###########svm 사용해보기
from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)
pre_svm = clf.predict(x_test)
#성능 평가
print(confusion_matrix(y_test,pre_svm))
#평가결과 확인
print(classification_report(y_test,pre_svm)) 
print(classifier.score(x_test, y_test)) 

###########dicision Tree 써보기
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydot
dtree = DecisionTreeClassifier(random_state= 0)
dtree.fit(x_train,y_train)
pre_tree = dtree.predict(x_test)
#성능 평가
print(confusion_matrix(y_test,pre_tree))
#평가결과 확인
print(classification_report(y_test,pre_tree))
print(classifier.score(x_test, y_test))

###########로지스틱
from sklearn.linear_model import LogisticRegression
logi = LogisticRegression( random_state= 0)
logi.fit(x_train,y_train)
pre_logi = logi.predict(x_test)
#성능 평가
print(confusion_matrix(y_test,pre_logi))
#평가결과 확인
print(classification_report(y_test,pre_logi)) 
print(classifier.score(x_test, y_test))#'''

d_all.loc[d_all['y'] == 1]