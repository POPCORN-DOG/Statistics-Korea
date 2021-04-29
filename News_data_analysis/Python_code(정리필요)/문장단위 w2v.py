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
okt = Okt()

with open('C:/Users/USER/Desktop/백업/text.json') as f:
    words = json.load(f) #형태소 분석 결과 로드
data = pd.read_excel('C:/Users/USER/Desktop/news_url.xlsx', header = None,names=["a",'date','title','text','category','url'])
## 문장단위w2v 해보기
#문장단위 형태소 분석 해서 저장하기
'''data = pd.read_excel('C:/Users/USER/Desktop/news_url.xlsx', header = None,names=["a",'date','title','text','category','url'])
list_text = data['text']
list_sentenc = [okt.nouns(a) for i in tqdm(list_text) for a in i.split('.')] #문장(온점)을 기준으로 나누어서 명사화
with open('C:/Users/USER/Desktop/백업/text_sentence.json', 'w') as make_file:
    json.dump(list_sentenc, make_file)#'''
with open('C:/Users/USER/Desktop/백업/text_sentence.json') as f:
    list_sentenc = json.load(f) #형태소 분석 결과 로드
list_sentence = [i for i in list_sentenc if len(i) >= 5] #명사가 5개 이상인 문장만 취급

embedding_model = Word2Vec(list_sentence, size=100, window = 5, min_count=100, workers=4, iter=100)
#embedding_model.wv.save_word2vec_format('C:/Users/USER/Desktop/백업/news_w2v_sentence')# 모델 저장
embedding_model = KeyedVectors.load_word2vec_format('C:/Users/USER/Desktop/백업/news_w2v') # 모델 로드
print(embedding_model.most_similar(positive=["일자리"], topn=100)) #관련단어 검색

#문장단위 스코어링
sc = pd.read_excel('C:/Users/USER/Desktop/scoreing수정.xlsx')
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
data_all = pd.read_csv('C:/Users/USER/Desktop/data_all.csv') #, header = None,names=["a",'date','title','text','category','url'])
vecdata['score'] = data_all['score']
vecdata['y'] = scoring['y']
# 문장단위 스코어링 저장
#vecdata.to_csv('C:/Users/USER/Desktop/scoreing_sentence.csv', header=True)

#test train set 나누기
x = vecdata.iloc[:,1:101]
y = vecdata.iloc[:,102]
x_train, x_test, y_train, y_test = skl.model_selection.train_test_split(x, y,random_state=20)

##smote 사용해보기
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

sm = SMOTE()
x_smote, y_smote = sm.fit_sample(x_train,y_train)
#x_train, x_test, y_train, y_test = skl.model_selection.train_test_split(x_smote, y_smote,random_state=20)
#정규화시키는 함수
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


#정규화시키는 함수
scaler = skl.preprocessing.StandardScaler()

#x_train 정규화
scaler.fit(x_train) #x_train 데이터에 대해서 평균과 표준편차 계산
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

classifier = KNeighborsClassifier(n_neighbors= 2)
classifier.fit(x_train,y_train)
knn_pre = classifier.predict(x_test)

#성능 평가
print(confusion_matrix(y_test,knn_pre))

#평가결과 확인
print(classification_report(y_test,knn_pre)) #'''
print(classifier.score(x_test, y_test))



a= np.array([[1,2,3],[1,2,3]])
b= np.array([[1,2,3],[1,2,5]])
c= a+b

