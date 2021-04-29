# 2021-04-27(화) 수정

## 네이버 영화 리뷰 Word2Vec

# 영화 리뷰를 스토리, 연기, 배우, 감독 으로 범주를 나누어 보자

# =======================================================
# 1. 파일 읽어오기 
# =======================================================

# https://cyc1am3n.github.io/2018/11/10/classifying_korean_movie_review.html
# with 구문은 파일을 읽고 처리할 때 쓰인다. 
# open함수를 쓰면 close를 꼭 해줘야 하는데 with 구문안에 쓴다면 따로 close를 안해도 된다
def read_data(filename):   
    with open(filename, 'r',encoding='UTF8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:] # txt 파일의 헤더(id document label)는 제외하기
    return data

train_data = read_data('C:/Users/USER/Desktop/nsmc-master/ratings_train.txt')

import urllib.request
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
train_data = pd.read_table('ratings.txt')


# =======================================================
# 2. 데이터 전처리
# =======================================================
from konlpy.tag import Okt
okt = Okt()

from tqdm import tqdm
#asd = [];train_pos = [];as2 = []
#for i in tqdm(range(len(train_data))): #train data
#    aaa = okt.pos(train_data[i][1])
#    for j in aaa:
#        as2.append((str(j[0])))
#    asd.append(as2)
#    asd.append(train_data[i][2])
#    train_pos.append(asd)
#    asd = []
#    as2 = []

# 태깅한건 json 파일로 저장하기
import json
#with open('C:/Users/USER/Desktop/백업/train_word.json','w',encoding='utf-8') as make_file:
#    json.dump(train_pos,make_file,indent='\t')
# 불러오기
with open('C:/Users/USER/Desktop/백업/train_word.json') as f:
    train_pos = json.load(f)

for i in range(len(train_pos)):
    train_pos[i] = train_pos[i][0]
    if len(train_pos[i]) < 4:
        del train_pos[i]
        del train_data[i]


# Word2Vec embedding
from gensim.models import Word2Vec
embedding_model = Word2Vec(train_pos, size=100, window = 5, min_count=40, workers=4, iter=100, sg=0)

# check embedding result
print(embedding_model.most_similar(positive=["스토리"], topn=10))

#모델 저장 및 불러오기
from gensim.models import KeyedVectors
#embedding_model.wv.save_word2vec_format('C:/Users/USER/Desktop/백업/movie_w2v') # 모델 저장
embedding_model = KeyedVectors.load_word2vec_format('C:/Users/USER/Desktop/백업/movie_w2v') # 모델 로드

# 연기력에 대한 내용의 리뷰일수록 숫자가 높게나오는 함수를 만들기
import numpy as np

avg_dist = []
dist = []
for i in tqdm(range(len(train_pos))):
    for k in train_pos[i]:
        try:
            dist.append(embedding_model.similarity('스토리',k)) #연기력과 비교하여 similarity 구하기
        except:
            continue
    if len(dist) > 15:
        dist = sorted(dist, reverse=True)[:15] #게시물이 길어질수록 낮게 나오는 경향을 없어기 위함
    np.mean(dist)
    avg_dist.append([train_data[i][1],np.mean(dist)])
    dist = []


score = [avg_dist[i][1] for i in range(len(avg_dist))]


score2 = [i for i in score if str(i) != 'nan']
np.max(score2)
#분포보기
import matplotlib.pyplot as plt
plt.hist(score2)

#연기력에 관련된 리뷰 뽑기
score_20 = [i for i in score2 if i > 0.25]
avg_dist[score.index(score_20[0])][0]
rev = [avg_dist[score.index(score_20[i])][0] for i in range(len(score_20))]