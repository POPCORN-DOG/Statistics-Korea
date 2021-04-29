import numpy as np
import pandas as pd
from tqdm import tqdm

###파일 읽어오기
def read_data(filename):   #인터넷 배낀거
    #with 구문은 파일을 읽고 처리할 때 쓰인다. open함수를 쓰면 close를 꼭 해줘야 하는데 with 구문안에 쓴다면 따로 close를 안해도 된다
    with open(filename, 'r',encoding='UTF8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        # txt 파일의 헤더(id document label)는 제외하기
        data = data[1:]
    return data

def read_data2(filename): #공부해서 내가 만든것
    data = []
    with open(filename,'r',encoding='utf8') as f:
        for i in f.read().splitlines():
            asdf = i.split('\t')
            data.append(asdf)
        data = data[1:] #첫 행 제외하기
    return data
test_data = read_data2('C:/Users/USER/Desktop/nsmc-master/ratings_test.txt')
train_data = read_data('C:/Users/USER/Desktop/nsmc-master/ratings_train.txt')

###데이터 전처리
from konlpy.tag import Okt
okt = Okt()

#잘 나뉘나 확인
okt.pos(train_data[0][1])
train_data[0][2]

##리뷰를 형태소 분석기로 나누기
asd = [];test_pos = []; as2 = []
for i in tqdm(range(len(test_data))): #test data
    aaa = okt.pos(test_data[i][1])
    for j in aaa:
        as2.append((str(j[0])+"/"+str(j[1])))
    asd.append(as2)
    asd.append(test_data[i][2])
    test_pos.append(asd)
    asd = []
    as2 = []

asd = [];train_pos = [];as2 = []
for i in tqdm(range(len(train_data))): #train data
    aaa = okt.pos(train_data[i][1])
    for j in aaa:
        as2.append((str(j[0])+"/"+str(j[1])))
    asd.append(as2)
    asd.append(train_data[i][2])
    train_pos.append(asd)
    asd = []
    as2 = []

# 태깅한건 json 파일로 저장하기
#import json
#with open('C:/Users/USER/Desktop/백업/train.json','w',encoding='utf-8') as make_file:
#    json.dump(train_pos,make_file,indent='\t')
#with open('C:/Users/USER/Desktop/백업/test.json', 'w', encoding='utf-8') as make_file:
#    json.dump(test_pos, make_file, indent='\t')
# 불러오기
#    with open('C:/Users/USER/Desktop/백업/train.json') as f:
#        train_pos = json.load(f)
#    with open('C:/Users/USER/Desktop/백업/test.json') as f:
#        test_pos = json.load(f)

## 상위 빈도 탐색
tokens=[]
tokens_test=[]
for d in tqdm(train_pos):
    for i in d[0]:
        tokens.append(i)
for d in tqdm(test_pos):
    for i in d[0]:
        tokens_test.append(i)
import nltk
text = nltk.Text(tokens,name='NMSC')
text[0:10]
len(set(text.tokens))
text.vocab().most_common(10) #상위 10개 빈도 보여주기

#빈도 순 50개 단어 시각화
import matplotlib.pyplot as plt
from matplotlib import font_manager,rc

font_fname = 'c:/windows/fonts/gulim.ttc'
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font',family = font_name)

plt.figure(figsize=(20,10))
text.plot(50)

##bag of words  countvectorization 상위 15000개의 단어로 데이터 정제하여 학습
## 리뷰 한개마다 긍정인지 부정인지를 나타내는 데이터 만들기

# 단어 빈도수가 높은 10000개 단어만 사용
selected_words = [f[0] for  f in text.vocab().most_common(10000)]

#위에서 만든 selected_words의 갯수에 따라서 해당 텍스트가 얼만큼 표현되는지 빈도를 만들기 위한 함수
def term_frequency(doc):
    return [doc.count(word) for word in selected_words]

#학습용 테스트용 분류
train_x = [term_frequency(i) for i, _ in tqdm(train_pos)] # train_pos list안에 첫번째꺼 쓰고 두번째거 버린다는 뜻
test_x = [term_frequency(i) for i, _ in tqdm(test_pos)]
train_y = [i for _, i in tqdm(train_pos)]
test_y = [i for _, i in tqdm(test_pos)]

#모델링을 하기 위해 리스트형식을 array로 바꾸고 데이터타입도 실수로 바꿔준다
x_train = np.asarray(train_x).astype('float32')
x_test = np.asarray(test_x).astype('float32')

y_train = np.asarray(train_y).astype('float32')
y_test = np.asarray(test_y).astype('float32')

### 모델링 해보기
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics

#tensorflow.keras 를 활용하여 모델 층 입력하기 ( 자세한 원리 공부 필요)
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))#10000개를 추출했으므로 shape는10000
model.add(layers.Dense(64, activation='relu')) # 2개의 dense층은 64개의 유닛을 가지고 relu함수를 사용
model.add(layers.Dense(1, activation='sigmoid')) #마지막층은 시그모이드 함수를 사용하여 긍정리뷰일 확률을 출력

#모델생성 손실함수로 binary_crossentropy사용 / rmsprop 옵티마이저를 통해서 경사하강법 진행 (자세한건 잘 모르겠음)
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
             loss=losses.binary_crossentropy,
             metrics=[metrics.binary_accuracy])
#모델학습
model.fit(x_train,y_train, epochs=10, batch_size=512)
results = model.evaluate(x_test,y_test)

#예측결과
results

#새로운 데이터 예측하기
a = "정말 재미가 드럽게 없어요"
a2=["/".join(i) for i in okt.pos(a)]
tf = term_frequency(a2)
data = np.expand_dims(np.asarray(tf).astype('float32'),axis=0)
score = float(model.predict(data))
print("[{}]는 {:.2f}% 확률로 긍정리뷰입니다.".format(a,score*100))

#모델 저장하기
#from keras.models import load_model
#model.save('C:/Users/USER/Desktop/백업/model.h5')


### 저장한 파일을 바탕으로 간단 구동하기 ####################################################################
import json
from tqdm import tqdm
import numpy as np
# 토큰화 파일 불러오기
with open('C:/Users/USER/Desktop/백업/train.json') as f:
    train_pos = json.load(f)
with open('C:/Users/USER/Desktop/백업/test.json') as f:
    test_pos = json.load(f)

#상위 빈도 만개 사용
tokens = [i for d in tqdm(train_pos) for i in d[0]]
import nltk
text = nltk.Text(tokens,name='NMSC')
selected_words = [f[0] for  f in text.vocab().most_common(10000)]

#모델 불러오기
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
from keras.models import load_model
from konlpy.tag import Okt
okt = Okt()
def term_frequency(doc):
    return [doc.count(word) for word in selected_words]
def data_predict(text):
    ttf = term_frequency(["/".join(i) for i in okt.pos(text)])
    d = np.expand_dims(np.asarray(ttf).astype('float32'),axis=0)
    sc = float(model2.predict(d))
    print("[{}]는 {:.2f}% 확률로 긍정리뷰입니다.".format(text,sc*100))

model2 = load_model('C:/Users/USER/Desktop/백업/model.h5')
data_predict("별로")



