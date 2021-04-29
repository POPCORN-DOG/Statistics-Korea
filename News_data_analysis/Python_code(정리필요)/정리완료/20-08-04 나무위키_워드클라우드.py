import pandas #데이터를 다루기위해 필요한 기본 패키지
from collections import Counter
from konlpy.tag import Okt

with open('C:/Users/USER/Desktop/백업/대구일과학고나무위키.txt', encoding='utf8') as t:
    texts = t.read()

#텍스트를 한 리스트에 모아서 넣기
texts_2 =[texts]


# 자연어 처리기술 중 하나인 okt를 이용하여 글자를 나누고 품사를 정의
morphs = []
okt = Okt()
for sentence in texts_2:
    morphs.append(okt.pos(sentence))
print(morphs)

noun_adj_adv_list=[]
for sentence in morphs :
    for word, tag in sentence :
        if tag in ['Noun'] and ['Verb'] and ("편집" not in word) and ("학생" not in word)and ("학교" not in word)and ("학년"not in word) \
                and("주로"not in word)and("시간"not in word)and("경우"not in word)and("통해"not in word)and("정도"not in word)\
                and("재"not in word):
            noun_adj_adv_list.append(word)
print(noun_adj_adv_list)
noun_adj_adv_list = [i for i in noun_adj_adv_list if len(i)>1] #한글자 제외
count = Counter(noun_adj_adv_list) # 숫자들 세주는 counter함수
words = dict(count.most_common()) # 사전형식으로 바꿈

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
#주피터 에서  %matplotlib inline
import matplotlib
from matplotlib import rc


#글씨체
rc('font', family='H2HDRM')

#이미지 적용
from os import path
from PIL import Image
import os
import numpy as np
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
mask = np.array(Image.open(path.join(d,'C:/Users/USER/Desktop/백업/로고.png')))

from wordcloud import ImageColorGenerator
image_c = ImageColorGenerator(mask)
wc = WordCloud(
    font_path = 'C:/Windows/Fonts/H2HDRM.TTF',    # 맥에선 한글폰트 설정 잘해야함.
    background_color='white',                             # 배경 색깔 정하기
    width = 800,
    height = 800,
    mask = mask
)
wc.generate_from_frequencies(words)
plt.imshow(wc.recolor(color_func=image_c), interpolation="bilinear")
plt.axis('off')
plt.show()