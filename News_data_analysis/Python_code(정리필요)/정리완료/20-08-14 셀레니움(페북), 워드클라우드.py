###셀레니움 해보기
import os
import re
from selenium import webdriver
from bs4 import BeautifulSoup #크롤링 도구
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd
from urllib.request import urlopen
driver = webdriver.Chrome('C:/r-selenium/chromedriver.exe')  # 크롬 드라이버 연결
driver.get('https://www.facebook.com/dg1sBamBoo/')  #url 이동


driver.set_window_size(1000,1000) #창크기 조절
driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
last_height = driver.execute_script('return document.body.scrollHeight')

while True:   # 무한 스크롤
    last_height = driver.execute_script('return document.body.scrollHeight')
    driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
    time.sleep(2.7)
    last_height_new = driver.execute_script('return document.body.scrollHeight')
    if last_height_new == last_height:
        break
    last_height = last_height_new

soup = BeautifulSoup(driver.page_source, 'html.parser')


ss = soup.select('.userContent')
te = [ii.text.replace("더 보기","") for ii in ss]
tex = [i[i.find(">")+1 : ] for i in te]
text = ''.join(tex)


from collections import Counter
from konlpy.tag import Okt
okt = Okt()
# 자연어 처리기술 중 하나인 okt를 이용하여 글자를 나누고 품사를 정의
text = [text]
morphs = []
for sentence in text:
    morphs.append(okt.pos(sentence,norm = True,stem=True))
print(morphs[0:10])



#명사만 가져오고 것, 내, 나 등 의미가 없는 명사들을 제외하는 코드 작성
noun_adj_adv_list=[]
for sentence in morphs :
    for word, tag in sentence :
        if tag in ['Noun'] and ['Verb'] and ("것" not in word) and ("내" not in word)and ("나" not in word)and ("수"not in word) \
                and("게"not in word)and("말"not in word)and("등"not in word)and("고"not in word)and("이"not in word)\
                and("재"not in word):
            noun_adj_adv_list.append(word)
print(noun_adj_adv_list)
noun_adj_adv_list = [i for i in noun_adj_adv_list if len(i)>1] #한글자 제외

count = Counter(noun_adj_adv_list) # 숫자들 세주는 counter함수
words = dict(count.most_common()) # 사전형식으로 바꿈


#words를 json으로 저장
import json
with open('C:/Users/USER/Desktop/백업/대구일과학고.json','w',encoding='utf-8') as make_file:
    json.dump(words,make_file,indent='\t')
# 불러오기
    with open('C:/Users/USER/Desktop/백업/대구일과학고.json') as f:
        words = json.load(f)



#워드클라우드 만들기 pip install pytagcloud 설치 필요
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

############################대구 동부고

###셀레니움 해보기
import os
import re
from selenium import webdriver
from bs4 import BeautifulSoup #크롤링 도구
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd
from urllib.request import urlopen
driver = webdriver.Chrome('C:/r-selenium/chromedriver.exe')  # 크롬 드라이버 연결
driver.get('https://www.facebook.com/%EB%8F%99%EB%B6%80%EA%B3%A0-%EB%8C%80%EC%8B%A0-%EC%A0%84%ED%95%B4%EB%93%9C%EB%A6%BD%EB%8B%88%EB%8B%A4-1103341589795335/')  #url 이동


driver.set_window_size(1000,1000) #창크기 조절

while True:   # 무한 스크롤
    last_height = driver.execute_script('return document.body.scrollHeight')
    driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
    time.sleep(2.7)
    last_height_new = driver.execute_script('return document.body.scrollHeight')
    if last_height_new == last_height:
        break
    last_height = last_height_new

soup = BeautifulSoup(driver.page_source, 'html.parser')

ss = soup.select('.userContent')
te = [ii.text.replace("더 보기","") for ii in ss]
tex = [i[i.find(")")+1 : ] for i in te]
text = ''.join(tex)
from collections import Counter
from konlpy.tag import Okt
okt = Okt()
# 자연어 처리기술 중 하나인 okt를 이용하여 글자를 나누고 품사를 정의
text = [text]
morphs = []
for sentence in text:
    morphs.append(okt.pos(sentence,norm = True,stem=True))
print(morphs[0:10])

#명사만 가져오고 것, 내, 나 등 의미가 없는 명사들을 제외하는 코드 작성
noun_adj_adv_list=[]
for sentence in morphs :
    for word, tag in sentence :
        if tag in ['Noun'] and ['Verb'] and ("제보" not in word) and ("학년" not in word)and ("나" not in word)and ("수"not in word) \
                and("게"not in word)and("말"not in word)and("등"not in word)and("고"not in word)and("이"not in word)\
                and("재"not in word):
            noun_adj_adv_list.append(word)
print(noun_adj_adv_list)
noun_adj_adv_list = [i for i in noun_adj_adv_list if len(i)>1] #한글자 제외
count = Counter(noun_adj_adv_list) # 숫자들 세주는 counter함수
words = dict(count.most_common()) # 사전형식으로 바꿈

with open('C:/Users/USER/Desktop/백업/대구동부고.json','w',encoding='utf-8') as make_file:
    json.dump(words,make_file,indent='\t')

with open('C:/Users/USER/Desktop/백업/대구동부고.json') as f:
    words = json.load(f)


#워드클라우드 만들기 pip install pytagcloud 설치 필요
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
mask = np.array(Image.open(path.join(d,'C:/Users/USER/Desktop/백업/대구동부고.png')))

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

##########################뉴스기사 단어 분석
from konlpy.tag import Okt
okt = Okt()
from tqdm import tqdm

data = pd.read_csv('C:/Users/USER/Desktop/백업/news.csv', encoding='cp949')
data = data.drop_duplicates() #중복제거

x=[]
for i in range(len(words)):
    x.append([words[i],data['category'][i]])
x1 = [i[0] for i in x if i[1] == '세계']
x2 = []
for i in x1:
    for k in i:
        x2.append(k)
print(pd.Series(x2).value_counts().head(30))

