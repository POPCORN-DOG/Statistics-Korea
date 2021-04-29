####크롤링 기본 (네이버 쇼핑 best 100 크롤링 해보기)####

#필요 패키지 설치
from urllib.request import urlopen #url의 html 을 가져 오기 위한 패키지
from bs4 import BeautifulSoup  #크롤링 필수 패키지 설치하려면 cmd창에서 pip install bs4
import pandas #데이터를 다루기위해 필요한 기본 패키지

# BeautifulSoup 시작하기
webpage = urlopen('https://search.shopping.naver.com/best100v2/detail.nhn?catId=50000006&listType=B10002')
soup = BeautifulSoup(webpage, 'html.parser') # 위 url 의 html 가져오기 파이썬에서 보기 위해 파싱작업을 하는것

#이름 가져오기
a=soup.select('.cont') #class는 앞에 .붙히기 a=soup.find_all('p', {'class' : 'cont'}) 이렇게도 가능
a[0].text.replace("\n","") #첫번째 방법
a[0].find('a')['title'] #두가지 방법 다 가능

#url 가져오기
a[0].find('a')['href']

#가격 가져오기
b=soup.select('.num')
b[0].text

#list에 저장하기
rank = []; id = []; price = []; url =[]

for i in range(0,100):
    rank.append(i+1)
    id.append(a[i].find('a')['title'])
    price.append(b[i].text)
    url.append(a[i].find('a')['href'])

#data 합치기
data = pandas.DataFrame({'rank':rank, 'id':id, 'price':price, 'url':url})

#csv로 내보내기
data.to_csv('C:/Users/USER/Desktop/파이선/rank100.csv',encoding='utf-8-sig')


####크롤링 중급 및 워드클라우드 해보기 ####
####크롤링 중급 및 워드클라우드 해보기 ####

#필요 패키지 설치
from urllib.request import urlopen #url의 html 을 가져 오기 위한 패키지
from bs4 import BeautifulSoup  #크롤링 필수 패키지 설치하려면 cmd창에서 pip install bs4
import pandas #데이터를 다루기위해 필요한 기본 패키지

#네이버 랭킹뉴스 파싱하기
webpage = urlopen('https://news.naver.com/main/ranking/popularDay.nhn?rankingType=popular_day&sectionId=101&date=20200610')
soup = BeautifulSoup(webpage, 'html.parser')

# 기사의 url 불러오기
aa = soup.select('.ranking_headline > a')
aa[0]['href']

#기사의 url을 받아 파싱하기
news_url = urlopen('https://news.naver.com' + aa[0]['href'])
news_soup = BeautifulSoup(news_url, 'html.parser')

#본문가져오기
news_text = news_soup.select('._article_body_contents')
news_text[0].text.replace("\n\n\n\n\n// flash 오류를 우회하기 위한 함수 추가\nfunction _flash_removeCallback() {}\n\n","") #앞의 반복문구 지우기
#하나의 url에서 다른 url을 들어가 다시 파싱한 후 본문을 가져왔다. for문을 사용해서 30개의 뉴스의 본문을 다 가져와보자

#들어갈 url 리스트 만들고 본문 가져오기
texts = []
for i in range(0,30):
    news_urls = urlopen('https://news.naver.com' + aa[i]['href'])
    news_soups = BeautifulSoup(news_urls, 'html.parser')
    news_texts = news_soups.select('._article_body_contents')
    text = news_texts[0].text.replace("\n\n\n\n\n// flash 오류를 우회하기 위한 함수 추가\nfunction _flash_removeCallback() {}\n\n", "")
    texts.append(text)
    print(i)

#konpy 깔았음 okt도 깔았음
#pip install konpy 안되면 구글 참조 트위터는 okt로 바뀐듯
#워드 클라우드에 필요한 패키지 설치및 임포트
from collections import Counter
from konlpy.tag import Okt



#텍스트를 한 리스트에 모아서 넣기
texts_2 =[""]
for i in range(0,30):
    texts_2[0] = texts_2[0] + texts[i]
texts_2[0]

# 자연어 처리기술 중 하나인 okt를 이용하여 글자를 나누고 품사를 정의
morphs = []
okt = Okt()
for sentence in texts_2:
    morphs.append(okt.pos(sentence))
print(morphs)

#명사만 가져오고 것, 내, 나 등 의미가 없는 명사들을 제외하는 코드 작성
noun_adj_adv_list=[]
for sentence in morphs :
    for word, tag in sentence :
        if tag in ['Noun'] and ("것" not in word) and ("내" not in word)and ("나" not in word)and ("수"not in word) \
                and("게"not in word)and("말"not in word)and("등"not in word)and("고"not in word)and("이"not in word)\
                and("재"not in word):
            noun_adj_adv_list.append(word)
print(noun_adj_adv_list)


count = Counter(noun_adj_adv_list) # 숫자들 세주는 counter함수
words = dict(count.most_common()) # 사전형식으로 바꿈

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

#워드클라우드 설정하기
wordcloud = WordCloud(
    font_path = 'C:/Windows/Fonts/H2HDRM.TTF',    # 맥에선 한글폰트 설정 잘해야함.
    background_color='white',                             # 배경 색깔 정하기
    width = 800,
    height = 800
).generate_from_frequencies(words)

#내보내기
plt.imshow(wordcloud)
plt.axis('off')
plt.show()





