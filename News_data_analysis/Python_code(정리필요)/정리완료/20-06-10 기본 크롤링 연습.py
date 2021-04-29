# 2021-04-25(일) 수정

from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas
#import 색 보이기 Unresolved references

# 1. urlopen
html = urlopen('http://www.naver.com')
bsob = BeautifulSoup(html, "html.parser")

# 2. 출력 시행
print(bsob) # 전체 출력
print(bsob.head.title) # 태그로 구성된 트리에서 title 태그만 출력

# 3. tag a와 href속성 출력
bsob.find_all('a')[0].text.strip()
bsob.find_all('a')[0].get('href')

for link in bsob.find_all('a'):
    print(link.text.strip(), link.get('href')) # a 태그로 둘러싸인 텍스트와 a태그의 href속성 출력

# 4. 교보문고 주간순위 책 정보 가져오기
html = urlopen('http://www.kyobobook.co.kr/bestSellerNew/bestseller.laf')
bs = BeautifulSoup(html,'html.parser')

# 5. div에서 class: detail의 href 가져오기
# <div class="detail">
# <a href="http://www.kyobobook.co.kr/.....

book_page_urls = []
for cover in bs.find_all('div',{'class' : 'detail'}):
    link = cover.select('a')[0].get('href')
    book_page_urls.append(link) #url list 만들기

len(book_page_urls)

# 6. enumerate로 index반환 및 확인
for index, urls in enumerate(book_page_urls):
    print(index, urls)


# ===================================================
# 7. 정보 크롤링
list = []
for index, urls in enumerate(book_page_urls):
    html = urlopen(urls)
    bs = BeautifulSoup(html, 'html.parser')

    title = bs.find('meta', {'property':'rb:itemName'}).get('content')
    author = bs.select('span.name')[0].text
    url = bs.find('meta',{'property':'rb:salePrice'}).get('content')

    print(index+1,title, author, url)
    
    list.append(index+1)
    list.append(title)
    list.append(author)
    list.append(url)

list2 = [list[i * 4: (i +1) * 4]for i in range((len(list)+4-1) // 4)]
df = pandas.DataFrame(list2)
df = pandas.DataFrame(list2, columns=['index', 'title', 'author', 'url'])
df.title[2] = "코로나 투자 전쟁"

for i in range(df):
    df.author[i] = df.author[i].replace()

for i in range(1,6):
    try:
        print('kk' + str(i) +'hj'/0)
    except:
        print('kk' + str(i) + 'hj')

df = pd.DataFrame(columns=['a','b','c'])
df.a = list_2 # 칼럼에 리스트 삽입
df.b = list_1_2
list_2 + list_2_2 # 일자로 결합

df[0:10] #행 가져오기



