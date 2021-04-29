# 홈플러스 만들기
library(rvest)
library(dplyr)
library(stringr)
library(tidyr)
##### 중분류 가져오기
url <- 'http://www.homeplus.co.kr/app.exhibition.main.Main.ghs?paper_no=category'
html <- read_html(url)
html0 <- html %>% html_nodes('.cat-dep1') %>% html_nodes('a') %>% html_text() # 성공
# 쓰기
write.csv(html0,'C:/Users/user/Desktop/백업파일/소비자물가조사/홈플러스/홈플러스중분류.csv')




#코드 가져오기
#하나 가져와보기
url <- 'http://www.homeplus.co.kr/app.exhibition.main.Main.ghs?paper_no=category'
html <- read_html(url)
html1 <- html %>% html_nodes('.cat-dep1') %>% html_nodes('a') %>% html_attr('onclick')
substr(html1[1],14,18) # 성공
code <- substr(html1,14,18) #코드 list 만들기
# for문으로 코드를 붙여 url list 만들기 

url <- 'http://www.homeplus.co.kr/app.exhibition.category.Category.ghs?comm=category.list&cid='
urllist <- NULL

for (i in 1:30) {
  urllist[i] <- paste0(url,code[i])
}
# 확인하기
urllist # 성공

### 소분류 하기

html2<- read_html(urllist[2])

#html3<- html2 %>% html_nodes('.menu') %>% html_nodes('.last') %>% html_text()
html3<- html2 %>% html_nodes('.exhibition-tab') %>% html_text()
html3_1<- gsub("[[:cntrl:]]","",html3)
html3_1<-gsub('   ','',html3_1)
html3_2<-strsplit(html3_1,split = ' ')
html3_2 <- as.data.frame(html3_2)  #데이터 프레임으로 가져오기 성공
#### 파이프라인으로 한줄로 만들기
html3_2<-as.data.frame(html2 %>% html_nodes('.exhibition-tab') %>% html_text() %>%
  str_replace_all(pattern = '[[:cntrl:]]',replacement = "") %>% str_replace_all(pattern = '   ',replacement = "") %>%
  strsplit(split = ' ')) %>% rename(starts_with("c") = "dd")

cbind(html0[1],html3_2[[1]])



##for 문 돌려보기
a<- NULL
rb<- NULL
for (i in 1:30) {
  a<- read_html(urllist[i])
  assign(paste0('v',i),as.data.frame(a %>% html_nodes('.exhibition-tab') %>% html_text() %>%
           str_replace_all(pattern = '[[:cntrl:]]',replacement = "") %>% str_replace_all(pattern = '   ',replacement = "") %>%
           strsplit(split = ' ')))
  assign(paste0('vv',i),cbind(html0[i],get(paste0('v',i))))
  assign(names(get(paste0('vv',i))[2]),c("c..사과.배.토마토....감.곶감.홍시....수박.메론.참외....블루베리.석류...")) 
  rb<-bind_rows(rb,get(paste0('vv',i)))
}

write.csv(rb,'C:/Users/USER/Desktop/백업/홈플러스 소분류.csv')

unite(rb,col = '소분류', rb[2:31], sep = "")

as.character(rb[2:31])

str(rb)
is(rb$`html0[i]`)
