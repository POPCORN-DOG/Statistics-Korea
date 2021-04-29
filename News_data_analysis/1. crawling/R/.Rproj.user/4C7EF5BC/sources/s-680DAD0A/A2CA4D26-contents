####크롤링 기본 네이버 쇼핑 베스트 100 크롤링 하기####

#필요 패키지 깔기
#install.packages(c('rvest','dplyr','stringr'))
library(rvest)  # 크롤링 필수 패키지
library(dplyr)  # 파이프라인 등 분석을 편하게 하게하는 패키지
library(stringr)  # 텍스트 파일을 다룰 수 있게하는 패키지

#해당 페이지 url 가져오기
html <- read_html('https://search.shopping.naver.com/best100v2/detail.nhn?catId=50000001&listType=B10002')

#class는 앞에 .을 찍고 id는 #을 붙이고, 처음 하나가 아닌 여러개를 가져올 경우 뒤에 s를 붙인다.
html %>% html_nodes('.cont') %>% html_text() # 뽑히긴 하지만 길이가 긴 상품명은 뒤에 ...으로 표시된다

#class = cont 안에 a태그 안에 title을 보면 잘 나와있으므로 그걸 가져온다
html %>% html_nodes('.cont') %>% html_nodes('a') %>% html_attr('title') # 성공


# 이것을 이용 하여 상품명, 가격 , url 정보를 가져와보자
id <- html %>% html_nodes('.cont') %>% html_nodes('a') %>% html_attr('title')
price <- html %>% html_nodes('.num') %>% html_text()
url <- html %>% html_nodes('.cont') %>% html_nodes('a') %>% html_attr('href')

#하나로 합쳐 data set을 만들고 csv로 내보내자
data <- cbind.data.frame(id,price,url)
write.csv(data,"C:/Users/USER/Desktop/백업/best100.csv")

#### 크롤링 중급 및 워드클라우드 만들기 ####
#네이버 랭킹 뉴스 본문 크롤링 해서 워드클라우드 만들기

#크롤링 필요 패키지 깔기
#install.packages(c('rvest','dplyr','stringr'))
library(rvest)  # 크롤링 필수 패키지
library(dplyr)  # 파이프라인 등 분석을 편하게 하게하는 패키지
library(stringr)  # 텍스트 파일을 다룰 수 있게하는 패키지


#url list 만들기

# 네이버 랭킹 페이지에서 각 뉴스의 url을 가져 온다
html <- read_html('https://news.naver.com/main/ranking/popularDay.nhn?rankingType=popular_day&sectionId=101&date=20200610')

url30 <- html %>%html_nodes('.ranking_headline') %>% html_node('a') %>% html_attr('href')

#앞에 https://news.naver.com 가 빠져있음 붙여야 한다.

url_list <- NULL
for( i in 1:30){
  url_list[i] <- paste0('https://news.naver.com',url30[i]) 
}# utl list가 만들어 졌다

# 첫번째 url의 본문을 가져와 보자
html_1 <- read_html(url_list[1])
html_1 %>% html_node('#articleBodyContents') %>% html_text()
# 두번째 url의 본문을 가져와 보자
html_1 <- read_html(url_list[2])
html_1 %>% html_node('#articleBodyContents') %>% html_text()

# 공통적으로 앞의 글자가 중복이된다 그걸 제거해주자
html_1 <- read_html(url_list[1])
html_1 %>% html_node('#articleBodyContents') %>% html_text() %>% 
  str_replace('\n\t\n\t\n\n\n\n// flash 오류를 우회하기 위한 함수 추가\nfunction _flash_removeCallback',"") %>% 
  str_replace('\nfunction _flash_removeCallback',"") #문자열에 중괄호가 들어가면 오류

#위의 코드를 바탕으로 for문을 이용하여 모든 본문을 가져오자 
text_list <- ""
for (i in 1:length(url_list)) {
  html <- read_html(url_list[i])
  tt<-html %>% html_node('#articleBodyContents') %>% html_text() %>% 
    str_replace('\n\t\n\t\n\n\n\n// flash 오류를 우회하기 위한 함수 추가\nfunction _flash_removeCallback',"") %>% 
    str_replace('\nfunction _flash_removeCallback',"") 
  tt<- str_sub(tt,7,length(tt)-20)
  text_list <- paste0(text_list,tt) #30개의 본문을 가져와서 합침
}

#워드 클라우드 준비
#사전 준비 1. rJava가 안깔려 있다면 깔아야 한다.
#사전 준비 2. rJava가 설치가 안된다면 java를 깔아야 한다
#사전 준비 2-1 java를 깔때 설치 경로를 파악해놓고  
Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\jre1.8.0_251') #명령어를 실행합니다(해당 설치경로로 해야함)
install.packages('rJava')
#이제 시작

#konlp를 깔아 야 하는데 최신 버젼에는 깔리지 않을 것이다.

#rTools를 인터넷에서 깐 후 https://cran.r-project.org/bin/windows/Rtools/
#https://cran.r-project.org/src/contrib/Archive/ 에서 직접 tar.gz를 다운해서 
# install.packages ("C:/Users/USER/Desktop/백업/KoNLP_0.80.2.tar.gz", repos=NULL, type="source")를 실행 (파일경로로 해야함)
# 그후  'hash', 'tau', 'Sejong', 'RSQLite', 'devtools'등의 패키지를 다운 받으라고 하면 다운을 받는다
#그럼 done(konlp)가 나온다. 그 후 패키지 부착
install.packages ("C:/Users/USER/Desktop/백업/KoNLP_0.80.1.tar.gz", repos=NULL, type="source")
install.packages(c('hash', 'tau', 'Sejong', 'RSQLite', 'devtools'))
install.packages('wordcloud')
install.packages('ellipsis')
library(rJava)
library(KoNLP)
library(wordcloud)
library(RColorBrewer)
#install.packages ("C:/Users/USER/Desktop/백업/NIADic_0.0.1.tar.gz", repos=NULL, type="source")
useNIADic() # 사전 불러오기

text_list <- gsub("\\d+","", text_list)   # 날짜를 제외하는 것을 의미
text_list <- gsub("[A-z]","", text_list)  # 영문을 제외하는 것을 의미
text_list <- gsub("[[:cntrl:]]","", text_list)   # 특수문자를 제외하는 것을 의미
text_list <- str_remove_all(text_list,"기자")
text_list <- str_replace_all(text_list,"/"," ")
#9개의 기준으로 품사 분석 
s09 <- SimplePos22(text_list)  #SimplePos09 간단한 버젼
s09 <- unlist(s09)
head(s09,100)
divide <- NULL
for (i in 1:length(s09)) {
  asd <- unlist(s09[i] %>% str_split("\\+")) #특수문자로 구분하려면 \\앞에 붙이기
  for (j in 1:length(asd)){
    asd_1 <- asd[j] %>% str_split("/")
    단어 <- asd_1[[1]][1]
    품사 <- asd_1[[1]][2]
    cb <- cbind(단어,품사)
    divide <- rbind(divide,cb)
  }
}
write.csv(divide,'Rkonlp.csv')

#문장에서 명사로 된 단어만 추출하여 다시 저장하고 빈도 분석
nouns <- extractNoun(text_list)
nouns2 <- unlist(nouns)
wordcount <- table(nouns2)
df_word <- as.data.frame(wordcount, stringsAsFactors = F)

#한글자 지우기
df_word <- filter(df_word, nchar(nouns2) >= 2)

# 상위 30개 출력해보기
df_word %>% arrange(desc(Freq)) %>% head(100)


# 먼저 색상 코드를 만든다
pal <- brewer.pal(8,'Dark2')


#지우는 코드
df_word <-df_word[!(df_word$nouns2 == '상황'),]

#원하는 단어가 몇개 있는지 보기

df_word[which(df_word$nouns2 == '힐링'),]
# 워드클라우드는 함수 실행마다 다른 워드클라우드를 만드는 데 set.seed(123)으로 난수를 고정하면 
# 일정모양의 워드클라우드가 나온다


#해보기
wordcloud(words = df_word$nouns2, #단어
          freq = df_word$Freq, #빈도
          min.freq = 10, #최소단어 빈도
          max.words =  150, #표현단어 수
          random.order = F , # 고빈도 단어 중앙배치
          rot.per = 0.2, #회전 비율
          scale = c(2.5,0.4), #가장 빈도가 큰 단어와 가장 빈도가 작은단어 폰트사이의 크기차이
          colors = pal #색상
          )


##ANN 인공신경망 만들어 보기
install.packages('neuralnet')
library(neuralnet)

#데이터 로드
df <- read.csv('C:/Users/USER/Desktop/백업/drinker.csv')
str(df)
df_s<-scale(df[,1:9])
colnames(df)[10] <- 'drinker'

# 편의상 나눔 
df_train <- df_s[1:500,]
fd_test <- df_s[501:740,]

#모델 생성
df_model <- neuralnet(formula = drinker~., data = df_train)
#hidden 값 늘려서 생성
df_model2 <- neuralnet(formula = drinker~., data = df_train, hidden = 5)
plot(df_model2)

