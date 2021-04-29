####
install.packages('rvest')
library(rvest)
library(dplyr)
###url list 만들기
urls <- NULL
ba <- 'http://www.donga.com/news/search?query=미국증시&check_news=1&more=1&sorting=1&p='
for(i in 0:9){
  urls[i+1] <- paste0(ba,i * 15 + 1)
}
urls
###html 가져와서 url주소 찾기
html <- read_html(urls[1])

html2<-html_nodes(html,'.searchCont')  ### class는 앞에 .을 찍고 id는 앞에 #을 붙임

html3<-html_nodes(html2,'a')

links<-html_attr(html3,'href')

links<- unique(links)  ### %>% 으로 한번에 해도됨

grep('pdf',links)
links[-grep('pdf',links)]

### for문으로 10페이지까지 url 가져오기
links <- NULL
html <- NULL
for(i in 1:10){
  html <- read_html(urls[i])
  links <- c(links, html %>% html_nodes('.searchCont') %>%
               html_nodes('a') %>% html_attr('href') %>%
               unique()  )
}
links<-links[-grep('pdf',links)]

###신문에 내용 찾기
txts<- NULL
for(x in links){
  html <- read_html(x)
  txts <- c(txts, html %>% html_nodes('.article_txt') %>% html_text())
}

#hhtml<- read_html(links[1])
#hhtml2 <- hhtml %>% html_nodes('.article_txt') %>% html_text()
#html_

write.table(txts,'C:/Users/user/Desktop/qgis/202002_위치정보요약DB_전체분/cr.txt')

head(txts)
#이후 정규식을 사용하여 자료 정리
library(stringr)
#영어 지우기
gsub('[A-z]',"",txts[1])

