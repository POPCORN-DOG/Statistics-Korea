#롯데마트몰
library(rvest)
library(dplyr)
library(stringr)
html <- read_html('http://www.lottemart.com/category/categoryList.do?CategoryID=C0010145')

html1<- html %>% html_nodes('#divCat1') %>% html_nodes('a') %>% html_text()
write.csv(as.data.frame(html1),'C:/Users/user/Desktop/백업파일/소비자물가조사/롯데마트몰중분류.csv')

url<-'http://www.lottemart.com/category/categoryList.do?CategoryID=C0010145'
html2<- html %>% html_nodes('#divCat1') %>% html_nodes('a') %>% html_attr('onclick')

html2[2]
aa<-substr(html2,13,20)
aa<-aa[2:64]
###for문돌리기 
url<-'http://www.lottemart.com/category/categoryList.do?CategoryID='
urllist <- NULL

for (i in 1:63) {
  urllist[i]<-paste0(url,aa[i])
  
}
urllist #완성
#소분류 가져오기

sohtml<- read_html(urllist[1])
sohtml2 <-sohtml %>% html_nodes('.prod-sorting-area') %>% html_nodes('a') %>% html_text()
sohtml2 <- sohtml2[1] #하나 해보기 성공

solist <- NULL
a<- NULL
df1<-data.frame(NULL)
df2<-data.frame(NULL)

for (i in 1:27){
  a <- read_html(urllist[i])
  assign(paste0('v',i),cbind(html1[i],a %>% html_nodes('.prod-sorting-area') %>% html_nodes('a') %>% html_text()))  
    #solist <- c(solist,a %>% html_nodes('.prod-sorting-area') %>% html_nodes('a') %>% html_text())
  df1 <- rbind(df1,get(paste0('v',i)))
  }


for (i in 31:63){
  a <- read_html(urllist[i])
  assign(paste0('v',i),cbind(html1[i],a %>% html_nodes('.prod-sorting-area') %>% html_nodes('a') %>% html_text()))  
  #solist <- c(solist,a %>% html_nodes('.prod-sorting-area') %>% html_nodes('a') %>% html_text())
  df2 <- rbind(df2,get(paste0('v',i)))
}
df<-rbind(df1,df2)


write.csv(as.data.frame(solist),'C:/Users/user/Desktop/백업파일/소비자물가조사/롯데마트몰소분류.csv')

write.csv(df,'C:/Users/user/Desktop/백업파일/소비자물가조사/롯데마트몰소분류2.csv')

