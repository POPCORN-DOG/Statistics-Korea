#롯데온
library(rvest)
library(dplyr)
library(stringr)
library(readxl)
library(readr)
#테스트
html <- read_html('https://search.shopping.naver.com/search/all.nhn?query=%EB%B3%B4%EB%85%B8%20%ED%8F%AC%EB%A5%B4%EC%B9%98%EB%8B%88%EB%B2%84%EC%84%AF%EC%8A%A4%ED%94%84(51.6g)&mall=596&pagingIndex=1&pagingSize=80&exused=true')

url<-URLencode(iconv('옛날국수소면+"이마트몰"', to = "UTF-8")) #url에 직접 한글을 치면 오류나옴. utf-8로 바꿔줘야함
html1 <- read_html(paste0('https://search.shopping.naver.com/search/all.nhn?query=',url))

html2<- html  %>% html_nodes('._productSet_total') %>% html_text() %>%
  str_replace_all(pattern = '[[:cntrl:]]',replacement = "")   #검색결과 갯수 파악

html2<- html  %>% html_nodes('.tit') %>% html_nodes('.link') %>% html_text() %>%
  str_replace_all(pattern = '[[:cntrl:]]',replacement = "")  #검색결과 상품명 파악 

html3<- html1  %>% html_nodes('.price') %>% html_text() %>%
  str_replace_all(pattern = '[[:cntrl:]]',replacement = "") %>%
  str_replace_all(pattern = '                    ',replacement = "") #가격 평균 구하기


html3
# 테스트 list 해보기

dt<-read.csv('C:/Users/USER/Desktop/백업/t2.csv')
a<-as.vector(dt[,1])
Encoding(a)
traaaaa<- read.csv('C:/Users/USER/Desktop/백업/traaaaa.csv')
encode<- NULL
url_list<- NULL
fi1<-NULL
itemlist1 <- NULL
for(i in 1:450) {
  ww<- NULL
  encode[i] <- URLencode(iconv(a[i], to = "UTF-8"))
  url_list[i] <-paste0("https://search.shopping.naver.com/search/all.nhn?query=",encode[i])  
  kkh <- read_html(url_list[i]) %>% html_nodes('._productSet_total') %>% html_text() %>%
    str_replace_all(pattern = '[[:cntrl:]]',replacement = "")   #검색결과 갯수 파악
  fi1[i] <- paste0(kkh," ")
  kkh2 <- read_html(url_list[i]) %>% html_nodes('.tit') %>% html_nodes('.link') %>% html_text() %>%
    str_replace_all(pattern = '[[:cntrl:]]',replacement = "")
  for ( j in 1:length(kkh2)){
    ww<- paste(ww,kkh2[j], sep = " / ")
       
  }
  itemlist1[i] <- paste0(ww," ") 

  for (p in 1:25) {
    traaaaa<- read.csv('C:/Users/USER/Desktop/백업/traaaaa.csv')
    
  }
  print(i)
  
    }
fi1
itemlist1


b<-as.vector(dt[,2])

encode<- NULL
url_list<- NULL
fi2<-NULL
for(i in 1:450) {
  encode[i] <- URLencode(iconv(b[i], to = "UTF-8"))
  url_list[i] <-paste0("https://search.shopping.naver.com/search/all.nhn?query=",encode[i])  
  kkh <- read_html(url_list[i]) %>% html_nodes('._productSet_total') %>% html_text() %>%
    str_replace_all(pattern = '[[:cntrl:]]',replacement = "")   #검색결과 갯수 파악
  fi2[i] <- paste0(kkh," ")
  for (j in 1:30) {
    traaaaa<- read.csv('C:/Users/USER/Desktop/백업/traaaaa.csv')
    
  }
  print(i)
}
fi2

c<-as.vector(dt[,3])

encode<- NULL
url_list<- NULL
fi3<-NULL
for(i in 1:450) {
  encode[i] <- URLencode(iconv(c[i], to = "UTF-8"))
  url_list[i] <-paste0("https://search.shopping.naver.com/search/all.nhn?query=",encode[i])  
  kkh <- read_html(url_list[i]) %>% html_nodes('._productSet_total') %>% html_text() %>%
    str_replace_all(pattern = '[[:cntrl:]]',replacement = "")   #검색결과 갯수 파악
  fi3[i] <- paste0(kkh," ")
  for (j in 1:30) {
  traaaaa<- read.csv('C:/Users/USER/Desktop/백업/traaaaa.csv')
  
  }
  print(i)
}
fi3

d<-as.vector(dt[,4])

encode<- NULL
url_list<- NULL
fi4<-NULL
for(i in 1:450) {
  encode[i] <- URLencode(iconv(d[i], to = "UTF-8"))
  url_list[i] <-paste0("https://search.shopping.naver.com/search/all.nhn?query=",encode[i])  
  kkh <- read_html(url_list[i]) %>% html_nodes('._productSet_total') %>% html_text() %>%
    str_replace_all(pattern = '[[:cntrl:]]',replacement = "")   #검색결과 갯수 파악
  fi4[i] <- paste0(kkh," ")
  for (j in 1:30) {
    traaaaa<- read.csv('C:/Users/USER/Desktop/백업/traaaaa.csv')
    
  }
  print(i)
}
fi4

f1 <- as.data.frame(fi1)
f2 <- as.data.frame(fi2)
f3 <- as.data.frame(fi3)
f4 <- as.data.frame(fi4)

fin <- cbind(f1,f2)
fin <- cbind(fin,f3)
fin <- cbind(fin,f4)

write.csv(fin,'C:/Users/USER/Desktop/백업/수량파악.csv')


