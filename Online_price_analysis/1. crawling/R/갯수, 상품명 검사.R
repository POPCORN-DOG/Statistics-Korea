#더 정확한 url로 바꾼 후 상품 갯수, 상품명 검사
library(rvest)
library(dplyr)
library(stringr)
library(readxl)
library(readr)

item_name <- read.csv('C:/Users/USER/Desktop/백업/item.csv')
a<-as.vector(item_name[,1])
Encoding(a)

#이마트몰 검사 
encode<- NULL
url_list<- NULL
fi1<-NULL
itemlist1 <- NULL
for(i in 1:450) {
  ww<- NULL
  qq<- NULL
  encode[i] <- URLencode(iconv(a[i], to = "UTF-8"))
  
  qq <-paste0("https://search.shopping.naver.com/search/all.nhn?query=",encode[i]) 
  
  url_list[i] <-paste0(qq,"&mall=596&pagingIndex=1&pagingSize=80&exused=true")  
  
  kkh <- read_html(url_list[i]) %>% html_nodes('._productSet_total') %>% html_text() %>%
    str_replace_all(pattern = '[[:cntrl:]]',replacement = "")   #검색결과 갯수 파악
  
  fi1[i] <- paste0(kkh," ")
  
  kkh2 <- read_html(url_list[i]) %>% html_nodes('.tit') %>% html_nodes('.link') %>% html_text() %>%
    str_replace_all(pattern = '[[:cntrl:]]',replacement = "")
  
  for ( j in 1:length(kkh2)){
    ww<- paste(ww,kkh2[j], sep = " / ")
    
  }
  itemlist1[i] <- paste0(ww," ") 
  
  for (p in 1:60) {
    traaaaa<- read.csv('C:/Users/USER/Desktop/백업/traaaaa.csv')
    
  }
  print(i)
  
}
head(url_list)
fi1
itemlist1

#롯데온 검사
encode<- NULL
url_list2<- NULL
fi2<-NULL
itemlist2 <- NULL
for(i in 1:450) {
  ww<- NULL
  qq<- NULL
  encode[i] <- URLencode(iconv(a[i], to = "UTF-8"))
  qq <-paste0("https://search.shopping.naver.com/search/all.nhn?query=",encode[i]) 
  url_list2[i] <-paste0(qq,"&mall=1243359&pagingIndex=1&pagingSize=80&exused=true")  
  kkh <- read_html(url_list2[i]) %>% html_nodes('._productSet_total') %>% html_text() %>%
    str_replace_all(pattern = '[[:cntrl:]]',replacement = "")   #검색결과 갯수 파악
  fi2[i] <- paste0(kkh," ")
  kkh2 <- read_html(url_list2[i]) %>% html_nodes('.tit') %>% html_nodes('.link') %>% html_text() %>%
    str_replace_all(pattern = '[[:cntrl:]]',replacement = "")
  for ( j in 1:length(kkh2)){
    ww<- paste(ww,kkh2[j], sep = " / ")
    
  }
  itemlist2[i] <- paste0(ww," ") 
  
  for (p in 1:70) {
    traaaaa<- read.csv('C:/Users/USER/Desktop/백업/traaaaa.csv')
    
  }
  print(i)
  
}
head(url_list2)
head(fi2)
head(itemlist2)


#홈플러스 검사
encode<- NULL
url_list3<- NULL
fi3<-NULL
itemlist3 <- NULL
for(i in 1:450) {
  ww<- NULL
  qq<- NULL
  encode[i] <- URLencode(iconv(a[i], to = "UTF-8"))
  qq <-paste0("https://search.shopping.naver.com/search/all.nhn?query=",encode[i]) 
  url_list3[i] <-paste0(qq,"&mall=108756&pagingIndex=1&pagingSize=80&exused=true")  
  kkh <- read_html(url_list3[i]) %>% html_nodes('._productSet_total') %>% html_text() %>%
    str_replace_all(pattern = '[[:cntrl:]]',replacement = "")   #검색결과 갯수 파악
  fi3[i] <- paste0(kkh," ")
  kkh2 <- read_html(url_list3[i]) %>% html_nodes('.tit') %>% html_nodes('.link') %>% html_text() %>%
    str_replace_all(pattern = '[[:cntrl:]]',replacement = "")
  for ( j in 1:length(kkh2)){
    ww<- paste(ww,kkh2[j], sep = " / ")
    
  }
  itemlist3[i] <- paste0(ww," ") 
  
  for (p in 1:70) {
    traaaaa<- read.csv('C:/Users/USER/Desktop/백업/traaaaa.csv')
    
  }
  print(i)
  
}
head(url_list3)
head(fi3)
head(itemlist3)

#롯데슈퍼 검사

encode<- NULL
url_list4<- NULL
fi4<-NULL
itemlist4 <- NULL
for(i in 1:450) {
  ww<- NULL
  qq<- NULL
  encode[i] <- URLencode(iconv(a[i], to = "UTF-8"))
  qq <-paste0("https://search.shopping.naver.com/search/all.nhn?query=",encode[i]) 
  url_list4[i] <-paste0(qq,"&mall=235018&pagingIndex=1&pagingSize=80&exused=true")  
  kkh <- read_html(url_list4[i]) %>% html_nodes('._productSet_total') %>% html_text() %>%
    str_replace_all(pattern = '[[:cntrl:]]',replacement = "")   #검색결과 갯수 파악
  fi4[i] <- paste0(kkh," ")
  kkh2 <- read_html(url_list4[i]) %>% html_nodes('.tit') %>% html_nodes('.link') %>% html_text() %>%
    str_replace_all(pattern = '[[:cntrl:]]',replacement = "")
  for ( j in 1:length(kkh2)){
    ww<- paste(ww,kkh2[j], sep = " / ")
    
  }
  itemlist4[i] <- paste0(ww," ") 
  
  for (p in 1:70) {
    traaaaa<- read.csv('C:/Users/USER/Desktop/백업/traaaaa.csv')
    
  }
  print(i)
  
}
head(url_list4)
head(fi4)
head(itemlist4)

# 합치기
# 일단 두개 합쳐보기
url_list_df <- as.data.frame(url_list)
fi1_df <- as.data.frame(fi1)
itemlist1_df <- as.data.frame(itemlist1)

url_list2_df <- as.data.frame(url_list2)
fi2_df <- as.data.frame(fi2)
itemlist2_df <- as.data.frame(itemlist2)

url_list3_df <- as.data.frame(url_list3)
fi3_df <- as.data.frame(fi3)
itemlist3_df <- as.data.frame(itemlist3)

url_list4_df <- as.data.frame(url_list4)
fi4_df <- as.data.frame(fi4)
itemlist4_df <- as.data.frame(itemlist4)





test <- cbind(url_list_df,fi1_df,itemlist1_df,url_list2_df[1:209,],fi2_df[1:209,],itemlist2_df[1:209,]
              )
write.csv(test,'C:/Users/USER/Desktop/백업/수량파악test12.csv')


#가격파악

encode<- NULL
url_list<- NULL
fi1<-NULL
itemlist1 <- NULL
for(i in 1:450) {
  ww<- NULL
  qq<- NULL
  encode[i] <- URLencode(iconv(a[i], to = "UTF-8"))
  qq <-paste0("https://search.shopping.naver.com/search/all.nhn?query=",encode[i]) 
  url_list[i] <-paste0(qq,"&mall=596&pagingIndex=1&pagingSize=80&exused=true")  
  kkh2 <- read_html(url_list[i]) %>% html_nodes('.price') %>% html_text() %>%
    str_replace_all(pattern = '[[:cntrl:]]',replacement = "") %>%
    str_replace_all(pattern = '                    ',replacement = "") %>%
    str_replace_all(pattern = '가격비교',replacement = "")
  for ( j in 1:length(kkh2)){
    ww<- paste(ww,kkh2[j], sep = " / ")
    
  }
  itemlist1[i] <- paste0(ww," ") 
  
  for (p in 1:40) {
    traaaaa<- read.csv('C:/Users/USER/Desktop/백업/traaaaa.csv')
    
  }
  print(i)
  
}
head(url_list)
fi1
itemlist1


# 롯데온 롯데마트 동시 검색
#롯데온 검사

item_name_new <- read.csv('C:/Users/USER/Desktop/백업/item_new.csv')
b<-as.vector(item_name_new[,1])
Encoding(b)

encode<- NULL
url_list2<- NULL
fi2<-NULL
itemlist2 <- NULL
for(i in 1:450) {
  ww<- NULL
  qq<- NULL
  encode[i] <- URLencode(iconv(b[i], to = "UTF-8"))
  qq <-paste0("https://search.shopping.naver.com/search/all.nhn?query=",encode[i]) 
  url_list2[i] <-paste0(qq,"&mall=107396%5E1243359&pagingIndex=1&pagingSize=80&exused=true")  
  kkh <- read_html(url_list2[i]) %>% html_nodes('._productSet_total') %>% html_text() %>%
    str_replace_all(pattern = '[[:cntrl:]]',replacement = "")   #검색결과 갯수 파악
  fi2[i] <- paste0(kkh," ")
  kkh2 <- read_html(url_list2[i]) %>% html_nodes('.tit') %>% html_nodes('.link') %>% html_text() %>%
    str_replace_all(pattern = '[[:cntrl:]]',replacement = "")


  for ( j in 1:length(kkh2)){
    ww<- paste(ww,kkh2[j], sep = " / ")
    
  }
  itemlist2[i] <- paste0(ww," ") 
  
  for (p in 1:70) {
    traaaaa<- read.csv('C:/Users/USER/Desktop/백업/traaaaa.csv')
    
  }
  print(i)
  
}
head(url_list2)
head(fi2)
head(itemlist2)

url_list2_df <- as.data.frame(url_list2)
fi2_df <- as.data.frame(fi2)
itemlist2_df <- as.data.frame(itemlist2)

test <- cbind(url_list2_df,fi2_df,itemlist2_df)

write.csv(test,'C:/Users/USER/Desktop/백업/수량파악test123.csv')

############## 2가지 방식 같이 써보고 검색결과 낮은것 선택해서 추가
# 롯데온 롯데마트 동시 검색
#롯데온 검사
encode<- NULL
url_list2<- NULL
fi2<-NULL
itemlist2 <- NULL
for(i in 260:450) {
  ww<- NULL
  qq<- NULL
  encode[i] <- URLencode(iconv(a[i], to = "UTF-8"))
  
  qq <-paste0("https://search.shopping.naver.com/search/all.nhn?query=",encode[i]) 
  url_list2[i] <-paste0(qq,"&mall=107396%5E1243359&pagingIndex=1&pagingSize=80&exused=true")  
  kkh <- read_html(url_list2[i]) %>% html_nodes('._productSet_total') %>% html_text() %>%
    str_replace_all(pattern = '[[:cntrl:]]',replacement = "")   #검색결과 갯수 파악
  kkh2 <- read_html(url_list2[i]) %>% html_nodes('.tit') %>% html_nodes('.link') %>% html_text() %>%
    str_replace_all(pattern = '[[:cntrl:]]',replacement = "")
  kkh_0 <- kkh
  kkh2_0 <- kkh2
  lenk <- length(kkh2)
  if(length(kkh) != 0){
    print("있음")
    fi2[i] <- paste0(kkh," ")
    for ( j in 1:length(kkh2)){
      ww<- paste(ww,kkh2[j], sep = " / ")
      
    }
    itemlist2[i] <- paste0(ww," ") 
  }else{
    encode[i] <- URLencode(iconv(b[i], to = "UTF-8"))
    qq <-paste0("https://search.shopping.naver.com/search/all.nhn?query=",encode[i])
    url_list2[i] <-paste0(qq,"&mall=107396%5E1243359&pagingIndex=1&pagingSize=80&exused=true")  
    kkh <- read_html(url_list2[i]) %>% html_nodes('._productSet_total') %>% html_text() %>%
      str_replace_all(pattern = '[[:cntrl:]]',replacement = "")   #검색결과 갯수 파악
    kkh2 <- read_html(url_list2[i]) %>% html_nodes('.tit') %>% html_nodes('.link') %>% html_text() %>%
      str_replace_all(pattern = '[[:cntrl:]]',replacement = "")
    
    
    fi2[i] <- paste0(kkh," ")
    for ( j in 1:length(kkh2)){
      ww<- paste(ww,kkh2[j], sep = " / ")
    }
    print("없음")
  }
  itemlist2[i] <- paste0(ww," ") 
  
  
  for (p in 1:80) {
    traaaaa<- read.csv('C:/Users/USER/Desktop/백업/traaaaa.csv')
    
  }
  print(i)
  
}

url_list2[1:15]
fi2[1:15]
itemlist2

url_list2_fin_df <- as.data.frame(url_list_fin)
fi2_df <- as.data.frame(fi2)
itemlist2_df <- as.data.frame(itemlist2)

test <- cbind(url_list2_fin_df,fi2_df,itemlist2_df)
write.csv(test,'C:/Users/USER/Desktop/백업/수량파악좋은거골라서.csv')

# ###################옵션 선택할 경우
# if(lenk <= length(kkh2)){
#  print("옵션 1 선택")
#  fi2[i] <- paste0(kkh_0," ")
#  for ( j in 1:length(kkh2_0)){
#    ww<- paste(ww,kkh2_0[j], sep = " / ")
#  }else{
#    print("옵션 2 선택")
#    fi2[i] <- paste0(kkh," ")
#    for ( j in 1:length(kkh2)){
#      ww<- paste(ww,kkh2[j], sep = " / ")
#    }


encode<- NULL
url_list2<- NULL
fi2<-NULL
itemlist2 <- NULL
url_list2_b<- NULL
fi2_b<-NULL
itemlist2_b <- NULL
url_list_fin <- NULL
for(i in 324:450) {
  ww<- NULL
  qq<- NULL
  encode[i] <- URLencode(iconv(a[i], to = "UTF-8"))
  qq <-paste0("https://search.shopping.naver.com/search/all.nhn?query=",encode[i]) 
  url_list2[i] <-paste0(qq,"&mall=107396%5E1243359&pagingIndex=1&pagingSize=80&exused=true")  
  kkh <- read_html(url_list2[i]) %>% html_nodes('._productSet_total') %>% html_text() %>%
    str_replace_all(pattern = '[[:cntrl:]]',replacement = "")   #검색결과 갯수 파악
  fi2[i] <- paste0(kkh," ")
  kkh2 <- read_html(url_list2[i]) %>% html_nodes('.tit') %>% html_nodes('.link') %>% html_text() %>%
    str_replace_all(pattern = '[[:cntrl:]]',replacement = "")
  url_list_fin[i] <-url_list2[i]

  
    encode[i] <- URLencode(iconv(b[i], to = "UTF-8"))
  qq <-paste0("https://search.shopping.naver.com/search/all.nhn?query=",encode[i]) 
  url_list2_b[i] <-paste0(qq,"&mall=107396%5E1243359&pagingIndex=1&pagingSize=80&exused=true")  
  kkh_b <- read_html(url_list2_b[i]) %>% html_nodes('._productSet_total') %>% html_text() %>%
    str_replace_all(pattern = '[[:cntrl:]]',replacement = "")   #검색결과 갯수 파악
  fi2_b[i] <- paste0(kkh_b," ")
  kkh2_b <- read_html(url_list2_b[i]) %>% html_nodes('.tit') %>% html_nodes('.link') %>% html_text() %>%
    str_replace_all(pattern = '[[:cntrl:]]',replacement = "")
  
  
  if(length(kkh) == 0){
    lenk <- 10000000
    print("옵션 1 없음")
  }else{
    if(length(kkh_b) == 0){
      lenkb <- 10000000
      print("옵션 2 없음")
    }else{ lenkb <- length(kkh_b)}
    lenk <- length(kkh)
  }
  
  
   if(lenk < lenkb ){
    print("옵션 1 선택")
    fi2[i] <- paste0(kkh," ")
    url_list_fin[i] <-url_list2_b[i]
    for ( j in 1:length(kkh2)){
      ww<- paste(ww,kkh2[j], sep = " / ")
      }
    }else{
      print("옵션 2 선택")
      fi2[i] <- paste0(kkh_b," ")
      
      for ( j in 1:length(kkh2_b)){
        ww<- paste(ww,kkh2_b[j], sep = " / ")
      }
    }
  
  
  
  #  for ( j in 1:length(kkh2)){
#    ww<- paste(ww,kkh2[j], sep = " / ")
#    
#  }
  itemlist2[i] <- paste0(ww," ") 
  
  for (p in 1:90) {
    traaaaa<- read.csv('C:/Users/USER/Desktop/백업/traaaaa.csv')
    
  }
  print(i)
  
}
fi2_b[20:40]


length(read_html(url_list2[4]) %>% html_nodes('.tit') %>% html_nodes('.link') %>% html_text() %>%
  str_replace_all(pattern = '[[:cntrl:]]',replacement = "")) 
length(NULL)

if(length(kkh) == 0){
  lenk <- 10000000
  print("옵션 1 없음")
}else{
  if(length(kkh_b) == 0){
    lenkb = 10000000
    print("옵션 2 없음")
  }else{ lenkb <- length(kkh_b)}
  lenk <- length(kkh)
}

