###########아이깨끗해 해보기 ############
#cmd 창에서 cd C:\r-selenium 엔터 java -Dwebdriver.gecko.driver="geckodriver.exe" -jar selenium-server-standalone-4.0.0-alpha-1.jar -port 4445 엔터
library(dplyr)
library(httr)
library(jsonlite)
library(RSelenium)
library(stringr)
library(XML)
library(rvest)
library(readxl)
library(readr)
library(rJava)

remdr <- remoteDriver(remoteServerAddr = "localhost",port = 4445L,
                      browserName = 'chrome') # 기본으로 해야하는것 cmd창에서 뭐뭐 치고 들어가야함
remdr$open() # 창을 염
#아이 깨끗해 검색창 나오기
remdr$navigate('https://search.shopping.naver.com/search/all?baseQuery=%EC%95%84%EC%9D%B4%EA%B9%A8%EB%81%97%ED%95%B4%20250ml&frm=NVSHTTL&pagingIndex=1&pagingSize=40&productSet=total&query=%EC%95%84%EC%9D%B4%EA%B9%A8%EB%81%97%ED%95%B4%20250ml&sort=rel&timestamp=&viewType=list')


remdr$setWindowSize(width = 1300, height = 1000) # 창조절
remdr$executeScript("window.scrollTo(0, 0);")

#밑에서 url 주소 가져오기
remdr$executeScript("window.scrollTo(0,document.body.scrollHeight);")
remdr$executeScript("window.scrollTo(0, 0);")
remdr$executeScript("window.scrollTo(0,document.body.scrollHeight);")
a<-remdr$findElement(using = 'class', value = 'basicList_link__1MaTN')
stop<-a$getElementAttribute('href')
stop<-stop[[1]]

#스크롤 내리면서 상품명 판매처 가격 배송비 검색
remdr$executeScript("window.scrollTo(0, 0);")
i <- 1; url <- ""; list_id <- NULL; list_shop <- NULL; list_price <- NULL; list_dp <- NULL
while(stop != url){ #반복횟수와 스크롤 내리는 횟수 조절해야함
  cat(i)
  Sys.sleep(rnorm(n=1,mean = 0.7,sd=0.18))
  i = i + 1
  remdr$executeScript(paste0("window.scrollBy(0, 90);"))
  items <- remdr$findElements(using = 'class', value = 'basicList_mall__sbVax')
  item_list <- lapply(items, function(x){x$getElementText()})
  item1 <- str_split(item_list[[1]],'\n')
  
  a<-remdr$findElement(using = 'class', value = 'basicList_link__1MaTN')
  url<-a$getElementAttribute('href')
  url<-url[[1]]
  
  if( item1[[1]][1] == "" ){
    items <- remdr$findElements(using = 'css selector', value = 'img')
    item_list <- lapply(items, function(x){x$getElementAttribute("alt")})
    item1 <- str_split(item_list[[5]],'\n')
    list_shop <- rbind(list_shop,item1[[1]][1]) #이미지 판매처 넣기
    #    Sys.sleep(0.3)
    
    
    items <- remdr$findElements(using = 'class', value = 'basicList_title__3P9Q7')
    item_list <- lapply(items, function(x){x$getElementText()})
    item1 <- str_split(item_list[[1]],'\n')
    list_id <- rbind(list_id,item1[[1]][1]) #상품명 넣기
    #    Sys.sleep(0.3)
    
    
    items <- remdr$findElements(using = 'class', value = 'basicList_price__2r23_')
    item_list <- lapply(items, function(x){x$getElementText()})
    item1 <- str_split(item_list[[1]],'\n')
    list_price <- rbind(list_price,item1[[1]][1] %>% str_replace_all('\\D','') %>% as.integer()) #가격 넣기
    #    Sys.sleep(0.3)
    
    qwe<-remdr$findElements(using = 'class', value = 'basicList_mall_area__lIA7R')
    item_list <- lapply(qwe, function(x){x$getElementText()})
    qwe1<-unlist(str_split(item_list[[1]],'\n'))
    if(length(qwe1[str_detect(qwe1,"배송비")]) == 0){
      list_dp <- rbind(list_dp, '0')
    }else{
      list_dp <- rbind(list_dp,qwe1[str_detect(qwe1,"배송비")] %>% str_replace_all('\\D','') %>% as.integer())
    }  #배송비 넣기
    #    Sys.sleep(0.3)
    print("이미지 판매처")
    
  }else{
    
    if(item1[[1]][1] == "쇼핑몰별 최저가"){
      html2 <-read_html(remdr$getPageSource()[[1]])
      aa_p <- html2 %>% html_node('.basicList_mall_area__lIA7R') %>% html_nodes('.basicList_price__2r23_') %>% html_text() %>% 
        str_replace_all(',','') %>% as.integer()
      aa_s <- html2 %>% html_node('.basicList_mall_area__lIA7R') %>% html_nodes('.basicList_mall_name__1XaKA') %>% html_text()
      
      for(j in 1:5){
        
        items <- remdr$findElements(using = 'class', value = 'basicList_title__3P9Q7')
        item_list <- lapply(items, function(x){x$getElementText()})
        item1 <- str_split(item_list[[1]],'\n')
        list_id <- rbind(list_id,item1[[1]][1])
        #      Sys.sleep(0.3)
        
        list_shop <-rbind(list_shop,aa_s[j])
        
        items <- remdr$findElements(using = 'class', value = 'price_num__2WUXn')
        item_list <- lapply(items, function(x){x$getElementText()})
        item1 <- str_split(item_list[[1]],'\n')
        list_price <- rbind(list_price,aa_p[j])
        #      Sys.sleep(0.3)
        
        list_dp <- rbind(list_dp,'0')
      }
      print('가격비교')
   
    }else{
      list_shop <- rbind(list_shop,item1[[1]][1]) #판매처 넣기 
      #      Sys.sleep(0.3)
      
      
      items <- remdr$findElements(using = 'class', value = 'basicList_title__3P9Q7')
      item_list <- lapply(items, function(x){x$getElementText()})
      item1 <- str_split(item_list[[1]],'\n')
      list_id <- rbind(list_id,item1[[1]][1]) #상품명 넣기
      #      Sys.sleep(0.3)
      
      
      items <- remdr$findElements(using = 'class', value = 'basicList_price__2r23_')
      item_list <- lapply(items, function(x){x$getElementText()})
      item1 <- str_split(item_list[[1]],'\n')
      list_price <- rbind(list_price,item1[[1]][1] %>% str_replace_all('\\D','') %>% as.integer()) #가격 넣기
      #      Sys.sleep(0.3)
      qwe<-remdr$findElements(using = 'class', value = 'basicList_mall_area__lIA7R')
      item_list <- lapply(qwe, function(x){x$getElementText()})
      qwe1<-unlist(str_split(item_list[[1]],'\n'))
      if(length(qwe1[str_detect(qwe1,"배송비")]) == 0){
        list_dp <- rbind(list_dp, '0')
      }else{
        list_dp <- rbind(list_dp,qwe1[str_detect(qwe1,"배송비")] %>% str_replace_all('\\D','') %>% as.integer())
      } 
      #      Sys.sleep(0.3)
      print('기본형식')
      
    }
    Sys.sleep(0.5)
  
  }
  
  if(i>230) break
}; print(paste0(i,'번 끝'));for(i in 1:length(lapply(remdr$findElements(using = 'class', value = 'basicList_mall__sbVax'), function(x){x$getElementText()}))){ # 마지막 5개 
  
  items <- remdr$findElements(using = 'class', value = 'basicList_mall__sbVax')
  item_list <- lapply(items, function(x){x$getElementText()})
  item1 <- str_split(item_list[[i]],'\n')
  
  if( item1[[1]][1] == ""){
    items <- remdr$findElements(using = 'css selector', value = 'img')
    item_list <- lapply(items, function(x){x$getElementAttribute("alt")})
    item1 <- str_split(item_list[[3*i+3]],'\n')
    list_shop <- rbind(list_shop,item1[[1]][1]) #이미지 판매처 넣기
    #    Sys.sleep(0.3)
    
    
    items <- remdr$findElements(using = 'class', value = 'basicList_title__3P9Q7')
    item_list <- lapply(items, function(x){x$getElementText()})
    item1 <- str_split(item_list[[i]],'\n')
    list_id <- rbind(list_id,item1[[1]][1]) #상품명 넣기
    #    Sys.sleep(0.3)
    
    
    items <- remdr$findElements(using = 'class', value = 'basicList_price__2r23_')
    item_list <- lapply(items, function(x){x$getElementText()})
    item1 <- str_split(item_list[[i]],'\n')
    list_price <- rbind(list_price,item1[[1]][1] %>% str_replace_all('\\D','') %>% as.integer()) #가격 넣기
    #    Sys.sleep(0.3)
    
    
    qwe<-remdr$findElements(using = 'class', value = 'basicList_mall_area__lIA7R')
    item_list <- lapply(qwe, function(x){x$getElementText()})
    qwe1<-unlist(str_split(item_list[[i]],'\n'))
    if(length(qwe1[str_detect(qwe1,"배송비")]) == 0){
      list_dp <- rbind(list_dp, '0')
    }else{
      list_dp <- rbind(list_dp,qwe1[str_detect(qwe1,"배송비")] %>% str_replace_all('\\D','') %>% as.integer())
    }
    #    Sys.sleep(0.3)
    print("이미지 판매처")
    cat(i)
  }else{
    
    if(item1[[1]][1] == "쇼핑몰별 최저가"){
      html2 <-read_html(remdr$getPageSource()[[i]])
      aa_p <- html2 %>% html_node('.basicList_mall_area__lIA7R') %>% html_nodes('.basicList_price__2r23_') %>% html_text() %>% 
        str_replace_all(',','') %>% as.integer()
      aa_s <- html2 %>% html_node('.basicList_mall_area__lIA7R') %>% html_nodes('.basicList_mall_name__1XaKA') %>% html_text()
      
      for(j in 1:5){
        
        items <- remdr$findElements(using = 'class', value = 'basicList_title__3P9Q7')
        item_list <- lapply(items, function(x){x$getElementText()})
        item1 <- str_split(item_list[[i]],'\n')
        list_id <- rbind(list_id,item1[[1]][1])
        #      Sys.sleep(0.3)
        
        list_shop <-rbind(list_shop,aa_s[j])
        
        items <- remdr$findElements(using = 'class', value = 'price_num__2WUXn')
        item_list <- lapply(items, function(x){x$getElementText()})
        item1 <- str_split(item_list[[i]],'\n')
        list_price <- rbind(list_price,aa_p[j])
        #      Sys.sleep(0.3)
        
        list_dp <- rbind(list_dp,'0')
      }
      print('가격비교')
      cat(i)
    }else{
      list_shop <- rbind(list_shop,item1[[1]][1]) #판매처 넣기 
      #      Sys.sleep(0.3)
      
      
      items <- remdr$findElements(using = 'class', value = 'basicList_title__3P9Q7')
      item_list <- lapply(items, function(x){x$getElementText()})
      item1 <- str_split(item_list[[i]],'\n')
      list_id <- rbind(list_id,item1[[1]][1]) #상품명 넣기
      #      Sys.sleep(0.3)
      
      
      items <- remdr$findElements(using = 'class', value = 'basicList_price__2r23_')
      item_list <- lapply(items, function(x){x$getElementText()})
      item1 <- str_split(item_list[[i]],'\n')
      list_price <- rbind(list_price,item1[[1]][1] %>% str_replace_all('\\D','') %>% as.integer()) #가격 넣기
      #      Sys.sleep(0.3)
      qwe<-remdr$findElements(using = 'class', value = 'basicList_mall_area__lIA7R')
      item_list <- lapply(qwe, function(x){x$getElementText()})
      qwe1<-unlist(str_split(item_list[[i]],'\n'))
        if(length(qwe1[str_detect(qwe1,"배송비")]) == 0){
          list_dp <- rbind(list_dp, '0')
        }else{
          list_dp <- rbind(list_dp,qwe1[str_detect(qwe1,"배송비")] %>% str_replace_all('\\D','') %>% as.integer())
        }
      #      Sys.sleep(0.3)
      

      print('기본형식')
    
    }
  }
}; print('끝')
list_item <- unique(cbind.data.frame(list_id,list_shop,list_price,list_dp))  
list_item[is.na(list_item)] <-0 

