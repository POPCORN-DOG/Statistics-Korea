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
remdr$navigate('https://search.shopping.naver.com/search/all?baseQuery=%EC%95%84%EC%9D%B4%EA%B9%A8%EB%81%97%ED%95%B4%20250ml&frm=NVSHTTL&pagingIndex=1&pagingSize=80&productSet=total&query=%EC%95%84%EC%9D%B4%EA%B9%A8%EB%81%97%ED%95%B4%20250ml&sort=rel&timestamp=&viewType=list')
remdr$navigate('https://search.shopping.naver.com/search/all.nhn?origQuery=%EA%B7%B8%EB%A6%B0%EC%A0%9C%EC%95%BD%20%EC%86%8C%EB%8F%85%EC%9A%A9%20%EC%97%90%ED%83%84%EC%98%AC%201l%20-2%EA%B0%9C%20-2%EB%B3%91%20-20%ED%86%B5%20-20%EA%B0%9C&pagingIndex=1&pagingSize=40&viewType=list&sort=rel&frm=NVSHATC&query=%EA%B7%B8%EB%A6%B0%EC%A0%9C%EC%95%BD%20%EC%86%8C%EB%8F%85%EC%9A%A9%20%EC%97%90%ED%83%84%EC%98%AC%201l&xq=2%EA%B0%9C%202%EB%B3%91%2020%ED%86%B5%2020%EA%B0%9C')

remdr$setWindowSize(width = 1300, height = 1000) # 창조절
remdr$executeScript("window.scrollTo(0, 0);")
#remdr$executeScript("window.scrollTo(0,document.body.scrollHeight);")스크롤 밑으로 내리기

#밑에서 url 주소 가져오기
remdr$executeScript("window.scrollTo(0,document.body.scrollHeight);")
a<-remdr$findElement(using = 'class', value = 'basicList_link__1MaTN')
stop<-a$getElementAttribute('href')
stop<-stop[[1]]

#스크롤 내리면서 상품명 판매처 가격 배송비 검색
list_id <- NULL; list_shop <- NULL; list_price <- NULL; list_dp <- NULL;
remdr$executeScript("window.scrollTo(0, 0);")
for(i in 1:170){ #반복횟수와 스크롤 내리는 횟수 조절해야함
  remdr$executeScript(paste0("window.scrollTo(0, 100 *",i,");"))
  
  items <- remdr$findElements(using = 'class', value = 'basicList_mall__sbVax')
  item_list <- lapply(items, function(x){x$getElementText()})
  item1 <- str_split(item_list[[1]],'\n')
  
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
    
    
    items <- remdr$findElements(using = 'class', value = 'price_num__2WUXn')
    item_list <- lapply(items, function(x){x$getElementText()})
    item1 <- str_split(item_list[[1]],'\n')
    list_price <- rbind(list_price,item1[[1]][1] %>% str_replace_all(',','') %>% str_sub(1,-2) %>% as.integer()) #가격 넣기
    #    Sys.sleep(0.3)
    
    items <- remdr$findElements(using = 'class', value = 'basicList_option__3eF2s')
    item_list <- lapply(items, function(x){x$getElementText()})
    item1 <- str_split(item_list[[1]],'\n')
    list_dp <- rbind(list_dp,item1[[1]][1] %>% str_replace_all('배송비','') %>% 
                       str_replace_all('무료','0') %>% str_replace_all(',','') %>% str_sub(1,-2) %>% as.integer()) #배송비 넣기
    #    Sys.sleep(0.3)
    print("이미지 판매처")
    cat(i)
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
      cat(i)
    }else{
      list_shop <- rbind(list_shop,item1[[1]][1]) #판매처 넣기 
      #      Sys.sleep(0.3)
      
      
      items <- remdr$findElements(using = 'class', value = 'basicList_title__3P9Q7')
      item_list <- lapply(items, function(x){x$getElementText()})
      item1 <- str_split(item_list[[1]],'\n')
      list_id <- rbind(list_id,item1[[1]][1]) #상품명 넣기
      #      Sys.sleep(0.3)
      
      
      items <- remdr$findElements(using = 'class', value = 'price_num__2WUXn')
      item_list <- lapply(items, function(x){x$getElementText()})
      item1 <- str_split(item_list[[1]],'\n')
      list_price <- rbind(list_price,item1[[1]][1] %>% str_replace_all(',','') %>% str_sub(1,-2) %>% as.integer()) #가격 넣기
      #      Sys.sleep(0.3)
      
      items <- remdr$findElements(using = 'class', value = 'basicList_option__3eF2s')
      item_list <- lapply(items, function(x){x$getElementText()})
      item1 <- str_split(item_list[[1]],'\n')
      list_dp <- rbind(list_dp,item1[[1]][1] %>% str_replace_all('배송비','') %>% 
                         str_replace_all('무료','00') %>% str_replace_all(',','') %>% str_sub(1,-2) %>% as.integer()) #배송비 넣기
      #      Sys.sleep(0.3)
      print('기본형식')
      cat(i)
    }
  }
}; print(paste0(i,'번 끝'))

#합치기
list_item <- unique(cbind.data.frame(list_id,list_shop,list_price,list_dp))  
write.csv(list_item,'아이깨끗해.csv')
#수정해야 할 것 가격비교에서 5개가 안된 것도 5개로 나오고 na로 표시됨 삭제해야함, 배송비 na로 표시되는거 0으로 바꿔야함


ck_id <- NULL
remdr$executeScript("window.scrollTo(0,document.body.scrollHeight);")
a<-remdr$findElement(using = 'class', value = 'basicList_link__1MaTN')
stop<-a$getElementAttribute('href')
stop<-stop[[1]]
remdr$executeScript("window.scrollTo(0,0);")

i <- 1
url <- ""
list_id <- NULL; list_shop <- NULL; list_price <- NULL; list_dp <- NULL
while(stop != url){ #반복횟수와 스크롤 내리는 횟수 조절해야함
  i = i + 1
  remdr$executeScript(paste0("window.scrollBy(0, 100);"))
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
    
    
    items <- remdr$findElements(using = 'class', value = 'price_num__2WUXn')
    item_list <- lapply(items, function(x){x$getElementText()})
    item1 <- str_split(item_list[[1]],'\n')
    list_price <- rbind(list_price,item1[[1]][1] %>% str_replace_all(',','') %>% str_sub(1,-2) %>% as.integer()) #가격 넣기
    #    Sys.sleep(0.3)
    
    items <- remdr$findElements(using = 'class', value = 'basicList_option__3eF2s')
    item_list <- lapply(items, function(x){x$getElementText()})
    item1 <- str_split(item_list[[1]],'\n')
    list_dp <- rbind(list_dp,item1[[1]][1] %>% str_replace_all('배송비','') %>% 
                       str_replace_all('무료','0') %>% str_replace_all(',','') %>% str_sub(1,-2) %>% as.integer()) #배송비 넣기
    #    Sys.sleep(0.3)
    print("이미지 판매처")
    #cat(i)
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
      #cat(i)
    }else{
      list_shop <- rbind(list_shop,item1[[1]][1]) #판매처 넣기 
      #      Sys.sleep(0.3)
      
      
      items <- remdr$findElements(using = 'class', value = 'basicList_title__3P9Q7')
      item_list <- lapply(items, function(x){x$getElementText()})
      item1 <- str_split(item_list[[1]],'\n')
      list_id <- rbind(list_id,item1[[1]][1]) #상품명 넣기
      #      Sys.sleep(0.3)
      
      
      items <- remdr$findElements(using = 'class', value = 'price_num__2WUXn')
      item_list <- lapply(items, function(x){x$getElementText()})
      item1 <- str_split(item_list[[1]],'\n')
      list_price <- rbind(list_price,item1[[1]][1] %>% str_replace_all(',','') %>% str_sub(1,-2) %>% as.integer()) #가격 넣기
      #      Sys.sleep(0.3)
      
      items <- remdr$findElements(using = 'class', value = 'basicList_option__3eF2s')
      item_list <- lapply(items, function(x){x$getElementText()})
      item1 <- str_split(item_list[[1]],'\n')
      list_dp <- rbind(list_dp,item1[[1]][1] %>% str_replace_all('배송비','') %>% 
                         str_replace_all('무료','00') %>% str_replace_all(',','') %>% str_sub(1,-2) %>% as.integer()) #배송비 넣기     
      #      Sys.sleep(0.3)
      print('기본형식')
      #cat(i)
    }
    Sys.sleep(0.5)
  }
  
  if(i>200) break
}; print(paste0(i,'번 끝'));
list_item <- unique(cbind.data.frame(list_id,list_shop,list_price,list_dp))  

#뒤에 4개 넣기
remdr$findElements(using = 'class', value = 'basicList_mall__sbVax')
hh<-lapply(remdr$findElements(using = 'class', value = 'basicList_mall__sbVax'), function(x){x$getElementText()})

for(i in 1:5){ # 마지막 5개 

  items <- remdr$findElements(using = 'class', value = 'basicList_mall__sbVax')
  item_list <- lapply(items, function(x){x$getElementText()})
  item1 <- str_split(item_list[[i]],'\n')
  
  if( item1[[1]][1] == "" ){
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
    
    
    items <- remdr$findElements(using = 'class', value = 'price_num__2WUXn')
    item_list <- lapply(items, function(x){x$getElementText()})
    item1 <- str_split(item_list[[i]],'\n')
    list_price <- rbind(list_price,item1[[1]][1] %>% str_replace_all(',','') %>% str_sub(1,-2) %>% as.integer()) #가격 넣기
    #    Sys.sleep(0.3)
    
    items <- remdr$findElements(using = 'class', value = 'basicList_option__3eF2s')
    item_list <- lapply(items, function(x){x$getElementText()})
    ss<-unlist(item_list)
    item_list<-ss[str_detect(ss,"배송비")]
    item1 <- str_split(item_list[i],'\n')
    list_dp <- rbind(list_dp,item1[[1]][1] %>% str_replace_all('배송비','') %>% 
                       str_replace_all('무료','0') %>% str_replace_all(',','') %>% str_sub(1,-2) %>% as.integer()) #배송비 넣기
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
      
      
      items <- remdr$findElements(using = 'class', value = 'price_num__2WUXn')
      item_list <- lapply(items, function(x){x$getElementText()})
      item1 <- str_split(item_list[[i]],'\n')
      list_price <- rbind(list_price,item1[[1]][1] %>% str_replace_all(',','') %>% str_sub(1,-2) %>% as.integer()) #가격 넣기
      #      Sys.sleep(0.3)
      
      items <- remdr$findElements(using = 'class', value = 'basicList_option__3eF2s')
      item_list <- lapply(items, function(x){x$getElementText()})
      ss<-unlist(item_list)
      item_list<-ss[str_detect(ss,"배송비")]
      item1 <- str_split(item_list[[i]],'\n')
      list_dp <- rbind(list_dp,item1[[1]][1] %>% str_replace_all('배송비','') %>% 
                         str_replace_all('무료','00') %>% str_replace_all(',','') %>% str_sub(1,-2) %>% as.integer()) #배송비 넣기
      #      Sys.sleep(0.3)
      print('기본형식')
      cat(i)
    }
  }
}; print(paste0(i,'번 끝'))
