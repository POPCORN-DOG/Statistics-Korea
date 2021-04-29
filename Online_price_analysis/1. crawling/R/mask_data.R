#함수

#cmd에서 cd C:\r-selenium 후 java -Dwebdriver.gecko.driver="geckodriver.exe" -jar selenium-server-standalone-4.0.0-alpha-1.jar -port 4445 엔터
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
                      browserName = 'chrome') # 기본
remdr$open() # 창을 염
remdr$setWindowSize(width = 1300, height = 1000) #창크기 조절

#크롤링 함수
shopping_kkh <- function(x){
  remdr$navigate(x)
  Sys.sleep(3)
  remdr$executeScript("window.scrollTo(0, 0);")
  Sys.sleep(0.3)
  remdr$executeScript("window.scrollTo(0,document.body.scrollHeight);")
  Sys.sleep(1)
  remdr$executeScript("window.scrollTo(0, 0);")
  Sys.sleep(0.3)
  remdr$executeScript("window.scrollTo(0,document.body.scrollHeight);")
  Sys.sleep(1)
  a<-remdr$findElement(using = 'class', value = 'basicList_link__1MaTN')
  stop<-a$getElementAttribute('href')
  stop<-stop[[1]]
  Sys.sleep(0.3)
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
      items <- remdr$findElements(using = 'css selector', 
                                  value = paste0('#__next > div > div.container > div > div.style_content_wrap__1PzEo > div.style_content__2T20F > ul > div > div > div > div:nth-child(',i,') > li > div > div.basicList_mall_area__lIA7R > div.basicList_mall_title__3MWFY > a.basicList_mall__sbVax > img'))
      item_list <- lapply(items, function(x){x$getElementAttribute("alt")})
      list_shop <- rbind(list_shop,unlist(item_list)) #이미지 판매처 넣기
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
  list_item <- list_item %>% filter(!is.na(list_shop))
  return(list_item)
}
shopping_kkh_fix <- function(x){
  remdr$navigate(x)
  remdr$executeScript("window.scrollTo(0, 0);")
  remdr$executeScript("window.scrollTo(0,document.body.scrollHeight);")
  Sys.sleep(1)
  remdr$executeScript("window.scrollTo(0, 0);")
  html <- read_html(remdr$getPageSource()[[1]])
  list_id <- NULL; list_shop <- NULL; list_price <- NULL; list_dp <- NULL;
  
  box <- html %>% html_nodes('.basicList_inner__eY_mq')
  if (length(box)==0){print('검색결과 없음')}else{
    list_id <- NULL; list_shop <- NULL; list_price <- NULL; list_dp <- NULL;
    items_t <- remdr$findElements(using = 'class', value = 'basicList_mall__sbVax')
    item_list_t <- lapply(items_t, function(x){x$getElementText()})
    
    for (i in 1:length(item_list_t)) {
      item_1 <- str_split(item_list_t[[i]],'\n') # 3가지 타입 구분하기 1 8 12
      id <- box[i] %>% html_nodes('.basicList_info_area__17Xyo') %>%  html_nodes('.basicList_link__1MaTN') %>% html_text()
      
      
      if( item_1[[1]][1] == "" ){ #이미지 판매처
        items <- box[i] %>%  html_nodes('.basicList_mall_title__3MWFY') %>% html_nodes('img') %>% html_attr('alt')
        list_shop <- rbind(list_shop,items) #이미지 판매처 넣기
        #    Sys.sleep(0.3)
        
        list_id <- rbind(list_id,id) #상품명 넣기
        #    Sys.sleep(0.3)
        
        items <- box[i] %>% html_nodes('.basicList_price_area__1UXXR') %>% html_text() %>% str_replace_all('\\D','  ') %>%
          str_sub(1,-3) %>% str_replace_all(' ','') %>% as.integer()
        list_price <- rbind(list_price,items) #가격 넣기
        #    Sys.sleep(0.3)
        
        items <- box[i] %>% html_nodes('.basicList_mall_option__1qEUo') %>% html_nodes('.basicList_option__3eF2s') %>% html_text()
        list_dp <- rbind(list_dp,items[1] %>%  
                           str_replace_all('무료','0') %>% str_replace_all('\\D','') %>% as.integer()) #배송비 넣기
        #    Sys.sleep(0.3)
        print("이미지 판매처")
        cat(i)
      }else{
        
        if(item_1[[1]][1] == "쇼핑몰별 최저가"){
          aa_s <- box[i] %>% html_nodes('.basicList_mall_list__vIiQw') %>% html_nodes('.basicList_mall_name__1XaKA') %>% html_text()
          aa_p <- box[i] %>% html_nodes('.basicList_mall_list__vIiQw') %>% html_nodes('.basicList_price__2r23_') %>% html_text()%>%
            str_replace_all('\\D','') %>% as.integer()
          
          for(j in 1:5){
            
            list_id <- rbind(list_id,id)#상품명 넣기
            #      Sys.sleep(0.3)
            
            list_shop <-rbind(list_shop,aa_s[j])
            
            list_price <- rbind(list_price,aa_p[j])
            #      Sys.sleep(0.3)
            
            list_dp <- rbind(list_dp,'0' %>% as.integer())
          }
          print('가격비교')
          cat(i)
        }else{
          list_shop <- rbind(list_shop,item_1[[1]][1]) #판매처 넣기 
          #      Sys.sleep(0.3)
          
          list_id <- rbind(list_id,id)#상품명 넣기  
          #      Sys.sleep(0.3)
          
          items <- box[i] %>% html_nodes('.basicList_price_area__1UXXR') %>% html_text() %>% str_replace_all('\\D','  ') %>%
            str_sub(1,-3) %>% str_replace_all(' ','') %>% as.integer()
          list_price <- rbind(list_price,items) #가격 넣기
          #      Sys.sleep(0.3)
          
          items <- box[i] %>% html_nodes('.basicList_mall_option__1qEUo') %>% html_nodes('.basicList_option__3eF2s') %>% html_text()
          list_dp <- rbind(list_dp,items[1] %>%  
                             str_replace_all('무료','0') %>% str_replace_all('\\D','') %>% as.integer()) #배송비 넣기
          #      Sys.sleep(0.3)
          print('기본형식')
          cat(i)
        }
      }
    }}; print(paste0(i,'번 끝'))
  list_item <- unique(cbind.data.frame(list_id,list_shop,list_price,list_dp)) 
  list_item[is.na(list_item)] <- 0
  list_item <- list_item %>% filter(!is.na(list_shop))
  return(list_item)
}

#찾고 싶은 상품 검색 후
url <- remdr$getCurrentUrl()
#list_kkh <- shopping_kkh_fix(a23[[1]][1])
#write.csv(list_kkh,'C:/Users/USER/Desktop/t.csv')
a<-shopping_kkh_fix(url)
write.csv(a,'C:/Users/USER/Desktop/t.csv')

url_list <- read.csv('C:/Users/USER/Desktop/예방품목url.csv')
aa<- NULL
df <- NULL
for (i in 1:length(url_list[,1])){
  aa <- shopping_kkh_fix(url_list[,1][i])
  aa$url <- url_list[,1][i]
  df <- rbind(df,aa)
  Sys.sleep(1)
}
item_list <- read.csv('C:/Users/USER/Desktop/예방품목.csv')
data <- merge(item_list,df,by='url')
write.csv(data,'C:/Users/USER/Desktop/예방품목검색결과.csv')

head(mask_data,14)