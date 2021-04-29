#기본 크롤링
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
                      browserName = 'chrome') # 기본으로 해야하는것 cmd창에서 뭐뭐 치고 들어가야

remdr$open() # 창을 염
remdr$navigate('https://search.shopping.naver.com/search/all?frm=NVSHATC&pagingIndex=1&pagingSize=40&productSet=total&query=%EC%95%84%EC%9D%B4%EA%B9%A8%EB%81%97%ED%95%B4&sort=rel&timestamp=&viewType=list')
remdr$executeScript("window.scrollTo(0, 0);")
remdr$executeScript("window.scrollTo(0,document.body.scrollHeight);")
remdr$executeScript("window.scrollTo(0, 0);")
html <- read_html(remdr$getPageSource()[[1]])
list_id <- NULL; list_shop <- NULL; list_price <- NULL; list_dp <- NULL;

box <- html %>% html_nodes('.basicList_inner__eY_mq')
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
    aa_p <- box[13] %>% html_nodes('.basicList_mall_list__vIiQw') %>% html_nodes('.basicList_price__2r23_') %>% html_text()%>%
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
}; print(paste0(i,'번 끝'))
list_item <- unique(cbind.data.frame(list_id,list_shop,list_price,list_dp)) 
list_item[is.na(list_item)] <- 0
list_item <- list_item %>% filter(!is.na(list_shop))


