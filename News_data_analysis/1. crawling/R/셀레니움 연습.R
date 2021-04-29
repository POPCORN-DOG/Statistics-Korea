# 셀레니움 해보기

#cmd 창에서 cd C:\r-selenium 엔터 java -Dwebdriver.gecko.driver="geckodriver.exe" -jar selenium-server-standalone-4.0.0-alpha-1.jar -port 4445 엔터

# java install
Sys.setenv(JAVA_HOME = 'C:/Program Files/Java/jre1.8.0_251')
install.packages("rJava")
library(rJava)

# selenium package download
install.packages(c("dplyr", "httr", "jsonlite", "RSelenium", "stringr"))                 

library(dplyr)
library(httr)
library(jsonlite)
library(RSelenium)
library(stringr)
library(XML)
library(rvest)
library(readxl)
library(readr)

#login naver 
remdr <- remoteDriver(remoteServerAddr = "localhost",port = 4445L,
                      browserName = 'chrome') # 기본으로 해야하는것 cmd창에서 뭐뭐 치고 들어가야함

remdr$open() # 창을 염
remdr$navigate('https://nid.naver.com/nidlogin.login?mode=form&url=https%3A%2F%2Fwww.naver.com') #창에 url 지정
remdr$getCurrentUrl() 
id <- remdr$findElement(using = 'id', value = 'id') # 아이디 쓰는 input의 아이디가 id
pw <- remdr$findElement(using = 'id', value = 'pw') # 비밀번호 쓰는 input의 아이디가 pw

btn <- remdr$findElement(using = 'class', value = 'btn_global') # 로그인 버튼 누르는 class는 btn_global

# setElementAttribute() 사용해서 값 넣기
id$setElementAttribute('value', 'rlgus2007')
pw$setElementAttribute('value', 'pwpwpwpwpwpw')
btn$clickElement()

remdr$getCurrentUrl() #바뀐 url 확인

# 네이버 쇼핑 구매목록 가져오기 
#내 쇼핑 페이지로 이동
page <- remdr$navigate("http://order.pay.naver.com/home")
#구매목록 가져오기
items<- remdr$findElements(using = "class", value = 'name')
items
item_list <- lapply(items, function(x){x$getElementText()})
item_list
i_df <- as.data.frame(item_list)
i_df
item_list_vec<-unlist(item_list)
typeof(item_list) #데이터 타입 알아보기
#성공

remdr$navigate('https://search.shopping.naver.com/search/all?baseQuery=%EC%95%84%EC%9D%B4%EA%B9%A8%EB%81%97%ED%95%B4%20250ml&frm=NVSHATC&pagingIndex=2&pagingSize=20&productSet=total&query=%EC%95%84%EC%9D%B4%EA%B9%A8%EB%81%97%ED%95%B4%20250ml&sort=rel&timestamp=&viewType=thumb') #창에 url 지
remdr$setWindowSize(width = 1000, height = 2000)
remdr$executeScript("window.scrollTo(0, 800);")
items <- remdr$findElements(using = 'class', value = 'imgList_list_item__226HB')
item_list <- lapply(items, function(x){x$getElementText()})
item_list[[1]]
tt<- str_split(item_list[[1]],'\n')
tt[[1]][1]
remdr$quit()



#####정리#####
item_name <- NULL  #1번째 list
item_price <- NULL #2 번째 list
item_3 <- NULL #3번째 list
item_4 <- NULL #4
item_5 <- NULL #5
for(i in 1:4){
  remdr$open()
  remdr$navigate(paste0('https://search.shopping.naver.com/search/all?baseQuery=%EC%95%84%EC%9D%B4%EA%B9%A8%EB%81%97%ED%95%B4%20250ml&frm=NVSHATC&pagingIndex=',i,'&pagingSize=20&productSet=total&query=%EC%95%84%EC%9D%B4%EA%B9%A8%EB%81%97%ED%95%B4%20250ml&sort=rel&timestamp=&viewType=thumb'))
  remdr$setWindowSize(width = 1300, height = 3000)
  remdr$executeScript("window.scrollTo(0, 1000);")
  items <- remdr$findElements(using = 'class', value = 'imgList_info_area__-L6s4')
  item_list <- lapply(items, function(x){x$getElementText()})
  for(j in 1:25){
    item1 <- str_split(item_list[[j]],'\n')
    item_name <- rbind(item_name,item1[[1]][1])
    item_price <- rbind(item_price,item1[[1]][2])
    item_3 <- rbind(item_3,item1[[1]][3])
    item_4 <- rbind(item_4,item1[[1]][4])
    item_5 <- rbind(item_5,item1[[1]][5])
    }
  remdr$quit()
}

test_data <- cbind.data.frame(item_name,item_price,item_3,item_4,item_5)

write.csv(test_data,'test_data.csv')


######리스트형 일때 #########

remdr$open()
remdr$navigate('https://search.shopping.naver.com/search/all?baseQuery=%EC%95%84%EC%9D%B4%EA%B9%A8%EB%81%97%ED%95%B4%20250ml&frm=NVSHTTL&pagingIndex=2&pagingSize=80&productSet=total&query=%EC%95%84%EC%9D%B4%EA%B9%A8%EB%81%97%ED%95%B4%20250ml&sort=rel&timestamp=&viewType=list')
remdr$setWindowSize(width = 1300, height = 3000)
remdr$executeScript("window.scrollTo(0, 1500);")
items <- remdr$findElements(using = 'class', value = 'basicList_inner__eY_mq')
item_list <- lapply(items, function(x){x$getElementText()})
item1 <- str_split(item_list[[6]],'\n')
item1


items <- remdr$findElements(using = 'class', value = 'basicList_mall__sbVax')
items <- remdr$findElements(using = 'xpath', value = '//*[@id="__next"]/div/div[2]/div/div[4]/div[1]/ul/div/div/div/div/li/div/div[3]/div[1]/a[1]/img') #가격비교용
item_list <- lapply(items, function(x){x$getElementText()})
item_list <- lapply(items, function(x){x$getElementAttribute("alt")})

item_list[[4]]
items[[5]]$getElementAttribute("alt")
items[[5]]$getElementText()

remdr$executeScript("window.blur(-10000);")
#하나하나 검사

remdr$setWindowSize(width = 1300, height = 1000)

for(i in 1:180){
  remdr$executeScript(paste0("window.scrollTo(0, 100 *",i,");"))
  items <- remdr$findElements(using = 'class', value = 'basicList_title__3P9Q7')
  item_list <- lapply(items, function(x){x$getElementText()})
  item1 <- str_split(item_list[[1]],'\n')
  item_name <- rbind(item_name,item1[[1]][1])
  Sys.sleep(0.3)
    }

a<-unique(item_name)

#행마다 걸러내고 비교상푸 ㅁ걸러내고

#비교 상품 만들기
items <- remdr$findElements(using = 'class', value = 'basicList_mall_name__1XaKA')
item_list <- lapply(items, function(x){x$getElementText()}) #이름 가져오기

items <- remdr %>% findElements(using = 'class', value = 'basicList_price__2r23_')
item_list <- lapply(items, function(x){x$getElementText()}) #하는중

#가격,이름 가져오기 
html2 <-read_html(remdr$getPageSource()[[1]])
html2 %>% html_node('.basicList_mall_area__lIA7R') %>% html_nodes('.basicList_price__2r23_') %>% html_text()
html2 %>% html_node('.basicList_mall_area__lIA7R') %>% html_nodes('.basicList_mall_name__1XaKA') %>% html_text()

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

#스크롤 내리면서 상품명 판매처 가격 배송비 검색
list_id <- NULL; list_shop <- NULL; list_price <- NULL; list_dp <- NULL

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
#스크롤 맨 위로 올리는 것도 찾아봐야 함
##############
items <-remdr$findElements(using = 'xpath', value = '//*[@id="__next"]/div/div[2]/div/div[3]/div[1]/ul/div/div')
item_list <- lapply(items, function(x){x$getElementText()})
item1 <- str_split(item_list[[1]],'\n')
