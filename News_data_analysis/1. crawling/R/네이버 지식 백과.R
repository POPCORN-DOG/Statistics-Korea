# 네이버 지식백과 크롤링

# 라이브러리 ####
library(rvest)
library(dplyr)
library(stringr)
library(readxl)
library(readr)
### test ####
# url 가져오기 
html_test <- read_html('https://terms.naver.com/list.nhn?cid=43702&categoryId=43702')

trurl <- html_test %>% html_nodes('.subject_item')
code <- substr(trurl,61,65)

#체집되는지 검사
urllist <- NULL
url <- 'https://terms.naver.com/list.nhn?cid=43702&categoryId='
for (i in 1:10) {
  urllist[i] <- paste0(url,code[i])
} # 확인하기

urllist # 성공

html_gi <- read_html(urllist[2])

data <- html_gi %>% html_node('.content_list') %>%  html_nodes('.title') %>% html_text() %>% 
  str_replace_all(pattern = '[[:cntrl:]]',replacement = "")
data_1 <- as.data.frame(data) #한국 1체집 성공

#20,000건이 넘는 것은 안나옴

#건수 가져와서 페이지 계산하기
aa <-   html_gi %>% html_node('.path_area') %>% html_node('.count') %>%  html_text() %>%
  str_replace_all(pattern = ',',replacement = "") %>% str_sub(1,-2) %>% as.integer()%/%15 + 1

##########################지역/국가 크롤링########(2만건 이하)###########################

#지역 국가 아시아 해보기
html_test <- read_html('https://terms.naver.com/list.nhn?cid=43702&categoryId=43702') #지역 국가 url

trurl <- html_test %>% html_nodes('.subject_item')
code <- substr(trurl,61,65)
urllist <- NULL
url <- 'https://terms.naver.com/list.nhn?cid=43702&categoryId='
for (i in 1:length(trurl)) { 
  urllist[i] <- paste0(url,code[i]) #중분류 url 리스트 만들기 
} 

data_list <- NULL
data_scale <- NULL

for(k in 2:length(trurl)){
  page_html <- read_html(urllist[k]) #마지막 페이지수 구하기
  page <-  page_html %>% html_node('.path_area') %>% html_node('.count') %>%  html_text() %>%
    str_replace_all(pattern = ',',replacement = "") %>% str_sub(1,-2) %>% as.integer()%/%15 + 1
  for(i in 1:page){
    html_ggi <- read_html(paste0(urllist[k],'&page=',i))
    data <- html_ggi %>% html_node('.content_list') %>%  html_nodes('.title') %>% html_text() %>% 
    str_replace_all(pattern = '[[:cntrl:]]',replacement = "") %>% str_sub(1,-3)
    scale1 <- html_ggi %>% html_nodes('.path_area') %>% html_nodes('.selected') %>% html_text()
    for(j in 1:15){
      data_list <- rbind(data_list,data[j])
      data_scale <- rbind(data_scale,scale1)
    }
    print(i)
  }
}
data_asia <- cbind(data_scale,data_list)

write.csv(data_asia,'test.csv')
# test 성공

#페이지수 계산 하고 크롤링 하기

###한국 가다나 순 크롤링 ####
# 가나다 list 만들기 
ganada <- list('ㄱ','ㄴ','ㄷ','ㄹ','ㅁ','ㅂ','ㅅ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ','0-9','A-Z','기타')

korea_url <- 'https://terms.naver.com/list.nhn?cid=43703&categoryId=43703&so=st3.asc&viewType=&categoryType=&index='

#url list 만들기
url_list20000 <- NULL
for(i in 1:length(ganada)){
  url_list20000[i] <- paste0(korea_url,ganada[i]) 
}

data_list <- NULL
data_scale <- NULL

for(k in 1:length(ganada)){
  page_html <- read_html(url_list20000[k]) #마지막 페이지수 구하기
  page <-  page_html %>% html_node('.path_area') %>% html_node('.count') %>%  html_text() %>%
    str_replace_all(pattern = ',',replacement = "") %>% str_sub(1,-2) %>% as.integer()%/%15 + 1
  for(i in 1:page){
    html_ggi <- read_html(paste0(url_list20000[k],'&page=',i))
    data <- html_ggi %>% html_node('.content_list') %>%  html_nodes('.title') %>% html_text() %>% 
      str_replace_all(pattern = '[[:cntrl:]]',replacement = "") %>% str_sub(1,-3)
    scale1 <- html_ggi %>% html_nodes('.path_area') %>% html_nodes('.selected') %>% html_text()
    for(j in 1:15){
      data_list <- rbind(data_list,data[j])
      data_scale <- rbind(data_scale,scale1)
    }
    print(i)
  }
}
data_korea <- cbind(data_scale,data_list)
write.csv(data_korea,'korea.csv')
