#크롤링 필요 패키지 깔기
#install.packages(c('rvest','dplyr','stringr'))
library(rvest)  # 크롤링 필수 패키지
library(dplyr)  # 파이프라인 등 분석을 편하게 하게하는 패키지
library(stringr)  # 텍스트 파일을 다룰 수 있게하는 패키지

urllist <- list('https://www.trendforce.com/price','https://www.trendforce.com/price/flash','https://www.trendforce.com/price/ssd','https://www.trendforce.com/price/lcd',
                'https://www.trendforce.com/price/pv')

for( i in 1:5){
  html <- read_html(urllist[[i]])
  a <- html %>% html_nodes('table')
  b<-html %>% html_nodes('.text-right')
  
  if( i == 1){
    name = 'DRAM'
  }else{
    asd <- urllist[[i]] %>% str_split('/')
    name <- asd[[1]][length(asd[[1]])]    
  }
  for(j in 1:length(a)){
    aa <- a[j] %>% html_table()
    dt <- aa[[1]]
    dt$update <-  b[j] %>% html_text %>% str_remove_all('\\D')
    write.csv(dt,paste0('C:/Users/USER/Desktop/', name, j,'.csv'))
  }
}
