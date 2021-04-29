##### xlsx파일 버전 (구버전)
# 0> 필요한 패키지 로드 =====================================================
install.packages('openxlsx')
install.packages('dplyr')
install.packages('rvest')
install.packages('stringr')
install.packages("progress")
library(progress)
library(openxlsx)
library(dplyr)
library(rvest)
library(stringr)
library(tidyr)


choose_file_name = function(file='Select_data_file') {
  
  filename<-tcltk::tk_choose.files(file)
  data <- read.xlsx(filename)
  
  # 1> 소재지주소 na인 경우 도로명 주소 가져오기 ==============================
  data_doro <- data[is.na(data$소재지전체주소),'도로명전체주소'] %>% str_replace_all("[[:space:]]","+");
  row_number <- which(is.na(data$소재지전체주소), arr.ind=TRUE)
  
  
  # 2> 도로명 주소 > 소재지 주소 =============================================
  pb <- progress_bar$new(format = "downloading [:bar] :current/:total (:percent) in :elapsedfull/:eta (도로명주소변환)",
                         total = length(data_doro), clear = FALSE, width= 60)
  
  for(i in 1:length(data_doro)){
    
    # progress bar
    pb$tick()
    
    # 괄호,쉼표 기준으로 데이터를 분류 후 제일 첫 번째 텍스트(주소)를 가져온다.
    a <- data_doro[i] %>% strsplit('[(]|,')
    doro_address <- str_trim(a[[1]][1])
    
    # 제일 첫 번째 텍스트(주소)로 url 생성
    url <- paste0('https://www.juso.go.kr/support/AddressMainSearch.do?searchKeyword='
                  ,doro_address,
                  '&dsgubuntext=&dscity1text=&dscounty1text=&dsemd1text=&dsri1text=&dssan1text=&dsrd_nm1text=')
    
    
    # 해당 주소를 도로명주소 웹을 통해 소재지주소로 변환한다.
    tryCatch(
      data$소재지전체주소[row_number[i]] <- 
        read_html(url) %>%
        html_nodes(".subject_area") %>% 
        html_nodes(".subejct_2") %>% 
        html_nodes(".roadNameText") %>% 
        html_text() %>% 
        str_replace_all("\t|\n|\r","") %>% 
        str_replace_all("  "," "),
      error = function(e) {}
    )
    #  cat(i,row_number[i],data$소재지전체주소[row_number[i]],'\n')
  }
  
  
  
  # 3> 만약 동 주소를 검색하지 못했을 때 NA값을 도로명 주소로 대체한다.
  df = data.frame(x=as.factor(data$소재지전체주소),y=as.factor(data$도로명전체주소))
  df2 = as.data.frame(t(apply(df,1, function(x) { return(c(x[!is.na(x)],x[is.na(x)]) )} )))
  data$소재지전체주소 <- df2[,1]
  
  
  
  # 4> 해당 주소의 읍면동 주소와 주소코드를 저장한다.
  folder <- unlist(strsplit(filename, "/(?!.*/)", perl = TRUE))[1]
  data2 <- read.xlsx(paste0(folder,'/(필수!) 주소코드&읍면동.xlsx'))
  
  
  pb <- progress_bar$new(format = "downloading [:bar] :current/:total (:percent) in :elapsedfull/:eta (주소코드)",
                         total = nrow(data2), clear = FALSE, width= 60)
  
  for (i in 1:nrow(data2)){
    pb$tick()
    
    row_number <- which(str_detect(data$소재지전체주소, data2[i,2]), arr.ind=TRUE)
    for (j in row_number){
      data[j,'주소코드'] <- data2[i,1]
      data[j,'주소명'] <- data2[i,2]
    }
  }
  
  
  # 5> 파일 저장
  write.xlsx(data,paste0(unlist(strsplit(filename, '[.(?!.*)]', perl = TRUE))[1],'_완료.xlsx'))
  print('success!!')
}


choose_file_name()

##### csv 버젼 (신버전)

# 0> 필요한 패키지 로드 =====================================================
install.packages('openxlsx')
install.packages('dplyr')
install.packages('rvest')
install.packages('stringr')
install.packages("progress")
library(progress)
library(openxlsx)
library(dplyr)
library(rvest)
library(stringr)
library(tidyr)

choose_file_name = function(file='Select_data_file') {
  
  filename<-tcltk::tk_choose.files(file)
  data <- read.csv(filename,stringsAsFactors=F,na.strings = c(""," ",NA))
  
  # 1> 소재지주소 na인 경우 도로명 주소 가져오기 ==============================
  data_doro <- data[is.na(data$소재지전체주소),'도로명전체주소'] %>% str_replace_all("[[:space:]]","+");
  row_number <- which(is.na(data$소재지전체주소), arr.ind=TRUE)
  
  
  # 2> 도로명 주소 > 소재지 주소 =============================================
  pb <- progress_bar$new(format = "downloading [:bar] :current/:total (:percent) in :elapsedfull/:eta (도로명주소변환)",
                         total = length(data_doro), clear = FALSE, width= 60)
  
  for(i in 1:length(data_doro)){
    
    # progress bar
    pb$tick()
    
    # 괄호,쉼표 기준으로 데이터를 분류 후 제일 첫 번째 텍스트(주소)를 가져온다.
    a <- data_doro[i] %>% strsplit('[(]|,')
    doro_address <- str_trim(a[[1]][1])
    
    # 제일 첫 번째 텍스트(주소)로 url 생성
    url <- paste0('https://www.juso.go.kr/support/AddressMainSearch.do?searchKeyword='
                  ,doro_address,
                  '&dsgubuntext=&dscity1text=&dscounty1text=&dsemd1text=&dsri1text=&dssan1text=&dsrd_nm1text=')
    
    
    # 해당 주소를 도로명주소 웹을 통해 소재지주소로 변환한다.
    tryCatch(
      data$소재지전체주소[row_number[i]] <- 
        read_html(url) %>% 
        html_nodes(".subject_area") %>% 
        html_nodes(".subejct_2") %>% 
        html_nodes(".roadNameText") %>% 
        html_text() %>% 
        str_replace_all("\t|\n|\r","") %>% 
        str_replace_all("  "," "),
      error = function(e) {}
    )
    #cat(i,row_number[i],data$소재지전체주소[row_number[i]],'\n')
  }
  
  
  
  # 3> 만약 동 주소를 검색하지 못했을 때 NA값을 도로명 주소로 대체한다.
  df = data.frame(x=as.factor(data$소재지전체주소),y=as.factor(data$도로명전체주소))
  df2 = as.data.frame(t(apply(df,1, function(x) { return(c(x[!is.na(x)],x[is.na(x)]) )} )))
  data$소재지전체주소 <- df2[,1]
  
  
  
  # 4> 해당 주소의 읍면동 주소와 주소코드를 저장한다.
  folder <- unlist(strsplit(filename, "/(?!.*/)", perl = TRUE))[1]
  data2 <- read.csv(paste0(folder,'/(필수!) 주소코드&읍면동.csv'),stringsAsFactors=F)
  
  
  pb <- progress_bar$new(format = "downloading [:bar] :current/:total (:percent) in :elapsedfull/:eta (주소코드)",
                         total = nrow(data2), clear = FALSE, width= 60)
  
  for (i in 1:nrow(data2)){
    pb$tick()
    
    row_number <- which(str_detect(data$소재지전체주소, data2[i,2]), arr.ind=TRUE)
    for (j in row_number){
      data[j,'주소코드'] <- data2[i,1]
      data[j,'주소명'] <- data2[i,2]
    }
  }
  
  
  # 5> 파일 저장
  write.xlsx2(data,paste0(unlist(strsplit(filename, '[.(?!.*)]', perl = TRUE))[1],'_완료.xlsx'))
  print('success!!')
}


choose_file_name()






