#인터파크
library(rvest)
library(dplyr)
library(stringr)
library(tidyr)

#코드 가져오기

html_c<- read_html('http://www.interpark.com/malls/index.html?smid1=header&smid2=logo')

html_c %>% html_nodes('.interparkNavigation') %>% html_nodes('.allCategory') %>% 
  html_nodes('.allCategoryWrapper') %>% html_nodes('.linkName')

html_c %>% html_nodes('.interparkNavigation') %>% html_nodes('.allCategory') %>% html_children()
  
html_a <- read_html('http://shopping.interpark.com/display/main.do?dispNo=001310')
html_a %>% html_nodes('.ct0 on') %>% html_text()
