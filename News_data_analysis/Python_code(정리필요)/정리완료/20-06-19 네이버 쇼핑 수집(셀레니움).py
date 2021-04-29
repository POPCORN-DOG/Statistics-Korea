help('modules')
###셀레니움 해보기
import os
import re
from selenium import webdriver
from bs4 import BeautifulSoup #크롤링 도구
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd

driver = webdriver.Chrome('C:/r-selenium/chromedriver.exe')  # 크롬 드라이버 연결
driver.get('http://naver.com')  #url 이동


driver.set_window_size(1000,1000) #창크기 조절
elem1 = driver.find_element_by_id('query') # 검색창 선택
elem1.send_keys('국민청원')

elem2= driver.find_element_by_id('search_btn')
elem2.click()

elem3 = driver.find_element_by_xpath('//*[@id="main_pack"]/div[2]/ul/li/dl/dt/a')
elem3.click()

driver.set_window_size(1200,1000) #창크기 조절
elem4 = driver.find_element_by_class_name('sf-with-ul')

driver.quit()

print('pp')
## kati 해보기
driver = webdriver.Chrome('C:/r-selenium/chromedriver.exe')
driver.get('https://www.kati.net/statistics/monthlyPerformanceByProduct.do')
but = driver.find_element_by_id('btnStatSearchItem')
but.click()

driver.find_element_by_xpath('//*[@id="statSearchItem"]/div/div/div[1]/div[2]/div[2]/a[5]').click()
driver.find_element_by_xpath('//*[@id="statSearchItem"]/div/div/div[1]/div[2]/div[2]/div[5]/div[1]/div[1]/div/ul/li[1]/a').click()
driver.find_element_by_xpath('//*[@id="statSearchItem"]/div/div/div[1]/div[2]/div[2]/div[5]/div[1]/div[1]/div/ul/li[2]/a').click()
driver.find_element_by_xpath('//*[@id="statSearchItem"]/div/div/div[1]/div[2]/div[2]/div[5]/div[1]/div[1]/div/ul/li[3]/a').click()
driver.find_element_by_xpath('//*[@id="statSearchItem"]/div/div/div[1]/div[2]/div[2]/div[5]/div[1]/div[1]/div/ul/li[4]/a').click()

mindex = driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(4) > ul ').text.split('>')
len(mindex)

for i in range(1,len(mindex)+1): #농산물 중분류클릭
    driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(4) > ul > li:nth-child(' + str(i) + ')').click()
    time.sleep(0)
sindex = driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(1) > ul > li:nth-child(1) > ul').text.split('\n')
len(sindex)

#농산품
list_1 = []
list_1_2 = []
list_1_3 = []
    try:
        for k in range(1, 43):
            sindex = driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(1) > ul > li:nth-child('+str(k)+')> ul').text.split('\n')
            for i in range(1,len(sindex)+1): #소분류 클릭 후 가져오기
                driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(1) > ul > li:nth-child('+str(k)+') > ul > li:nth-child('+str(i)+') > span > label').click()
                time.sleep(1.5)
                for j in range(0,len(driver.find_elements_by_class_name('align-left'))):
                    print(driver.find_element_by_css_selector(
                        '#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(1) > ul > li:nth-child(' + str(
                            k) + ') > ul > li:nth-child(' + str(i) + ') > span').text)
                    list_1.append(driver.find_elements_by_class_name('align-left')[j].text)
                    list_1_2.append(driver.find_element_by_css_selector(
                        '#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(1) > ul > li:nth-child(' + str(
                            k) + ') > ul > li:nth-child(' + str(i) + ') > span').text)
                    list_1_3.append(driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(1) > ul > li:nth-child('+str(k)+') > a').text)

    except:
        for m in range(k-1, 43):
            sindex = driver.find_element_by_css_selector(
                '#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(1) > ul > li:nth-child(' + str(
                    m) + ')> ul').text.split('\n')
            for i in range(1, len(sindex) + 1):  # 소분류 클릭 후 가져오기
                driver.find_element_by_css_selector(
                    '#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(1) > ul > li:nth-child(' + str(
                        m) + ') > ul > li:nth-child(' + str(i) + ') > span > label').click()
                time.sleep(1.5)
                for j in range(0, len(driver.find_elements_by_class_name('align-left'))):
                    print(driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(1) > ul > li:nth-child('+str(m)+') > ul > li:nth-child('+str(i)+') > span').text)
                    list_1.append(driver.find_elements_by_class_name('align-left')[j].text)
                    list_1_2.append(driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(1) > ul > li:nth-child('+str(m)+') > ul > li:nth-child('+str(i)+') > span').text)
                    list_1_3.append(driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(1) > ul > li:nth-child('+str(m)+') > a').text)

data =pd.DataFrame({"중분류": list_1_3, "소분류" : list_1_2, "항목" : list_1})

list_1 = []
list_2 = []
list_3 = []
list_4 = []
data.to_csv('C:/HANSSAK/SecureGate/download/농산품.csv',encoding='utf-8-sig')

#축산품
list_2 = []
list_2_2 = []
list_2_3 = []
    try:
        for k in range(1, 27):
            sindex = driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(2) > ul > li:nth-child('+str(k)+')> ul').text.split('\n')
            for i in range(1,len(sindex)+1): #소분류 클릭 후 가져오기
                driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(2) > ul > li:nth-child('+str(k)+') > ul > li:nth-child('+str(i)+') > span > label').click()
                time.sleep(1.5)
                for j in range(0,len(driver.find_elements_by_class_name('align-left'))):
                    print(driver.find_element_by_css_selector(
                        '#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(2) > ul > li:nth-child(' + str(
                            k) + ') > ul > li:nth-child(' + str(i) + ') > span').text)
                    list_2.append(driver.find_elements_by_class_name('align-left')[j].text)
                    list_2_2.append(driver.find_element_by_css_selector(
                        '#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(2) > ul > li:nth-child(' + str(
                            k) + ') > ul > li:nth-child(' + str(i) + ') > span').text)
                    list_2_3.append(driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(2) > ul > li:nth-child('+str(k)+') > a').text)

    except:
        for m in range(k-1, 27):
            sindex = driver.find_element_by_css_selector(
                '#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(2) > ul > li:nth-child(' + str(
                    m) + ')> ul').text.split('\n')
            for i in range(1, len(sindex) + 1):  # 소분류 클릭 후 가져오기
                driver.find_element_by_css_selector(
                    '#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(2) > ul > li:nth-child(' + str(
                        m) + ') > ul > li:nth-child(' + str(i) + ') > span > label').click()
                time.sleep(1.5)
                for j in range(0, len(driver.find_elements_by_class_name('align-left'))):
                    print(driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(2) > ul > li:nth-child('+str(m)+') > ul > li:nth-child('+str(i)+') > span').text)
                    list_2.append(driver.find_elements_by_class_name('align-left')[j].text)
                    list_2_2.append(driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(2) > ul > li:nth-child('+str(m)+') > ul > li:nth-child('+str(i)+') > span').text)
                    list_2_3.append(driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(2) > ul > li:nth-child('+str(m)+') > a').text)
data2 =pd.DataFrame({"중분류": list_2_3, "소분류" : list_2_2, "항목" : list_2})
data2.to_csv('C:/HANSSAK/SecureGate/download/축산물.csv',encoding='utf-8-sig')

#임산물
list_3 = []
list_3_2 = []
list_3_3 = []
    try:
        for k in range(1, 25):
            sindex = driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(3) > ul > li:nth-child('+str(k)+')> ul').text.split('\n')
            for i in range(1,len(sindex)+1): #소분류 클릭 후 가져오기
                driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(3) > ul > li:nth-child('+str(k)+') > ul > li:nth-child('+str(i)+') > span > label').click()
                time.sleep(1.5)
                for j in range(0,len(driver.find_elements_by_class_name('align-left'))):
                    print(driver.find_element_by_css_selector(
                        '#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(3) > ul > li:nth-child(' + str(
                            k) + ') > ul > li:nth-child(' + str(i) + ') > span').text)
                    list_3.append(driver.find_elements_by_class_name('align-left')[j].text)
                    list_3_2.append(driver.find_element_by_css_selector(
                        '#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(3) > ul > li:nth-child(' + str(
                            k) + ') > ul > li:nth-child(' + str(i) + ') > span').text)
                    list_3_3.append(driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(3) > ul > li:nth-child('+str(k)+') > a').text)

    except:
        for m in range(k-1, 25):
            sindex = driver.find_element_by_css_selector(
                '#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(3) > ul > li:nth-child(' + str(
                    m) + ')> ul').text.split('\n')
            for i in range(1, len(sindex) + 1):  # 소분류 클릭 후 가져오기
                driver.find_element_by_css_selector(
                    '#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(3) > ul > li:nth-child(' + str(
                        m) + ') > ul > li:nth-child(' + str(i) + ') > span > label').click()
                time.sleep(1.5)
                for j in range(0, len(driver.find_elements_by_class_name('align-left'))):
                    print(driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(3) > ul > li:nth-child('+str(m)+') > ul > li:nth-child('+str(i)+') > span').text)
                    list_3.append(driver.find_elements_by_class_name('align-left')[j].text)
                    list_3_2.append(driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(3) > ul > li:nth-child('+str(m)+') > ul > li:nth-child('+str(i)+') > span').text)
                    list_3_3.append(driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(3) > ul > li:nth-child('+str(m)+') > a').text)
data3 =pd.DataFrame({"중분류": list_3_3, "소분류" : list_3_2, "항목" : list_3})
data3.to_csv('C:/HANSSAK/SecureGate/download/임산물.csv',encoding='utf-8-sig')

#수산물

list_4 = []
list_4_2 = []
list_4_3 = []
    try:
        for k in range(1, 13):
            sindex = driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(4) > ul > li:nth-child('+str(k)+')> ul').text.split('\n')
            for i in range(1,len(sindex)+1): #소분류 클릭 후 가져오기
                driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(4) > ul > li:nth-child('+str(k)+') > ul > li:nth-child('+str(i)+') > span > label').click()
                time.sleep(1.5)
                for j in range(0,len(driver.find_elements_by_class_name('align-left'))):
                    print(driver.find_element_by_css_selector(
                        '#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(4) > ul > li:nth-child(' + str(
                            k) + ') > ul > li:nth-child(' + str(i) + ') > span').text)
                    list_4.append(driver.find_elements_by_class_name('align-left')[j].text)
                    list_4_2.append(driver.find_element_by_css_selector(
                        '#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(4) > ul > li:nth-child(' + str(
                            k) + ') > ul > li:nth-child(' + str(i) + ') > span').text)
                    list_4_3.append(driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(4) > ul > li:nth-child('+str(k)+') > a').text)

    except:
        for m in range(k-1, 25):
            sindex = driver.find_element_by_css_selector(
                '#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(4) > ul > li:nth-child(' + str(
                    m) + ')> ul').text.split('\n')
            for i in range(1, len(sindex) + 1):  # 소분류 클릭 후 가져오기
                driver.find_element_by_css_selector(
                    '#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(4) > ul > li:nth-child(' + str(
                        m) + ') > ul > li:nth-child(' + str(i) + ') > span > label').click()
                time.sleep(1.5)
                for j in range(0, len(driver.find_elements_by_class_name('align-left'))):
                    print(driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(4) > ul > li:nth-child('+str(m)+') > ul > li:nth-child('+str(i)+') > span').text)
                    list_4.append(driver.find_elements_by_class_name('align-left')[j].text)
                    list_4_2.append(driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(4) > ul > li:nth-child('+str(m)+') > ul > li:nth-child('+str(i)+') > span').text)
                    list_4_3.append(driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(4) > ul > li:nth-child('+str(m)+') > a').text)
data4 = pd.DataFrame({"중분류": list_4_3, "소분류" : list_4_2, "항목" : list_4})
data4.to_csv('C:/HANSSAK/SecureGate/download/수산물.csv',encoding='utf-8-sig')



#######한꺼번에
for m in range(2,5):
    mindex = driver.find_element_by_css_selector(
        '#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(' + str(
            m) + ') > ul ').text.split('>')
    for n in range(1,len(mindex)):
        driver.find_element_by_css_selector(
        '#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child('+str(m)+') > ul > li:nth-child(' + str(
         n) + ')').click()
        time.sleep(0.1)
    time.sleep(10)
    for k in range(1,len(mindex)):
        sindex = driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child('+str(m)+') > ul > li:nth-child('+str(k)+')> ul').text.split('\n')
        time.sleep(0.2)
        for i in range(1,len(sindex)+1): #소분류 클릭 후 가져오기
            driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child('+str(m)+') > ul > li:nth-child('+str(k)+') > ul > li:nth-child('+str(i)+') > span > label').click()
            time.sleep(2)
            for j in range(0,len(driver.find_elements_by_class_name('align-left'))):
                print(driver.find_elements_by_class_name('align-left')[j].text)
                if m == 1:
                    list_1.append(driver.find_elements_by_class_name('align-left')[j].text)
                    time.sleep(0.2)
                elif m == 2:
                    list_2.append(driver.find_elements_by_class_name('align-left')[j].text)
                    time.sleep(0.2)
                elif m == 3:
                    list_3.append(driver.find_elements_by_class_name('align-left')[j].text)
                    time.sleep(0.2)
                else:
                    list_4.append(driver.find_elements_by_class_name('align-left')[j].text)
                    time.sleep(0.2)



driver.find_element_by_css_selector('#statSearchItem > div > div > div.ly-wrap.w1100 > div.ly-contents > div.tab-menu.type-cont.mt20 > div:nth-child(10) > div.half-area.mt20 > div:nth-child(1) > div > ul > li:nth-child(1) > ul > li:nth-child(5) > ul > li:nth-child(7)').click()
time.sleep(2)
for j in range(0,len(driver.find_elements_by_class_name('align-left'))):
     print(driver.find_elements_by_class_name('align-left')[j].text)
