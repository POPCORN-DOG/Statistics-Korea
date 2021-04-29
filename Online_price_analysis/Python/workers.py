#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from multiprocessing import Pool
import time
from tqdm import *
import json
import requests

def review_num(url):
    resp = []
    url = 'https://comment.daum.net/apis/v1/ui/single/main/' + url.replace('https://v.daum.net/v/','@')
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJmb3J1bV9rZXkiOiJuZXdzIiwidXNlcl92aWV3Ijp7ImlkIjoxNjI1NTk5NiwiaWNvbiI6Imh0dHBzOi8vdDEuZGF1bWNkbi5uZXQvcHJvZmlsZS9SVlE4TnNOS1RTYzAiLCJwcm92aWRlcklkIjoiREFVTSIsImRpc3BsYXlOYW1lIjoi7JiB7KeEIn0sImdyYW50X3R5cGUiOiJhbGV4X2NyZWRlbnRpYWxzIiwic2NvcGUiOltdLCJleHAiOjE2MDQ0OTM1MzAsImF1dGhvcml0aWVzIjpbIlJPTEVfREFVTSIsIlJPTEVfSURFTlRJRklFRCIsIlJPTEVfVVNFUiJdLCJqdGkiOiI3MTg2MTI2MC1mNWE4LTRjZDUtOGI2OC00OWY4ZmZjMzliYTEiLCJmb3J1bV9pZCI6LTk5LCJjbGllbnRfaWQiOiIyNkJYQXZLbnk1V0Y1WjA5bHI1azc3WTgifQ.tEwOwxWFDizwBtbRxCMoMK_2eTEcnN_Pqg0KqIc39zQ',
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36'
    }
    resp = json.loads(requests.get(url,headers=headers,verify=False).text.replace('post\":{\"','').replace('}}','}'))
    return resp

def _foo(my_number):
   square = my_number * my_number
#   time.sleep(1)
   return square 

def worker(x): 
    return x * x


import hmac
import string
import hashlib
import base64
import time
import requests
import json
import warnings

from base64 import b64decode, b64encode
from bs4 import BeautifulSoup

def pypapago2(search):
    
    # key값을 알기위한 코드 ===================================================
    # timestamp를 기준으로 mp5()암호화
    timestamp = int(time.time()*1000)

    # =======================================================================
    key = ("v1.5.2_0d13cb6cf4").encode('utf-8')
    msg = '3cb80585-364a-4b66-b3a4-b778b28df550'+\
        "\n"+\
        'https://papago.naver.com/apis/langs/dect' +\
        "\n"+\
        str(timestamp)

    hmac_result = hmac.new(key, msg.encode('utf-8'), hashlib.md5)
    key_code = b64encode(hmac_result.digest()).decode('utf-8')

    #print(b64encode(hmac_result.digest()).decode('utf-8'))
    #print(base64.b64encode(hmac_result.digest()).decode('utf-8'))
    # ======================================================================
    # 언어 감지
    # 알아낸 key_code 값으로 파싱 시작
    
    url = 'https://papago.naver.com/apis/langs/dect'
    
    headers = {
        'Authorization':'PPG 3cb80585-364a-4b66-b3a4-b778b28df550:'+ str(key_code),
        'Timestamp': str(timestamp),
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36'
    }
    
    data = {
        'query': search
    }
    
    resp = requests.post(url,headers=headers,data=data,verify=False)
    lang = resp.json()['langCode']
    
    # ======================================================================
    # 언어감지 값으로 결과 도축

    key = ("v1.5.2_0d13cb6cf4").encode('utf-8')
    msg = '3cb80585-364a-4b66-b3a4-b778b28df550'+\
        "\n"+\
        'https://papago.naver.com/apis/n2mt/translate' +\
        "\n"+\
        str(timestamp)

    hmac_result = hmac.new(key, msg.encode('utf-8'), hashlib.md5)
    key_code = b64encode(hmac_result.digest()).decode('utf-8')
    
    url = 'https://papago.naver.com/apis/n2mt/translate'

    headers = {
        'Authorization':'PPG 3cb80585-364a-4b66-b3a4-b778b28df550:'+ str(key_code),
        'Timestamp': str(timestamp),
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36'
    }
        
    language = {
        '영어':'en',
        '한국어':'ko',
        '일본어':'ja'
    }
    
    data = {
        'deviceId': '3cb80585-364a-4b66-b3a4-b778b28df550',
        'locale': 'ko',
        'dict': 'true',
        'dictDisplay': '30',
        'honorific': 'false',
        'instant': 'false',
        'paging': 'false',
        'source': str(lang),
        'target': 'en',
        'text': search
    }
    
    resp = requests.post(url,headers=headers,data=data,verify=False).text
    translatedText = json.loads(resp)['translatedText']
    return translatedText
