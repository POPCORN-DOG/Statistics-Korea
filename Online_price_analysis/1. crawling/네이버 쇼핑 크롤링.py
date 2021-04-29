from urllib.request import urlopen
import urllib.request as req
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from urllib.request import urlopen #url의 html 을 가져 오기 위한 패키지
from bs4 import BeautifulSoup  #크롤링 필수 패키지 설치하려면 cmd창에서 pip install bs4
from selenium import webdriver
from bs4 import BeautifulSoup #크롤링 도구
from selenium.webdriver.common.keys import Keys
import time
from tqdm import tqdm
import os
import re
import json
import datetime
import matplotlib.pyplot as plt
import pylab as pl
import random
import string
import glob
import requests

header = {
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.183 Safari/537.36 Edg/86.0.622.63"
}
r = requests.get('https://search.shopping.naver.com/search/all?query=%EB%A7%88%EC%8A%A4%ED%81%AC&cat_id=&frm=NVSHATC', verify = False, headers = header)


r.text

soup =  BeautifulSoup(r.text,'lxml')
soup.select('body > script')[0].text
a2 = soup.select('body > script')[0].text
json.loads(a2)
