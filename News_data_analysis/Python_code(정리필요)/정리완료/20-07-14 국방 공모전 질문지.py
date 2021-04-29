import numpy as np
import pandas as pd

data = pd.read_csv('C:/Users/USER/Desktop/백업/data.csv',encoding='cp949')

#칼럼 이름 바꾸기
data.rename(columns={data.columns[0] : 'index'})

a1 = input('운동명을 입력하세요 :')
data.loc[data['카테고리'] == a1,['추천 운동','운동방법']]

name = input('이름을 입력해 주세요 : ')
sex = input('성별을 입력하세요 : (남자,여자)')
age = int(input('나이를 입력하세요 :'))
height = int(input('키를 입력하세요 :'))
weight = int(input('몸무게를 입력하세요 :'))
play = input('활동수준을 입력 해 주세요 (번호로 입력): \n① 활동이 적거나 운동을 안할경우\n'
             '② 가벼운 활동 및 주2회 운동을 하는경우\n'
             '③ 보통의 활동 밑 주4회 운동을 하는경우\n'
             '④ 적극적인 활동 및 매일 운동을 하는경우')
BMI = round(weight / (height/100 * height/100),2)
if sex == '남자':
    ba = 66.47 + (13.75 * weight) + (5 * height) - (age * 6.76)
else:
    ba = 655.1 + (9.56 * weight) + (1.85 * height) - (4.68 * age)

if play == '1':
    ac = ba * 0.2
elif play == '2':
    ac = ba * 0.375
elif play == '3':
    ac = ba * 0.555
elif play == '4':
    ac = ba * 0.725
else:
    print('활동수준 오류입니다.')

if BMI < 25:
    BMI_s = "정상"
elif BMI < 30:
    BMI_s = '과체중'
elif BMI < 35:
    BMI_s = '비만'
else:
    BMI_s = '고도비만'
print(str(name) + "님의 BMI 지수는 " + str(BMI) + '이며 ' + BMI_s + '상태입니다.\n'
       '또한 기초 대사량은 ' + str(ba) +'Kcal 이며\n'
        '하루에 필요한 대사량은 (기초대사량 + 활동대사량) '  + str(ba+ac) + 'Kcal 입니다.')