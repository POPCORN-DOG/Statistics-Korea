# 2021-04-25(일) 수정

# https://jfun.tistory.com/48
# 파이썬을 이용한 빅데이터 분석
# 아이리스 데이터로 인공신경망 실습

# (데이터가 안불러질 경우)
# 터미널 창에서 pip uninstall scikit-learn 후 pip install -U scikit-learn 하기 

# ===============================================================
# 1. iris data set 로드
# ===============================================================

# sklearn 라이브러리의 datasets 모듈에서 load_iris를 import한다.
from sklearn.datasets import load_iris

# Bunch 형태의 데이터셋으로 구성되어 있다.
# data: 아이리스 데이터
# feature_names: irir data set의 특징
# target: setosa, verginia, virginica 
iris = load_iris()
iris.keys()
iris.feature_names

# 독립변수로만 구성된 데이터를 NumPy 형태로 가지고 있다.
iris_data = iris.data
iris_data[0:10] # 또는 iris['data'][0:10]

# iris.target은 붓꽃 데이터 세트에서 레이블을 NumPy로 가지고 있다.
iris_label = iris.target

print('iris target값:', iris_label[[0, 50, 100]])
print('iris target명:', iris.target_names)


# ===============================================================
# 2. 데이터를 train, test set 분리
# ===============================================================

# 사이킷런은 train_test_split() API를 제공한다.

from sklearn.model_selection import train_test_split

x=iris['data']   # 독립변수
y=iris['target'] # 종속변수
x_train, x_test, y_train, y_test = train_test_split(x,y)
# training data set - 75%
# test data set - 25% 로 분할된다.


# ===============================================================
# 3. StandardScaler 데이터 정규화
# ===============================================================
# 데이터의 범위를 평균 0, 표준편차 1의 범위로 바꿔주는 모듈

# 정규화시키는 함수
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# x_train 정규화
# x_train 데이터에 대해서 평균과 표준편차 계산
scaler.fit(x_train) 

# x_train, x_test 데이터 정규화 및 변환
# scaler.transform
# transform이라는 모듈이 보이는데 이 모듈은 데이터를 정규화 형식으로 변환한다.
# 예를 들어 10, 50, 40, 20, 100, 70, 60, 100 등의 값을 
# scaler.transform 모듈을 사용할 경우 0, 0.5, 0.4, 0.2, 1, 0.7, 0.6, 1로 정규화 된다.
# 이렇게 해서 X_train과 X_test 두 개의 값을 정규화 한다
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# ===============================================================
# 4. MLP 알고리즘 불러오기 및 hidden layer할당
# ===============================================================

# 다층신경망 분류알고리즘을 불러와서 은닉층을 설정하는 부분이다.
# sklearn.neural_network 라이브러리의 MLPClassifier라는 모듈을 import한다.
from sklearn.neural_network import MLPClassifier

# MLPClassifier는 다중신경망 분류 알고리즘을 저장하고 있는 모듈
# 함수의 하이퍼 파라미터로 3개의 은닉층을 만들고 각 계층별로 10개의 노드씩 할당
# mlp라는 변수에 MLPClassifier() 함수를 실행한 결과를 저장한다.
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10)) 


# ===============================================================
# 5. x_train, y_train 데이터 학습
# ===============================================================

# 설정한 은닉층 3계층의 신경망에 x_train과 y_train 데이터를 학습
mlp.fit(x_train,y_train) 

#out:
#MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#       beta_2=0.999, early_stopping=False, epsilon=1e-08,
#       hidden_layer_sizes=(10, 10, 10), learning_rate='constant',
#       learning_rate_init=0.001, max_iter=200, momentum=0.9,
#       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
#       random_state=None, shuffle=True, solver='adam', tol=0.0001,
#       validation_fraction=0.1, verbose=False, warm_start=False)

# Activation : 다층 신경망에서 사용하는 활성화 함수
# Alpha : 신경망 내의 정규화 파라미터
# Batch_size : 최적화를 시키기 위한 학습 최소 크기
# Epsilon : 수치 안정성을 위한 오차 값
# Learning_rate_init : 가중치를 업데이트 할 때 크기를 제어
# Max_iter : 최대 반복 횟수
# Hidden_layer_sizes : 히든 레이어의 크기
# Learning_rate : 단계별로 움직이는 학습 속도
# Shuffle : 데이터를 학습 시 데이터들의 위치를 임의적으로 변경하는 지의 여부
# Solver : 가중치 최적화를 위해 사용하는 함수
# Validation_fraction : training data를 학습 시 validation의 비율
# Validation : training data를 학습 시 데이터가 유의미한지를 검증하는 데이터


# ===============================================================
# 6. 예측한 X_test를 변수 predictions에 저장
# ===============================================================

# 이전에 X_train과 y_train을 이용하여 학습한 결과 모델을 mlp에 넣었었다.
# 이번에는 이 mlp를 기반으로 X_test를 예측해 predictions라는 변수에 저장해 보겠다.
# 예측한 x_test를 저장
prediction = mlp.predict(x_test)


# ===============================================================
# 7. 학습 성능 평가
# ===============================================================

# 테스트 결과를 이용하여 학습한 모델의 성능을 평가해 보겠다.
# 평가를 위해 classification_report와 confusion_matrix 모듈을 사용하겠다.
# sklearn.metrics 라이브러리에서 classification_report와 confusion_matrix모듈을 import 한다.
# confusion_matrix모듈을 이용해 실제 값 y_test와 예측 값을 비교한다.
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,prediction))
#[[ 6  0  0] 6개가 제대로 분류되었다는 의미 실제 값 setosa에 대해 예측 값 setosa가 나온 경우
# [ 0 11  4] vericolor라는 클래스는 11개를 분류했지만 4개의 경우 예측값이 잘못되어 virginica로 분류한 경우
# [ 0  1 16]]

# ===============================================================
# 8. Precision, Recall, F1-Score를 이용한 평가
# ===============================================================

# Classification_report모듈에서는 Precision(정확률), Recall(재현률), f1-score를 계산해준다.

# 실제 결과와 예측 결과를 비교했을 때 둘 다 참인 경우를 TP(True Positive)라고 하며 
# 실제 결과는 참이지만 예측 결과가 거짓인 경우를 FN(False Negative)라 한다.
# 또한 실제 결과는 거짓인데 예측 결과에서 참인 경우를 FP(False Positive)라 하며, 
# 둘 다 거짓인 경우를 TN 즉, True Negative라 한다.

# ex> Iris data를 이용했을 때 테스트 데이터를 setosa로 제대로 판별한 경우가 정확률이 된다.

# 또한 테스트 데이터 중 아이리스 데이터의 setosa가 10개인 경우 분류 결과 8이라는 숫자가 나왔다면 재현율은 0.8이 된다.

# 마지막으로 F1-score는 정확률과 재현율을 둘 다 이용해 계산한다.
# 평과 결과 
from sklearn.metrics import classification_report
print(classification_report(y_test,prediction))


