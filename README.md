# pp-인코딩과 범주화
## 2.인코딩 방법
### 2.1 범주형 데이터 -> 이산 수치형 데이터
  -  테스트를 위한 데이터 세트 생성하기
    
```python
import pandas as pd

df = pd.DataFrame({'weight':[40, 80, 60, 50, 90], # feature: weight, continuous
                   'height':[162, 155, 182, 173, 177], # feature: height, continuous
                   'sex':['f', 'm', 'm', 'f', 'm'], # feature: sex, categorical
                   'blood_type':['O', 'A', 'B', 'O', 'A'], # feature: blood_type, categorical
                   'health':['good', 'excellent', 'bad', 'bad', 'good'], # target: health, categorical
                   })
df
```
|    |   weight |   height | sex   | blood_type   | health    |
|---:|---------:|---------:|:------|:-------------|:----------|
|  0 |       40 |      162 | f     | O            | good      |
|  1 |       80 |      155 | m     | A            | excellent |
|  2 |       60 |      182 | m     | B            | bad       |
|  3 |       50 |      173 | f     | O            | bad       |
|  4 |       90 |      177 | m     | A            | good      |
  
  
  - originalEncoder
    - 범주형 데이터를 정수로 인코딩
    - 여러 컬럼(독립변수)에 사용 가능
```python
from sklearn.preprocessing import OrdinalEncoder

# 데이터프레임 복사
df_oe = df.copy()

# OrdinalEncoder에 대한 객체 생성
oe = OrdinalEncoder()

# 데이터로 oe 학습
oe.fit(df)

# 학습된 결과 
print(f'{oe.categories_=}')
# OrdinalEncoder는 수치형 weight와 height도 범주형으로 인식하여 변경하므로 주의

# 학습된 결과를 적용하여 변환
df_oe = pd.DataFrame(oe.transform(df), columns=df.columns)
df_oe
```
oe.categories_=[array([40, 50, 60, 80, 90], dtype=int64), array([155, 162, 173, 177, 182], dtype=int64), array(['f', 'm'], dtype=object), array(['A', 'B', 'O'], dtype=object), array(['bad', 'excellent', 'good'], dtype=object)]
|    |   weight |   height |   sex |   blood_type |   health |
|---:|---------:|---------:|------:|-------------:|---------:|
|  0 |        0 |        1 |     0 |            2 |        2 |
|  1 |        3 |        0 |     1 |            0 |        1 |
|  2 |        2 |        4 |     1 |            1 |        0 |
|  3 |        1 |        2 |     0 |            2 |        0 |
|  4 |        4 |        3 |     1 |            0 |        2 |
```python
# OrdinalEncoder 수정된 사용

# 데이터프레임 복사
df_oe = df.copy()

# OrdinalEncoder에 대한 객체 생성
oe = OrdinalEncoder()

# 데이터로 oe 학습
oe.fit(df[['sex', 'blood_type']])

# 학습된 결과 
print(f'{oe.categories_=}')

# 학습된 결과를 적용하여 삽입
df_oe.iloc[:,2:4] = oe.transform(df[['sex', 'blood_type']])
df_oe
```
oe.categories_=[array(['f', 'm'], dtype=object), array(['A', 'B', 'O'], dtype=object)]
|    |   weight |   height |   sex |   blood_type | health    |
|---:|---------:|---------:|------:|-------------:|:----------|
|  0 |       40 |      162 |     0 |            2 | good      |
|  1 |       80 |      155 |     1 |            0 | excellent |
|  2 |       60 |      182 |     1 |            1 | bad       |
|  3 |       50 |      173 |     0 |            2 | bad       |
|  4 |       90 |      177 |     1 |            0 | good      |
```python
# 디코딩(decoding)
oe.inverse_transform(df_oe.iloc[:,2:4])
```
```python
array([['f', 'O'],
       ['m', 'A'],
       ['m', 'B'],
       ['f', 'O'],
       ['m', 'A']], dtype=object)
```
  - LabelEncoder
      - 범주형 데이터를 정수로 인코딩
      - 하나의 컬럼(종속 변수, 타겟)에만 사용 가능
```python
from sklearn.preprocessing import LabelEncoder

# 데이터프레임 복사
df_le = df.copy()

# LabelEncoder는 하나의 변수에 대해서만 변환 가능
# LabelEncoder 객체 생성과 fit을 동시에 적용
health_le = LabelEncoder().fit(df.health)
df_le['health'] = health_le.transform(df.health)
df_le
```
|    |   weight |   height | sex   | blood_type   |   health |
|---:|---------:|---------:|:------|:-------------|---------:|
|  0 |       40 |      162 | f     | O            |        2 |
|  1 |       80 |      155 | m     | A            |        1 |
|  2 |       60 |      182 | m     | B            |        0 |
|  3 |       50 |      173 | f     | O            |        0 |
|  4 |       90 |      177 | m     | A            |        2 |
```python
# fit_transform() 메서드를 사용하여 한번에 인코딩 수행가능

# 데이터프레임 복사
df_le = df.copy()

# LabelEncoder 객체 생성과 fit을 동시에 적용
df_le['health'] = LabelEncoder().fit_transform(df.health)
df_le
```
|    |   weight |   height | sex   | blood_type   |   health |
|---:|---------:|---------:|:------|:-------------|---------:|
|  0 |       40 |      162 | f     | O            |        2 |
|  1 |       80 |      155 | m     | A            |        1 |
|  2 |       60 |      182 | m     | B            |        0 |
|  3 |       50 |      173 | f     | O            |        0 |
|  4 |       90 |      177 | m     | A            |        2 |
  - TargetEncoder 적용
      - 범주형 데이터를 특정한 컬럼(타겟)의 값의 크기와 비례한 숫자로 인코딩
```python
from sklearn.preprocessing import TargetEncoder

# 데이터프레임 복사
df_te = df.copy()

# TargetEncoder에 대한 객체 생성
# smooth는 정밀도를 조정하고 target_type은 인코딩 타입을 지정
te = TargetEncoder(smooth=0, target_type='continuous')

# 데이터로 te 학습
# 타겟을 weight라고 가정하고 blood_type을 인코딩
# blood_type_target은 weight와 비례하여 인코딩된 값
# 인코딩이 되는 값은 2차원으로 변환해야 함
te.fit(df['blood_type'].values.reshape(-1, 1), df.weight)

# 학습된 결과 
print(f'{te.categories_=}')

# 학습된 결과를 적용하여 새로운 컬럼 삽입
df_te['blood_type_target'] = te.transform(df['blood_type'].values.reshape(-1, 1))
df_te
```
te.categories_=[array(['A', 'B', 'O'], dtype=object)]
|    |   weight |   height | sex   | blood_type   | health    |   blood_type_target |
|---:|---------:|---------:|:------|:-------------|:----------|--------------------:|
|  0 |       40 |      162 | f     | O            | good      |                  45 |
|  1 |       80 |      155 | m     | A            | excellent |                  85 |
|  2 |       60 |      182 | m     | B            | bad       |                  60 |
|  3 |       50 |      173 | f     | O            | bad       |                  45 |
|  4 |       90 |      177 | m     | A            | good      |                  85 |

### 2.2 범주형 데이터 -> 이진 데이터
  - 원핫인코딩(one-hot encoding)
      - 하나의 컬럼에 있는 범주형 데이터를 여러개의 이진수 컬럼(수치형 데이터)로 인코딩
      - one-of-K 인코딩이라고도 함
```python
from sklearn.preprocessing import OneHotEncoder

# 데이터프레임 복사
df_ohe = df.copy()

# OneHotEncoder에 대한 객체 생성 후 fit
ohe = OneHotEncoder().fit(df_ohe[['blood_type']])

# 학습된 결과 
print(f'{ohe.categories_=}')

# 학습된 결과를 적용하여 새로운 컬럼 삽입
# OneHotEncoder는 결과를 sparse matrix로 반환하므로 toarray()를 통해 ndarray로 변환
df_ohe[ohe.categories_[0]] = ohe.transform(df_ohe[['blood_type']]).toarray()
df_ohe
```
ohe.categories_=[array(['A', 'B', 'O'], dtype=object)]
|    |   weight |   height | sex   | blood_type   | health    |   A |   B |   O |
|---:|---------:|---------:|:------|:-------------|:----------|----:|----:|----:|
|  0 |       40 |      162 | f     | O            | good      |   0 |   0 |   1 |
|  1 |       80 |      155 | m     | A            | excellent |   1 |   0 |   0 |
|  2 |       60 |      182 | m     | B            | bad       |   0 |   1 |   0 |
|  3 |       50 |      173 | f     | O            | bad       |   0 |   0 |   1 |
|  4 |       90 |      177 | m     | A            | good      |   1 |   0 |   0 |

  - Dummy encoding
      - pandas에서 제공하는 get_dummies는 One-hot encoding과 같은 동일한 기능
        - 여러 컬럼을 한 번에 변환 가능
      - 회귀분석에서 범주형 변수를 고려할 때 사용
```python
pd.get_dummies(df, columns=['sex', 'blood_type'], drop_first=False)
```
|weight |height |health | sex_female | sex_male | blood_type_A | blood_type_B | blood_type_O |
|------:|------:|------:|-----------:|---------:|-------------:|-------------:|-------------:|
|     0 |    40 |   162 |       good |     True |        False |        False |         True |
|     1 |    80 |   155 |  excellent |    False |         True |        False |        False |
|     2 |    60 |   182 |        bad |    False |        False |         True |        False |
|     3 |    50 |   173 |        bad |     True |        False |        False |         True |
|     4 |    90 |   177 |       good |    False |         True |        False |        False |

### 연속 수치형 데이터 -> 이진 데이터
  - Binerizer
      - 연속 수치형 데이터를 기준값(threshold)을 기준으로 이진수로 인코딩
```python
from sklearn.preprocessing import Binarizer

# 데이터 불러오기
df_bin = df.copy()

# Binarizer 객체 생성과 fit, transform을 동시에 적용
# Binarizer는 수치형 변수에 대해서만 변환 가능
df_bin['weight_bin'] = Binarizer(threshold=50).fit_transform(df.weight.values.reshape(-1,1))
df_bin['height_bin'] = Binarizer(threshold=170).fit_transform(df.height.values.reshape(-1,1))
df_bin
```

|    |   weight |   height | sex   | blood_type   | health    |   weight_bin |   height_bin |
|---:|---------:|---------:|:------|:-------------|:----------|-------------:|-------------:|
|  0 |       40 |      162 | f     | O            | good      |            0 |            0 |
|  1 |       80 |      155 | m     | A            | excellent |            1 |            0 |
|  2 |       60 |      182 | m     | B            | bad       |            1 |            1 |
|  3 |       50 |      173 | f     | O            | bad       |            0 |            1 |
|  4 |       90 |      177 | m     | A            | good      |            1 |            1 |

  -LabelBinerizer
    - 연속형 데이터를 이진수 컬럼으로 인코딩
    - 하나의 컬럼(종속변수, 타겟)에만 사용 가능
```python
from sklearn.preprocessing import LabelBinarizer

# 데이터프레임 복사
df_lb = df.copy()

# LabelBinarizer 객체 생성과 fit을 적용
lb = LabelBinarizer().fit(df.health)

# lb.classes_ : LabelBinarizer가 인코딩한 클래스 확인
print(f'{lb.classes_ = }')

# lb.transform() : 인코딩 변환
health_lb = lb.transform(df.health)
print('health_lb = \n', health_lb)

# 인코딩된 데이터를 데이터프레임으로 변환
df_lb[lb.classes_] = health_lb
df_lb
```
```python
lb.classes_ = array(['bad', 'excellent', 'good'], dtype='<U9')
health_lb = 
 [[0 0 1]
 [0 1 0]
 [1 0 0]
 [1 0 0]
 [0 0 1]]
```

|    |   weight |   height | sex   | blood_type   | health    |   bad |   excellent |   good |
|---:|---------:|---------:|:------|:-------------|:----------|------:|------------:|-------:|
|  0 |       40 |      162 | f     | O            | good      |     0 |           0 |      1 |
|  1 |       80 |      155 | m     | A            | excellent |     0 |           1 |      0 |
|  2 |       60 |      182 | m     | B            | bad       |     1 |           0 |      0 |
|  3 |       50 |      173 | f     | O            | bad       |     1 |           0 |      0 |
|  4 |       90 |      177 | m     | A            | good      |     0 |           0 |      1 |

  - MultiLabelBinerizer
      - multi-class(여러개의 범주가 있는) 데이터를 이진수 컬럼으로 인코딩
      -  하나의 컬럼(종속변수, 타겟)에만 사용
```python
from sklearn.preprocessing import MultiLabelBinarizer

# 데이터프레임 복사
df_mlb = df.copy()

# multi-class를 위한 컬럼 추가
df_mlb['test'] = [['math', 'english'], ['math', 'science'], ['science'], ['math', 'english'], 
                           ['science']] # target: test, categorical, multi-class
df_mlb
```
|    |   weight |   height | sex   | blood_type   | health    | test                |
|---:|---------:|---------:|:------|:-------------|:----------|:--------------------|
|  0 |       40 |      162 | f     | O            | good      | ['math', 'english'] |
|  1 |       80 |      155 | m     | A            | excellent | ['math', 'science'] |
|  2 |       60 |      182 | m     | B            | bad       | ['science']         |
|  3 |       50 |      173 | f     | O            | bad       | ['math', 'english'] |
|  4 |       90 |      177 | m     | A            | good      | ['science']         |

```python
# MultiLabelBinarizer 객체를 생성하고 fit() 메소드를 호출하여 클래스를 인코딩
mlb = MultiLabelBinarizer().fit(df_mlb.test)

# classes_ 속성을 사용하면 어떤 클래스가 인코딩되었는지 확인 가능
print(f'{mlb.classes_ = }')

# 인코딩된 데이터를 데이터프레임으로 변환
df_mlb[mlb.classes_] = mlb.transform(df_mlb.test)
df_mlb
```
mlb.classes_ = array(['english', 'math', 'science'], dtype=object)
|    |   weight |   height | sex   | blood_type   | health    | test                |   english |   math |   science |
|---:|---------:|---------:|:------|:-------------|:----------|:--------------------|----------:|-------:|----------:|
|  0 |       40 |      162 | f     | O            | good      | ['math', 'english'] |         1 |      1 |         0 |
|  1 |       80 |      155 | m     | A            | excellent | ['math', 'science'] |         0 |      1 |         1 |
|  2 |       60 |      182 | m     | B            | bad       | ['science']         |         0 |      0 |         1 |
|  3 |       50 |      173 | f     | O            | bad       | ['math', 'english'] |         1 |      1 |         0 |
|  4 |       90 |      177 | m     | A            | good      | ['science']         |         0 |      0 |         1 |

## 3.범주화
### 3.1 범주화(Discritization)
- 연속형 변수를 구간별로 나누어 범주형 변수로 변환하는 것
- quantization 또는 binning이라고도 함
- k-bins discretization
```python
from sklearn.preprocessing import KBinsDiscretizer

# 데이터프레임 복사
df_kbd = df.copy()

# KBinsDiscretizer 객체 생성과 fit을 적용
kbd = KBinsDiscretizer(n_bins=3, encode='ordinal').fit(df[['weight', 'height']])

# kbd.transform() : 인코딩 변환
# 인코딩된 데이터를 데이터프레임으로 변환
df_kbd[['weight_bin', 'height_bin']] = kbd.transform(df[['weight', 'height']])
df_kbd
```
|    |   weight |   height | sex   | blood_type   | health    |   weight_bin |   height_bin |
|---:|---------:|---------:|:------|:-------------|:----------|-------------:|-------------:|
|  0 |       40 |      162 | f     | O            | good      |            0 |            0 |
|  1 |       80 |      155 | m     | A            | excellent |            2 |            0 |
|  2 |       60 |      182 | m     | B            | bad       |            1 |            2 |
|  3 |       50 |      173 | f     | O            | bad       |            0 |            1 |
|  4 |       90 |      177 | m     | A            | good      |            2 |            2 |
---
# pp-피쳐엔지니어링
## 2. 피쳐 추출
### 2.4 Scikit-Learn으로 PCA와 LDA 수행하기
```python
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# iris 데이터셋을 로드
iris = datasets.load_iris()

X = iris.data # iris 데이터셋의 피쳐들
y = iris.target # iris 데이터셋의 타겟
target_names = list(iris.target_names) # iris 데이터셋의 타겟 이름

print(f'{X.shape = }, {y.shape = }') # 150개 데이터, 4 features
print(f'{target_names = }')
```
```python
X.shape = (150, 4), y.shape = (150,)
target_names = ['setosa', 'versicolor', 'virginica']
```
```python
# PCA의 객체를 생성, 차원은 2차원으로 설정(현재는 4차원)
pca = PCA(n_components=2)

# PCA를 수행. PCA는 비지도 학습이므로 y값을 넣지 않음
pca_fitted = pca.fit(X)

print(f'{pca_fitted.components_ = }')  # 주성분 벡터
print(f'{pca_fitted.explained_variance_ratio_ = }') # 주성분 벡터의 설명할 수 있는 분산 비율

X_pca = pca_fitted.transform(X) # 주성분 벡터로 데이터를 변환
print(f'{X_pca.shape = }')  # 4차원 데이터가 2차원 데이터로 변환됨
```
```python
pca_fitted.components_ = array([[ 0.36138659, -0.08452251,  0.85667061,  0.3582892 ],
       [ 0.65658877,  0.73016143, -0.17337266, -0.07548102]])
pca_fitted.explained_variance_ratio_ = array([0.92461872, 0.05306648])
X_pca.shape = (150, 2)
```
```python
# LDA의 객체를 생성. 차원은 2차원으로 설정(현재는 4차원)
lda = LinearDiscriminantAnalysis(n_components=2)

# LDA를 수행. LDA는 지도학습이므로 타겟값이 필요
lda_fitted = lda.fit(X, y)

print(f'{lda_fitted.coef_=}') # LDA의 계수
print(f'{lda_fitted.explained_variance_ratio_=}') # LDA의 분산에 대한 설명력

X_lda = lda_fitted.transform(X)
print(f'{X_lda.shape = }')  # 4차원 데이터가 2차원 데이터로 변환됨
```
```python
lda_fitted.coef_=array([[  6.31475846,  12.13931718, -16.94642465, -20.77005459],
       [ -1.53119919,  -4.37604348,   4.69566531,   3.06258539],
       [ -4.78355927,  -7.7632737 ,  12.25075935,  17.7074692 ]])
lda_fitted.explained_variance_ratio_=array([0.9912126, 0.0087874])
X_lda.shape = (150, 2)
```
```python
# 시각화 하기
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Seaborn을 이용하기 위해 데이터프레임으로 변환
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_lda = pd.DataFrame(X_lda, columns=['LD1', 'LD2'])
y = pd.Series(y).replace({0:'setosa', 1:'versicolor', 2:'virginica'})

# subplot으로 시각화
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

sns.scatterplot(df_pca, x='PC1', y='PC2', hue=y, style=y, ax=ax[0], palette='Set1')
ax[0].set_title('PCA of IRIS dataset')

sns.scatterplot(df_lda, x='LD1', y='LD2', hue=y, style=y, ax=ax[1], palette='Set1')
ax[1].set_title('LDA of IRIS dataset')

plt.show()
```
![subplot 시각화 이미지](./images/데전이미지.png)
