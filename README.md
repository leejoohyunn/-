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

---
<p>$\bf{\rm{\color{#5ad7b7}Example}}$</p>

```python
import pandas as pd

df_example = pd.DataFrame({
    'age': [25, 30, 22, 35, 28],                   # feature: age,
continuous
    'income': [50000, 75000, 60000, 90000, 80000], # feature: income, continuous
    'education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master'],  # feature: education, categorical
    'marital_status': ['Single', 'Married', 'Single', 'Married', 'Divorced'],  # feature: marital_status, categorical
    'purchase': ['Yes', 'No', 'Yes', 'Yes', 'No'],  # target: purchase, categorical
})
```
|  | age|  income|  education|  marital_status|  purchase|
|-:|---:|-------:|----------:|---------------:|---------:|
| 0|  25|   50000|   Bachelor|          Single|       Yes|
| 1|  30|   75000|     Master|         Married|        No|
| 2|  22|   60000|        PhD|          Single|       Yes|
| 3|  35|   90000|   Bachelor|         Married|       Yes|
| 4|  28|   80000|     Master|        Divorced|        No|

--- 
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

---
<p>$\bf{\rm{\color{#5ad7b7}Example}}$</p>

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
|  | age|  income|  education|  marital_status|  purchase|
|-:|---:|-------:|----------:|---------------:|---------:|
| 0|  25|   50000|        0.0|             2.0|       1.0|
| 1|  30|   75000|        1.0|             1.0|       0.0|
| 2|  22|   60000|        2.0|             2.0|       1.0|
| 3|  35|   90000|        0.0|             1.0|       1.0|
| 4|  28|   80000|        1.0|             0.0|       0.0|
---

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

---
<p>$\bf{\rm{\color{#5ad7b7}Example}}$</p>

```pyhon
# OrdinalEncoder 수정된 사용

# 데이터프레임 복사
df_oe = df.copy()

# OrdinalEncoder에 대한 객체 생성
oe = OrdinalEncoder()

# 데이터로 oe 학습
oe.fit(df[['education', 'marital_status']])

# 학습된 결과 
print(f'{oe.categories_=}')

# 학습된 결과를 적용하여 삽입
df_oe.iloc[:,2:4] = oe.transform(df[['education', 'marital_status']])
df_oe
```
|  | age|  income|  education|  marital_status|  purchase|
|-:|---:|-------:|----------:|---------------:|---------:|
| 0|  25|   50000|        0.0|             2.0|       Yes|
| 1|  30|   75000|        1.0|             1.0|        No|
| 2|  22|   60000|        2.0|             2.0|       Yes|
| 3|  35|   90000|        0.0|             1.0|       Yes|
| 4|  28|   80000|        1.0|             0.0|        No|
---


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

---
<p>$\bf{\rm{\color{#5ad7b7}Example}}$</p>

```python
# 디코딩(decoding)
oe.inverse_transform(df_oe.iloc[:,2:4])  
```
```pyhton

oe.categories_=[array(['Bachelor', 'Master', 'PhD'], dtype=object), array(['Divorced', 'Married', 'Single'], dtype=object)]
array([['Bachelor', 'Single'],
       ['Master', 'Married'],
       ['PhD', 'Single'],
       ['Bachelor', 'Married'],
       ['Master', 'Divorced']], dtype=object)
```
---

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

---
<p>$\bf{\rm{\color{#5ad7b7}Example}}$</p>

```python
from sklearn.preprocessing import LabelEncoder

# 데이터프레임 복사
df_le = df.copy()

# LabelEncoder는 하나의 변수에 대해서만 변환 가능
# LabelEncoder 객체 생성과 fit을 동시에 적용
purchase_le = LabelEncoder().fit(df.purchase)
df_le['purchase'] = purchase_le.transform(df.purchase)
df_le
```

|  | age|  income|  education|  marital_status|  purchase|
|-:|---:|-------:|----------:|---------------:|---------:|
| 0|  25|   50000|   Bachelor|          Single|         1|
| 1|  30|   75000|     Master|         Married|         0|
| 2|  22|   60000|        PhD|          Single|         1|
| 3|  35|   90000|   Bachelor|         Married|         1|
| 4|  28|   80000|     Master|        Divorced|         0|
---

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

---
<p>$\bf{\rm{\color{#5ad7b7}Example}}$</p>

```python
# fit_transform() 메서드를 사용하여 한번에 인코딩 수행가능

# 데이터프레임 복사
df_le = df.copy()

# LabelEncoder 객체 생성과 fit을 동시에 적용
df_le['purchase'] = LabelEncoder().fit_transform(df.purchase)
df_le
```
|  | age|  income|  education|  marital_status|  purchase|
|-:|---:|-------:|----------:|---------------:|---------:|
| 0|  25|   50000|   Bachelor|          Single|         1|
| 1|  30|   75000|     Master|         Married|         0|
| 2|  22|   60000|        PhD|          Single|         1|
| 3|  35|   90000|   Bachelor|         Married|         1|
| 4|  28|   80000|     Master|        Divorced|         0|
---
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

---
<p>$\bf{\rm{\color{#5ad7b7}Example}}$</p>

```python
from sklearn.preprocessing import TargetEncoder

# 데이터프레임 복사
df_te = df.copy()

# TargetEncoder에 대한 객체 생성
# smooth는 정밀도를 조정하고 target_type은 인코딩 타입을 지정
te = TargetEncoder(smooth=0, target_type='continuous')

# 데이터로 te 학습
# 타겟을 weight라고 가정하고 marital_status을 인코딩
# marital_status_target은 weight와 비례하여 인코딩된 값
# 인코딩이 되는 값은 2차원으로 변환해야 함
te.fit(df['marital_status'].values.reshape(-1, 1), df.weight)

# 학습된 결과 
print(f'{te.categories_=}')

# 학습된 결과를 적용하여 새로운 컬럼 삽입
df_te['marital_status_target'] = te.transform(df['marital_status'].values.reshape(-1, 1))
df_te
```
te.categories_=[array(['Divorced', 'Married', 'Single'], dtype=object)]
|    |   weight |   height | sex   | blood_type   | health    | marital_status   |   marital_status_target |
|---:|---------:|---------:|:------|:-------------|:----------|:-----------------|------------------------:|
|  0 |       40 |      162 | f     | O            | good      | Single           |                      50 |
|  1 |       80 |      155 | m     | A            | excellent | Married          |                      65 |
|  2 |       60 |      182 | m     | B            | bad       | Single           |                      50 |
|  3 |       50 |      173 | f     | O            | bad       | Married          |                      65 |
|  4 |       90 |      177 | m     | A            | good      | Divorced         |                      90 |
---

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

---
<p>$\bf{\rm{\color{#5ad7b7}Example}}$</p>

```python
from sklearn.preprocessing import OneHotEncoder

# 데이터프레임 복사
df_ohe = df.copy()

# OneHotEncoder에 대한 객체 생성 후 fit
ohe = OneHotEncoder().fit(df_ohe[['marital_status']])

# 학습된 결과 
print(f'{ohe.categories_=}')

# 학습된 결과를 적용하여 새로운 컬럼 삽입
# OneHotEncoder는 결과를 sparse matrix로 반환하므로 toarray()를 통해 ndarray로 변환
df_ohe[ohe.categories_[0]] = ohe.transform(df_ohe[['marital_status']]).toarray()
df_ohe
```
ohe.categories_=[array(['Divorced', 'Married', 'Single'], dtype=object)]
|    |   weight |   height | sex   | blood_type   | health    | marital_status   |   Divorced |   Married |   Single |
|---:|---------:|---------:|:------|:-------------|:----------|:-----------------|-----------:|----------:|---------:|
|  0 |       40 |      162 | f     | O            | good      | Single           |          0 |         0 |        1 |
|  1 |       80 |      155 | m     | A            | excellent | Married          |          0 |         1 |        0 |
|  2 |       60 |      182 | m     | B            | bad       | Single           |          0 |         0 |        1 |
|  3 |       50 |      173 | f     | O            | bad       | Married          |          0 |         1 |        0 |
|  4 |       90 |      177 | m     | A            | good      | Divorced         |          1 |         0 |        0 |
---

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

---
<p>$\bf{\rm{\color{#5ad7b7}Example}}$</p>

```python
# One-hot encode specified columns
df_encoded = pd.get_dummies(df, columns=['education', 'marital_status'], drop_first=False)

# Display the encoded DataFrame
df_encoded
```
|    |   age |   income | purchase   | education_Bachelor   | education_Master   | education_PhD   | marital_status_Divorced   | marital_status_Married   | marital_status_Single   |
|---:|------:|---------:|:-----------|:---------------------|:-------------------|:----------------|:--------------------------|:-------------------------|:------------------------|
|  0 |    25 |    50000 | Yes        | True                 | False              | False           | False                     | False                    | True                    |
|  1 |    30 |    75000 | No         | False                | True               | False           | False                     | True                     | False                   |
|  2 |    22 |    60000 | Yes        | False                | False              | True            | False                     | False                    | True                    |
|  3 |    35 |    90000 | Yes        | True                 | False              | False           | False                     | True                     | False                   |
|  4 |    28 |    80000 | No         | False                | True               | False           | True                      | False                    | False                   |
---

### 2.3 연속 수치형 데이터 -> 이진 데이터
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

---
<p>$\bf{\rm{\color{#5ad7b7}Example}}$</p>

```python
from sklearn.preprocessing import Binarizer

# 데이터 불러오기
df_bin = df.copy()

# Binarizer 객체 생성과 fit, transform을 동시에 적용
# Binarizer는 수치형 변수에 대해서만 변환 가능
df_bin['age_bin'] = Binarizer(threshold=25).fit_transform(df.age.values.reshape(-1,1))
# age 기준을 25살로 설정
df_bin['income_bin'] = Binarizer(threshold=77000).fit_transform(df.income.values.reshape(-1,1))
df_bin
# income 기준을 77000으로 설정
```
|    |   age |   income | education   | marital_status   | purchase   |   age_bin |   income_bin |
|---:|------:|---------:|:------------|:-----------------|:-----------|----------:|-------------:|
|  0 |    25 |    50000 | Bachelor    | Single           | Yes        |         0 |            0 |
|  1 |    30 |    75000 | Master      | Married          | No         |         1 |            0 |
|  2 |    22 |    60000 | PhD         | Single           | Yes        |         0 |            0 |
|  3 |    35 |    90000 | Bachelor    | Married          | Yes        |         1 |            1 |
|  4 |    28 |    80000 | Master      | Divorced         | No         |         1 |            1 |
---
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

---
<p>$\bf{\rm{\color{#5ad7b7}Example}}$</p>

```python
from sklearn.preprocessing import LabelBinarizer

# 데이터프레임 복사
df_lb = df.copy()

# LabelBinarizer 객체 생성과 fit을 적용
lb = LabelBinarizer().fit(df.purchase)

# lb.classes_ : LabelBinarizer가 인코딩한 클래스 확인
print(f'{lb.classes_ = }')

# lb.transform() : 인코딩 변환
purchase_lb = lb.transform(df.purchase)
print('purchase_lb = \n', purchase_lb)

# 인코딩된 데이터를 데이터프레임으로 변환
df_lb[lb.classes_] = purchase_lb
df_lb
```
---
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
![subplot 시각화 이미지](https://github.com/leejoohyunn/images/blob/main/%EB%8D%B0%EC%A0%84%EC%9D%B4%EB%AF%B8%EC%A7%80.png)
## 3.피쳐 선택 기법
  - 종속변수 활용여부에 따라
      - supervised: 종속변수를 활용해 선택
      - unsupervised: 독립변수들 만으로 선택
  - 선택 메커니즘에 따라
      - Filter: 통계적인 방법으로 선택
      - Wrapper: 모델을 활용해 선택
      - Embedded: 모델 훈련 과정에서 자동으로 선택
      - Hybrid: Filter + Wrapper
   
### 3.1 필터 기법(Filter Method)
  - 분산 기반 선택(Variance-based Selection)
```python
from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold

# iris 데이터셋을 로드
iris = datasets.load_iris()

X = iris.data # iris 데이터셋의 피쳐들
y = iris.target # iris 데이터셋의 타겟
X_names = iris.feature_names # iris 데이터셋의 피쳐 이름
y_names = iris.target_names # iris 데이터셋의 타겟 이름

# 분산이 0.2 이상인 피쳐들만 선택하도록 학습
sel = VarianceThreshold(threshold=0.2).fit(X)
print(f'{sel.variances_ = }') # 각 피쳐의 분산 확인

# 분산이 0.2 이상인 피쳐들만 선택 적용
X_selected = sel.transform(X) # 분산이 0.2 이상인 피쳐들만 선택
X_selected_names = [X_names[i] for i in sel.get_support(indices=True)] # 선택된 피쳐들의 이름

print(f'{X_selected_names = }')
print(f'{X_selected[:5] = }')
```
```python

sel.variances_ = array([0.68112222, 0.18871289, 3.09550267, 0.57713289])
X_selected_names = ['sepal length (cm)', 'petal length (cm)', 'petal width (cm)']
X_selected[:5] = array([[5.1, 1.4, 0.2],
       [4.9, 1.4, 0.2],
       [4.7, 1.3, 0.2],
       [4.6, 1.5, 0.2],
       [5. , 1.4, 0.2]])
```
> F-value
>   - 두 모집단(확률변수)의 분산의 비율을 나타내는 값
>   - ANOVA, Regression에서는 모형이 설명하는 분산/잔자의 분산
>       - F-value가 크면 모형이 잘 설명하고 있다는 의미

> 상호정보량(mutual information)
>   - 하나의 확률변수가 다른 하나의 확률변수에 대해 제공하는 정보의 양
>   - 두 확률변수가 공유하는 엔트로피
>       - 두 확률변수가 독립이라면, 상호정보량은 0
>       - 두 확률변수의 상관관계가 강할수록 상호정보량이 커짐

> 카이제곱 테스트
>   - 범주형 데이터에서 두 요인간 독립성 검정에서 사용
>     - 카이제곱 value가 크면 두 요인간 독립이 아니라는 의미(즉, 상관관계가 있음)

```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, f_regression, chi2

# k개의 베스트 피쳐를 선택
sel_fc = SelectKBest(f_classif, k=2).fit(X, y)
print('f_classif: ')
print(f'{sel_fc.scores_ = }')
print(f'{sel_fc.pvalues_ = }')
print(f'{sel_fc.get_support() = }')
print('Selected features: ', [X_names[i] for i in sel_fc.get_support(indices=True)]) # 선택된 피쳐들의 이름

sel_fr = SelectKBest(f_regression, k=2).fit(X, y)
print('\nf_regression: ')
print(f'{sel_fr.scores_ = }')
print(f'{sel_fr.pvalues_ = }')
print(f'{sel_fr.get_support() = }')
print('Selected features: ', [X_names[i] for i in sel_fr.get_support(indices=True)]) # 선택된 피쳐들의 이름

sel_chi2 = SelectKBest(chi2, k=2).fit(X, y)
print('\nchi2: ')
print(f'{sel_chi2.scores_ = }')
print(f'{sel_chi2.pvalues_ = }')
print(f'{sel_chi2.get_support() = }')
print('Selected features: ', [X_names[i] for i in sel_chi2.get_support(indices=True)]) # 선택된 피쳐들의 이름
```
```python
f_classif: 
sel_fc.scores_ = array([ 119.26450218,   49.16004009, 1180.16118225,  960.0071468 ])
sel_fc.pvalues_ = array([1.66966919e-31, 4.49201713e-17, 2.85677661e-91, 4.16944584e-85])
sel_fc.get_support() = array([False, False,  True,  True])
Selected features:  ['petal length (cm)', 'petal width (cm)']

f_regression: 
sel_fr.scores_ = array([ 233.8389959 ,   32.93720748, 1341.93578461, 1592.82421036])
sel_fr.pvalues_ = array([2.89047835e-32, 5.20156326e-08, 4.20187315e-76, 4.15531102e-81])
sel_fr.get_support() = array([False, False,  True,  True])
Selected features:  ['petal length (cm)', 'petal width (cm)']

chi2: 
sel_chi2.scores_ = array([ 10.81782088,   3.7107283 , 116.31261309,  67.0483602 ])
sel_chi2.pvalues_ = array([4.47651499e-03, 1.56395980e-01, 5.53397228e-26, 2.75824965e-15])
sel_chi2.get_support() = array([False, False,  True,  True])
Selected features:  ['petal length (cm)', 'petal width (cm)']
```
### 3.2 래퍼 기법(wrapper method)
>svc(Support Vector Classification)와 SVM(Support Vector Machine)
>    - SVC(Support Vector Classification)는 분류를 위한 서포트 벡터 머신
>        - 오차 계산 시, margin 안쪽에 있는 데이터를 기준으로 오차 계산
>    - SVR(Support Vector Regression)는 회귀를 위한 서포트 벡터 머신
>        - 오차 계산 시, margin 안쪽에 있는 데이터는 오차로 계산하지 않음
> ![Svc와 Svm 이미지](https://github.com/leejoohyunn/images/blob/main/Svc%EC%99%80Svr.png)

```python
# RFE(Recursive Feature Elimination) 적용
from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE, RFECV, SelectFromModel, SequentialFeatureSelector
from sklearn.svm import SVC, SVR

# iris 데이터셋 로드
X, y = load_iris(return_X_y=True)

# 분류기 SVC 객체 생성, 선형분류, 3개의 클래스 
svc = SVR(kernel="linear", C=3)

# RFE 객체 생성, 2개의 피쳐 선택, 1개씩 제거 
rfe = RFE(estimator=svc, n_features_to_select=2, step=1)
# RFE+CV(Cross Validation), 5개의 폴드, 1개씩 제거
rfe_cv = RFECV(estimator=svc, step=1, cv=5) 

# 데이터셋에 RFE 적용
rfe.fit(X, y)
print('RFE Rank: ', rfe.ranking_)

# rank가 1인 피쳐들만 선택
X_selected = rfe.transform(X) 
X_selected_names = [X_names[i] for i in rfe.get_support(indices=True)] # 선택된 피쳐들의 이름

print(f'{X_selected_names = }')
print(f'{X_selected[:5] = }')

# 데이터셋에 RFECV 적용
rfe_cv.fit(X, y)
print('RFECV Rank: ', rfe_cv.ranking_)

# rank가 1인 피쳐들만 선택
X_selected = rfe_cv.transform(X) 
X_selected_names = [X_names[i] for i in rfe_cv.get_support(indices=True)] # 선택된 피쳐들의 이름

print(f'{X_selected_names = }')
print(f'{X_selected[:5] = }')
```
```python
RFE Rank:  [2 3 1 1]
X_selected_names = ['petal length (cm)', 'petal width (cm)']
X_selected[:5] = array([[1.4, 0.2],
       [1.4, 0.2],
       [1.3, 0.2],
       [1.5, 0.2],
       [1.4, 0.2]])
RFECV Rank:  [1 2 1 1]
X_selected_names = ['sepal length (cm)', 'petal length (cm)', 'petal width (cm)']
X_selected[:5] = array([[5.1, 1.4, 0.2],
       [4.9, 1.4, 0.2],
       [4.7, 1.3, 0.2],
       [4.6, 1.5, 0.2],
       [5. , 1.4, 0.2]])
```
```python
# SFS(Sequential Feature Selector) : 순차적으로 특성을 선택하는 방법

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# 데이터를 로드하고, 분류기를 초기화한 후 SFS를 적용
X, y = load_iris(return_X_y=True)
knn = KNeighborsClassifier(n_neighbors=3)
sfs = SequentialFeatureSelector(knn, n_features_to_select=2, direction='backward')

# SFS를 학습하고, 선택된 특성을 출력
sfs.fit(X, y)
print('SFS selected: ', sfs.get_support())

# 선택된 피쳐들만 선택
X_selected = sfs.transform(X) 
X_selected_names = [X_names[i] for i in sfs.get_support(indices=True)] # 선택된 피쳐들의 이름

print(f'{X_selected_names = }')
print(f'{X_selected[:5] = }')
```
```python
SFS selected:  [False False  True  True]
X_selected_names = ['petal length (cm)', 'petal width (cm)']
X_selected[:5] = array([[1.4, 0.2],
       [1.4, 0.2],
       [1.3, 0.2],
       [1.5, 0.2],
       [1.4, 0.2]])
```
### 3.3 임베디드 기법(Embedded Mthod)
  - 임베디드 기법
      - SelectFromModel
          - 의사결정나무 기반 알고리즘에서 변수를 선택하는 기법
```python
from sklearn.feature_selection import SelectFromModel
from sklearn import tree
from sklearn.datasets import load_iris

# 데이터를 로드하고, 분류기를 초기화한 후 SFS를 적용
X, y = load_iris(return_X_y=True)
clf = tree.DecisionTreeClassifier()
sfm = SelectFromModel(estimator=clf)

# 모형 구조 확인 및 출력을 pandas로 설정
sfm.set_output(transform='pandas')
```
# 아 이거 몰라

```python
# 모형 학습
sfm.fit(X, y)
print('SFM threshold: ', sfm.threshold_)

# 선택된 피쳐들만 선택
X_selected = sfm.transform(X) 
X_selected.columns = [X_names[i] for i in sfm.get_support(indices=True)] # 선택된 피쳐들의 이름

X_selected.head()
```
# 시댕

---
# pp-파이프라인
## 1.파이프라인
### 1.3파이프라인을 이요해 연결형 추정기 만들기
  - 파이프라인을 사용하지 않은 경우
```python
from sklearn.feature_selection import SelectKBest, f_classif # 피처선택 메서드
from sklearn.preprocessing import StandardScaler # 데이터 표준화
from sklearn.tree import DecisionTreeClassifier # 의사결정나무 분류기
from sklearn.datasets import load_iris # iris 데이터세트
 
# iris 데이터세트 로드
X, y = load_iris(return_X_y=True)
 
## 피쳐 선택
feat_sel = SelectKBest(f_classif, k=2)
X_selected = feat_sel.fit_transform(X, y)
print('Selected features:', feat_sel.get_feature_names_out())

## 표준화
scaler = StandardScaler()
scaler.fit(X_selected)
X_transformed = scaler.transform(X_selected)
print('Standard Scaled: \n', X_transformed[:5, :])

## 모델 학습
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_transformed, y)
print('Estimate : ', clf.predict(X_transformed)[:3])
print('Accuracy : ', clf.score(X_transformed, y))
```
```python
Selected features: ['x2' 'x3']
Standard Scaled: 
 [[-1.34022653 -1.3154443 ]
 [-1.34022653 -1.3154443 ]
 [-1.39706395 -1.3154443 ]
 [-1.2833891  -1.3154443 ]
 [-1.34022653 -1.3154443 ]]
Estimate :  [0 0 0]
Accuracy :  0.9733333333333334
```
  - 파이프라인을 사용한 경우
      - 파이프라인은 (key, value)의 리스트를 구성해 만듦
      -  파이프라인을 사용하면, 변환된 데이터를 별도로 저장하지 않고 연속적으로 사용하므로 속도 개선 및 메모리 절약됨
```python
from sklearn.pipeline import Pipeline # 파이프라인 구성을 위한 함수
from sklearn.feature_selection import SelectKBest, f_classif # 피처선택 메서드
from sklearn.preprocessing import StandardScaler # 데이터 표준화
from sklearn.tree import DecisionTreeClassifier # 의사결정나무 분류기
from sklearn.datasets import load_iris # iris 데이터세트
 
# iris 데이터세트 로드
X, y = load_iris(return_X_y=True)
 
## pipeline 구축
pipeline = Pipeline([
    ('Feature_Selection', SelectKBest(f_classif, k=2)), ## 피쳐 선택
    ('Standardization', StandardScaler()),  ## 표준화
    ('Decision_Tree', DecisionTreeClassifier(max_depth=3)) ## 학습 모델
])
display(pipeline) # 파이프라인 그래프로 구성 확인

pipeline.fit(X, y) ## 모형 학습
print('Estimate : ', pipeline.predict(X)[:3]) ## 예측
print('Accuracy : ', pipeline.score(X, y)) ## 성능 평가
```
# 아 이거 또 나옴
  - make_pipeline() 함수를 사용하여 파이프라인을 만들 수 있음
# 아 제발
  - 파이프라인 내부의 중간결과 확인하기
      - pipeline의 인덱스나 named_steps로 확인이 가능
```python
# pipiline의 Feature_Selection step의 결과 확인
# pipeline.named_steps['Feature_Selection'] == pipeline[0]
# pipeline.named_steps['Standardization'] == pipeline[1]
# pipeline.named_steps['Decision_Tree'] == pipeline[2]
print('Selected features:', pipeline.named_steps['Feature_Selection'].get_feature_names_out())
X_transformed = pipeline[1].transform(X_selected)
print('Standard Scaled: \n', X_transformed[:5, :])
```
```python
Selected features: ['x2' 'x3']
Standard Scaled: 
 [[-1.34022653 -1.3154443 ]
 [-1.34022653 -1.3154443 ]
 [-1.39706395 -1.3154443 ]
 [-1.2833891  -1.3154443 ]
 [-1.34022653 -1.3154443 ]]
```
## 2.파이프라인의 결합
### 2.1 수치형 데이터 파이프라인 처리
```python
import seaborn as sns
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 데이터 로드
df = sns.load_dataset('diamonds')
print(df.info())
X = df.drop('price', axis=1)
y = df['price']

# 데이터를 유형에 따라 분리
numeric_col = list(X.select_dtypes(exclude='category').columns)
category_col = list(X.select_dtypes(include='category').columns)
print(f'numeric_col: {numeric_col}')
print(f'category_col: {category_col}')
```
```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 53940 entries, 0 to 53939
Data columns (total 10 columns):
 #   Column   Non-Null Count  Dtype   
---  ------   --------------  -----   
 0   carat    53940 non-null  float64 
 1   cut      53940 non-null  category
 2   color    53940 non-null  category
 3   clarity  53940 non-null  category
 4   depth    53940 non-null  float64 
 5   table    53940 non-null  float64 
 6   price    53940 non-null  int64   
 7   x        53940 non-null  float64 
 8   y        53940 non-null  float64 
 9   z        53940 non-null  float64 
dtypes: category(3), float64(6), int64(1)
memory usage: 3.0 MB
None
numeric_col: ['carat', 'depth', 'table', 'x', 'y', 'z']
category_col: ['cut', 'color', 'clarity']
```
```python
# 파이프라인 구축
numeric_pipeline = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='mean')), # 평균값으로 Nan값 채워주기
        ('scaler', StandardScaler()) # 표준화
    ])

display(numeric_pipeline) # 파이프라인 그래프로 구성 확인

# 파이프라인 학습
numerical_data_piped = numeric_pipeline.fit_transform(X[numeric_col])
pd.DataFrame(numerical_data_piped, columns=numeric_col).head()
```
# ㅋㅋㅋ 그만 나와라
### 2.2 범주형 데이터 파이프라인 처리
```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# 파이프라인 구축
category_pipeline = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # 비어있는 값을 'missing'으로 채우기
        ('onehot', OneHotEncoder(sparse_output=False)), # Onehotencoder
    ])

display(category_pipeline) # 파이프라인 그래프로 구성 확인

# 파이프라인 학습
category_data_piped = category_pipeline.fit_transform(X[category_col])
# Onehotencoder의 컬럼명을 확인
category_colnames = category_pipeline[1].get_feature_names_out(category_col)
# 파이프라인 이후 데이터(array형 -> 데이터프레임)
pd.DataFrame(category_data_piped, columns=category_colnames).head()
```
# 이런
### 2.3 수치형 + 범주형 파이프라인 결합한 파이프라인
```python
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

# numeric & category 파이프라인 합치기
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_pipeline, numeric_col),
        ('category', category_pipeline, category_col)
    ])

pipe = make_pipeline(preprocessor, LinearRegression())
display(pipe) # 파이프라인 그래프로 구성 확인
pipe.fit(X,y)

print('Estimate : ', pipe.predict(X))
print('Accuracy : ', pipe.score(X, y))
```
# ㅋ
### 2.4 ColumnTransformer
```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

data_df = pd.DataFrame({
    "height":[165,  np.nan, 182],
    "weight":[70,   62,     np.nan],
    "age"   :[np.nan,18,    15]
})

# SimpleImputer를 사용해서 height의 null 값들은 평균으로 출력하고 나머지 column은 통과
col_transformer = ColumnTransformer([
    ("Impute_mean", SimpleImputer(strategy="mean"), ["height"])
    ], 
    remainder="passthrough"
)

display(col_transformer) # 파이프라인 그래프로 구성 확인
print(data_df)
print(col_transformer.fit_transform(data_df))
```
```python
# SimpleImputer를 사용해서 mean과 median 값을 null에 넣고 
# 나머지 열(column)에 대한 값은 상수로 -1 값을 넣어 줌
col_transformer2 = ColumnTransformer([
    ("Impute_mean"  , SimpleImputer(strategy="mean")  , ["height"]),
    ("Impute_median", SimpleImputer(strategy="median"), ["weight"])
    ],
    remainder=SimpleImputer(strategy="constant", fill_value=-1)
)

display(col_transformer2) # 파이프라인 그래프로 구성 확인
print(data_df)
print(col_transformer2.fit_transform(data_df))
```
# ㅋㅋ


