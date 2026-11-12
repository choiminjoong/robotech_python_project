import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 의료 기초 데이터 분석 미니 프로젝트
# Numpy + Pandas + Matplotlib


# 줄 정렬 옵션
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.width', 140)
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 200)


# =====================================
# [1] 데이터 준비
# =====================================
data = {
    '측정일': [
        '2026-02-01','2026-02-01','2026-02-02',
        '2026-02-02','2026-02-03','2026-02-03',
        '2026-02-04','2026-02-04','2026-02-05',
        '2026-02-05','2026-02-06','2026-02-06',
        '2026-02-07','2026-02-07','2026-02-08'
    ],
    '환자ID': [
        'P001','P002','P003','P004','P005',
        'P006','P007','P008','P009','P010',
        'P011','P012','P013','P014','P015'
    ],
    '성별': [
        'F','M','M','F','F','M','M','F',
        'F','M','F','M','M','F','F'
    ],
    '나이': [
        25,31,44,np.nan,52,39,28,33,
        47,61,29,42,np.nan,36,55
    ],
    '키': [
        '162cm','175cm','168cm','160cm','158cm',
        '180cm','172cm','165cm','170cm','177cm',
        '159cm','174cm','169cm','163cm','157cm'
    ],
    '몸무게': [
        '54kg','78kg','70kg','52kg','60kg',
        '90kg','68kg','58kg','73kg','82kg',
        '50kg','76kg','71kg','57kg','66kg'
    ],
    'SBP(수축기)': [
        108,135,142,118,150,130,125,112,
        140,160,105,138,145,120,155
    ],
    'DBP(이완기)': [
        70,88,92,75,95,85,80,72,
        90,100,68,89,96,78,97
    ],
    '심박수': [
        72,84,90,68,95,78,70,75,
        88,92,66,80,86,73,89
    ],
    '공복혈당': [
        92,110,125,98,130,np.nan,105,90,
        118,140,88,112,128,100,np.nan
    ],
    '흡연': [
        'N','Y','Y','N','N','Y','N','N',
        'Y','Y','N','Y','Y','N','N'
    ],
    '운동빈도(주)': [
        3,1,0,2,np.nan,1,4,2,
        1,0,5,1,0,3,np.nan
    ],
}

df = pd.DataFrame(data)


print('=====================================')
print('[2] 데이터 탐색')
print('=====================================')

print(f'\n[head()]: \n{df.head()}')
print(f'\n[info()]: ')
print(df.info())

num_cols=['나이', 'SBP(수축기)', 'DBP(이완기)', '심박수', '공복혈당', '운동빈도(주)']
print(f'\n[describe]: {df[num_cols].describe}')


print('=====================================')
print('[3] 데이터 상세 및 변환')
print('=====================================')

# 3-1 타입변환: 날짜 문자열 -> datetime
print(f'\n변환 전 측정일 type: {df["측정일"].dtype}')
df['측정일'] = pd.to_datetime(df['측정일'])
print(f'변환 후 측정일 type: {df["측정일"].dtype}')

# 3-2 단위 제거: '170cm' -> 170, '70kg' => 70
print(f'\n단위 제거 전 키 type: {df["키"].dtype}')
print(f'\n단위 제거 전 몸무게 type: {df["몸무게"].dtype}')
df['키_cm'] = df['키'].str.replace('cm', '', regex=False).astype(int)
df['몸무게_kg'] = df['몸무게'].str.replace('kg', '', regex=False).astype(int)
print(f'\n단위 제거 후 키 type: {df["키"].dtype}')
print(f'\n단위 제거 후 몸무게 type: {df["몸무게"].dtype}')

# 3-3 결측지 확인
# 나이 => 평균값, 공복혈당: 중앙값, 운동빈도: 0
print(f'\n결측지 수: \n{df.isnull().sum()}')

df['나이'] = df['나이'].fillna(df['나이'].mean())
df['공복혈당'] = df['공복혈당'].fillna(df['공복혈당'].median())
df['운동빈도(주)'] = df['운동빈도(주)'].fillna(0) 

print(f'\n결측지 처리 후 개수: \n{df.isnull().sum()}')

# 파생 컬럼 생성
# BMI = 몸무게kg / 키m / ** 2
df['BMI'] = df['몸무게_kg']/((df['키_cm']/100)**2)
print(f'\nBMI: \n{df["BMI"].head()}')
df['맥박압_pp'] = df['SBP(수축기)'] - df['DBP(이완기)']
print(f'\n맥박압: \n{df["맥박압_pp"].head()}')
df['평균동맥압(MAP)'] = (df['SBP(수축기)'] + (2 * df['DBP(이완기)']))/3
print(f'\n평균동맥압: \n{df["평균동맥압(MAP)"].head()}')

# 범주화(카테고리 만둘기): 혈압 단계 분류
def bp_category(sbp, dbp):
    if 140 <= sbp or dbp >= 90:
        return '고혈압2기'
    elif 130 <= sbp <= 139 or 80 <= dbp < 90:
        return '고혈압1기'
    elif 120 <= sbp <= 129 and dbp < 80:
        return '주의'
    elif sbp < 120 and dbp < 80:
        return '정상'
    else:
        return '기타'

# lambda: 이름 없이 간단하게 작성하는 함수(익명함수)    
# lambda 매개변수: 변환할 식
# apply(): 행 단위로 함수를 적용할 때 자주 사용
df['혈압분류'] = df.apply(
    lambda row: bp_category(
        row['SBP(수축기)'],
        row['DBP(이완기)']
    ), axis=1
)
print(f'\n혈압분류 결과: \n{df[["SBP(수축기)", "DBP(이완기)","혈압분류"]].head()}')

# BMI 분류
def bmi_category(bmi):
    if bmi < 18.5:
        return '저체중'
    elif 18.5 <= bmi < 23:
        return '정상'
    elif 23 <= bmi < 25:
        return '과체중'
    else:
        return '비만'
df['BMI분류'] = df.apply(
    lambda row: bmi_category(
        row['BMI']
    ), axis=1
)
print(f'\nBMI분류 결과: \n{df[["BMI", "BMI분류"]].head()}')