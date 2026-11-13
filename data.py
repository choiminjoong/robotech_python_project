import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


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


print('=====================================')
print('[4] 분석')
print('=====================================')

# 최고/최저: BMI
max_bmi_row = df.loc[df['BMI'].idxmax()]
min_bmi_row = df.loc[df['BMI'].idxmin()]
print(f'\nmax_bmi_row: \n{max_bmi_row[["환자ID", "성별", "나이", "키_cm", "몸무게_kg", "BMI", "BMI분류"]]}')
print(f'\nmin_bmi_row: \n{min_bmi_row[["환자ID", "성별", "나이", "키_cm", "몸무게_kg", "BMI", "BMI분류"]]}')

# 혈압 분류별 평균 지표
# groupby(): 데이터 묶는 기능
bp_group = df.groupby('혈압분류')[['SBP(수축기)', 'DBP(이완기)', '맥박압_pp', '평균동맥압(MAP)', 'BMI', '운동빈도(주)']].mean()
print(f'\n혈압 분류별 평균: \n {bp_group}')

# 성별 평균 비교
gender_mean = df.groupby('성별')[['SBP(수축기)', 'DBP(이완기)', '맥박압_pp', '평균동맥압(MAP)', 'BMI', '운동빈도(주)']].mean()

print('\n[성별 평균 비교]')
print(gender_mean)


# 연령별 그룹 평균
# pd.cut()
bins = [0, 29, 39, 49, 59, 120]
labels = ['20대', '30대', '40대', '50대', '60대이상']
df['나이대'] = pd.cut(df['나이'], bins=bins, labels=labels, right=True)

age_group = df.groupby('나이대')[['SBP(수축기)', 'DBP(이완기)', '공복혈당', 'BMI']].mean()
print('\[나이대별 평균]')
print(age_group)

# 상관관계(수치형 컬럼)
# corr
corr_cols = ['나이', '키_cm', '몸무게_kg', 'SBP(수축기)', 'DBP(이완기)', '공복혈당', '운동빈도(주)', 'BMI', '맥박압_pp', '평균동맥압(MAP)']
corr_df = df[corr_cols].corr()
print('\n[상관관계]')
print(corr_df)
print('\n[상관관계 - 소수점 정리]')
print(corr_df.round(2))

# 조건 필터링
rist_df = df[
    (df['혈압분류'] == '고혈압2기') |
    (df['공복혈당'] >= 126) |
    (df['BMI'] >= 25.0)
].copy()

print('\n[관심 대상 목록]')
print(rist_df[['환자ID', '성별', '나이', 'SBP(수축기)', 'DBP(이완기)', '혈압분류', '공복혈당', 'BMI', 'BMI분류', '운동빈도(주)']])

# 교차분석
# crosstab()
cross = pd.crosstab(df['성별'], df['혈압분류'])
print(cross)

# 비율분석
# value_counts(): 각 항목이 몇번 나왔는지 세는 함수
# normalize=True 비율로 바꿔줌
bp_ratio = df['혈압분류'].value_counts(normalize=True)*100
print(f'\n혈압 분류 비율(%): \n{bp_ratio.round()}')

# 상위데이터 확인
# nlargest() : 데이터에서 값이 가장 큰 상위 N개를 뽑는 함수
top3_bmi = df.nlargest(3, 'BMI')
print(f'\n[BMI 상위 3명]: \n{top3_bmi[["환자ID", "BMI", "BMI분류"]]}')


# 조건 + 그룹분석
ht_df = df[df['혈압분류'] == '고혈압2기']
ht_mean = ht_df.groupby('성별')[['BMI','공복혈당']].mean()

print(f'\n[고혈압2기 환자 - 성별 평균]: \n{ht_mean}')

# 순위 만들기
# rank() : 데이터에 순위를 매기는 함수
# ascending=True는 작은 값부터 1등으로 순위를 부여함
df['BMI순위'] = df['BMI'].rank(ascending=True)

# BMI 순위별로 보는 게 아니라 데이터 순서대로 출력
print(f'\n[BMI 순위](환자ID 기준 순서): \n{df[["환자ID", "BMI", "BMI순위"]]}')

# BMI 순위별로 출력
df_sorted = df.sort_values('BMI순위')
print(f'\n[BMI 순위](BMI순위 기준 순서): \n{df_sorted[["환자ID", "BMI", "BMI순위"]]}')

# 위험 점수 만들기
df['위험점수'] = (
    (df['혈압분류'] == '고혈압2기').astype(int) +
    (df['공복혈당'] >= 126).astype(int) +
    (df['BMI'] >= 25).astype(int)
)

print(f'\n[위험점수]: \n{df[["환자ID", "위험점수"]]}')


print('=====================================')
print('[5] 시각화')
print('=====================================')

# 혈압분포 히스토그램

# 원하는 폰트 파일 경로 문자열로 입력
font_path = r'/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
# 지정한 폰트 파일 경로에서 폰트 추출
font_name = fm.FontProperties(
    fname=font_path
).get_name()

# matplotlib의 전역 폰트 설정 변경
matplotlib.rc('font', family=font_name)

plt.figure() # 그래프 공간 생성
plt.hist(df['SBP(수축기)'], bins=8) # 구간 설정
plt.title('SBP DIST') # 그래프 위 제목(수축기 혈압(SBP) 분포)
plt.xlabel('SBP(mmHg)') # 가로축 의미 설명
plt.ylabel('빈도') # 세로축 의미 설명(빈도)
plt.tight_layout() # 글자가 잘리지 않게 자동 정렬
#plt.show()

plt.figure()
plt.hist(df['DBP(이완기)'], bins=8) # 구간 설정
plt.title('DBP DIST') # 그래프 위 제목(이완기 혈압(DBP) 분포)
plt.xlabel('DBP(mmHg)')
plt.ylabel('frequency')
plt.tight_layout()
#plt.show()


# 혈압분류별 평균 SBP 막대그래프
bp_sbp_mean = df.groupby('혈압분류')['SBP(수축기)'].mean().sort_values(ascending=False)

plt.close('all')
plt.figure()
plt.bar(bp_sbp_mean.index, bp_sbp_mean.values)
plt.title('Avg SBP by Blood Category')
plt.xlabel('Blood Pressure Category')
plt.ylabel('Average SBP(mmHg)')
plt.tight_layout()
# plt.show()


# 산점도 : BMI vs SBP
plt.close('all')
plt.figure()
plt.scatter(df['BMI'], df['SBP(수축기)'])
plt.title('BMI vs SBP Scatter plot')
plt.xlabel('BMI')
plt.ylabel('SBP(mmHg)')
plt.tight_layout()
# plt.show()


# 측정일 기준 SBP 추이(날짜별 평균)
date_sbp = df.groupby('측정일')['SBP(수축기)'].mean()

plt.close('all')
plt.figure()
plt.plot(date_sbp.index, date_sbp.values) # 선 그래프 생성
plt.title('Avg SBP Over Time') # 측정일별 평균 SBP 추이
plt.xlabel('Measurement Date')
plt.ylabel('Avg SBP(mmHg)')
plt.xticks(rotation=30) # x축 눈금 글자 30도 회전
plt.tight_layout()
# plt.show()


# BMI분류 인원 수(파이차트)
bmi_count = df['BMI분류'].value_counts()

plt.close('all')
plt.figure()
plt.pie(
    bmi_count.values,
    labels=bmi_count.index,
    autopct='%1.1f%%'
)
plt.title('BMI Category DIST')
plt.tight_layout()
# plt.show()


#혈압분류별 BMI 분포(박스플롯)
plt.close('all')
plt.figure()
df.boxplot(column='BMI', by='혈압분류')
plt.title('BMI DIST by Blood Category')
plt.suptitle('') # 자동으로 생성되는 상단 제목 제거
plt.xlabel('Blood Pressure Category')
plt.ylabel('BMI')
plt.tight_layout()
# plt.show()


#성별 BMI
plt.close('all')

plt.figure()
df[df['성별'] == 'M']['BMI'].hist(alpha=0.5, bins=8)
df[df['성별'] == 'F']['BMI'].hist(alpha=0.5, bins=8)
plt.title('BMI DIST by Gender')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.legend(['Male', 'Female']) # 범례 표시
plt.tight_layout()
# plt.show()


#상관관계 히트맵
corr_cols = [
    '나이',
    '키_cm',
    '몸무게_kg',
    'SBP(수축기)',
    'DBP(이완기)',
    '공복혈당',
    'BMI'
]
corr = df[corr_cols].corr()

plt.close('all')
plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr_cols)), corr_cols, rotation=45)
plt.yticks(range(len(corr_cols)), corr_cols, rotation=45)
plt.title('Correlation Heatmap')
plt.tight_layout()
# plt.show()


print('=====================================')
print('[6] 인사이트(자동 요약)')
print('=====================================')

# 혈압 분류 비율
bp_counts = df['혈압분류'].value_counts() # 혈압분류 개수 세기
bp_ratio = (bp_counts / len(df)) * 100 # 비율 계산

print('\n[혈압분류 분포(개수)]')
print(bp_counts)

print('\n[혈압분류 분포(비율%)]')
print(bp_ratio.round(1)) # 소수점 한 자리까지만 표시


# 위험 신호 : 공복혈당 평균과 126 이상 비율
mean_glu = df['공복혈당'].mean()
high_glu_ratio = (df['공복혈당'] >= 126).mean() * 100 # True=1, Flase=0

print(f'\n[공복혈당 평균 : {mean_glu: .2f} mg/dL]')
print(f'[공복혈당 126 이상 비율 : {high_glu_ratio: .1f}%]')


# 운동빈도와 주요 지표 관계
corr_exercise_sbp = df['운동빈도(주)'].corr(df['SBP(수축기)'])
corr_exercise_bmi = df['운동빈도(주)'].corr(df['BMI'])

print(f'\n[운동빈도 vs SBP 상관계수] : {corr_exercise_sbp: .2f}')
print(f'[운동빈도 vs BMI 상관계수] : {corr_exercise_bmi: .2f}')


#전략/제안 3줄
stratege = []

# 혈압2기 비율 기반 코멘트
ht2_ratio = (df['혈압분류'] == '고혈압2기').mean() * 100

if ht2_ratio >= 30:
    stratege.append('고혈압2기 비율이 높은 편이라, 생활습관(운동/체중/염분)관련 교육 및 추적 관리가 필요합니다')
else:
    stratege.append('고혈압2기 비율이 과도하게 높진 않지만, 고혈압1기 이상에 대한 예방 안내가 유효합니다.')    

# BMI 비만 비율 코멘트
obesity_ratio = (df['BMI분류'] == '비만').mean() * 100
stratege.append(f'BMI 기준 비만 비율이 {obesity_ratio: .1f}%로 확인되어, 체중 관리(식습관+운동) 프로그램을 고려할 수 있습니다.')

# 혈당 고위험 비율 코멘트
if high_glu_ratio > 20:
    stratege.append('공복혈당 126 이상 비율이 꽤 있어, 혈당 관련 추가 검사/상담 연계를 고려할 수 있습니다.')
else:
    stratege.append('공복혈당 고위험 비율이 높진 않지만, 식습관/활동량 관련 기본 가이드를 제공하면 좋습니다.')

print('\n[제안 3줄]')
for s in stratege:
    print(f'- {s}')


#위험 점수 기반 위험군 요약
high_risk = df[df['위험점수'] >= 2]
high_risk_ratio = (len(high_risk) / len(df)) * 100

print(f'\n[고위험군 비율(위험점수 2 이상)] : {high_risk_ratio: .1f}%')
print('[고위험군 목록]')
print(high_risk[['환자ID', 'BMI', '공복혈당', '혈압분류', '위험점수']])


# 최상위 위험 대상 Top 3
top_risk = df.sort_values(['위험점수', 'BMI'], ascending=True).head(3)
print('\n[최상위 위험 대상 Top 3]')
print(top_risk[['환자ID', '위험점수', 'BMI', '공복혈당', '혈압분류']])


# 전체 건강 상태 등급 평가
if (high_risk_ratio > 30) or (high_glu_ratio > 30):
    health_level = '위험'
elif (high_risk_ratio > 15) or (high_glu_ratio > 15):
    health_level = '주의'
else:
    health_level = '양호'

print(f'\n[전체 건강 상태 평가] => {health_level}')
