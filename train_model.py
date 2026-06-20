"""
TAAS 교통사고 위험도 예측 모델 학습 스크립트
데이터 출처: 도로교통공단 TAAS 교통사고분석시스템
 - 전국교통사고다발지역표준데이터 (data.go.kr / 15029185)
 - TAAS 시간대별/도로형태별/도로종류별/월별 교통사고 통계 (taas.koroad.or.kr)
"""

import pandas as pd
import numpy as np
import json
import joblib
import sys
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, recall_score, precision_score,
                              classification_report)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/tmp/taas_data.csv'

# ──────────────────────────────────────────────────────────
# 1. 데이터 로드
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("[1] TAAS 공공데이터 로드")
print("=" * 60)

df = pd.read_csv(DATA_PATH, encoding='euc-kr')
print(f"  총 레코드: {len(df):,} 건")
print(f"  연도 범위: {df['사고연도'].min()} ~ {df['사고연도'].max()}")
print(f"  사고유형 분포:\n{df['사고유형구분'].value_counts().to_string()}")

# ──────────────────────────────────────────────────────────
# 2. 피처 엔지니어링
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[2] 피처 엔지니어링")
print("=" * 60)

df['lat'] = pd.to_numeric(df['위도'], errors='coerce')
df['lng'] = pd.to_numeric(df['경도'], errors='coerce')
df['accident_count'] = pd.to_numeric(df['사고건수'], errors='coerce').fillna(0).astype(int)
df['death_count']    = pd.to_numeric(df['사망자수'], errors='coerce').fillna(0).astype(int)
df['casualty_count'] = pd.to_numeric(df['사상자수'], errors='coerce').fillna(0).astype(int)
df['year'] = pd.to_numeric(df['사고연도'], errors='coerce').fillna(2020).astype(int)

# 결측 / 범위 이상값 제거
before = len(df)
df = df.dropna(subset=['lat', 'lng']).copy()
df = df[(df['lat'].between(33.0, 38.9)) & (df['lng'].between(124.5, 130.5))].copy()
print(f"  결측·범위이상 제거: {before} → {len(df)} 건")

# 파생 피처
df['death_rate']    = df['death_count'] / df['accident_count'].clip(lower=1)
df['casualty_rate'] = df['casualty_count'] / df['accident_count'].clip(lower=1)
df['severity_index'] = df['death_rate'] * 3.0 + df['casualty_rate']

# 시도 추출
df['sido'] = df['사고다발지역시도시군구'].str.extract(
    r'^(서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)'
)
df['sido'] = df['sido'].fillna('기타')

# 시도별 핫스팟 밀도
sido_cnt = df.groupby('sido').size().reset_index(name='sido_hotspot_count')
df = df.merge(sido_cnt, on='sido', how='left')

# 사고유형 위험 가중치
TYPE_WEIGHT = {'보행어린이': 3.0, '스쿨존어린이': 3.5, '보행노인': 2.5, '자전거': 1.5}
df['type_weight'] = df['사고유형구분'].map(TYPE_WEIGHT).fillna(1.0)

# 연도 정규화
df['year_normalized'] = (df['year'] - df['year'].min()) / max(df['year'].max() - df['year'].min(), 1)

# 범주형 인코딩
le_type = LabelEncoder()
le_sido = LabelEncoder()
df['accident_type_enc'] = le_type.fit_transform(df['사고유형구분'].fillna('기타'))
df['sido_enc']          = le_sido.fit_transform(df['sido'])

print(f"  사고유형 레이블: {dict(enumerate(le_type.classes_))}")
print(f"  시도 레이블 수: {len(le_sido.classes_)}")

# ──────────────────────────────────────────────────────────
# 3. 타겟 변수 (3-class 위험도)
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[3] 위험도 레이블 생성")
print("=" * 60)

def assign_risk(count):
    if count >= 6: return 2    # 고위험
    elif count >= 4: return 1  # 중위험
    else: return 0             # 저위험

df['risk_level'] = df['accident_count'].apply(assign_risk)
RISK_LABEL = {0: '저위험', 1: '중위험', 2: '고위험'}

dist = df['risk_level'].value_counts().sort_index()
for lvl, cnt in dist.items():
    print(f"  {RISK_LABEL[lvl]}: {cnt:,}건 ({cnt/len(df)*100:.1f}%)")

# ──────────────────────────────────────────────────────────
# 4. 학습/테스트 분리 + SMOTE
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[4] 학습/테스트 분리 + SMOTE 클래스 불균형 처리")
print("=" * 60)

FEATURES = [
    'lat', 'lng',
    'accident_type_enc', 'sido_enc',
    'year_normalized', 'type_weight',
    'death_rate', 'casualty_rate', 'severity_index',
    'sido_hotspot_count',
]

X = df[FEATURES].values
y = df['risk_level'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  학습: {len(X_train):,} / 테스트: {len(X_test):,}")

sm = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"  SMOTE 전: {Counter(y_train)}")
print(f"  SMOTE 후: {Counter(y_train_res)}")

# ──────────────────────────────────────────────────────────
# 5. XGBoost 학습
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[5] XGBoost 모델 학습")
print("=" * 60)

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
model.fit(X_train_res, y_train_res,
          eval_set=[(X_test, y_test)],
          verbose=False)
print("  학습 완료")

# ──────────────────────────────────────────────────────────
# 6. 평가
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[6] 모델 평가")
print("=" * 60)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['저위험', '중위험', '고위험']))

macro_f1  = f1_score(y_test, y_pred, average='macro')
macro_rec = recall_score(y_test, y_pred, average='macro')
macro_pre = precision_score(y_test, y_pred, average='macro')

print(f"  ★ Macro F1       : {macro_f1:.4f}")
print(f"  ★ Macro Recall   : {macro_rec:.4f}")
print(f"  ★ Macro Precision: {macro_pre:.4f}")

print("\n  [특성 중요도 Top-10]")
for i, (feat, imp) in enumerate(
    sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])[:10]
):
    print(f"    {i+1:2}. {feat:<30} {imp:.4f}")

# ──────────────────────────────────────────────────────────
# 7. 시도별 위험 프로파일
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[7] 시도별 위험 프로파일")
print("=" * 60)

sido_profile = {}
for sido_name, group in df.groupby('sido'):
    proba = model.predict_proba(group[FEATURES].values)
    sido_profile[sido_name] = {
        'high_risk_ratio': float((proba[:, 2] > 0.5).mean()),
        'avg_risk_score':  float(proba[:, 2].mean()),
        'hotspot_count':   int(len(group)),
    }
    print(f"  {sido_name:<12} 핫스팟:{len(group):4d}  고위험비율:{sido_profile[sido_name]['high_risk_ratio']:.2%}")

# ──────────────────────────────────────────────────────────
# 8. hotspots.json 생성
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[8] hotspots.json 생성 (상위 50개)")
print("=" * 60)

df['risk_score'] = model.predict_proba(df[FEATURES].values)[:, 2]
top50 = df.nlargest(50, 'risk_score').copy()

hotspots = []
for _, row in top50.iterrows():
    hotspots.append({
        'lat':           round(float(row['lat']), 6),
        'lng':           round(float(row['lng']), 6),
        'name':          str(row['사고지역위치명']),
        'region':        str(row['사고다발지역시도시군구']),
        'accident_type': str(row['사고유형구분']),
        'accident_count': int(row['accident_count']),
        'death_count':   int(row['death_count']),
        'risk_score':    round(float(row['risk_score']), 4),
        'risk_level':    RISK_LABEL[int(row['risk_level'])],
        'year':          int(row['year']),
    })

with open('hotspots.json', 'w', encoding='utf-8') as f:
    json.dump(hotspots, f, ensure_ascii=False, indent=2)
print(f"  저장: hotspots.json ({len(hotspots)}개)")
for h in hotspots[:3]:
    print(f"    {h['region'][:25]:<25} score:{h['risk_score']:.3f}")

# ──────────────────────────────────────────────────────────
# 9. model.pkl 저장
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[9] model.pkl 저장")
print("=" * 60)

# TAAS 실통계 기반 조정 계수 ─────────────────────────────
# 시간대별 (출처: taas.koroad.or.kr 시간대별 교통사고 통계 2025)
_TIME = {
    0: 5566, 2: 3114, 4: 4405, 6: 11794,
    8: 22109, 10: 20675, 12: 22346, 14: 23561,
    16: 27278, 18: 26247, 20: 16327, 22: 10467,
}
_max_t = max(_TIME.values())
TIME_MULTIPLIER = {h: v / _max_t for h, v in _TIME.items()}

# 월별 계절성 (출처: taas.koroad.or.kr 월별 교통사고 통계 2025)
_MONTH = {
    1: 13740, 2: 13016, 3: 14620, 4: 16385,
    5: 16344, 6: 16175, 7: 17811, 8: 16651,
    9: 17430, 10: 17114, 11: 17442, 12: 17161,
}
_max_m = max(_MONTH.values())
MONTH_MULTIPLIER = {m: v / _max_m for m, v in _MONTH.items()}

# 도로형태별 (출처: taas.koroad.or.kr 도로형태별 교통사고 통계 2025)
_RF = {'교차로': 93792, '단일로': 82974, '기타': 17121, '철길건널목': 2}
_max_rf = max(_RF.values())
ROAD_FORM_MULTIPLIER = {k: v / _max_rf for k, v in _RF.items()}

# 날씨별 위험 계수 (출처: 도로교통공단 2024 교통사고통계 노면상태별 건당사망자 상대비)
# 건조(맑음)=1.0 기준; 비·눈·안개는 습윤/결빙노면과 시야저하로 사망위험 상승
WEATHER_MULTIPLIER = {
    '맑음': 1.00,
    '흐림': 1.12,
    '비':   1.48,
    '눈':   2.35,
    '안개': 1.85,
}

model_bundle = {
    'model':    model,
    'le_type':  le_type,
    'le_sido':  le_sido,
    'features': FEATURES,
    'risk_labels': RISK_LABEL,
    'metrics': {
        'macro_f1':        float(macro_f1),
        'macro_recall':    float(macro_rec),
        'macro_precision': float(macro_pre),
        'test_size':       int(len(X_test)),
        'train_size':      int(len(X_train_res)),
    },
    'adjustment_factors': {
        'time_multiplier':      TIME_MULTIPLIER,
        'month_multiplier':     MONTH_MULTIPLIER,
        'road_form_multiplier': ROAD_FORM_MULTIPLIER,
        'weather_multiplier':   WEATHER_MULTIPLIER,
    },
    'sido_profile': sido_profile,
    'data_info': {
        'source':         'TAAS 전국교통사고다발지역표준데이터 (data.go.kr/15029185)',
        'records':        int(len(df)),
        'years':          f"{int(df['year'].min())}-{int(df['year'].max())}",
        'accident_types': list(le_type.classes_),
        'sido_list':      list(le_sido.classes_),
    },
}

joblib.dump(model_bundle, 'model.pkl')
print(f"  저장: model.pkl")

print(f"\n{'='*60}")
print(f"  ✓ Macro F1      = {macro_f1:.4f}")
print(f"  ✓ Macro Recall  = {macro_rec:.4f}")
print(f"  ✓ model.pkl     = 저장됨")
print(f"  ✓ hotspots.json = {len(hotspots)}개 저장됨")
print(f"{'='*60}")
