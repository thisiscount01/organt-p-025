"""
교통사고 위험도 예측 AI 서버
TAAS 공공데이터 기반 XGBoost 모델

API:
  POST /api/predict   - 조건 입력 → 위험도 예측
  GET  /api/hotspots  - 전국 고위험 구간 50개
  GET  /api/model-info - 모델 메타데이터
"""

import os, json, time, math
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np

# ─── 앱 초기화 ────────────────────────────────────────────
app = Flask(__name__, static_folder='public', static_url_path='')

MODEL_PATH    = os.path.join(os.path.dirname(__file__), 'model.pkl')
HOTSPOTS_PATH = os.path.join(os.path.dirname(__file__), 'hotspots.json')

# ─── 모델 로드 ────────────────────────────────────────────
print("[서버 시작] model.pkl 로드 중...")
bundle = joblib.load(MODEL_PATH)
model        = bundle['model']
le_type      = bundle['le_type']
le_sido      = bundle['le_sido']
FEATURES     = bundle['features']
RISK_LABELS  = bundle['risk_labels']      # {0:'저위험', 1:'중위험', 2:'고위험'}
METRICS      = bundle['metrics']
ADJ          = bundle['adjustment_factors']
SIDO_PROFILE = bundle['sido_profile']
DATA_INFO    = bundle['data_info']

TIME_MUL     = ADJ['time_multiplier']     # {0: 0.20, 2: ..., 22: 0.38}
MONTH_MUL    = ADJ['month_multiplier']
ROAD_FORM_MUL = ADJ['road_form_multiplier']
WEATHER_MUL  = ADJ['weather_multiplier']

# hotspots.json 로드
with open(HOTSPOTS_PATH, encoding='utf-8') as f:
    HOTSPOTS = json.load(f)

print(f"[서버 시작] 로드 완료 | F1={METRICS['macro_f1']:.4f} "
      f"Recall={METRICS['macro_recall']:.4f} | 핫스팟 {len(HOTSPOTS)}개")


# ─── 유틸리티 ─────────────────────────────────────────────
def time_to_slot(hour: int) -> float:
    """시(0-23) → TAAS 2시간 슬롯 multiplier"""
    slot = (hour // 2) * 2        # 0,2,4,...,22
    return TIME_MUL.get(slot, 0.5)


def get_weather_multiplier(weather: str) -> float:
    return WEATHER_MUL.get(weather, 1.0)


def get_road_form_multiplier(road_form: str) -> float:
    return ROAD_FORM_MUL.get(road_form, 0.5)


def encode_sido(sido_name: str):
    """시도명 → sido_enc; 미등록 시도는 평균값 사용"""
    try:
        return le_sido.transform([sido_name])[0]
    except ValueError:
        # 미등록 시도: 첫 번째 존재하는 유사 시도명 탐색
        for cls in le_sido.classes_:
            if sido_name in cls or cls in sido_name:
                return le_sido.transform([cls])[0]
        return int(len(le_sido.classes_) // 2)   # fallback: 중간값


def get_sido_centroid(sido_name: str):
    """시도 → (lat_centroid, lng_centroid) — HOTSPOTS에서 추정"""
    lats, lngs = [], []
    for h in HOTSPOTS:
        region = h.get('region', '')
        if sido_name in region:
            lats.append(h['lat'])
            lngs.append(h['lng'])
    if lats:
        return float(np.mean(lats)), float(np.mean(lngs))
    # 광역시도 대표 좌표 (fallback)
    FALLBACK = {
        '서울': (37.5665, 126.9780), '경기': (37.4138, 127.5183),
        '부산': (35.1796, 129.0756), '인천': (37.4563, 126.7052),
        '대구': (35.8714, 128.6014), '대전': (36.3504, 127.3845),
        '광주': (35.1595, 126.8526), '울산': (35.5384, 129.3114),
        '강원': (37.8228, 128.1555), '충북': (36.6357, 127.4917),
        '충남': (36.5184, 126.8000), '전북': (35.7175, 127.1530),
        '전남': (34.8161, 126.4630), '경북': (36.4919, 128.8889),
        '경남': (35.2383, 128.6922), '세종': (36.4801, 127.2890),
        '제주': (33.4996, 126.5312),
    }
    for key, coord in FALLBACK.items():
        if key in sido_name:
            return coord
    return (36.5, 127.5)   # 한반도 중심


def compute_risk(sido_name: str, hour: int, month: int,
                 weather: str, road_form: str,
                 accident_type: str = '보행노인') -> dict:
    """
    실 XGBoost 예측 + TAAS 통계 기반 조정 계수 적용
    최종 risk_score = base_proba × time_mul × month_mul × weather_mul × road_mul
    """
    t0 = time.perf_counter()

    # 1) 기본 피처 벡터 구성
    lat, lng = get_sido_centroid(sido_name)
    sido_enc = encode_sido(sido_name)

    try:
        type_enc = le_type.transform([accident_type])[0]
    except ValueError:
        type_enc = 0

    TYPE_WEIGHT = {'보행어린이': 3.0, '스쿨존어린이': 3.5,
                   '보행노인': 2.5, '자전거': 1.5}
    type_weight = TYPE_WEIGHT.get(accident_type, 1.5)

    # 시도 프로파일 기반 파생 피처
    profile = SIDO_PROFILE.get(sido_name, {})
    hotspot_count = profile.get('hotspot_count', 500)

    X = np.array([[
        lat, lng, type_enc, sido_enc,
        0.7,            # year_normalized: 최신 연도 기준
        type_weight,
        0.01,           # death_rate: 일반 추정치
        0.85,           # casualty_rate: 일반 추정치
        0.04,           # severity_index
        hotspot_count,
    ]])

    # 2) XGBoost 예측 (고위험 확률)
    proba = model.predict_proba(X)[0]   # [저위험, 중위험, 고위험]
    base_score = float(proba[2])         # 고위험 확률

    # 3) TAAS 통계 기반 조정 계수 적용
    t_mul  = time_to_slot(hour)
    mo_mul = MONTH_MUL.get(month, 0.9)
    w_mul  = get_weather_multiplier(weather)
    rf_mul = get_road_form_multiplier(road_form)

    # 조정 계수를 base_score에 반영 (가중 결합)
    # base_score(ML) 60% + 상황 계수 40% 결합
    situation_score = (t_mul + mo_mul + w_mul * 0.5 + rf_mul) / 4.0
    final_score = 0.60 * base_score + 0.40 * situation_score

    # 정규화 (0-1 범위 클리핑)
    final_score = max(0.0, min(1.0, final_score))

    # 4) 위험 등급 결정
    if final_score >= 0.65:
        grade = '고위험'
        grade_en = 'HIGH'
        color = '#e74c3c'
    elif final_score >= 0.38:
        grade = '중위험'
        grade_en = 'MEDIUM'
        color = '#f39c12'
    else:
        grade = '저위험'
        grade_en = 'LOW'
        color = '#27ae60'

    # 5) 상황 설명 생성
    explanations = []
    if t_mul >= 0.85:
        explanations.append(f"현재 {hour}시는 교통량 최대 구간(사고 위험 최고조)")
    elif t_mul <= 0.30:
        explanations.append(f"현재 {hour}시는 교통량 저조 구간(상대적 안전)")
    else:
        explanations.append(f"현재 {hour}시는 보통 교통량 구간")

    if weather == '눈':
        explanations.append("눈/결빙: 제동거리 최대 3배 증가, 급제동 금지")
    elif weather == '안개':
        explanations.append("안개: 시거리 급감, 전조등 점등·감속 필수")
    elif weather == '비':
        explanations.append("비: 수막현상·시야저하, 안전거리 2배 확보 필요")
    elif weather == '흐림':
        explanations.append("흐림: 시야 다소 제한, 주의 운행 권고")

    if road_form == '교차로':
        explanations.append("교차로: 전체 사고 중 48% 발생 구간, 신호 준수 필수")
    elif road_form == '단일로':
        explanations.append("단일로: 전체 사고 중 43% 발생, 과속 주의")

    # 시도 위험 프로파일 언급
    if profile:
        hr = profile.get('high_risk_ratio', 0)
        if hr >= 0.25:
            explanations.append(f"{sido_name} 지역: 핫스팟 {profile.get('hotspot_count',0)}개 중 고위험 비율 {hr:.0%}")

    # TAAS 데이터 기반 월 위험도 언급
    if month in [7, 8, 11, 12]:
        explanations.append("하절기/동절기: TAAS 데이터상 사고 빈발 월")

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        'risk_score':       round(final_score, 4),
        'risk_grade':       grade,
        'risk_grade_en':    grade_en,
        'risk_color':       color,
        'base_ml_score':    round(float(base_score), 4),
        'adjustment_factors': {
            'time_factor':       round(t_mul, 3),
            'month_factor':      round(mo_mul, 3),
            'weather_factor':    round(w_mul, 3),
            'road_form_factor':  round(rf_mul, 3),
        },
        'class_probabilities': {
            '저위험': round(float(proba[0]), 4),
            '중위험': round(float(proba[1]), 4),
            '고위험': round(float(proba[2]), 4),
        },
        'explanations':     explanations,
        'inference_ms':     round(elapsed_ms, 2),
    }


# ─── API 라우트 ───────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('public', 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    POST /api/predict
    Body (JSON):
      sido     : str  — 시도명 (예: "서울", "경기", "부산")
      hour     : int  — 시간 0-23
      month    : int  — 월 1-12
      weather  : str  — 맑음|흐림|비|눈|안개
      road_form: str  — 교차로|단일로|기타
      accident_type: str (선택)
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': '요청 본문이 없습니다.'}), 400

        sido     = str(data.get('sido', '서울')).strip()
        hour     = int(data.get('hour', datetime.now().hour))
        month    = int(data.get('month', datetime.now().month))
        weather  = str(data.get('weather', '맑음'))
        road_form = str(data.get('road_form', '단일로'))
        acc_type  = str(data.get('accident_type', '보행노인'))

        # 입력값 검증
        if not (0 <= hour <= 23):
            return jsonify({'error': 'hour는 0-23 사이 정수입니다.'}), 400
        if not (1 <= month <= 12):
            return jsonify({'error': 'month는 1-12 사이 정수입니다.'}), 400
        if weather not in WEATHER_MUL:
            return jsonify({'error': f"weather는 {list(WEATHER_MUL.keys())} 중 하나입니다."}), 400

        result = compute_risk(sido, hour, month, weather, road_form, acc_type)

        return jsonify({
            'status': 'ok',
            'input': {
                'sido':     sido,
                'hour':     hour,
                'month':    month,
                'weather':  weather,
                'road_form': road_form,
                'accident_type': acc_type,
            },
            'result': result,
            'data_source': '도로교통공단 TAAS 전국교통사고다발지역표준데이터',
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/hotspots', methods=['GET'])
def hotspots():
    """GET /api/hotspots — 전국 고위험 구간 50개 (lat/lng/risk_score 포함)"""
    limit = min(int(request.args.get('limit', 50)), 50)
    return jsonify({
        'status': 'ok',
        'count':  len(HOTSPOTS[:limit]),
        'data':   HOTSPOTS[:limit],
        'source': '도로교통공단 TAAS 전국교통사고다발지역표준데이터 (data.go.kr/15029185)',
    })


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """GET /api/model-info — 모델 메타데이터"""
    return jsonify({
        'status': 'ok',
        'model': {
            'type':         'XGBoostClassifier',
            'task':         '3-class risk classification (저위험/중위험/고위험)',
            'n_estimators': 300,
            'features':     FEATURES,
            'num_features':  len(FEATURES),
            'smote_applied': True,
        },
        'metrics': METRICS,
        'data': DATA_INFO,
        'adjustment_factors': {
            'time_slots':  len(TIME_MUL),
            'months':      len(MONTH_MUL),
            'road_forms':  list(ROAD_FORM_MUL.keys()),
            'weather_conditions': list(WEATHER_MUL.keys()),
            'source': 'taas.koroad.or.kr 통계 (2025년)',
        },
        'sido_profiles': {
            k: {'high_risk_ratio': round(v['high_risk_ratio'], 4),
                'hotspot_count':   v['hotspot_count']}
            for k, v in SIDO_PROFILE.items()
        },
        'api_version': '1.0.0',
        'generated_at': DATA_INFO.get('years', ''),
    })


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': True})


# ─── 정적 파일 fallback ───────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return send_from_directory('public', 'index.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[서버] http://localhost:{port} 에서 실행 중")
    app.run(host='0.0.0.0', port=port, debug=False)
