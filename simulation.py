# -*- coding: utf-8 -*-
"""
Track B 학기 시뮬레이션
=========================================================
강의계획서의 단일 기말고사(S1) vs 격주 퀴즈(S2) 평가 체계를
ACT-R Base-Level Activation 공식으로 비교한다.

[강의 공식 — cognition3]
    B = ln( Σ_j  t_j^(-d) )
        B   : 기억 활성화 (클수록 잘 인출됨)
        t_j : j번째 학습/퀴즈로부터 기말까지 경과 시간
        d   : 개인 망각률

[파라미터 — d, theta 모두 개인별 변동]
    d (망각률)         : single 정답률에서 개인별 역산
    theta (퀴즈 부스트) : (repeated - single) 정답률 차이에서 개인별 추정
                          → 사람마다 퀴즈 효과가 다르므로 theta도 변동

실행:  python3 simulation.py
입력:  trackB_data.csv  (Google Sheets에서 export한 게임 응답)
=========================================================
"""

import csv, math, random, statistics as st

# ========================================================
# 상수
# ========================================================
T_TEST  = 25                         # 게임: 학습 후 시험까지 (초)
T_EXAM  = 105                         # 학기: 종강 (15주 × 7일)
QUIZZES = [14, 28, 42, 56, 70, 84, 98]   # 격주 퀴즈 시점 (일)
K       = 5                           # theta 보정 상수 (계획서값, 민감도 범위 [3,7])
N_Q     = 6                           # 게임 repeated 조건의 퀴즈 횟수
N_BOOT  = 1000                        # 부트스트랩 가상 학생 수
SEED    = 42

random.seed(SEED)


# ========================================================
# STEP 1. 개인 파라미터 추정 (d, theta — 둘 다 개인별 변동)
# ========================================================
def estimate_d(alpha):
    """single 조건 정답률 alpha에서 개인 망각률 d 추정.
       R(t)=t^(-d) 형태로 역산:  alpha = T_TEST^(-d)
       →  d = -ln(alpha) / ln(T_TEST)
       d 가 클수록 빨리 잊는 사람."""
    a = min(max(alpha, 0.02), 0.98)          # 0/1 극단값 방어
    d = -math.log(a) / math.log(T_TEST)
    return min(max(d, 0.10), 1.50)           # 합리적 범위로 clip


def estimate_theta(alpha, beta):
    """퀴즈 부스트 theta 추정 — 개인별로 다름.
       그 사람의 '퀴즈 효과' = beta - alpha
         beta  : repeated 조건(학습+퀴즈) 정답률
         alpha : single 조건(학습만) 정답률
       theta = 1 + K * (beta - alpha) / N_Q
         theta = 1  → 퀴즈가 학습과 동일 (부스트 없음)
         theta > 1  → 퀴즈가 학습보다 강함
       beta-alpha 가 사람마다 다르므로 theta도 사람마다 다름(변동)."""
    raw = 1 + K * (beta - alpha) / N_Q
    return max(1.0, min(raw, 3.0))           # [1.0, 3.0] 범위로 clip


def load_participants(csv_path="trackB_data.csv"):
    """게임 응답 CSV를 읽어 각 응답자의 (d, theta)를 추정."""
    test_ids = {"VERIFY_FULL_B", "DEMO_1", "DEMO_2", "FINAL_CHECK_B"}
    participants = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["participantId"]
            if pid in test_ids or not pid.startswith("PMP"):
                continue
            try:
                alpha = float(row["single_accuracy"])    # 개인 alpha
                beta  = float(row["repeated_accuracy"])  # 개인 beta
            except (ValueError, TypeError, KeyError):
                continue
            participants.append({
                "id":    pid,
                "alpha": alpha,
                "beta":  beta,
                "d":     estimate_d(alpha),               # ← 개인별 d
                "theta": estimate_theta(alpha, beta),     # ← 개인별 theta
            })
    return participants


# ========================================================
# STEP 2. 부트스트랩 — 69명 → N_BOOT명
#   (d, theta) 쌍을 통째로 복원추출 → 가상 학생이 실제 사람의
#   개인별 d·theta를 그대로 물려받음
# ========================================================
def bootstrap(participants, n=N_BOOT):
    return [random.choice(participants) for _ in range(n)]


# ========================================================
# STEP 3. 학기 시뮬레이션 — 강의 공식 B = ln(Σ t^(-d))
# ========================================================
def activation(d, theta, week, scenario):
    """w주차 내용의 기말(105일) 시점 활성화 B.

    S1 단일 기말 : 학습 1회
        B = ln( (T_EXAM - t0)^(-d) )
    S2 격주 퀴즈 : 학습 + 이후의 격주 퀴즈마다 인출
        B = ln( (T_EXAM - t0)^(-d)  +  theta * Σ (T_EXAM - q)^(-d) )
    """
    t0 = 7 * (week - 1)                       # w주차 학습 시점 (일)
    learn_term = (T_EXAM - t0) ** (-d)        # 학습 항

    if scenario == "S1":
        total = learn_term
    elif scenario == "S2":
        quiz_term = sum((T_EXAM - q) ** (-d)
                        for q in QUIZZES if q > t0)   # 학습 이후 퀴즈만
        total = learn_term + theta * quiz_term
    else:
        raise ValueError(scenario)

    return math.log(total)


def semester_mean(d, theta, scenario):
    """한 학생의 15주차 평균 활성화."""
    return st.mean(activation(d, theta, w, scenario) for w in range(1, 16))


def simulate(students, theta_override=None):
    """각 학생 × 두 시나리오 → 학기 평균 활성화.
       theta_override 가 주어지면 모든 학생에게 그 값을 강제(민감도용),
       None 이면 학생 개인별 theta 를 사용."""
    results = {"S1": [], "S2": []}
    for s in students:
        th = s["theta"] if theta_override is None else theta_override
        for sc in ("S1", "S2"):
            results[sc].append(semester_mean(s["d"], th, sc))
    return results


# ========================================================
# 실행
# ========================================================
def main():
    # --- STEP 1 ---
    parts = load_participants()
    n   = len(parts)
    ds  = [p["d"]     for p in parts]
    ths = [p["theta"] for p in parts]
    print(f"[STEP 1] 개인 파라미터 추정  (n={n})")
    print(f"  d     : 평균 {st.mean(ds):.3f}  SD {st.pstdev(ds):.3f}"
          f"  범위 [{min(ds):.2f}, {max(ds):.2f}]   ← 개인별 변동")
    print(f"  theta : 평균 {st.mean(ths):.3f}  SD {st.pstdev(ths):.3f}"
          f"  범위 [{min(ths):.2f}, {max(ths):.2f}]   ← 개인별 변동")

    # --- STEP 2 ---
    students = bootstrap(parts)
    print(f"\n[STEP 2] 부트스트랩  {n}명 → {len(students)}명  "
          f"((d, theta) 쌍 복원추출)")

    # --- STEP 3 ---
    res = simulate(students)
    m1, m2 = st.mean(res["S1"]), st.mean(res["S2"])
    print(f"\n[STEP 3] 학기 시뮬레이션 결과")
    print(f"  S1 단일 기말 : 평균 활성화 B = {m1:+.3f}")
    print(f"  S2 격주 퀴즈 : 평균 활성화 B = {m2:+.3f}")
    print(f"  차이 ΔB = {m2 - m1:+.3f}"
          f"   →  활성화 배율 e^ΔB = {math.exp(m2 - m1):.2f}배")

    # --- 민감도 1: theta=1 고정 (퀴즈 부스트 완전 제거) ---
    res0 = simulate(students, theta_override=1.0)
    d0 = st.mean(res0["S2"]) - st.mean(res0["S1"])
    print(f"\n[민감도 1] theta=1 (퀴즈 부스트 제거) 시"
          f"  ΔB = {d0:+.3f}  (e^ΔB = {math.exp(d0):.2f}배)")
    print("  → 부스트를 꺼도 우세 = '빈도 효과'만으로 결론 성립")

    # --- 민감도 2: 기억력 약자 추적 — d(망각률) 상하위 30% 이득 비교 ---
    gains = []
    for s in students:
        b1 = semester_mean(s["d"], s["theta"], "S1")
        b2 = semester_mean(s["d"], s["theta"], "S2")
        gains.append((s["d"], b2 - b1))
    gains.sort(key=lambda g: g[0])             # d 오름차순
    cut  = len(gains) * 30 // 100
    low  = st.mean(g[1] for g in gains[-cut:])  # d 높음 = 기억력 약자
    high = st.mean(g[1] for g in gains[:cut])   # d 낮음 = 기억력 강자
    print(f"\n[민감도 2] 기억력 약자 추적 — 분산 평가 이득 ΔB")
    print(f"  기억력 약자(d 상위30%): ΔB = {low:+.3f}  (e^ΔB={math.exp(low):.1f}배)")
    print(f"  기억력 강자(d 하위30%): ΔB = {high:+.3f}  (e^ΔB={math.exp(high):.1f}배)")
    print("  → 잘 잊는 사람일수록 격주 퀴즈의 이득이 크다")


if __name__ == "__main__":
    main()
