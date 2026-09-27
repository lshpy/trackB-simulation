# -*- coding: utf-8 -*-
"""
Track B semester simulation
=========================================================
Compares the syllabus assessment schemes, a single final exam (S1) vs
biweekly quizzes (S2), with the ACT-R base-level activation equation.

[Lecture equation — cognition3]
    B = ln( Σ_j  t_j^(-d) )
        B   : memory activation (higher = easier retrieval)
        t_j : time elapsed from the j-th study/quiz event to the final exam
        d   : individual decay (forgetting) rate

[Parameters — both d and theta vary per person]
    d (decay rate)       : back-calculated per person from single-condition accuracy
    theta (quiz boost)   : estimated per person from the (repeated - single) accuracy gap
                           -> the quiz effect differs between people, so theta varies too

Run:    python3 simulation.py
Input:  trackB_data.csv  (game responses exported from Google Sheets)
=========================================================
"""

import csv, math, random, statistics as st

# ========================================================
# Constants
# ========================================================
T_TEST  = 25                         # game: study-to-test delay (seconds)
T_EXAM  = 105                         # semester: end of term (15 weeks x 7 days)
QUIZZES = [14, 28, 42, 56, 70, 84, 98]   # biweekly quiz days
K       = 5                           # theta scaling constant (syllabus value, sensitivity range [3,7])
N_Q     = 6                           # number of quizzes in the game's repeated condition
N_BOOT  = 1000                        # number of bootstrap virtual students
SEED    = 42

random.seed(SEED)


# ========================================================
# STEP 1. Individual parameter estimation (d and theta both vary per person)
# ========================================================
def estimate_d(alpha):
    """Estimate the individual decay rate d from single-condition accuracy alpha.
       Inverting R(t)=t^(-d):  alpha = T_TEST^(-d)
       ->  d = -ln(alpha) / ln(T_TEST)
       Larger d = forgets faster."""
    a = min(max(alpha, 0.02), 0.98)          # guard against 0/1 extremes
    d = -math.log(a) / math.log(T_TEST)
    return min(max(d, 0.10), 1.50)           # clip to a plausible range


def estimate_theta(alpha, beta):
    """Estimate the quiz boost theta, which differs per person.
       The person's 'quiz effect' = beta - alpha
         beta  : accuracy in the repeated condition (study + quizzes)
         alpha : accuracy in the single condition (study only)
       theta = 1 + K * (beta - alpha) / N_Q
         theta = 1  -> a quiz counts the same as a study event (no boost)
         theta > 1  -> a quiz is stronger than a study event
       Since beta-alpha differs between people, theta varies too."""
    raw = 1 + K * (beta - alpha) / N_Q
    return max(1.0, min(raw, 3.0))           # clip to [1.0, 3.0]


def load_participants(csv_path="trackB_data.csv"):
    """Read the game-response CSV and estimate (d, theta) for each respondent."""
    test_ids = {"VERIFY_FULL_B", "DEMO_1", "DEMO_2", "FINAL_CHECK_B"}
    participants = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["participantId"]
            if pid in test_ids or not pid.startswith("PMP"):
                continue
            try:
                alpha = float(row["single_accuracy"])    # individual alpha
                beta  = float(row["repeated_accuracy"])  # individual beta
            except (ValueError, TypeError, KeyError):
                continue
            participants.append({
                "id":    pid,
                "alpha": alpha,
                "beta":  beta,
                "d":     estimate_d(alpha),               # <- individual d
                "theta": estimate_theta(alpha, beta),     # <- individual theta
            })
    return participants


# ========================================================
# STEP 2. Bootstrap — 69 people -> N_BOOT students
#   Resample (d, theta) pairs as a whole with replacement, so each virtual
#   student inherits a real person's individual d and theta together
# ========================================================
def bootstrap(participants, n=N_BOOT):
    return [random.choice(participants) for _ in range(n)]


# ========================================================
# STEP 3. Semester simulation — lecture equation B = ln(Σ t^(-d))
# ========================================================
def activation(d, theta, week, scenario):
    """Activation B at the final exam (day 105) of the content from week w.

    S1 single final : one study event
        B = ln( (T_EXAM - t0)^(-d) )
    S2 biweekly quizzes : study + a retrieval at every later biweekly quiz
        B = ln( (T_EXAM - t0)^(-d)  +  theta * Σ (T_EXAM - q)^(-d) )
    """
    t0 = 7 * (week - 1)                       # study day of week w
    learn_term = (T_EXAM - t0) ** (-d)        # study term

    if scenario == "S1":
        total = learn_term
    elif scenario == "S2":
        quiz_term = sum((T_EXAM - q) ** (-d)
                        for q in QUIZZES if q > t0)   # only quizzes after study
        total = learn_term + theta * quiz_term
    else:
        raise ValueError(scenario)

    return math.log(total)


def semester_mean(d, theta, scenario):
    """One student's activation averaged over the 15 weeks."""
    return st.mean(activation(d, theta, w, scenario) for w in range(1, 16))


def simulate(students, theta_override=None):
    """Each student x both scenarios -> semester-mean activation.
       If theta_override is given it is forced for all students (sensitivity analysis);
       if None, each student's own theta is used."""
    results = {"S1": [], "S2": []}
    for s in students:
        th = s["theta"] if theta_override is None else theta_override
        for sc in ("S1", "S2"):
            results[sc].append(semester_mean(s["d"], th, sc))
    return results


# ========================================================
# Main
# ========================================================
def main():
    # --- STEP 1 ---
    parts = load_participants()
    n   = len(parts)
    ds  = [p["d"]     for p in parts]
    ths = [p["theta"] for p in parts]
    print(f"[STEP 1] Individual parameter estimation  (n={n})")
    print(f"  d     : mean {st.mean(ds):.3f}  SD {st.pstdev(ds):.3f}"
          f"  range [{min(ds):.2f}, {max(ds):.2f}]   <- varies per person")
    print(f"  theta : mean {st.mean(ths):.3f}  SD {st.pstdev(ths):.3f}"
          f"  range [{min(ths):.2f}, {max(ths):.2f}]   <- varies per person")

    # --- STEP 2 ---
    students = bootstrap(parts)
    print(f"\n[STEP 2] Bootstrap  {n} people -> {len(students)} students  "
          f"((d, theta) pairs resampled with replacement)")

    # --- STEP 3 ---
    res = simulate(students)
    m1, m2 = st.mean(res["S1"]), st.mean(res["S2"])
    print(f"\n[STEP 3] Semester simulation results")
    print(f"  S1 single final     : mean activation B = {m1:+.3f}")
    print(f"  S2 biweekly quizzes : mean activation B = {m2:+.3f}")
    print(f"  Difference ΔB = {m2 - m1:+.3f}"
          f"   ->  activation ratio e^ΔB = {math.exp(m2 - m1):.2f}x")

    # --- Sensitivity 1: fix theta=1 (remove the quiz boost entirely) ---
    res0 = simulate(students, theta_override=1.0)
    d0 = st.mean(res0["S2"]) - st.mean(res0["S1"])
    print(f"\n[Sensitivity 1] theta=1 (quiz boost removed)"
          f"  ΔB = {d0:+.3f}  (e^ΔB = {math.exp(d0):.2f}x)")
    print("  -> S2 still wins without the boost: the 'frequency effect' alone supports the conclusion")

    # --- Sensitivity 2: weak-memory tracking — gain for top vs bottom 30% of d (decay rate) ---
    gains = []
    for s in students:
        b1 = semester_mean(s["d"], s["theta"], "S1")
        b2 = semester_mean(s["d"], s["theta"], "S2")
        gains.append((s["d"], b2 - b1))
    gains.sort(key=lambda g: g[0])             # ascending d
    cut  = len(gains) * 30 // 100
    low  = st.mean(g[1] for g in gains[-cut:])  # high d = weak memory
    high = st.mean(g[1] for g in gains[:cut])   # low d = strong memory
    print(f"\n[Sensitivity 2] Weak-memory tracking — gain ΔB from distributed assessment")
    print(f"  Weak memory   (top 30% of d)   : ΔB = {low:+.3f}  (e^ΔB={math.exp(low):.1f}x)")
    print(f"  Strong memory (bottom 30% of d): ΔB = {high:+.3f}  (e^ΔB={math.exp(high):.1f}x)")
    print("  -> the faster someone forgets, the more they gain from biweekly quizzes")


if __name__ == "__main__":
    main()
