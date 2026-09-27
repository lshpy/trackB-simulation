# Track B Semester Simulation

> ACT-R simulation comparing a single final exam with biweekly quizzes, for the IMEN315 Human Factors (Ergonomics) team project (Team 7).

Validates the assessment scheme proposed in the team's course syllabus. Using the ACT-R base-level activation
equation, it compares end-of-semester memory activation under a **single final exam (S1)** and
**biweekly quizzes (S2)**.

## Lecture equation (cognition3)

```
B = ln( Σ_j  t_j^(-d) )
```

- `B`   : memory activation (higher = easier retrieval)
- `t_j` : time elapsed from the j-th study/quiz event to the final exam
- `d`   : individual decay (forgetting) rate

## Parameters

| Parameter | Meaning | Estimation |
|---|---|---|
| `d` | individual decay rate | back-calculated per person from single-condition accuracy — **varies** |
| `theta` | quiz boost | estimated per person from the (repeated − single) accuracy gap — **varies** |

## Run

```bash
python3 simulation.py
```

- Input: `trackB_data.csv` (responses from the seat-position memory game, n=69)
- Output: STEP 1-3 plus sensitivity analyses (θ=1, and tracking of weak-memory students)

Standard library only; nothing to install.

## Status

Coursework (IMEN315 team project).

---
https://github.com/lshpy
