"""
EA Staffing Optimizer (hours-based) — run locally
=================================================

What it does
------------
Reads two Excel files located in the SAME folder as this script:
  - Example_SKills_Matrix.xlsx  (columns: Resource Name | Role | Skill | Level)
  - Example_Available_Hours.xlsx (columns: Resource Name | <date1> | <date2> | ... hours)

Builds three optimized staffing plans for a single project timeline:
  1) Proficiency-first     (maximize role proficiency)
  2) Development-first     (favor lower current level = higher growth opportunity)
  3) Mixed (60%/40%)       (blend proficiency and development)

Outputs
-------
./outputs/
  - team_prof.csv, team_dev.csv, team_mixed.csv
  - coverage_prof.csv, coverage_dev.csv, coverage_mixed.csv
  - history_prof.csv, history_dev.csv, history_mixed.csv
  - convergence_prof.png, convergence_dev.png, convergence_mixed.png

Quick start
-----------
python staffing_ea.py

Optional tweaks inside CONFIG:
  - TARGET_ROLES_DEFAULT   : which roles (if present in your data) to prioritize
  - NUM_DAYS               : how many date columns to treat as project timeline
  - REQUIREMENTS_MODE      : "auto" (set hours ~50% of total capacity) or "manual"
  - MANUAL_REQUIREMENTS    : dict of {role: {pd.Timestamp(date): hours, ...}}
  - MIX_WEIGHTS            : weight for mixed objective (proficiency, development)

Dependencies
------------
pip install pandas numpy matplotlib openpyxl
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import random

# -------------------------
# CONFIG — adjust as needed
# -------------------------
SKILLS_XLSX = "Example_SKills_Matrix.xlsx"       # long form: Resource Name | Role | Skill | Level
AVAIL_XLSX  = "Example_Available_Hours.xlsx"     # wide form:  Resource Name | <date1> | <date2> | ...

TARGET_ROLES_DEFAULT = ["WRITER", "DESIGNER", "EDITOR"]  # if present; otherwise first 3 roles found
NUM_DAYS = 10                                            # use the first N date columns as the project window

REQUIREMENTS_MODE = "auto"  # "auto" or "manual"
MIX_WEIGHTS = (0.60, 0.40)  # (proficiency_weight, development_weight)

# If REQUIREMENTS_MODE == "manual", fill this like:
# MANUAL_REQUIREMENTS = {
#     "WRITER":  {pd.Timestamp("2025-08-18"): 16, pd.Timestamp("2025-08-19"): 12, ...},
#     "DESIGNER":{pd.Timestamp("2025-08-18"):  8, pd.Timestamp("2025-08-19"): 10, ...},
#     "EDITOR":  {pd.Timestamp("2025-08-18"):  6, pd.Timestamp("2025-08-19"):  6, ...},
# }
MANUAL_REQUIREMENTS: dict[str, dict[pd.Timestamp, float]] = {}

# EA hyperparameters
SEED = 123
POP_SIZE = 250
GENERATIONS = 200
ELITISM = 6
TOURNAMENT = 4
MUT_RATE = 0.08

# -------------------------
# Utilities
# -------------------------
def here() -> Path:
    return Path(__file__).resolve().parent

def load_excel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.name} (expected in {path.parent})")
    # first sheet
    xl = pd.ExcelFile(path)
    return pd.read_excel(path, sheet_name=xl.sheet_names[0])

def normalize_names(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def ensure_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing required column(s): {missing}. Found: {list(df.columns)}")

def level_to_scores(level: str) -> tuple[float, float]:
    """Map textual Level to (proficiency, development-interest)."""
    if not isinstance(level, str):
        return (0.0, 0.0)
    lv = level.strip().lower()
    prof_map = {"basic": 2.0, "intermediate": 3.5, "advanced": 5.0}
    dev_map  = {"basic": 5.0, "intermediate": 3.0, "advanced": 1.0}
    return prof_map.get(lv, 0.0), dev_map.get(lv, 0.0)

# -------------------------
# Data model containers
# -------------------------
@dataclass(frozen=True)
class PersonRole:
    prof: float
    dev: float

# -------------------------
# Parsing
# -------------------------
def parse_skills(skills_df: pd.DataFrame) -> dict[str, dict[str, PersonRole]]:
    """
    From long-form skills: Resource Name | Role | Skill | Level
    Compute mean prof/dev per (Resource Name, Role).
    Returns: employees[name][ROLE] = PersonRole(prof, dev)
    """
    ensure_columns(skills_df, ["Resource Name", "Role", "Skill", "Level"], "Skills sheet")

    skills_df = skills_df.copy()
    skills_df["Resource Name"] = normalize_names(skills_df["Resource Name"])
    skills_df["Role"] = skills_df["Role"].astype(str).str.strip().str.upper()

    profs, devs = zip(*skills_df["Level"].astype(str).map(level_to_scores))
    skills_df["prof_score"] = list(profs)
    skills_df["dev_score"] = list(devs)

    agg = (skills_df
           .groupby(["Resource Name", "Role"], as_index=False)
           .agg(prof=("prof_score", "mean"),
                dev=("dev_score", "mean"),
                skills_count=("Skill", "count")))

    employees: dict[str, dict[str, PersonRole]] = {}
    for _, r in agg.iterrows():
        name = r["Resource Name"]
        role = r["Role"]
        if name not in employees:
            employees[name] = {}
        employees[name][role] = PersonRole(float(r["prof"]), float(r["dev"]))
    return employees

def parse_availability(avail_df: pd.DataFrame, num_days: int) -> tuple[list[pd.Timestamp], dict[str, dict[pd.Timestamp, float]]]:
    """
    From wide availability: Resource Name | date1 | date2 | ...
    Returns:
      WEEKS (list of pd.Timestamp) and
      availability[name][date] = hours
    """
    ensure_columns(avail_df, ["Resource Name"], "Availability sheet")
    avail_df = avail_df.copy()
    avail_df["Resource Name"] = normalize_names(avail_df["Resource Name"])

    # Date columns: anything except the name col
    date_cols = [c for c in avail_df.columns if c != "Resource Name"]
    # Coerce to datetime for keys, numeric for values
    def to_date(c):
        try:
            return pd.to_datetime(c)
        except Exception:
            return None
    dates = [to_date(c) for c in date_cols]
    keep_pairs = [(c, d) for c, d in zip(date_cols, dates) if d is not None]
    if not keep_pairs:
        raise ValueError("No parsable date columns found in availability sheet.")
    # sort by date and take first num_days
    keep_pairs.sort(key=lambda x: x[1])
    keep_pairs = keep_pairs[:num_days]
    week_cols = [c for (c, _) in keep_pairs]
    weeks = [d for (_, d) in keep_pairs]

    avail_df[week_cols] = avail_df[week_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    availability: dict[str, dict[pd.Timestamp, float]] = {}
    for _, row in avail_df.iterrows():
        name = row["Resource Name"]
        availability[name] = {d: float(row[c]) for c, d in keep_pairs}
    return weeks, availability

# -------------------------
# Requirements (project)
# -------------------------
def build_requirements_auto(target_roles: list[str],
                            employees: dict[str, dict[str, PersonRole]],
                            availability: dict[str, dict[pd.Timestamp, float]],
                            weeks: list[pd.Timestamp],
                            load_fraction: float = 0.50) -> dict[str, dict[pd.Timestamp, float]]:
    """
    For each role, take ~50% of total available hours across all weeks, split evenly by day.
    """
    req: dict[str, dict[pd.Timestamp, float]] = {}
    for role in target_roles:
        total = 0.0
        for name, roles in employees.items():
            if role in roles and name in availability:
                total += sum(availability[name].get(w, 0.0) for w in weeks)
        per_day = max(1.0, (load_fraction * total) / max(1, len(weeks)))
        req[role] = {w: float(round(per_day, 2)) for w in weeks}
    return req

# -------------------------
# EA (evolutionary algorithm)
# -------------------------
@dataclass
class Individual:
    assign: dict[str, str | None]  # employee -> role or None
    fitness: float = -1e9
    feasible: bool = False

def base_score(assign: dict[str, str | None],
               employees: dict[str, dict[str, PersonRole]],
               mode: str,
               mix_weights: tuple[float, float]) -> float:
    s = 0.0
    for name, role in assign.items():
        if role is None:
            continue
        person_roles = employees.get(name, {})
        pr = person_roles.get(role)
        if not pr:
            continue
        if mode == "prof":
            s += pr.prof
        elif mode == "dev":
            s += pr.dev
        else:
            s += mix_weights[0] * pr.prof + mix_weights[1] * pr.dev
    return s

def coverage(assign: dict[str, str | None],
             employees: dict[str, dict[str, PersonRole]],
             availability: dict[str, dict[pd.Timestamp, float]],
             target_roles: list[str],
             weeks: list[pd.Timestamp]) -> dict[str, dict[pd.Timestamp, float]]:
    cov = {r: {w: 0.0 for w in weeks} for r in target_roles}
    for name, role in assign.items():
        if role is None or role not in target_roles:
            continue
        if role not in employees.get(name, {}):
            continue
        for w in weeks:
            cov[role][w] += availability.get(name, {}).get(w, 0.0)
    return cov

def feasibility(assign: dict[str, str | None],
                employees: dict[str, dict[str, PersonRole]],
                availability: dict[str, dict[pd.Timestamp, float]],
                req: dict[str, dict[pd.Timestamp, float]],
                target_roles: list[str],
                weeks: list[pd.Timestamp]) -> tuple[float, float, dict[str, dict[pd.Timestamp, float]]]:
    cov = coverage(assign, employees, availability, target_roles, weeks)
    deficit = 0.0
    oversupply = 0.0
    for r in target_roles:
        for w in weeks:
            need = req[r][w]
            got = cov[r][w]
            if got < need:
                deficit += (need - got)
            else:
                oversupply += (got - need)
    return deficit, oversupply, cov

def fitness(assign: dict[str, str | None],
            employees: dict[str, dict[str, PersonRole]],
            availability: dict[str, dict[pd.Timestamp, float]],
            req: dict[str, dict[pd.Timestamp, float]],
            target_roles: list[str],
            weeks: list[pd.Timestamp],
            objective: str,
            mix_weights: tuple[float, float]) -> tuple[float, bool]:
    mode = "prof" if objective == "prof" else ("dev" if objective == "dev" else "mix")
    base = base_score(assign, employees, mode, mix_weights)
    deficit, oversupply, _ = feasibility(assign, employees, availability, req, target_roles, weeks)
    # heavy penalty for deficits; light for oversupply to keep teams lean
    return base - 5.0 * deficit - 0.05 * oversupply, (deficit == 0.0)

def rand_assign(employees: dict[str, dict[str, PersonRole]],
                target_roles: list[str]) -> dict[str, str | None]:
    a: dict[str, str | None] = {}
    for name, roles in employees.items():
        options = [None] + [r for r in target_roles if r in roles]
        a[name] = random.choice(options)
    return a

def mutate(a: dict[str, str | None],
           employees: dict[str, dict[str, PersonRole]],
           target_roles: list[str],
           rate: float) -> dict[str, str | None]:
    out = dict(a)
    for name, roles in employees.items():
        if random.random() < rate:
            options = [None] + [r for r in target_roles if r in roles]
            out[name] = random.choice(options)
    return out

def crossover(a: dict[str, str | None],
              b: dict[str, str | None]) -> dict[str, str | None]:
    out = {}
    for name in a.keys():
        out[name] = a[name] if random.random() < 0.5 else b[name]
    return out

def tournament_select(pop: list[Individual], k: int) -> Individual:
    cands = random.sample(pop, k)
    return max(cands, key=lambda x: x.fitness)

def run_ea(objective: str,
           employees: dict[str, dict[str, PersonRole]],
           availability: dict[str, dict[pd.Timestamp, float]],
           req: dict[str, dict[pd.Timestamp, float]],
           target_roles: list[str],
           weeks: list[pd.Timestamp],
           mix_weights: tuple[float, float],
           pop_size: int, generations: int, elitism: int, tournament: int, mut_rate: float,
           seed: int) -> tuple[Individual, pd.DataFrame]:
    random.seed(seed)
    pop: list[Individual] = [Individual(rand_assign(employees, target_roles)) for _ in range(pop_size)]
    history = []

    def eval_ind(ind: Individual) -> None:
        fit, feas = fitness(ind.assign, employees, availability, req, target_roles, weeks, objective, mix_weights)
        ind.fitness = fit
        ind.feasible = feas

    # initial evaluation
    for ind in pop:
        eval_ind(ind)

    for gen in range(generations):
        pop.sort(key=lambda x: x.fitness, reverse=True)
        best = pop[0]
        history.append({
            "generation": gen,
            "best_fitness": best.fitness,
            "avg_fitness": float(np.mean([p.fitness for p in pop])),
            "best_feasible": best.feasible
        })

        # next generation
        next_pop: list[Individual] = []
        next_pop.extend(pop[:elitism])
        while len(next_pop) < pop_size:
            p1 = tournament_select(pop, tournament)
            p2 = tournament_select(pop, tournament)
            child_assign = crossover(p1.assign, p2.assign)
            child_assign = mutate(child_assign, employees, target_roles, mut_rate)
            next_pop.append(Individual(child_assign))
        pop = next_pop

        for ind in pop:
            eval_ind(ind)

    pop.sort(key=lambda x: x.fitness, reverse=True)
    feas = [p for p in pop if p.feasible]
    best = feas[0] if feas else pop[0]
    return best, pd.DataFrame(history)

# -------------------------
# Reporting
# -------------------------
def make_team_table(sol: Individual,
                    employees: dict[str, dict[str, PersonRole]],
                    availability: dict[str, dict[pd.Timestamp, float]],
                    target_roles: list[str],
                    weeks: list[pd.Timestamp]) -> pd.DataFrame:
    rows = []
    for name, role in sol.assign.items():
        if role is None: 
            continue
        pr = employees.get(name, {}).get(role)
        if not pr:
            continue
        rec = {"employee": name, "assigned_role": role, "prof": pr.prof, "dev": pr.dev}
        for w in weeks:
            rec[f"{w:%Y-%m-%d}_hours"] = availability.get(name, {}).get(w, 0.0)
        rows.append(rec)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["assigned_role", "prof", "dev"], ascending=[True, False, False])
    return df

def make_coverage_table(sol: Individual,
                        employees: dict[str, dict[str, PersonRole]],
                        availability: dict[str, dict[pd.Timestamp, float]],
                        req: dict[str, dict[pd.Timestamp, float]],
                        target_roles: list[str],
                        weeks: list[pd.Timestamp]) -> pd.DataFrame:
    cov = coverage(sol.assign, employees, availability, target_roles, weeks)
    rows = []
    for r in target_roles:
        rec = {"role": r}
        for w in weeks:
            rec[f"{w:%Y-%m-%d}_required"] = req[r][w]
            rec[f"{w:%Y-%m-%d}_covered"]  = cov[r][w]
        rows.append(rec)
    return pd.DataFrame(rows)

def save_plot(hist: pd.DataFrame, title: str, out_path: Path) -> None:
    plt.figure()
    plt.plot(hist["generation"], hist["best_fitness"], label="Best")
    plt.plot(hist["generation"], hist["avg_fitness"], label="Average")
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# -------------------------
# Main
# -------------------------
def main():
    script_dir = here()
    out_dir = script_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    skills_df = load_excel(script_dir / SKILLS_XLSX)
    avail_df  = load_excel(script_dir / AVAIL_XLSX)

    # Parse
    employees = parse_skills(skills_df)
    weeks, availability = parse_availability(avail_df, NUM_DAYS)

    # intersect people present in both files
    common = [n for n in employees.keys() if n in availability]
    if not common:
        print("No overlapping names between skills and availability. Check 'Resource Name' values.", file=sys.stderr)
        sys.exit(1)
    employees = {n: employees[n] for n in common}
    availability = {n: availability[n] for n in common}

    # Role universe & target roles
    all_roles = sorted({r for m in employees.values() for r in m.keys()})
    target_roles = [r for r in TARGET_ROLES_DEFAULT if r in all_roles] or all_roles[:3]
    if not target_roles:
        raise RuntimeError("No roles found after parsing the skills file.")

    # Requirements
    if REQUIREMENTS_MODE == "manual":
        req = MANUAL_REQUIREMENTS
        # basic validation
        for r in target_roles:
            if r not in req:
                raise ValueError(f"Manual requirements missing role: {r}")
            for w in weeks:
                if w not in req[r]:
                    raise ValueError(f"Manual requirements missing {r} @ {w:%Y-%m-%d}")
    else:
        req = build_requirements_auto(target_roles, employees, availability, weeks, load_fraction=0.50)

    # Run EAs
    results = {}
    histories = {}
    for label in ["prof", "dev", "mixed"]:
        best, hist = run_ea(label, employees, availability, req, target_roles, weeks,
                            MIX_WEIGHTS, POP_SIZE, GENERATIONS, ELITISM, TOURNAMENT, MUT_RATE, SEED)
        results[label] = best
        histories[label] = hist
        hist.to_csv(out_dir / f"history_{label}.csv", index=False)
        save_plot(hist, f"EA Convergence — {label}", out_dir / f"convergence_{label}.png")

    # Save teams & coverage
    for label in ["prof", "dev", "mixed"]:
        team_df = make_team_table(results[label], employees, availability, target_roles, weeks)
        cov_df  = make_coverage_table(results[label], employees, availability, req, target_roles, weeks)
        team_df.to_csv(out_dir / f"team_{label}.csv", index=False)
        cov_df.to_csv(out_dir / f"coverage_{label}.csv", index=False)

    # Console summary
    print("\n=== Run complete ===")
    print(f"People counted: {len(employees)} | Roles: {target_roles}")
    print("Project days:", ", ".join([w.strftime("%Y-%m-%d") for w in weeks]))
    print("Outputs written to:", out_dir.resolve())
    for p in sorted(out_dir.iterdir()):
        print("-", p.name)

if __name__ == "__main__":
    main()
