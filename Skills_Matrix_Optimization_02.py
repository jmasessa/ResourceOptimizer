"""
Staffing picker with user prompts (Designer / Writer / Editor)
- Reads:
    Example_SKills_Matrix.xlsx  (columns: Resource Name | Role | Skill | Level)
    Example_Available_Hours.xlsx (columns: Resource Name | <date1> | <date2> ...)

Prompts for:
 1) Desired Designer Skillset
 2) Designer Effort Level (high/med/low)
 3) Desired Writer Skillset
 4) Writer Effort Level (high/med/low)
 5) Desired Editor Skillset
 6) Editor Effort Level (high/med/low)
 7) Project Start Date (YYYY-MM-DD)
 8) Project End Date   (YYYY-MM-DD)

Output:
 - Prints a neat summary
 - Writes selection_summary.csv with the chosen person per role and key metrics

Notes:
 - "Effort Level" means average available hours per weekday in the selected date range:
      high ≥ 5h/day, med ≥ 3h/day, low ≥ 1h/day.
   If nobody meets the threshold, we choose the closest (highest avg).
 - We maximize proficiency in the exact Skill you choose (not just the Role).
 - If one person would be selected for multiple roles, we automatically pick the
   next-best candidate for the later role(s) to avoid duplication.
"""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# ---------------- Configuration ----------------
# We’ll try both spellings to match your files.
SKILLS_FILES = ["Example_SKills_Matrix.xlsx", "Example_Skills_Matrix.xlsx"]
AVAIL_FILE   = "Example_Available_Hours.xlsx"

ROLES = ["DESIGNER", "WRITER", "EDITOR"]
EFFORT_THRESH = {"high": 5.0, "med": 3.0, "low": 1.0}

# Map textual Level → proficiency score (higher = better)
LEVEL_TO_PROF = {"basic": 2.0, "intermediate": 3.5, "advanced": 5.0}

# ---------------- Utilities ----------------
def find_skills_file() -> Path:
    here = Path(__file__).resolve().parent
    for name in SKILLS_FILES:
        p = here / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find any of: {SKILLS_FILES} in {here}")

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    here = Path(__file__).resolve().parent
    skills_path = find_skills_file()
    avail_path = here / AVAIL_FILE
    if not avail_path.exists():
        raise FileNotFoundError(f"Missing file: {AVAIL_FILE} in {here}")

    # Load first sheet in each workbook
    skills_xl = pd.ExcelFile(skills_path)
    avail_xl  = pd.ExcelFile(avail_path)
    skills_df = pd.read_excel(skills_path, sheet_name=skills_xl.sheet_names[0])
    avail_df  = pd.read_excel(avail_path,  sheet_name=avail_xl.sheet_names[0])
    return skills_df, avail_df

def normalize_frames(skills_df: pd.DataFrame, avail_df: pd.DataFrame):
    # Required columns
    for col in ["Resource Name", "Role", "Skill", "Level"]:
        if col not in skills_df.columns:
            raise ValueError(f"Skills sheet missing column: {col}")
    if "Resource Name" not in avail_df.columns:
        raise ValueError("Availability sheet missing column: Resource Name")

    # Normalize
    skills_df = skills_df.copy()
    avail_df  = avail_df.copy()
    skills_df["Resource Name"] = skills_df["Resource Name"].astype(str).str.strip()
    skills_df["Role"]  = skills_df["Role"].astype(str).str.strip().str.upper()
    skills_df["Skill"] = skills_df["Skill"].astype(str).str.strip()
    skills_df["Level"] = skills_df["Level"].astype(str).str.strip().str.lower()

    avail_df["Resource Name"] = avail_df["Resource Name"].astype(str).str.strip()

    # Coerce date columns
    date_cols = [c for c in avail_df.columns if c != "Resource Name"]
    dates = []
    keep = []
    for c in date_cols:
        try:
            d = pd.to_datetime(c)
            dates.append(d)
            keep.append(c)
        except Exception:
            pass
    if not keep:
        raise ValueError("No parseable date columns found in availability sheet.")
    # Sort columns by date
    keep_pairs = sorted(zip(keep, dates), key=lambda x: x[1])
    keep_cols = [c for c, _ in keep_pairs]
    keep_dates = [d for _, d in keep_pairs]
    avail_df[keep_cols] = avail_df[keep_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return skills_df, avail_df, keep_cols, keep_dates

def list_skills_per_role(skills_df: pd.DataFrame) -> dict[str, list[str]]:
    skills_per_role = {}
    for r in ROLES:
        found = (skills_df.loc[skills_df["Role"] == r, "Skill"]
                 .dropna().astype(str).str.strip().sort_values().unique().tolist())
        skills_per_role[r] = found
    return skills_per_role

def prompt_choice(prompt: str, valid: list[str]) -> str:
    vset = {v.lower(): v for v in valid}
    while True:
        s = input(prompt).strip()
        if s.lower() in vset:
            return vset[s.lower()]
        print(f"✋ Not found. Type one of: {', '.join(valid)}")

def prompt_effort(prompt: str) -> str:
    while True:
        s = input(prompt).strip().lower()
        if s in EFFORT_THRESH:
            return s
        print("✋ Please enter 'high', 'med', or 'low'.")

def prompt_date(prompt: str) -> pd.Timestamp:
    while True:
        s = input(prompt).strip()
        try:
            return pd.to_datetime(s)
        except Exception:
            print("✋ Please enter a valid date like 2025-08-18.")

def restrict_dates(all_dates: list[pd.Timestamp], start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    if start > end:
        start, end = end, start
    return [d for d in all_dates if (d >= start and d <= end and d.weekday() < 5)]  # weekdays only

def prof_for_employee_skill(skills_df: pd.DataFrame, name: str, role: str, skill: str) -> float:
    rows = skills_df[(skills_df["Resource Name"] == name) &
                     (skills_df["Role"] == role) &
                     (skills_df["Skill"].str.casefold() == skill.casefold())]
    if rows.empty:
        return 0.0
    # If multiple entries, average mapped scores
    prof_vals = rows["Level"].map(lambda lv: LEVEL_TO_PROF.get(lv.lower(), 0.0)).tolist()
    return float(np.mean(prof_vals)) if prof_vals else 0.0

def avg_weekday_hours(avail_df: pd.DataFrame, name: str, date_cols: list[str], dates: list[pd.Timestamp]) -> float:
    # Average hours per included weekday; missing name/date treated as 0
    if name not in set(avail_df["Resource Name"]):
        return 0.0
    r = avail_df.loc[avail_df["Resource Name"] == name]
    if r.empty or not dates:
        return 0.0
    # Map date -> column
    col_by_date = {d: c for c, d in zip(date_cols, dates)}
    vals = [float(r.iloc[0][col_by_date[d]]) if col_by_date[d] in r.columns else 0.0 for d in dates]
    return float(np.mean(vals)) if vals else 0.0

def pick_for_role(skills_df: pd.DataFrame,
                  avail_df: pd.DataFrame,
                  role: str,
                  desired_skill: str,
                  threshold_hours: float,
                  date_cols: list[str],
                  dates: list[pd.Timestamp],
                  already_chosen: set[str]) -> dict:
    # Candidates = everyone with this Role
    names = (skills_df.loc[skills_df["Role"] == role, "Resource Name"]
             .dropna().astype(str).str.strip().unique().tolist())
    if not names:
        return {"role": role, "chosen": None, "reason": "No candidates with this role."}

    # Score each candidate
    rows = []
    for name in names:
        prof = prof_for_employee_skill(skills_df, name, role, desired_skill)
        avg_h = avg_weekday_hours(avail_df, name, date_cols, dates)
        distance = max(0.0, threshold_hours - avg_h)  # 0 means meets threshold
        rows.append({
            "name": name,
            "prof_score": prof,
            "avg_weekday_hours": avg_h,
            "distance_to_threshold": distance
        })
    cand_df = pd.DataFrame(rows)
    if cand_df.empty:
        return {"role": role, "chosen": None, "reason": "No candidates scored."}

    # Rank:
    # 1) prefer those meeting threshold (distance == 0)
    # 2) higher prof_score
    # 3) higher avg_weekday_hours
    # If nobody meets threshold, pick closest (min distance), then prof, then hours.
    meets = cand_df[cand_df["distance_to_threshold"] == 0.0].copy()
    misses = cand_df[cand_df["distance_to_threshold"] > 0.0].copy()
    meets = meets.sort_values(["prof_score", "avg_weekday_hours"], ascending=[False, False])
    misses = misses.sort_values(["distance_to_threshold", "prof_score", "avg_weekday_hours"],
                                ascending=[True, False, False])
    ranked = pd.concat([meets, misses], ignore_index=True)

    # Avoid duplicate people across roles: pick first not already chosen
    for _, row in ranked.iterrows():
        if row["name"] not in already_chosen:
            chosen = row.to_dict()
            chosen["met_threshold"] = (chosen["distance_to_threshold"] == 0.0)
            chosen["desired_skill"] = desired_skill
            return {"role": role, "chosen": chosen, "reason": ""}

    # If all top picks already used, we’ll take the highest-ranked anyway
    fallback = ranked.iloc[0].to_dict()
    fallback["met_threshold"] = (fallback["distance_to_threshold"] == 0.0)
    fallback["desired_skill"] = desired_skill
    return {"role": role, "chosen": fallback, "reason": "Had to reuse because all others were already chosen."}

# ---------------- Main ----------------
def main():
    # Load & normalize
    skills_df, avail_df = load_inputs()
    skills_df, avail_df, date_cols, dates = normalize_frames(skills_df, avail_df)

    # Build role→skills lookup for validation
    skills_per_role = list_skills_per_role(skills_df)

    # ---- Prompts ----
    print("\nAvailable skills by role:")
    for r in ROLES:
        print(f" • {r}: {', '.join(skills_per_role[r]) or '(none)'}")

    designer_skill = prompt_choice("\nDesired Designer Skillset: ", skills_per_role["DESIGNER"]) if skills_per_role["DESIGNER"] else ""
    designer_eff   = prompt_effort("Designer Effort Level (high/med/low): ")

    writer_skill   = prompt_choice("\nDesired Writer Skillset: ", skills_per_role["WRITER"]) if skills_per_role["WRITER"] else ""
    writer_eff     = prompt_effort("Writer Effort Level (high/med/low): ")

    editor_skill   = prompt_choice("\nDesired Editor Skillset: ", skills_per_role["EDITOR"]) if skills_per_role["EDITOR"] else ""
    editor_eff     = prompt_effort("Editor Effort Level (high/med/low): ")

    start_date     = prompt_date("\nProject Start Date (YYYY-MM-DD): ")
    end_date       = prompt_date("Project End Date   (YYYY-MM-DD): ")

    # Filter the availability columns to the selected weekday date range
    picked_dates = restrict_dates(dates, start_date, end_date)
    if not picked_dates:
        print("\n✋ No weekday dates in the selected range that match your availability sheet columns.")
        print("   Check your dates or the availability file headers.")
        sys.exit(1)

    # Subset the mapped columns for the picked dates
    col_by_date = {d: c for c, d in zip(date_cols, dates)}
    picked_cols = [col_by_date[d] for d in picked_dates]

    # ---- Selection per role with thresholds ----
    thresholds = {
        "DESIGNER": EFFORT_THRESH[designer_eff],
        "WRITER":   EFFORT_THRESH[writer_eff],
        "EDITOR":   EFFORT_THRESH[editor_eff],
    }
    desired_skills = {
        "DESIGNER": designer_skill,
        "WRITER":   writer_skill,
        "EDITOR":   editor_skill,
    }

    chosen = []
    used_people = set()
    for role in ROLES:
        res = pick_for_role(skills_df, avail_df, role, desired_skills[role], thresholds[role],
                            picked_cols, picked_dates, used_people)
        chosen.append(res)
        if res["chosen"] and res["chosen"].get("name"):
            used_people.add(res["chosen"]["name"])

    # ---- Report ----
    print("\n=== Selection Summary ===")
    rows = []
    for res in chosen:
        role = res["role"]
        ch   = res["chosen"]
        if not ch:
            print(f"{role:8s}: (no candidate)  {res.get('reason','')}")
            rows.append({
                "role": role, "employee": None, "desired_skill": desired_skills[role],
                "prof_score": None, "avg_weekday_hours": None, "effort_threshold": thresholds[role],
                "met_threshold": None, "note": res.get("reason","")
            })
            continue
        print(f"{role:8s}: {ch['name']}  | skill='{ch['desired_skill']}' "
              f"| prof={ch['prof_score']:.2f} | avg/day={ch['avg_weekday_hours']:.2f}h "
              f"| needs≥{thresholds[role]:.1f} → {'OK' if ch['met_threshold'] else 'closest'} "
              f"{'(dup-resolved)' if res.get('reason') else ''}")
        rows.append({
            "role": role,
            "employee": ch["name"],
            "desired_skill": ch["desired_skill"],
            "prof_score": round(float(ch["prof_score"]), 3),
            "avg_weekday_hours": round(float(ch["avg_weekday_hours"]), 3),
            "effort_threshold": thresholds[role],
            "met_threshold": bool(ch["met_threshold"]),
            "note": res.get("reason","")
        })

    # Save CSV
    out_path = Path(__file__).resolve().parent / "selection_summary.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")

    # Helpful context
    print("\nDates used (weekdays):", ", ".join(d.strftime("%Y-%m-%d") for d in picked_dates))

if __name__ == "__main__":
    main()
