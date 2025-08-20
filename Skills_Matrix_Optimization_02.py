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
 - Prints a neat summary showing human-readable Skill Level (Basic/Intermediate/Advanced)
 - Writes selection_summary.csv with the chosen person per role and key metrics

Notes:
 - "Effort Level" means average available hours per weekday in the selected date range:
      high ≥ 5h/day, med ≥ 3h/day, low ≥ 1h/day.
   If nobody meets the threshold, we choose the closest (highest avg).
 - We maximize proficiency in the exact Skill you choose (not just the Role).
 - If one person would be selected for multiple roles, we automatically pick the
   next-best candidate for the later role(s) to avoid duplication.

Dependencies: pandas, openpyxl
  pip install pandas openpyxl
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

# Map textual Level → proficiency score (higher = better) for internal ranking
LEVEL_TO_PROF = {"basic": 2.0, "intermediate": 3.5, "advanced": 5.0}

# Human-readable ordering + canonical labels
LEVEL_ORDER = {"basic": 0, "intermediate": 1, "advanced": 2}
LEVEL_CANON = {"basic": "Basic", "intermediate": "Intermediate", "advanced": "Advanced"}

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
        found = (
            skills_df.loc[skills_df["Role"] == r, "Skill"]
            .dropna().astype(str).str.strip().sort_values().unique().tolist()
        )
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
    # weekdays only
    return [d for d in all_dates if (d >= start and d <= end and d.weekday() < 5)]

def skill_level_label(skills_df: pd.DataFrame, name: str, role: str, skill: str) -> str | None:
    """
    Return a clean Skill Level label for (person, role, skill).
    If multiple entries exist, choose the most frequent; tie → highest level.
    """
    rows = skills_df[
        (skills_df["Resource Name"] == name)
        & (skills_df["Role"] == role)
        & (skills_df["Skill"].str.casefold() == skill.casefold())
    ]
    if rows.empty:
        return None
    lv = rows["Level"].astype(str).str.strip().str.lower()
    counts = lv.value_counts()
    top = counts[counts == counts.max()].index.tolist()
    best = max(top, key=lambda k: LEVEL_ORDER.get(k, -1))
    return LEVEL_CANON.get(best, best.title())

def prof_for_employee_skill(skills_df: pd.DataFrame, name: str, role: str, skill: str) -> float:
    rows = skills_df[(skills_df["Resource Name"] == name)
                     & (skills_df["Role"] == role)
                     & (skills_df["Skill"].str.casefold() == skill.casefold())]
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

def pick_for_role(