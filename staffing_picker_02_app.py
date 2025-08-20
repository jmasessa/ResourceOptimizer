"""
Streamlit GUI — Staffing Picker (Designer / Writer / Editor)
===========================================================
- Reads two Excel files (same folder or via upload):
    • Example_SKills_Matrix.xlsx   (Resource Name | Role | Skill | Level)
    • Example_Available_Hours.xlsx (Resource Name | <date1> | <date2> ...)

- Lets the user pick per role:
    • Skill (from skills matrix for that role)
    • Preferred Level (Advanced / Intermediate / Basic)
    • Expected Effort (high / med / low) → avg weekday-hrs ≥ 5 / 3 / 1

- Tries to satisfy Preferred Level first; if nobody meets availability + level,
  falls back to any level keeping the same ranking rules.

Run:
  pip install streamlit pandas openpyxl
  streamlit run staffing_picker_app.py
"""
from __future__ import annotations
from pathlib import Path
from contextlib import suppress
import io
import pandas as pd
import numpy as np
import streamlit as st

# ---------------- Page setup ----------------
st.set_page_config(page_title="Staffing Picker", page_icon="🧩", layout="wide")
st.title("🧩 Staffing Picker — GUI")
st.caption("Choose skills, preferred levels, and dates; we'll suggest one person per role.")

# ---------------- Configuration ----------------
DEFAULT_SKILLS_FILES = ["Example_SKills_Matrix.xlsx", "Example_Skills_Matrix.xlsx"]
DEFAULT_AVAIL_FILE   = "Example_Available_Hours.xlsx"
ROLES = ["DESIGNER", "WRITER", "EDITOR"]
EFFORT_THRESH = {"high": 5.0, "med": 3.0, "low": 1.0}
LEVEL_TO_PROF = {"basic": 2.0, "intermediate": 3.5, "advanced": 5.0}
LEVEL_ORDER = {"basic": 0, "intermediate": 1, "advanced": 2}
LEVEL_CANON = {"basic": "Basic", "intermediate": "Intermediate", "advanced": "Advanced"}

# ---------------- Helpers ----------------
@st.cache_data(show_spinner=False)
def _load_excel_df(src: str | bytes, first_sheet_only: bool = True) -> pd.DataFrame:
    """Load first sheet of an Excel file from bytes or disk path."""
    if isinstance(src, (bytes, bytearray)):
        bio = io.BytesIO(src)
        xl = pd.ExcelFile(bio)
        return pd.read_excel(bio, sheet_name=xl.sheet_names[0]) if first_sheet_only else pd.read_excel(bio)
    path = Path(str(src))
    if not path.exists():
        raise FileNotFoundError(f"File not found: {src}")
    xl = pd.ExcelFile(path)
    return pd.read_excel(path, sheet_name=xl.sheet_names[0])

@st.cache_data(show_spinner=False)
def _find_default_skills_path() -> Path | None:
    here = Path.cwd()
    for name in DEFAULT_SKILLS_FILES:
        p = here / name
        if p.exists():
            return p
    return None

def hard_stop(msg: str):
    """Stop safely in Streamlit *and* when run as a plain Python script."""
    st.error(msg)
    with suppress(Exception):
        st.stop()
    raise SystemExit(msg)

@st.cache_data(show_spinner=False)
def normalize_frames(skills_df: pd.DataFrame, avail_df: pd.DataFrame):
    # Validate columns
    required = ["Resource Name", "Role", "Skill", "Level"]
    for col in required:
        if col not in skills_df.columns:
            raise ValueError(f"Skills sheet missing column: {col}")
    if "Resource Name" not in avail_df.columns:
        raise ValueError("Availability sheet missing column: Resource Name")

    skills_df = skills_df.copy()
    avail_df = avail_df.copy()

    skills_df["Resource Name"] = skills_df["Resource Name"].astype(str).str.strip()
    skills_df["Role"]  = skills_df["Role"].astype(str).str.strip().str.upper()
    skills_df["Skill"] = skills_df["Skill"].astype(str).str.strip()
    skills_df["Level"] = skills_df["Level"].astype(str).str.strip().str.lower()

    avail_df["Resource Name"] = avail_df["Resource Name"].astype(str).str.strip()

    # Parse date columns and sort
    date_cols = [c for c in avail_df.columns if c != "Resource Name"]
    pairs = []
    for c in date_cols:
        try:
            pairs.append((c, pd.to_datetime(c)))
        except Exception:
            pass
    if not pairs:
        raise ValueError("No parseable date columns found in availability sheet.")
    pairs.sort(key=lambda x: x[1])
    keep_cols = [c for c, _ in pairs]
    keep_dates = [d for _, d in pairs]

    avail_df[keep_cols] = avail_df[keep_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return skills_df, avail_df, keep_cols, keep_dates

def list_skills_per_role(skills_df: pd.DataFrame) -> dict[str, list[str]]:
    out = {}
    for r in ROLES:
        out[r] = (
            skills_df.loc[skills_df["Role"] == r, "Skill"]
            .dropna().astype(str).str.strip().sort_values().unique().tolist()
        )
    return out

def restrict_weekdays(dates: list[pd.Timestamp], start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    if start > end:
        start, end = end, start
    return [d for d in dates if (start <= d <= end and d.weekday() < 5)]

def skill_level_label(skills_df: pd.DataFrame, name: str, role: str, skill: str) -> str | None:
    rows = skills_df[(skills_df["Resource Name"] == name) & (skills_df["Role"] == role) & (skills_df["Skill"].str.casefold() == skill.casefold())]
    if rows.empty:
        return None
    lv = rows["Level"].astype(str).str.strip().str.lower()
    counts = lv.value_counts()
    top = counts[counts == counts.max()].index.tolist()
    best = max(top, key=lambda k: LEVEL_ORDER.get(k, -1))
    return LEVEL_CANON.get(best, best.title())

def prof_for_employee_skill(skills_df: pd.DataFrame, name: str, role: str, skill: str) -> float:
    rows = skills_df[(skills_df["Resource Name"] == name) & (skills_df["Role"] == role) & (skills_df["Skill"].str.casefold() == skill.casefold())]
    if rows.empty:
        return 0.0
    vals = rows["Level"].map(lambda lv: LEVEL_TO_PROF.get(str(lv).lower(), 0.0)).tolist()
    return float(np.mean(vals)) if vals else 0.0

def avg_weekday_hours(avail_df: pd.DataFrame, name: str, date_cols: list[str], dates: list[pd.Timestamp]) -> float:
    if name not in set(avail_df["Resource Name"]):
        return 0.0
    r = avail_df.loc[avail_df["Resource Name"] == name]
    if r.empty or not dates:
        return 0.0
    col_by_date = {d: c for c, d in zip(date_cols, dates)}
    vals = [float(r.iloc[0][col_by_date[d]]) if col_by_date[d] in r.columns else 0.0 for d in dates]
    return float(np.mean(vals)) if vals else 0.0

def pick_for_role_pref(skills_df: pd.DataFrame, avail_df: pd.DataFrame, role: str, desired_skill: str, preferred_level: str | None, threshold_hours: float, date_cols: list[str], dates: list[pd.Timestamp], already_chosen: set[str]) -> dict:
    names = skills_df.loc[skills_df["Role"] == role, "Resource Name"].dropna().astype(str).str.strip().unique().tolist()
    if not names:
        return {"role": role, "chosen": None, "reason": "No candidates with this role."}

    rows = []
    for name in names:
        prof = prof_for_employee_skill(skills_df, name, role, desired_skill)
        avg_h = avg_weekday_hours(avail_df, name, date_cols, dates)
        distance = max(0.0, threshold_hours - avg_h)
        level_label = skill_level_label(skills_df, name, role, desired_skill)
        rows.append({
            "name": name,
            "prof_score": prof,
            "avg_weekday_hours": avg_h,
            "distance_to_threshold": distance,
            "skill_level": level_label,
        })
    cand_df = pd.DataFrame(rows)
    if cand_df.empty:
        return {"role": role, "chosen": None, "reason": "No candidates scored."}

    def rank(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        meets = df[df["distance_to_threshold"] == 0.0].sort_values(["prof_score", "avg_weekday_hours"], ascending=[False, False])
        misses = df[df["distance_to_threshold"] > 0.0].sort_values(["distance_to_threshold", "prof_score", "avg_weekday_hours"], ascending=[True, False, False])
        return pd.concat([meets, misses], ignore_index=True)

    # Try preferred level first
    reason = ""
    if preferred_level:
        subset = cand_df[cand_df["skill_level"].str.lower() == preferred_level.lower()].copy()
        ranked = rank(subset)
        if ranked.empty:
            ranked = rank(cand_df)
            reason = "No candidates at preferred level; fell back to any level."
    else:
        ranked = rank(cand_df)

    for _, row in ranked.iterrows():
        if row["name"] not in already_chosen:
            chosen = row.to_dict()
            chosen["met_threshold"] = (chosen["distance_to_threshold"] == 0.0)
            chosen["desired_skill"] = desired_skill
            return {"role": role, "chosen": chosen, "reason": reason}

    fallback = ranked.iloc[0].to_dict()
    fallback["met_threshold"] = (fallback["distance_to_threshold"] == 0.0)
    fallback["desired_skill"] = desired_skill
    note = "Had to reuse because all others were already chosen."
    if reason:
        note = reason + " " + note
    return {"role": role, "chosen": fallback, "reason": note}

# ---------------- Data inputs (sidebar) ----------------
st.sidebar.header("Data Sources")
def_skills_path = _find_default_skills_path()
skills_path_str = str(def_skills_path) if def_skills_path else DEFAULT_SKILLS_FILES[0]
skills_choice = st.sidebar.text_input("Skills file name", skills_path_str, key="skills_file_input")
avail_choice  = st.sidebar.text_input("Availability file name", DEFAULT_AVAIL_FILE, key="avail_file_input")

st.sidebar.caption("Or upload files:")
uploaded_skills = st.sidebar.file_uploader("Upload Skills Matrix (.xlsx)", type=["xlsx"], key="skills_upload")
uploaded_avail  = st.sidebar.file_uploader("Upload Available Hours (.xlsx)", type=["xlsx"], key="avail_upload")

# Defensive load
try:
    skills_src = uploaded_skills.read() if uploaded_skills else skills_choice
    avail_src  = uploaded_avail.read()  if uploaded_avail  else avail_choice
    skills_df_raw = _load_excel_df(skills_src)
    avail_df_raw  = _load_excel_df(avail_src)
except Exception as e:
    hard_stop(f"Error loading files: {e}")

try:
    skills_df, avail_df, date_cols, dates = normalize_frames(skills_df_raw, avail_df_raw)
except Exception as e:
    hard_stop(f"Error parsing/normalizing sheets: {e}")

skills_by_role = list_skills_per_role(skills_df)

# ---------------- Controls ----------------
st.sidebar.header("Selections")
cols = st.columns(3)

def role_selects(role: str, key_prefix: str):
    opts = skills_by_role.get(role, [])
    if not opts:
        skill = st.selectbox(f"{role.title()} Skill", options=["(no skills found)"] , key=f"{key_prefix}_skill", disabled=True)
        level = st.selectbox(f"{role.title()} Level", options=["Advanced","Intermediate","Basic"], key=f"{key_prefix}_level", disabled=True)
        eff   = st.selectbox(f"{role.title()} Expected Effort", options=list(EFFORT_THRESH.keys()), key=f"{key_prefix}_eff")
        return None, None, eff
    skill = st.selectbox(f"{role.title()} Skill", options=opts, key=f"{key_prefix}_skill")
    level = st.selectbox(f"{role.title()} Level", options=["Advanced","Intermediate","Basic"], key=f"{key_prefix}_level")
    eff   = st.selectbox(f"{role.title()} Expected Effort", options=list(EFFORT_THRESH.keys()), key=f"{key_prefix}_eff")
    return skill, level, eff

with cols[0]:
    designer_skill, designer_level, designer_eff = role_selects("DESIGNER", "designer")
with cols[1]:
    writer_skill, writer_level, writer_eff = role_selects("WRITER", "writer")
with cols[2]:
    editor_skill, editor_level, editor_eff = role_selects("EDITOR", "editor")

min_date, max_date = min(dates), max(dates)
def_start = min_date
try:
    def_end = dates[min(len(dates)-1, 9)]
except Exception:
    def_end = max_date

date_range = st.date_input("Project Date Range", value=(def_start.date(), def_end.date()), min_value=min_date.date(), max_value=max_date.date(), key="date_range")
if isinstance(date_range, tuple):
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
else:
    start_date = end_date = pd.to_datetime(date_range)

run = st.button("Run Selection", type="primary", key="run_btn")

# ---------------- Run & Results ----------------
if run:
    picked_dates = restrict_weekdays(dates, start_date, end_date)
    if not picked_dates:
        st.warning("No weekday dates in that range match your availability columns.")
        st.stop()

    col_by_date = {d: c for c, d in zip(date_cols, dates)}
    picked_cols = [col_by_date[d] for d in picked_dates]

    thresholds = {"DESIGNER": EFFORT_THRESH[designer_eff], "WRITER": EFFORT_THRESH[writer_eff], "EDITOR": EFFORT_THRESH[editor_eff]}
    desired = {"DESIGNER": designer_skill, "WRITER": writer_skill, "EDITOR": editor_skill}
    preferred_levels = {"DESIGNER": designer_level, "WRITER": writer_level, "EDITOR": editor_level}

    chosen, used = [], set()
    for role in ROLES:
        if not desired.get(role):
            chosen.append({"role": role, "chosen": None, "reason": "No skill selected."})
            continue
        res = pick_for_role_pref(skills_df, avail_df, role, desired[role], preferred_levels[role], thresholds[role], picked_cols, picked_dates, used)
        chosen.append(res)
        if res.get("chosen") and res["chosen"].get("name"):
            used.add(res["chosen"]["name"])

    st.subheader("Results")
    rows = []
    grid = st.columns(3)
    for i, res in enumerate(chosen):
        role = res["role"]
        ch = res.get("chosen")
        with grid[i % 3]:
            st.markdown(f"### {role.title()}")
            if not ch:
                st.info(res.get("reason", "No candidate"))
                rows.append({"role": role, "employee": None, "desired_skill": desired.get(role), "preferred_level": preferred_levels.get(role), "actual_level": None, "avg_weekday_hours": None, "effort_threshold": thresholds.get(role), "met_threshold": None, "note": res.get("reason", "")})
            else:
                status = "✅ meets" if ch["met_threshold"] else "➕ closest"
                st.metric(label=f"{ch['name']} — {ch.get('skill_level') or 'N/A'}", value=f"{ch['avg_weekday_hours']:.2f} h/day", delta=f"target ≥ {thresholds[role]:.1f} ({status})")
                st.caption(f"Skill: {ch['desired_skill']} • Preferred level: {preferred_levels[role]} • Actual level: {ch.get('skill_level') or 'N/A'}")
                rows.append({"role": role, "employee": ch["name"], "desired_skill": ch["desired_skill"], "preferred_level": preferred_levels[role], "actual_level": ch.get("skill_level"), "avg_weekday_hours": round(float(ch["avg_weekday_hours"]), 3), "effort_threshold": float(thresholds[role]), "met_threshold": bool(ch["met_threshold"]), "note": res.get("reason", "")})

    result_df = pd.DataFrame(rows)
    st.dataframe(result_df, use_container_width=True)

    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download selection_summary.csv", data=csv, file_name="selection_summary.csv", mime="text/csv", key="dl_csv")

st.divider()
st.caption("Tip: change the selections and click ‘Run Selection’ again to explore alternatives. Weekends are automatically excluded.")