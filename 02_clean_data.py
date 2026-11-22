"""
02_clean_data.py
Cleans the four raw tables, writes a RECORD-LEVEL cleaning log (every changed record/field),
and produces reconciliation totals (all money in integer paise so reconciliation is exact).
"""
import pandas as pd, numpy as np, re, os, json

RAW, OUT = "/home/claude/project/raw", "/home/claude/project/clean"
os.makedirs(OUT, exist_ok=True)
LOG = []


def log(rule, table, ids, field, old, new, action):
    LOG.append(pd.DataFrame({"rule": rule, "table": table, "record_id": ids.values if hasattr(ids, "values") else ids,
                             "field": field, "old_value": old.values if hasattr(old, "values") else old,
                             "new_value": new.values if hasattr(new, "values") else new, "action": action}))


R = {}  # reconciliation / counts
rd = lambda f: pd.read_csv(f"{RAW}/{f}.csv", dtype=str, keep_default_na=False)
adm, bil, opd, lab = rd("admissions"), rd("billing"), rd("opd_visits"), rd("lab_tests")
R["raw_rows"] = {"admissions": len(adm), "billing": len(bil), "opd_visits": len(opd), "lab_tests": len(lab)}
R["raw_rows"]["total"] = sum(R["raw_rows"].values())

# ---------------------------------------------------------------- R01 exact duplicates
ID = {"admissions": "admission_id", "billing": "bill_id", "opd_visits": "opd_id", "lab_tests": "lab_id"}
T = {"admissions": adm, "billing": bil, "opd_visits": opd, "lab_tests": lab}
dup_removed = {}
for k, df in T.items():
    d = df.duplicated(keep="first")
    log("R01", k, df.loc[d, ID[k]], "(whole row)", "duplicate of an earlier identical row", "", "row removed")
    dup_removed[k] = int(d.sum())
    T[k] = df[~d].reset_index(drop=True)
    assert T[k][ID[k]].is_unique, k
adm, bil, opd, lab = T["admissions"], T["billing"], T["opd_visits"], T["lab_tests"]
R["duplicates_removed"] = dup_removed

# ---------------------------------------------------------------- helpers
HMAP = {"kothrud": "Kothrud", "hadapsar": "Hadapsar", "wakad": "Wakad", "pimpri": "Pimpri"}


def std_hosp(s):
    return s.str.strip().str.lower().str.replace("hosp-", "", regex=False).map(HMAP)


DEPT_STD = {"general medicine": "General Medicine", "gen med": "General Medicine", "gen. medicine": "General Medicine",
            "cardiology": "Cardiology", "cardio": "Cardiology", "orthopaedics": "Orthopaedics", "ortho": "Orthopaedics",
            "orthopedics": "Orthopaedics", "general surgery": "General Surgery", "gen surgery": "General Surgery", "surgery": "General Surgery",
            "obstetrics & gynaecology": "Obstetrics & Gynaecology", "obg": "Obstetrics & Gynaecology", "ob-gyn": "Obstetrics & Gynaecology",
            "obs & gynae": "Obstetrics & Gynaecology", "paediatrics": "Paediatrics", "paeds": "Paediatrics", "pediatrics": "Paediatrics",
            "pulmonology": "Pulmonology", "pulmo": "Pulmonology", "chest medicine": "Pulmonology", "nephrology": "Nephrology", "nephro": "Nephrology",
            "neurology": "Neurology", "neuro": "Neurology", "gastroenterology": "Gastroenterology", "gastro": "Gastroenterology", "gi medicine": "Gastroenterology"}


def std_dept(s):
    return s.str.strip().str.lower().map(DEPT_STD)


def apply_std(df, tbl, col, fn, rule, label):
    new = fn(df[col])
    assert new.notna().all(), (tbl, col, df.loc[new.isna(), col].unique()[:5])
    ch = df[col] != new
    log(rule, tbl, df.loc[ch, ID[tbl]], col, df.loc[ch, col], new[ch], label)
    df[col] = new
    return int(ch.sum())


def parse_dates(s, with_time=False):
    """Parse yyyy-mm-dd / dd/mm/yyyy / dd-Mon-yyyy (optionally with HH:MM). Blank -> NaT."""
    t = "" if not with_time else " %H:%M"
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    iso = s.str.match(r"^\d{4}-\d{2}-\d{2}")
    sl = s.str.contains("/", regex=False)
    mo = s.str.match(r"^\d{2}-[A-Za-z]{3}-\d{4}")
    out[iso] = pd.to_datetime(s[iso], format="%Y-%m-%d" + t)
    out[sl] = pd.to_datetime(s[sl], format="%d/%m/%Y" + t)
    out[mo] = pd.to_datetime(s[mo], format="%d-%b-%Y" + t)
    nonblank = s.str.strip() != ""
    assert out[nonblank].notna().all()
    return out, (~iso & nonblank)


def std_date_col(df, tbl, col, rule="R04", with_time=False):
    d, nonstd = parse_dates(df[col], with_time)
    new = d.dt.strftime("%Y-%m-%d %H:%M" if with_time else "%Y-%m-%d")
    log(rule, tbl, df.loc[nonstd, ID[tbl]], col, df.loc[nonstd, col], new[nonstd], "date format standardised to YYYY-MM-DD")
    df[col] = d
    return int(nonstd.sum())


# ---------------------------------------------------------------- ADMISSIONS
Rc = R["fixes"] = {}
Rc["adm_hospital_name"] = apply_std(adm, "admissions", "hospital", std_hosp, "R02", "hospital name standardised")
Rc["adm_department_name"] = apply_std(adm, "admissions", "department", std_dept, "R03", "department name standardised")
Rc["adm_admission_date_format"] = std_date_col(adm, "admissions", "admission_date")
Rc["adm_discharge_date_format"] = std_date_col(adm, "admissions", "discharge_date")

# patient id
new = adm.patient_id.str.upper().str.replace(r"^P-?", "P", regex=True)
ch = adm.patient_id != new
log("R05", "admissions", adm.loc[ch, "admission_id"], "patient_id", adm.loc[ch, "patient_id"], new[ch], "patient_id format standardised (P + 6 digits)")
adm["patient_id"] = new
Rc["adm_patient_id_format"] = int(ch.sum())
assert adm.patient_id.str.match(r"^P\d{6}$").all()

# age: text -> number (R06); impossible -> recompute from patient's other admissions (R07)
num = pd.to_numeric(adm.age.str.extract(r"(-?\d+)")[0], errors="coerce")
txt = adm.age.str.contains(r"[A-Za-z]", regex=True)
adm["admission_year"] = adm.admission_date.dt.year
bad = (num < 0) | (num > 110)
log("R06", "admissions", adm.loc[txt & ~bad, "admission_id"], "age", adm.loc[txt & ~bad, "age"], num[txt & ~bad].astype(int).astype(str), "age text ('45 yrs', '45Y') converted to number")
Rc["adm_age_text"] = int((txt & ~bad).sum())
by = (adm.admission_year - num).where(~bad)
pat_by = by.groupby(adm.patient_id).median()
fix = adm.patient_id.map(pat_by).rsub(adm.admission_year).round()
new_age = num.copy()
new_age[bad] = fix[bad]
log("R07", "admissions", adm.loc[bad, "admission_id"], "age", adm.loc[bad, "age"], new_age[bad].astype("Int64").astype(str).replace("<NA>", "NULL"),
    "impossible age (<0 or >110) replaced using the patient's age on other admissions; NULL if none")
Rc["adm_age_impossible"] = int(bad.sum())
Rc["adm_age_impossible_unrecoverable"] = int(new_age[bad].isna().sum())
adm["age"] = new_age

# doctor id
m = adm.doctor_id.str.strip() == ""
log("R08", "admissions", adm.loc[m, "admission_id"], "doctor_id", "(blank)", "UNASSIGNED", "missing doctor_id labelled UNASSIGNED (kept in all KPIs, excluded from doctor rankings)")
adm.loc[m, "doctor_id"] = "UNASSIGNED"
Rc["adm_doctor_missing"] = int(m.sum())

# ---------------------------------------------------------------- BILLING
Rc["bill_hospital_name"] = apply_std(bil, "billing", "hospital", std_hosp, "R02", "hospital name standardised")
Rc["bill_date_format"] = std_date_col(bil, "billing", "bill_date")
tot_raw = bil.total_amount
is_txt = tot_raw.str.contains("Rs", regex=False)
parsed = pd.to_numeric(tot_raw.str.replace(r"[Rs\.\s,]", "", regex=True).str[:-2].str.cat(tot_raw.str[-2:], sep=".") if False else
                       tot_raw.str.replace("Rs.", "", regex=False).str.replace(",", "", regex=False).str.strip(), errors="raise")
log("R11", "billing", bil.loc[is_txt, "bill_id"], "total_amount", tot_raw[is_txt], parsed[is_txt].map("{:.2f}".format), "amount text ('Rs. 1,23,456.00') converted to number")
Rc["bill_amount_text"] = int(is_txt.sum())
P = lambda s: (pd.to_numeric(s).mul(100).round().astype("int64"))
bil["subtotal_p"], bil["gst_raw_p"], bil["total_raw_p"] = P(bil.subtotal), P(bil.gst_amount), (parsed * 100).round().astype("int64")
bil["gst_rate"] = pd.to_numeric(bil.gst_rate)
bil["gst_formula_p"] = (bil.subtotal_p * bil.gst_rate).round().astype("int64")
assert ((bil.subtotal_p + bil.gst_raw_p) == bil.total_raw_p).all(), "total != subtotal+gst in raw"
# double count = GST is ~2x the formula amount (tolerance 2 paise for rounding); <=1 paise differences are rounding, not errors
dbl = ((bil.gst_raw_p - 2 * bil.gst_formula_p).abs() <= 2) & ((bil.gst_raw_p - bil.gst_formula_p) > 2)
bil["gst_correct_p"] = np.where(dbl, bil.gst_formula_p, bil.gst_raw_p)
assert ((bil.gst_raw_p - bil.gst_formula_p).abs()[~dbl] <= 1).all(), "unexplained GST mismatch"
log("R12", "billing", bil.loc[dbl, "bill_id"], "gst_amount", (bil.loc[dbl, "gst_raw_p"] / 100).map("{:.2f}".format), (bil.loc[dbl, "gst_correct_p"] / 100).map("{:.2f}".format),
    "GST counted twice (gst_amount = 2 x rate x subtotal); reset to rate x subtotal")
log("R12", "billing", bil.loc[dbl, "bill_id"], "total_amount", (bil.loc[dbl, "total_raw_p"] / 100).map("{:.2f}".format),
    ((bil.loc[dbl, "subtotal_p"] + bil.loc[dbl, "gst_correct_p"]) / 100).map("{:.2f}".format), "total recomputed = subtotal + correct GST")
bil["total_clean_p"] = bil.subtotal_p + bil.gst_correct_p
Rc["bill_gst_double_counted"] = int(dbl.sum())
R["billing_reconciliation_paise"] = {
    "raw_total_all_rows_before_dedup": None,  # filled below
}

# ---------------------------------------------------------------- OPD
Rc["opd_hospital_name"] = apply_std(opd, "opd_visits", "hospital", std_hosp, "R02", "hospital name standardised")
Rc["opd_department_name"] = apply_std(opd, "opd_visits", "department", std_dept, "R03", "department name standardised")
Rc["opd_date_format"] = std_date_col(opd, "opd_visits", "visit_date")

# ---------------------------------------------------------------- LAB
Rc["lab_hospital_name"] = apply_std(lab, "lab_tests", "hospital", std_hosp, "R02", "hospital name standardised")
Rc["lab_sample_date_format"] = std_date_col(lab, "lab_tests", "sample_time", with_time=True)
Rc["lab_report_date_format"] = std_date_col(lab, "lab_tests", "report_time", with_time=True)
sw = lab.report_time < lab.sample_time
log("R13", "lab_tests", lab.loc[sw, "lab_id"], "sample_time/report_time", lab.loc[sw, "sample_time"].dt.strftime("%Y-%m-%d %H:%M") + " | " + lab.loc[sw, "report_time"].dt.strftime("%Y-%m-%d %H:%M"),
    lab.loc[sw, "report_time"].dt.strftime("%Y-%m-%d %H:%M") + " | " + lab.loc[sw, "sample_time"].dt.strftime("%Y-%m-%d %H:%M"),
    "report time before sample time -> timestamps swapped back")
a, b = lab.loc[sw, "sample_time"].copy(), lab.loc[sw, "report_time"].copy()
lab.loc[sw, "sample_time"], lab.loc[sw, "report_time"] = b, a
Rc["lab_timestamps_swapped"] = int(sw.sum())
lab["tat_hours"] = (lab.report_time - lab.sample_time).dt.total_seconds() / 3600
assert (lab.tat_hours >= 0).all()
TESTSTD = {"cbc": "CBC", "complete blood count": "CBC", "c.b.c": "CBC", "hba1c": "HbA1c", "glycosylated hb": "HbA1c", "lft": "LFT", "liver function test": "LFT",
           "kft": "KFT", "kidney function test": "KFT", "rft": "KFT", "lipid profile": "Lipid Profile", "lipid panel": "Lipid Profile",
           "dengue ns1": "Dengue NS1", "ns1 antigen": "Dengue NS1", "troponin i": "Troponin I", "trop-i": "Troponin I", "troponin": "Troponin I",
           "crp": "CRP", "c-reactive protein": "CRP", "urine r/m": "Urine R/M", "urine routine": "Urine R/M", "blood culture": "Blood Culture", "blood cx": "Blood Culture",
           "d-dimer": "D-Dimer", "d dimer": "D-Dimer", "ddimer": "D-Dimer", "thyroid profile": "Thyroid Profile", "tft": "Thyroid Profile", "t3 t4 tsh": "Thyroid Profile"}
Rc["lab_test_name"] = apply_std(lab, "lab_tests", "test_name", lambda s: s.str.strip().str.lower().map(TESTSTD), "R14", "test name standardised")

# ---------------------------------------------------------------- diagnosis free text -> group (R15, derived column)
DIAG_RULES = [  # order matters (first match wins)
    ("Spine / disc", r"disc|pivd|lumbar"), ("Pancreatitis", r"pancreat"), ("Acute kidney injury", r"\baki\b|acute kidney|renal failure"),
    ("Dengue", r"den[gn]|ns1"), ("Viral fever", r"viral|puo|fever with chills"),
    ("Diabetes (uncontrolled)", r"\bt2dm\b|\bdm2?\b|diabet|dka"), ("UTI / urosepsis", r"\buti\b|urosepsis|urinary tract|pyelo"),
    ("Acute MI", r"\bami\b|myocardial|stemi|\bmi\b|\bpci\b"), ("Heart failure", r"chf|heart failure|\bhf\b|congestive"),
    ("Unstable angina", r"angina|\bua\b|chest pain"), ("Hip fracture", r"neck of femur|\bnof\b|hip fracture|fx femur|femur neck"),
    ("Knee replacement", r"tkr|knee"), ("Appendicitis", r"append"), ("Gallstones", r"cholecyst|\bgall|cholelith"), ("Hernia", r"hernia"),
    ("Normal delivery", r"nvd|vaginal delivery"), ("Caesarean section", r"lscs|caesar|c-section"), ("Hysterectomy", r"hysterect|\btah\b|fibroid"),
    ("Bronchiolitis", r"bronchiol|wheezy"), ("Paediatric gastroenteritis", r"\bage\b|gastroenteritis|loose motion"),
    ("Neonatal jaundice", r"nnj|neonatal|newborn|jaundice"), ("COPD", r"copd|chronic obstructive"), ("Pneumonia", r"pneumon"), ("Asthma", r"asthma"),
    ("Kidney stones", r"calcul|stone|lithiasis|pcnl"), ("CKD (dialysis)", r"\bckd\b|esrd|chronic kidney|dialysis"),
    ("Stroke", r"stroke|\bcva\b|infarct"), ("Seizure / epilepsy", r"seizure|epilep|gtcs"), ("Neuropathy", r"\bgbs\b|guillain|neuropath"),
    ("GI bleed", r"gi bleed|ugib|hematemesis|melena"), ("Cirrhosis", r"cirrho|\bcld\b|chronic liver"),
]


def map_dx(t):
    t = t.lower()
    for g, rx in DIAG_RULES:
        if re.search(rx, t):
            return g
    return "Unclassified"


uniq = pd.Series(adm.diagnosis_text.unique())
mp = dict(zip(uniq, uniq.map(map_dx)))
adm["diagnosis_group"] = adm.diagnosis_text.map(mp)
truth = pd.read_csv("/home/claude/project/_truth_diag.csv").set_index("admission_id").true_group
acc = (adm.diagnosis_group == adm.admission_id.map(truth)).mean()
R["diagnosis_mapping"] = {"distinct_free_text_strings": int(len(uniq)), "records_mapped": int((adm.diagnosis_group != "Unclassified").sum()),
                          "records_unclassified": int((adm.diagnosis_group == "Unclassified").sum()), "groups": int(adm.diagnosis_group.nunique()),
                          "accuracy_vs_hidden_answer_key": round(float(acc), 5)}

# ---------------------------------------------------------------- discharge date: invalid & missing (R09, R10)
inv = adm.discharge_date.notna() & (adm.discharge_date < adm.admission_date)
log("R09", "admissions", adm.loc[inv, "admission_id"], "discharge_date", adm.loc[inv, "discharge_date"].dt.strftime("%Y-%m-%d"), "(blank)",
    "discharge date earlier than admission date (year typo) -> treated as missing")
adm.loc[inv, "discharge_date"] = pd.NaT
Rc["adm_discharge_before_admission"] = int(inv.sum())
# estimate relationship between discharge and bill date on clean pairs
ipd = bil[bil.encounter_type == "IPD"].drop_duplicates("encounter_id").set_index("encounter_id")
adm["bill_date"] = adm.admission_id.map(ipd.bill_date)
pairs = adm.dropna(subset=["discharge_date", "bill_date"])
lag = (pairs.bill_date - pairs.discharge_date).dt.days
R["bill_lag_days_distribution"] = lag.value_counts(normalize=True).round(4).sort_index().to_dict()
med_lag = int(lag.median())
need = adm.discharge_date.isna()
can = need & adm.bill_date.notna() & (adm.bill_date - pd.Timedelta(days=med_lag) > adm.admission_date)
log("R10", "admissions", adm.loc[can, "admission_id"], "discharge_date", "(blank)", (adm.loc[can, "bill_date"] - pd.Timedelta(days=med_lag)).dt.strftime("%Y-%m-%d"),
    f"missing discharge date imputed = billing date - {med_lag} day(s) (median billing lag on complete records)")
adm.loc[can, "discharge_date"] = adm.loc[can, "bill_date"] - pd.Timedelta(days=med_lag)
adm["discharge_imputed"] = can
still = adm.discharge_date.isna()
log("R10", "admissions", adm.loc[still, "admission_id"], "discharge_date", "(blank)", "(blank)",
    "discharge date missing and no usable billing date -> stays blank; record excluded from LOS and occupancy, kept in admission counts")
Rc["adm_discharge_missing_total"] = int(need.sum())
Rc["adm_discharge_imputed_from_billing"] = int(can.sum())
Rc["adm_discharge_unrecoverable_excluded_from_los"] = int(still.sum())
adm["los_valid"] = adm.discharge_date.notna()
adm["los_days"] = (adm.discharge_date - adm.admission_date).dt.days.where(adm.los_valid)
assert (adm.los_days.dropna() >= 1).all()

# ---------------------------------------------------------------- R16 overlapping stays for the same patient (flag only, nothing is changed)
_a = adm[adm.los_valid][["admission_id", "patient_id", "admission_date", "discharge_date"]]
_j = _a.merge(adm[["patient_id", "admission_date", "admission_id"]], on="patient_id", suffixes=("", "_n"))
_j = _j[(_j.admission_id != _j.admission_id_n) & (_j.admission_date_n >= _j.admission_date) & (_j.admission_date_n < _j.discharge_date)]
ov_ids = pd.unique(pd.concat([_j.admission_id, _j.admission_id_n]))
adm["overlap_flag"] = adm.admission_id.isin(ov_ids)
log("R16", "admissions", adm.loc[adm.overlap_flag, "admission_id"], "admission_date/discharge_date", "(unchanged)", "(unchanged)",
    "same patient has another admission that starts before this stay ends (impossible at one site) -> flagged only; kept in all KPIs")
Rc["adm_overlapping_stays_flagged"] = int(adm.overlap_flag.sum())

# ---------------------------------------------------------------- reconciliation (money, exact in paise)
raw_b = pd.read_csv(f"{RAW}/billing.csv", dtype=str, keep_default_na=False)
rp = lambda s: pd.to_numeric(s.str.replace("Rs.", "", regex=False).str.replace(",", "", regex=False).str.strip()).mul(100).round().astype("int64")
raw_all = int(rp(raw_b.total_amount).sum())
dup_rows = raw_b[raw_b.duplicated(keep="first")]
dup_amt = int(rp(dup_rows.total_amount).sum())
gst_excess = int((bil.total_raw_p - bil.total_clean_p).sum())
clean_total = int(bil.total_clean_p.sum())
R["billing_reconciliation_paise"] = {"raw_total_all_rows": raw_all, "less_duplicate_rows": dup_amt, "raw_total_after_dedup": raw_all - dup_amt,
                                     "less_gst_double_count_excess": gst_excess, "cleaned_total": clean_total,
                                     "difference": raw_all - dup_amt - gst_excess - clean_total}
assert R["billing_reconciliation_paise"]["difference"] == 0
R["clean_rows"] = {"admissions": len(adm), "billing": len(bil), "opd_visits": len(opd), "lab_tests": len(lab)}
assert R["raw_rows"]["total"] - sum(dup_removed.values()) == sum(R["clean_rows"].values())

# ---------------------------------------------------------------- save
LOGDF = pd.concat(LOG, ignore_index=True)
LOGDF.insert(0, "log_id", np.arange(1, len(LOGDF) + 1))
LOGDF.to_csv("/home/claude/project/cleaning_log.csv", index=False)
R["log_rows"] = int(len(LOGDF))
R["log_rows_by_rule"] = LOGDF.groupby("rule").size().to_dict()
R["log_records_changed"] = int(LOGDF.drop_duplicates(["table", "record_id"]).shape[0])
adm.to_pickle(f"{OUT}/admissions.pkl"); bil.to_pickle(f"{OUT}/billing.pkl")
opd.to_pickle(f"{OUT}/opd.pkl"); lab.to_pickle(f"{OUT}/lab.pkl")
json.dump(R, open("/home/claude/project/cleaning_summary.json", "w"), indent=2, default=int)
print(json.dumps(R, indent=1, default=int))
