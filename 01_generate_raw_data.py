"""
01_generate_raw_data.py
Builds a FICTIONAL, anonymised raw extract for a 4-hospital diagnostic/hospital chain in Pune.
Exactly 500,000 raw rows across OPD visits, admissions, lab tests and billing, with the
real-world data problems named in the brief deliberately planted (see PLANTED dict at the end).
"""
import numpy as np, pandas as pd, os, json

rng = np.random.default_rng(2026)
RAW = "/home/claude/project/raw"
os.makedirs(RAW, exist_ok=True)
START, END = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")
TOTAL_ROWS = 500_000

HOSP = ["Kothrud", "Hadapsar", "Wakad", "Pimpri"]
BEDS = {"Kothrud": 120, "Hadapsar": 150, "Wakad": 90, "Pimpri": 110}
OCC = {"Kothrud": 0.76, "Hadapsar": 0.82, "Wakad": 0.62, "Pimpri": 0.74}
H_LOS = {("Kothrud", 2024): 1.00, ("Kothrud", 2025): 0.98, ("Hadapsar", 2024): 1.08, ("Hadapsar", 2025): 1.14,
         ("Wakad", 2024): 0.96, ("Wakad", 2025): 0.95, ("Pimpri", 2024): 1.05, ("Pimpri", 2025): 0.97}
H_RD = {"Kothrud": 0.90, "Hadapsar": 1.30, "Wakad": 0.85, "Pimpri": 1.00}
SEASON = [.98, .95, .97, .96, .94, .98, 1.08, 1.12, 1.10, 1.02, .98, 1.00]

DEPTS = ["General Medicine", "Cardiology", "Orthopaedics", "General Surgery", "Obstetrics & Gynaecology",
         "Paediatrics", "Pulmonology", "Nephrology", "Neurology", "Gastroenterology"]
D_LOS = [3.8, 4.6, 5.4, 4.9, 3.1, 3.4, 5.0, 5.6, 5.8, 4.4]
D_RD = [.09, .13, .05, .05, .03, .06, .14, .16, .10, .09]
D_EMG = [.60, .65, .50, .45, .55, .75, .80, .55, .80, .70]
D_PKG = [18000, 85000, 120000, 90000, 55000, 25000, 40000, 60000, 70000, 45000]
ROOM = {"Kothrud": 4800, "Hadapsar": 4200, "Wakad": 3800, "Pimpri": 4000}

# diagnosis groups: (dept index, group, los mult, readmit mult, free-text variants)
DIAG = [
 (0, "Dengue", 1.00, .6, ["Dengue fever", "denge fever w/ thrombocytopenia", "Dengue NS1 +ve", "dengue fvr, plt low"]),
 (0, "Viral fever", .70, .6, ["Viral fever", "viral fvr x 5 days", "PUO - viral", "fever with chills ? viral"]),
 (0, "Diabetes (uncontrolled)", 1.15, 1.6, ["Type 2 DM uncontrolled", "T2DM w/ DKA", "uncontrolled diabetes mellitus", "DM2, high sugars, foot ulcer"]),
 (0, "UTI / urosepsis", 1.05, 1.1, ["UTI", "Urosepsis", "urinary tract infection w/ fever", "uti - pyelonephritis"]),
 (1, "Acute MI", 1.15, 1.2, ["Acute MI", "acute myocardial infarction (STEMI)", "AMI - inf wall", "NSTEMI, PCI done"]),
 (1, "Heart failure", 1.20, 1.7, ["CHF", "Congestive heart failure", "heart failure, EF 30%", "decompensated HF"]),
 (1, "Unstable angina", .70, .7, ["Unstable angina", "chest pain ? UA", "angina pectoris unstable", "UA - cath advised"]),
 (2, "Hip fracture", 1.25, 1.0, ["# neck of femur", "fracture NOF - hemiarthroplasty", "hip fracture", "fx femur neck"]),
 (2, "Knee replacement", .85, .5, ["TKR - OA knee", "total knee replacement", "knee osteoarthritis, TKR rt", "OA both knees - TKR"]),
 (2, "Spine / disc", .80, .8, ["LS disc prolapse", "lumbar disc herniation", "PIVD L4-L5", "slip disc - discectomy"]),
 (3, "Appendicitis", .70, .6, ["Acute appendicitis", "appendicectomy", "appendicitis - lap appendectomy", "acute appendix"]),
 (3, "Gallstones", .85, .7, ["Cholelithiasis", "lap cholecystectomy", "gall stones w/ cholecystitis", "calculous cholecystitis"]),
 (3, "Hernia", .60, .5, ["Inguinal hernia repair", "hernia - mesh repair", "rt inguinal hernia", "incisional hernia"]),
 (4, "Normal delivery", .60, .5, ["Normal vaginal delivery", "NVD", "FTNVD, live baby", "vaginal delivery at term"]),
 (4, "Caesarean section", 1.15, .8, ["LSCS", "caesarean section", "C-section (previous LSCS)", "emergency LSCS - fetal distress"]),
 (4, "Hysterectomy", 1.00, .8, ["Hysterectomy - fibroid uterus", "TAH for fibroids", "abdominal hysterectomy", "fibroid uterus, menorrhagia"]),
 (5, "Bronchiolitis", 1.00, 1.2, ["Bronchiolitis", "acute bronchiolitis RSV", "bronchiolitis infant", "wheezy chest - infant"]),
 (5, "Paediatric gastroenteritis", .85, .8, ["AGE with dehydration", "acute gastroenteritis child", "loose motions & vomiting", "gastroenteritis, mod dehydration"]),
 (5, "Neonatal jaundice", .90, .7, ["Neonatal jaundice", "NNJ - phototherapy", "neonatal hyperbilirubinemia", "jaundice newborn"]),
 (6, "COPD", 1.20, 1.5, ["COPD exacerbation", "AECOPD", "COPD w/ resp failure", "chronic obstructive pulm disease"]),
 (6, "Pneumonia", 1.00, 1.0, ["Pneumonia", "community acq pneumonia (CAP)", "lobar pneumonia rt", "pneumonitis ? bacterial"]),
 (6, "Asthma", .75, 1.0, ["Acute severe asthma", "bronchial asthma exacerbation", "asthma attack", "status asthmaticus"]),
 (7, "CKD (dialysis)", 1.15, 1.5, ["CKD stage 5 on HD", "chronic kidney disease - dialysis", "ESRD for dialysis", "CKD 5, fluid overload"]),
 (7, "Acute kidney injury", 1.00, 1.0, ["AKI", "acute kidney injury", "AKI on CKD", "acute renal failure"]),
 (7, "Kidney stones", .60, .6, ["Renal calculus - PCNL", "kidney stone, ureteric colic", "nephrolithiasis", "urolithiasis rt"]),
 (8, "Stroke", 1.25, 1.1, ["Ischaemic stroke", "CVA - left MCA infarct", "stroke (cerebral infarct)", "acute stroke, hemiparesis"]),
 (8, "Seizure / epilepsy", .80, 1.1, ["Seizure disorder", "epilepsy - breakthrough seizures", "first episode seizure", "GTCS, ? epilepsy"]),
 (8, "Neuropathy", 1.00, .9, ["GBS", "Guillain-Barre syndrome", "peripheral neuropathy", "polyneuropathy - IVIG"]),
 (9, "GI bleed", 1.15, 1.2, ["Upper GI bleed", "UGIB - hematemesis", "melena ? peptic ulcer", "GI bleeding, Hb low"]),
 (9, "Pancreatitis", 1.10, 1.1, ["Acute pancreatitis", "pancreatitis - alcohol related", "acute pancreatitis, gallstone", "necrotising pancreatitis"]),
 (9, "Cirrhosis", 1.10, 1.7, ["Cirrhosis w/ ascites", "decompensated chronic liver disease", "CLD with ascites", "liver cirrhosis, hepatic encephalopathy"]),
]
DG = pd.DataFrame(DIAG, columns=["dept_i", "group", "los_m", "rd_m", "variants"])

# ------------------------------------------------------------------ doctor master
FN = ["Anil", "Sunita", "Rahul", "Priya", "Vikram", "Neha", "Amit", "Kavita", "Sandeep", "Meera", "Rohan", "Anjali",
      "Suresh", "Pooja", "Manoj", "Divya", "Nitin", "Swati", "Ajay", "Shweta", "Mahesh", "Rekha", "Vivek", "Asha"]
LN = ["Kulkarni", "Deshmukh", "Patil", "Joshi", "Shah", "Iyer", "Gupta", "Bhosale", "Jadhav", "Pawar", "Mehta",
      "Naik", "Kale", "Chavan", "Rao", "Sharma", "More", "Gokhale", "Apte", "Thakur", "Menon", "Sawant", "Gaikwad", "Shinde"]
doc_rows = []
did = 1
for h in HOSP:
    for di, d in enumerate(DEPTS):
        for k in range(3):
            doc_rows.append(dict(doctor_id=f"DR{did:03d}", doctor_name=f"Dr. {FN[(did*5)%24]} {LN[(did*7 + did//24)%24]}",
                                 hospital=h, department=d, dept_i=di, years_experience=int(rng.integers(3, 31))))
            did += 1
DOC = pd.DataFrame(doc_rows)
z1, z2 = rng.normal(size=len(DOC)), rng.normal(size=len(DOC))
DOC["eff_los"] = 0.14 * z1                                   # practice-pattern effect on LOS (log scale)
DOC["eff_rd"] = 0.30 * (0.45 * z1 + np.sqrt(1 - .45 ** 2) * z2)   # correlated effect on readmission
DOC["w"] = rng.lognormal(0, .3, len(DOC))
DOC["hd"] = DOC["hospital"].map({h: i for i, h in enumerate(HOSP)}) * 10 + DOC["dept_i"]
HD_DOCS = {hd: g.index.values for hd, g in DOC.groupby("hd")}
DOC[["doctor_id", "doctor_name", "hospital", "department", "years_experience"]].to_csv(f"{RAW}/doctors.csv", index=False)
pd.DataFrame({"hospital": HOSP, "licensed_beds": [BEDS[h] for h in HOSP]}).to_csv(f"{RAW}/hospitals.csv", index=False)

# ------------------------------------------------------------------ patient pool
NP = 250_000
age_grp = rng.choice(3, NP, p=[.12, .55, .33])
age2025 = np.where(age_grp == 0, rng.integers(0, 15, NP), np.where(age_grp == 1, rng.integers(15, 60, NP), rng.integers(60, 92, NP)))
PAT = pd.DataFrame({"birth_year": 2025 - age2025, "sex": rng.choice(["M", "F"], NP),
                    "payer": rng.choice(["Self-pay", "Insurance", "Corporate", "Govt scheme"], NP, p=[.38, .34, .10, .18])})
PAT["chronic"] = np.minimum(3, rng.poisson(np.clip(age2025 / 45, 0, 2), NP))
pw = rng.gamma(2.0, 1, NP) * (1 + .3 * PAT["chronic"]) * np.where(age2025 >= 65, 1.3, 1.0)
PW = pw / pw.sum()

DEPT_W = {  # age/sex category -> dept weights (order of DEPTS)
    "child": [.10, 0, .02, .04, 0, .80, .04, 0, 0, 0],
    "f_fertile": [.18, .02, .06, .10, .38, 0, .04, .03, .04, .08],
    "m_adult": [.25, .14, .15, .16, 0, 0, .08, .06, .08, .08],
    "f_other": [.25, .12, .13, .12, .06, 0, .08, .06, .08, .10],
    "elder": [.27, .18, .11, .08, .00, 0, .12, .07, .10, .07],
}
DEPT_W = {k: np.array(v) / sum(v) for k, v in DEPT_W.items()}
PAYER_LOS = {"Self-pay": 1.0, "Insurance": 1.10, "Corporate": 1.04, "Govt scheme": 1.15}


def dept_for(pidx, admit):
    age = admit.dt.year.values - PAT.birth_year.values[pidx]
    sex = PAT.sex.values[pidx]
    cat = np.where(age < 15, "child", np.where(age >= 60, "elder",
          np.where(sex == "M", "m_adult", np.where(age <= 44, "f_fertile", "f_other"))))
    out = np.zeros(len(pidx), int)
    for c, w in DEPT_W.items():
        m = cat == c
        out[m] = rng.choice(10, m.sum(), p=w)
    return out, age


def diag_for(dept_i, months):
    out = np.zeros(len(dept_i), int)
    for d in range(10):
        sub = DG[DG.dept_i == d]
        idx = sub.index.values
        for season in (True, False):
            m = (dept_i == d) & (np.isin(months, [7, 8, 9, 10]) == season)
            if not m.any():
                continue
            w = np.ones(len(idx))
            if d == 0:
                w = np.array([3.0 if season else .4, 1, 1, 1])
            out[m] = rng.choice(idx, m.sum(), p=w / w.sum())
    return out


def make_adm(hosp_i, admit, pidx, force_dept=None, rd_scale=1.0):
    n = len(admit)
    dept_i, age = dept_for(pidx, admit)
    if force_dept is not None:
        dept_i = force_dept
    dgi = diag_for(dept_i, admit.dt.month.values)
    hd = hosp_i * 10 + dept_i
    W = np.stack([np.pad(DOC.w.values[HD_DOCS[k]], (0, 0)) for k in sorted(HD_DOCS)])
    W = W / W.sum(1, keepdims=True)
    hd_pos = {k: i for i, k in enumerate(sorted(HD_DOCS))}
    pos = np.array([hd_pos[k] for k in hd])
    pick = (rng.random(n)[:, None] > np.cumsum(W[pos], axis=1)).sum(1).clip(0, 2)
    didx = np.array([HD_DOCS[k][p] for k, p in zip(hd, pick)])
    h_names = np.array(HOSP)[hosp_i]
    yr = admit.dt.year.values
    h_los = np.array([H_LOS[(h, y)] for h, y in zip(h_names, yr)])
    emerg = rng.random(n) < np.array(D_EMG)[dept_i]
    payer = PAT.payer.values[pidx]
    mean = (np.array(D_LOS)[dept_i] * DG.los_m.values[dgi] * h_los * np.exp(DOC.eff_los.values[didx])
            * np.where(np.isin(admit.dt.month.values, [7, 8, 9]), 1.05, 1.0) * (1 + .010 * np.clip(age - 45, 0, None))
            * np.where(emerg, 1.10, 1.0) * np.array([PAYER_LOS[p] for p in payer])) * 0.72
    los = 1 + rng.gamma(3.0, np.clip(mean - 1, .3, None) / 3.0)
    los = np.maximum(1, np.rint(los)).astype(int)
    r = rng.random(n)
    status = np.where(r < np.array([.012, .03, .008, .01, .001, .004, .028, .02, .03, .02])[dept_i], "Expired",
                      np.where(r > .98, "LAMA", "Discharged"))
    rdp = (np.array(D_RD)[dept_i] * DG.rd_m.values[dgi] * np.exp(DOC.eff_rd.values[didx]) * np.array([H_RD[h] for h in h_names])
           * (1 + .25 * PAT.chronic.values[pidx]) * np.where(age >= 65, 1.2, 1.0) * rd_scale)
    rdp = np.where(status == "Expired", 0, np.clip(rdp, .005, .5))
    return pd.DataFrame(dict(hosp_i=hosp_i, hospital=h_names, patient_idx=pidx, age=age, sex=PAT.sex.values[pidx],
                             dept_i=dept_i, diag_i=dgi, doc_i=didx, admission_type=np.where(emerg, "Emergency", "Elective"),
                             payer=payer, admit=admit.values, los=los, status=status, rdp=rdp))


# ------------------------------------------------------------------ base admissions
parts = []
ELOS = 4.55
months = pd.period_range("2024-01", "2025-12", freq="M")
for hi, h in enumerate(HOSP):
    for p in months:
        occ = min(.95, OCC[h] * SEASON[p.month - 1] * rng.normal(1, .025))
        n = int(round(BEDS[h] * p.days_in_month * occ / (ELOS * H_LOS[(h, 2024)]) / 1.085))
        days = rng.integers(0, p.days_in_month, n)
        parts.append((hi, p.start_time + pd.to_timedelta(days, "D")))
hi_arr = np.concatenate([np.full(len(d), hi) for hi, d in parts])
adm_dates = pd.Series(pd.DatetimeIndex(np.concatenate([d.values for _, d in parts])))
pidx0 = rng.choice(NP, len(adm_dates), p=PW)
base = make_adm(hi_arr, adm_dates, pidx0)

# ------------------------------------------------------------------ readmissions (2 rounds)
allp = [base]
cur = base
for rnd in range(2):
    flag = rng.random(len(cur)) < cur.rdp.values * (1.0 if rnd == 0 else .7)
    src = cur[flag]
    disch = pd.to_datetime(src.admit) + pd.to_timedelta(src.los, "D")
    gap = 1 + np.minimum(29, np.floor(rng.exponential(9, len(src)))).astype(int)
    new_admit = (disch + pd.to_timedelta(gap, "D")).reset_index(drop=True)
    keep = (new_admit <= END).values
    src, new_admit = src[keep], new_admit[keep]
    same_h = rng.random(len(src)) < .85
    new_h = np.where(same_h, src.hosp_i.values, rng.integers(0, 4, len(src)))
    same_d = rng.random(len(src)) < .8
    fd = np.where(same_d, src.dept_i.values, 0)
    nxt = make_adm(new_h, new_admit, src.patient_idx.values, force_dept=fd, rd_scale=1.0)
    allp.append(nxt)
    cur = nxt
adm = pd.concat(allp, ignore_index=True)
adm["admit"] = pd.to_datetime(adm.admit)
adm["discharge"] = adm.admit + pd.to_timedelta(adm.los, "D")
adm = adm[adm.admit <= END].sort_values(["admit", "hosp_i"]).reset_index(drop=True)
adm["admission_id"] = [f"ADM-{i + 1:07d}" for i in range(len(adm))]
adm["patient_id"] = [f"P{i + 1:06d}" for i in adm.patient_idx]
adm["doctor_id"] = DOC.doctor_id.values[adm.doc_i]
adm["department"] = np.array(DEPTS)[adm.dept_i]
adm["diag_text"] = [DG.variants[d][rng.integers(0, 4)] for d in adm.diag_i]
adm["true_group"] = DG.group.values[adm.diag_i]
N_ADM = len(adm)
print("unique admissions:", N_ADM, "mean LOS:", adm.los.mean().round(2))

# ------------------------------------------------------------------ billing (IPD for every admission + OPD bills)
gst_choices = np.array([.05, .12, .18])
ipd_sub = (adm.los.values * np.array([ROOM[h] for h in adm.hospital]) +
           np.array(D_PKG)[adm.dept_i] * rng.lognormal(0, .35, N_ADM) * DG.los_m.values[adm.diag_i])
ipd_sub = np.round(ipd_sub, 2)
ipd_rate = rng.choice(gst_choices, N_ADM, p=[.6, .25, .15])
lag = np.where(rng.random(N_ADM) < .70, 0, 1)
ipd_bill = pd.DataFrame(dict(encounter_id=adm.admission_id, encounter_type="IPD", hospital=adm.hospital,
                             bill_date=adm.discharge + pd.to_timedelta(lag, "D"), subtotal=ipd_sub, gst_rate=ipd_rate))
N_OPD_BILL = 10_000
opd_bill_n = N_OPD_BILL

# ------------------------------------------------------------------ row budget -> OPD size
DUP = .012
adm_dup = int(round(N_ADM * DUP))
bill_unique = N_ADM + opd_bill_n
bill_dup = int(round(bill_unique * .01))
lab_raw = 170_000
lab_dup = int(round(lab_raw * .01 / 1.01))
opd_raw = TOTAL_ROWS - (N_ADM + adm_dup) - (bill_unique + bill_dup) - lab_raw
opd_dup = int(round(opd_raw * .01 / 1.01))
opd_unique = opd_raw - opd_dup
lab_unique = lab_raw - lab_dup
print("opd_raw", opd_raw, "lab_raw", lab_raw, "bill_raw", bill_unique + bill_dup, "adm_raw", N_ADM + adm_dup)

# ------------------------------------------------------------------ OPD visits
od = rng.integers(0, (END - START).days + 1, opd_unique)
opd_date = START + pd.to_timedelta(od, "D")
opd_h = rng.choice(4, opd_unique, p=np.array([BEDS[h] for h in HOSP]) / sum(BEDS.values()))
opd_p = rng.choice(NP, opd_unique, p=PW)
opd_dept, _ = dept_for(opd_p, pd.Series(opd_date))
hd = opd_h * 10 + opd_dept
opd_doc = np.array([HD_DOCS[k][rng.integers(0, 3)] for k in hd])
opd = pd.DataFrame(dict(opd_id=[f"OPD-{i + 1:07d}" for i in range(opd_unique)],
                        patient_id=[f"P{i + 1:06d}" for i in opd_p], hospital=np.array(HOSP)[opd_h],
                        department=np.array(DEPTS)[opd_dept], doctor_id=DOC.doctor_id.values[opd_doc],
                        visit_date=opd_date, visit_type=rng.choice(["New", "Follow-up"], opd_unique, p=[.45, .55]),
                        consultation_fee=rng.choice([300, 400, 500, 600, 800, 1000], opd_unique, p=[.2, .25, .25, .15, .1, .05]),
                        wait_minutes=np.round(rng.gamma(3, 9, opd_unique)).astype(int)))
opd = opd.sort_values("visit_date").reset_index(drop=True)
opd["opd_id"] = [f"OPD-{i + 1:07d}" for i in range(len(opd))]

# OPD bills
ob = opd.sample(opd_bill_n, random_state=7)
opd_bill = pd.DataFrame(dict(encounter_id=ob.opd_id.values, encounter_type="OPD", hospital=ob.hospital.values,
                             bill_date=ob.visit_date.values, subtotal=(ob.consultation_fee.values * rng.choice([1, 1, 1, 2, 3], opd_bill_n)).astype(float),
                             gst_rate=rng.choice(gst_choices, opd_bill_n, p=[.6, .25, .15])))
bill = pd.concat([ipd_bill, opd_bill], ignore_index=True)
bill["bill_date"] = pd.to_datetime(bill.bill_date)
bill["gst_amount"] = np.round(bill.subtotal * bill.gst_rate, 2)
bill["total_amount"] = np.round(bill.subtotal + bill.gst_amount, 2)
bill["bill_id"] = [f"BL-{i + 1:07d}" for i in range(len(bill))]
bill["payment_status"] = rng.choice(["Paid", "Pending", "Partially paid"], len(bill), p=[.82, .10, .08])
# drop 0.8% of IPD bills (admissions with no bill)
drop = (bill.encounter_type == "IPD") & (rng.random(len(bill)) < .008)
bill = bill[~drop].reset_index(drop=True)
# keep row budget exact: top up/trim OPD bills so bill_unique matches
diff = bill_unique - len(bill)
if diff > 0:
    extra = opd_bill.sample(diff, random_state=11, replace=True).copy()
    extra["gst_amount"] = np.round(extra.subtotal * extra.gst_rate, 2)
    extra["total_amount"] = np.round(extra.subtotal + extra.gst_amount, 2)
    extra["bill_id"] = [f"BL-{len(bill) + i + 1:07d}" for i in range(diff)]
    extra["payment_status"] = "Paid"
    extra["encounter_id"] = rng.choice(opd.opd_id.values, diff, replace=False)
    bill = pd.concat([bill, extra], ignore_index=True)

bill['bill_id'] = [f'BL-{i + 1:07d}' for i in range(len(bill))]

# ------------------------------------------------------------------ lab tests
TESTS = {"CBC": (["CBC", "Complete Blood Count", "cbc ", "C.B.C"], 2, 350), "HbA1c": (["HbA1c", "Glycosylated Hb", "HBA1C"], 6, 600),
         "LFT": (["LFT", "Liver Function Test", "lft"], 5, 700), "KFT": (["KFT", "Kidney Function Test", "RFT"], 4, 650),
         "Lipid Profile": (["Lipid Profile", "lipid panel", "LIPID PROFILE"], 6, 800), "Dengue NS1": (["Dengue NS1", "NS1 antigen", "dengue ns1 "], 3, 900),
         "Troponin I": (["Troponin I", "Trop-I", "TROPONIN"], 1.5, 1200), "CRP": (["CRP", "C-reactive protein", "crp"], 3, 500),
         "Urine R/M": (["Urine R/M", "Urine Routine", "urine r/m "], 2, 150), "Blood Culture": (["Blood Culture", "blood cx", "BLOOD CULTURE"], 48, 1500),
         "D-Dimer": (["D-Dimer", "d dimer", "DDIMER"], 3, 1100), "Thyroid Profile": (["Thyroid Profile", "TFT", "T3 T4 TSH"], 8, 750)}
tkeys = list(TESTS)
tsel = rng.integers(0, len(tkeys), lab_unique)
on_adm = rng.random(lab_unique) < .55
adm_pick = rng.integers(0, N_ADM, lab_unique)
opd_pick = rng.integers(0, len(opd), lab_unique)
enc = np.where(on_adm, adm.admission_id.values[adm_pick], opd.opd_id.values[opd_pick])
lab_h = np.where(on_adm, adm.hospital.values[adm_pick], opd.hospital.values[opd_pick])
lab_p = np.where(on_adm, adm.patient_id.values[adm_pick], opd.patient_id.values[opd_pick])
base_t = np.where(on_adm, (adm.admit.values[adm_pick] + (rng.random(lab_unique) * adm.los.values[adm_pick] * 86400 * 1e9).astype("timedelta64[ns]")),
                  opd.visit_date.values[opd_pick] + (rng.random(lab_unique) * 8 * 3600 * 1e9).astype("timedelta64[ns]") + np.timedelta64(9, "h"))
tat = np.array([TESTS[tkeys[i]][1] for i in tsel]) * rng.lognormal(0, .35, lab_unique)
base_t = pd.to_datetime(base_t)
lab = pd.DataFrame(dict(lab_id=[f"LAB-{i + 1:07d}" for i in range(lab_unique)], patient_id=lab_p, encounter_id=enc, hospital=lab_h,
                        test_name=[TESTS[tkeys[i]][0][rng.integers(0, len(TESTS[tkeys[i]][0]))] for i in tsel],
                        sample_time=base_t.round("min"), report_time=(base_t + pd.to_timedelta(tat, "h")).round("min"),
                        price=[TESTS[tkeys[i]][2] for i in tsel],
                        result_flag=rng.choice(["Normal", "High", "Low", "Critical"], lab_unique, p=[.55, .25, .15, .05])))

# ------------------------------------------------------------------ plant data-quality problems
P = {}


def hosp_noise(s, p=.08):
    s = s.copy()
    m = rng.random(len(s)) < p
    k = rng.integers(0, 4, len(s))
    v = s.values.astype(object)
    for i in np.where(m)[0]:
        b = v[i]
        v[i] = [b.lower(), b.upper(), b + " ", f"Hosp-{b}"][k[i]]
    return pd.Series(v, index=s.index), int(m.sum())


DEPT_VAR = {"General Medicine": ["Gen Med", "Gen. Medicine", "GENERAL MEDICINE"], "Cardiology": ["Cardio", "CARDIOLOGY", "cardiology "],
            "Orthopaedics": ["Ortho", "Orthopedics", "ORTHOPAEDICS"], "General Surgery": ["Gen Surgery", "Surgery", "GENERAL SURGERY"],
            "Obstetrics & Gynaecology": ["OBG", "OB-GYN", "Obs & Gynae"], "Paediatrics": ["Paeds", "Pediatrics", "PAEDIATRICS"],
            "Pulmonology": ["Pulmo", "Chest Medicine", "pulmonology "], "Nephrology": ["Nephro", "NEPHROLOGY", "nephrology "],
            "Neurology": ["Neuro", "NEUROLOGY", "neurology "], "Gastroenterology": ["Gastro", "GI Medicine", "GASTROENTEROLOGY"]}


def dept_noise(s, p=.08):
    v = s.values.astype(object).copy()
    m = rng.random(len(v)) < p
    for i in np.where(m)[0]:
        v[i] = DEPT_VAR[v[i]][rng.integers(0, 3)]
    return pd.Series(v, index=s.index), int(m.sum())


def fmt_dates(s, p_slash=.10, p_mon=.05, with_time=False):
    r = rng.random(len(s))
    out = s.dt.strftime("%Y-%m-%d %H:%M" if with_time else "%Y-%m-%d")
    m1 = r < p_slash
    m2 = (r >= p_slash) & (r < p_slash + p_mon)
    out[m1] = s[m1].dt.strftime("%d/%m/%Y %H:%M" if with_time else "%d/%m/%Y")
    out[m2] = s[m2].dt.strftime("%d-%b-%Y %H:%M" if with_time else "%d-%b-%Y")
    return out


def inr(x):
    s = f"{x:.2f}"
    ip, dp = s.split(".")
    if len(ip) > 3:
        head, tail = ip[:-3], ip[-3:]
        head = ",".join([head[max(i - 2, 0):i] for i in range(len(head), 0, -2)][::-1])
        ip = head + "," + tail
    return f"Rs. {ip}.{dp}"


# ---- admissions
A = pd.DataFrame(dict(admission_id=adm.admission_id, patient_id=adm.patient_id, age=adm.age.astype(object), sex=adm.sex,
                      hospital=adm.hospital, department=adm.department, doctor_id=adm.doctor_id, admission_type=adm.admission_type,
                      payer_category=adm.payer, admission_date=adm.admit, discharge_date=adm.discharge,
                      diagnosis_text=adm.diag_text, discharge_status=adm.status))
n = len(A)
miss_dis = rng.random(n) < .035
yr_typo = (~miss_dis) & (rng.random(n) < .004)
disch = A.discharge_date.copy()
disch[yr_typo] = disch[yr_typo] - pd.DateOffset(years=1)
A["discharge_date"] = disch
A["discharge_date"] = fmt_dates(A.discharge_date)
A["admission_date"] = fmt_dates(A.admission_date)
A.loc[miss_dis, "discharge_date"] = ""
P["adm_missing_discharge"] = int(miss_dis.sum())
P["adm_discharge_year_typo"] = int(yr_typo.sum())
miss_doc = rng.random(n) < .015
A.loc[miss_doc, "doctor_id"] = ""
P["adm_missing_doctor"] = int(miss_doc.sum())
A["hospital"], P["adm_hosp_variants"] = hosp_noise(A.hospital)
A["department"], P["adm_dept_variants"] = dept_noise(A.department)
agev = A.age.astype(object).copy()
t = rng.random(n)
agev[t < .06] = [f"{a} yrs" for a in adm.age[t < .06]]
agev[(t >= .06) & (t < .10)] = [f"{a}Y" for a in adm.age[(t >= .06) & (t < .10)]]
bad_age = rng.random(n) < .001
agev[bad_age] = rng.choice([250, 199, 999, -4], bad_age.sum()).astype(str)
A["age"] = agev
P["adm_age_text_or_bad"] = int(((t < .10) | bad_age).sum())
pid = A.patient_id.copy()
pm = rng.random(n) < .02
pid[pm] = ["p-" + x[1:] for x in pid[pm]]
A["patient_id"] = pid
P["adm_patient_id_variants"] = int(pm.sum())
dups = A.sample(adm_dup, random_state=21)
A = pd.concat([A, dups], ignore_index=True).sample(frac=1, random_state=5).reset_index(drop=True)
A.to_csv(f"{RAW}/admissions.csv", index=False)
P["adm_exact_duplicates"] = adm_dup

# ---- billing
B = bill[["bill_id", "encounter_id", "encounter_type", "hospital", "bill_date", "subtotal", "gst_rate", "gst_amount", "total_amount", "payment_status"]].copy()
nb = len(B)
dbl = (rng.random(nb) < .04) & (B.gst_rate > 0)
B.loc[dbl, "gst_amount"] = np.round(B.loc[dbl, "gst_amount"] * 2, 2)
B.loc[dbl, "total_amount"] = np.round(B.loc[dbl, "subtotal"] + B.loc[dbl, "gst_amount"], 2)
P["bill_gst_double_counted"] = int(dbl.sum())
B["bill_date"] = fmt_dates(B.bill_date)
B["hospital"], P["bill_hosp_variants"] = hosp_noise(B.hospital)
txt = rng.random(nb) < .02
B["total_amount"] = B.total_amount.astype(object)
B.loc[txt, "total_amount"] = [inr(x) for x in bill.loc[txt, "total_amount"].where(~dbl[txt], B.loc[txt, "total_amount"].astype(float)).astype(float)] if False else \
    [inr(float(x)) for x in B.loc[txt, "total_amount"]]
P["bill_amount_as_text"] = int(txt.sum())
bd = B.sample(bill_dup, random_state=31)
B = pd.concat([B, bd], ignore_index=True).sample(frac=1, random_state=6).reset_index(drop=True)
B.to_csv(f"{RAW}/billing.csv", index=False)
P["bill_exact_duplicates"] = bill_dup

# ---- OPD
O = opd.copy()
O["visit_date"] = fmt_dates(O.visit_date)
O["hospital"], P["opd_hosp_variants"] = hosp_noise(O.hospital)
O["department"], P["opd_dept_variants"] = dept_noise(O.department)
od_ = O.sample(opd_dup, random_state=41)
O = pd.concat([O, od_], ignore_index=True).sample(frac=1, random_state=7).reset_index(drop=True)
O.to_csv(f"{RAW}/opd_visits.csv", index=False)
P["opd_exact_duplicates"] = opd_dup

# ---- lab
L = lab.copy()
neg = rng.random(len(L)) < .003
s_, r_ = L.sample_time.copy(), L.report_time.copy()
L.loc[neg, "sample_time"], L.loc[neg, "report_time"] = r_[neg], s_[neg]
P["lab_sample_report_swapped"] = int(neg.sum())
L["sample_time"] = fmt_dates(L.sample_time, with_time=True)
L["report_time"] = fmt_dates(L.report_time, with_time=True)
L["hospital"], P["lab_hosp_variants"] = hosp_noise(L.hospital)
ld = L.sample(lab_dup, random_state=51)
L = pd.concat([L, ld], ignore_index=True).sample(frac=1, random_state=8).reset_index(drop=True)
L.to_csv(f"{RAW}/lab_tests.csv", index=False)
P["lab_exact_duplicates"] = lab_dup

tot = {k: len(v) for k, v in dict(admissions=A, billing=B, opd_visits=O, lab_tests=L).items()}
tot["TOTAL"] = sum(tot.values())
print(tot)
assert tot["TOTAL"] == TOTAL_ROWS, tot
json.dump(dict(rows=tot, planted=P), open("/home/claude/project/planted_problems_ANSWER_KEY.json", "w"), indent=2)
adm[["admission_id", "true_group"]].to_csv("/home/claude/project/_truth_diag.csv", index=False)
print(P)
