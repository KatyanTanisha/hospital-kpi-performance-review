"""
03_analysis.py  -  KPI calculation, rankings, contribution analysis, rupee impact, charts, RAG scorecard.
Reads the cleaned tables from 02_clean_data.py. Every KPI is built from additive numerators/denominators so
segment totals reconcile to the overall total exactly (see `recon`).
"""
import pandas as pd, numpy as np, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

C = "/home/claude/project/clean"
OUT = "/home/claude/project/outputs"
CH = f"{OUT}/charts"
os.makedirs(CH, exist_ok=True)
adm, bil = pd.read_pickle(f"{C}/admissions.pkl"), pd.read_pickle(f"{C}/billing.pkl")
BEDS = {"Kothrud": 120, "Hadapsar": 150, "Wakad": 90, "Pimpri": 110}
HOSP = list(BEDS)
doc = pd.read_csv("/home/claude/project/raw/doctors.csv")
DNAME = dict(zip(doc.doctor_id, doc.doctor_name))
END_DATA = pd.Timestamp("2025-12-31")
RES = {}

TEAL, ORANGE, GREY, RED, AMBER, GREEN, NAVY = "#0F766E", "#EA580C", "#94A3B8", "#DC2626", "#F59E0B", "#16A34A", "#1E293B"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 12, "figure.dpi": 110, "savefig.bbox": "tight"})

# =========================================================== 1. trusted analysis table
ipd = bil[bil.encounter_type == "IPD"].set_index("encounter_id")
adm["revenue_p"] = adm.admission_id.map(ipd.subtotal_p)          # revenue excluding GST
adm["billed_total_p"] = adm.admission_id.map(ipd.total_clean_p)  # including (correct) GST
# 30-day readmission = ANY later admission of the same patient (any hospital) that starts 1-30 days after this stay's discharge
_a = adm[["admission_id", "patient_id", "discharge_date", "los_valid", "discharge_status"]]
_j = _a[_a.los_valid].merge(adm[["patient_id", "admission_date", "admission_id"]].rename(columns={"admission_date": "admission_date_n", "admission_id": "admission_id_n"}), on="patient_id")
_j = _j[_j.admission_id != _j.admission_id_n]
_gap = (_j.admission_date_n - _j.discharge_date).dt.days
_hit = set(_j.loc[(_gap >= 1) & (_gap <= 30), "admission_id"])
adm["readmit_eligible"] = adm.los_valid & (adm.discharge_status != "Expired") & (adm.discharge_date <= pd.Timestamp("2025-12-01"))
adm["readmit_30d"] = adm.readmit_eligible & adm.admission_id.isin(_hit)
adm = adm.sort_values("admission_id").reset_index(drop=True)
adm["year"] = adm.admission_date.dt.year
adm["month"] = adm.admission_date.dt.to_period("M").astype(str)
adm["los_n"] = adm.los_valid.astype(int)
adm["los_sum"] = adm.los_days.fillna(0)
adm["rd_n"] = adm.readmit_eligible.astype(int)
adm["rd_hit"] = adm.readmit_30d.astype(int)
adm["doctor_name"] = adm.doctor_id.map(DNAME).fillna("UNASSIGNED")
assert adm.admission_id.is_unique

# patient-days (midnights) that fall inside each calendar month
months = pd.period_range("2024-01", "2025-12", freq="M")
v = adm[adm.los_valid]
pdays = []
for p in months:
    ms, me = p.start_time, (p + 1).start_time
    lo = np.maximum(v.admission_date.values, np.datetime64(ms))
    hi = np.minimum(v.discharge_date.values, np.datetime64(me))
    nights = np.clip((hi - lo) / np.timedelta64(1, "D"), 0, None).astype(int)
    d = pd.DataFrame({"hospital": v.hospital.values, "payer_category": v.payer_category.values, "month": str(p), "patient_days": nights})
    pdays.append(d[d.patient_days > 0])
PD = pd.concat(pdays)
bed_days = pd.DataFrame([(h, str(p), BEDS[h] * p.days_in_month) for h in HOSP for p in months], columns=["hospital", "month", "bed_days"])
PD_hm = PD.groupby(["hospital", "month"], as_index=False).patient_days.sum().merge(bed_days, on=["hospital", "month"], how="right").fillna(0)
PD_hm["year"] = PD_hm.month.str[:4].astype(int)
cols = ["admission_id", "patient_id", "hospital", "department", "doctor_id", "doctor_name", "admission_type", "payer_category", "diagnosis_group", "age", "sex",
        "admission_date", "discharge_date", "discharge_imputed", "los_valid", "los_days", "discharge_status", "revenue_p", "billed_total_p", "readmit_eligible", "readmit_30d", "overlap_flag", "month", "year"]
trusted = adm[cols].copy()
trusted["revenue_inr"] = trusted.revenue_p / 100
trusted["billed_total_inr"] = trusted.billed_total_p / 100
trusted.drop(columns=["revenue_p", "billed_total_p"]).to_csv(f"{OUT}/analysis_table_clean.csv", index=False)

# =========================================================== 2. KPI engine
def seg(df, by, pdf=None):
    g = df.groupby(by).agg(admissions=("admission_id", "size"), los_n=("los_n", "sum"), los_sum=("los_sum", "sum"),
                           rd_elig=("rd_n", "sum"), rd_hit=("rd_hit", "sum"), revenue_p=("revenue_p", "sum")).reset_index()
    g["alos"] = g.los_sum / g.los_n
    g["readmit_rate"] = g.rd_hit / g.rd_elig
    return g


tot = seg(adm.assign(all="All"), "all").iloc[0]
RES["overall"] = {k: float(tot[k]) for k in ["admissions", "los_n", "los_sum", "rd_elig", "rd_hit", "alos", "readmit_rate"]}
yr = seg(adm, "year")
RES["year"] = yr.set_index("year").to_dict("index")

pd_year = PD_hm.groupby("year")[["patient_days", "bed_days"]].sum()
pd_year["occupancy"] = pd_year.patient_days / pd_year.bed_days
RES["occupancy_year"] = pd_year.to_dict("index")
RES["occupancy_overall"] = float(PD_hm.patient_days.sum() / PD_hm.bed_days.sum())

# monthly (chain)
mon = seg(adm, "month")
mo_occ = PD_hm.groupby("month")[["patient_days", "bed_days"]].sum()
mon = mon.merge(mo_occ, on="month")
mon["occupancy"] = mon.patient_days / mon.bed_days
mon["year"] = mon.month.str[:4].astype(int)
mon["mm"] = mon.month.str[5:].astype(int)
prev = mon[["mm", "year", "alos", "occupancy", "readmit_rate"]].copy()
prev["year"] += 1
mon = mon.merge(prev, on=["mm", "year"], how="left", suffixes=("", "_prev_year"))
for k in ["alos", "occupancy", "readmit_rate"]:
    mon[f"{k}_vs_prev_year"] = mon[k] - mon[f"{k}_prev_year"]
    mon[f"{k}_vs_overall"] = mon[k] - {"alos": tot.alos, "occupancy": RES["occupancy_overall"], "readmit_rate": tot.readmit_rate}[k]

# location x year, location x month
loc = seg(adm, ["hospital", "year"])
lo_occ = PD_hm.groupby(["hospital", "year"])[["patient_days", "bed_days"]].sum().reset_index()
loc = loc.merge(lo_occ, on=["hospital", "year"])
loc["occupancy"] = loc.patient_days / loc.bed_days
loc["revenue_per_patient_day"] = loc.revenue_p / 100 / loc.los_sum   # approx (bills / LOS days of admissions)
lp = loc[["hospital", "year", "alos", "occupancy", "readmit_rate"]].copy(); lp["year"] += 1
loc = loc.merge(lp, on=["hospital", "year"], how="left", suffixes=("", "_prev_year"))
for k in ["alos", "occupancy", "readmit_rate"]:
    loc[f"{k}_vs_prev_year"] = loc[k] - loc[f"{k}_prev_year"]
    loc[f"{k}_vs_chain"] = loc[k] - loc.year.map({y: (yr.set_index("year")[k].loc[y] if k != "occupancy" else pd_year.occupancy.loc[y]) for y in (2024, 2025)})
lm = seg(adm, ["hospital", "month"]).merge(PD_hm[["hospital", "month", "patient_days", "bed_days"]], on=["hospital", "month"])
lm["occupancy"] = lm.patient_days / lm.bed_days

# category (payer) x year
cat = seg(adm, ["payer_category", "year"])
pc = PD.assign(year=PD.month.str[:4].astype(int)).groupby(["payer_category", "year"]).patient_days.sum().reset_index()
cat = cat.merge(pc, on=["payer_category", "year"])
cat["occupancy_points"] = cat.patient_days / cat.year.map(pd_year.bed_days)     # share of ALL bed-days; adds up to occupancy
cat["share_of_patient_days"] = cat.patient_days / cat.year.map(pd_year.patient_days)
cp = cat[["payer_category", "year", "alos", "readmit_rate"]].copy(); cp["year"] += 1
cat = cat.merge(cp, on=["payer_category", "year"], how="left", suffixes=("", "_prev_year"))
for k in ["alos", "readmit_rate"]:
    cat[f"{k}_vs_prev_year"] = cat[k] - cat[f"{k}_prev_year"]
    cat[f"{k}_vs_chain"] = cat[k] - cat.year.map(yr.set_index("year")[k])
dept = seg(adm, ["department", "year"])

# ------- reconciliation (segment totals vs overall; difference must be exactly 0)
rows = []
def rc(name, overall, parts):
    rows.append({"check": name, "overall_total": float(overall), "sum_of_segments": float(parts), "difference": float(overall) - float(parts)})
rc("Admissions: sum of months = overall", tot.admissions, mon.admissions.sum())
rc("Admissions: sum of locations = overall", tot.admissions, loc.admissions.sum())
rc("Admissions: sum of payer categories = overall", tot.admissions, cat.admissions.sum())
rc("Admissions: sum of departments = overall", tot.admissions, dept.admissions.sum())
rc("LOS days: sum of months = overall", tot.los_sum, mon.los_sum.sum())
rc("LOS days: sum of locations = overall", tot.los_sum, loc.los_sum.sum())
rc("LOS days: sum of categories = overall", tot.los_sum, cat.los_sum.sum())
rc("Patient-days (occupancy numerator): months = locations", mon.patient_days.sum(), lm.patient_days.sum())
rc("Patient-days: payer categories = locations", cat.patient_days.sum(), loc.patient_days.sum())
rc("Patient-days vs LOS days (stays all inside window)", tot.los_sum, PD_hm.patient_days.sum() + 0)  # informational; difference = nights after 31 Dec 2025
rc("30-day readmissions: months = overall", tot.rd_hit, mon.rd_hit.sum())
rc("30-day readmissions: locations = overall", tot.rd_hit, loc.rd_hit.sum())
rc("Readmission-eligible stays: categories = overall", tot.rd_elig, cat.rd_elig.sum())
rc("Revenue (paise, ex-GST): locations = overall", adm.revenue_p.sum(), loc.revenue_p.sum())
CS = json.load(open("/home/claude/project/cleaning_summary.json"))
rc("Billed total (paise): analysis billing table = cleaning-script reconciliation total", bil.total_clean_p.sum(), CS["billing_reconciliation_paise"]["cleaned_total"])
RECON = pd.DataFrame(rows)
RES["recon_max_abs_diff_excl_informational"] = float(RECON.drop(index=9).difference.abs().max())

# =========================================================== 3. performer ranking (doctors, dept-units, patients)
base = adm[adm.los_valid].copy()
# expected = average of peers at the SAME hospital with the same department, diagnosis group and admission type (fair to doctors at busy sites)
exp_los = base.groupby(["hospital", "department", "diagnosis_group", "admission_type"]).los_days.transform("mean")
adm["exp_los"] = exp_los.reindex(adm.index)
rd_el = adm[adm.readmit_eligible]
adm["exp_rd"] = rd_el.groupby(["hospital", "department", "diagnosis_group"]).rd_hit.transform("mean").reindex(adm.index)
# chain-wide expectation (used for department units so hospital-level differences stay visible)
adm["exp_los_chain"] = base.groupby(["department", "diagnosis_group", "admission_type"]).los_days.transform("mean").reindex(adm.index)
adm["exp_rd_chain"] = rd_el.groupby(["department", "diagnosis_group"]).rd_hit.transform("mean").reindex(adm.index)
adm["exp_los_chain_sum"] = adm.exp_los_chain.where(adm.los_valid, 0).fillna(0)
adm["exp_rd_chain_sum"] = adm.exp_rd_chain.where(adm.readmit_eligible, 0).fillna(0)
adm["exp_los_sum"] = adm.exp_los.where(adm.los_valid, 0).fillna(0)
adm["exp_rd_sum"] = adm.exp_rd.where(adm.readmit_eligible, 0).fillna(0)
adm["rev_for_rate"] = adm.revenue_p.where(adm.los_valid)
adm["los_for_rate"] = adm.los_sum.where(adm.revenue_p.notna(), 0)


def perf(by, min_n=100, chain=False):
    el, er = ("exp_los_chain_sum", "exp_rd_chain_sum") if chain else ("exp_los_sum", "exp_rd_sum")
    g = adm.groupby(by).agg(admissions=("admission_id", "size"), los_n=("los_n", "sum"), los_obs=("los_sum", "sum"), los_exp=(el, "sum"),
                            rd_elig=("rd_n", "sum"), rd_obs=("rd_hit", "sum"), rd_exp=(er, "sum"),
                            rev=("rev_for_rate", "sum"), rev_days=("los_for_rate", "sum")).reset_index()
    g = g[g.los_n >= min_n].copy()
    g["alos"] = g.los_obs / g.los_n
    g["los_index"] = g.los_obs / g.los_exp            # >1 = slower than case-mix-expected
    g["readmit_rate"] = g.rd_obs / g.rd_elig
    g["readmit_oe"] = g.rd_obs / g.rd_exp
    g["rev_per_bed_day"] = g.rev / 100 / g.rev_days
    g["score"] = 100 * (.5 * (1 - g.los_index.rank(pct=True)) + .5 * (1 - g.readmit_oe.rank(pct=True)))   # revenue/bed-day shown for context only (not case-mix adjusted)
    g["rank"] = g.score.rank(ascending=False, method="first").astype(int)
    n10 = max(1, int(round(.10 * len(g))))
    g["band"] = np.where(g["rank"] <= n10, "Top 10%", np.where(g["rank"] > len(g) - n10, "Bottom 10%", "Middle 80%"))
    return g.sort_values("rank")


DOCS = perf("doctor_id")
DOCS = DOCS[DOCS.doctor_id != "UNASSIGNED"].copy()
n10 = int(round(.10 * len(DOCS)))
DOCS["rank"] = DOCS.score.rank(ascending=False, method="first").astype(int)
DOCS["band"] = np.where(DOCS["rank"] <= n10, "Top 10%", np.where(DOCS["rank"] > len(DOCS) - n10, "Bottom 10%", "Middle 80%"))
DOCS = DOCS.merge(doc.rename(columns={"department": "dept", "hospital": "hosp"}), on="doctor_id").sort_values("rank")
UNITS = perf(["hospital", "department"], chain=True)
PAT = adm.groupby("patient_id").agg(admissions=("admission_id", "size"), revenue=("revenue_p", "sum"), los=("los_sum", "sum"), age=("age", "mean"),
                                    emergency_share=("admission_type", lambda s: (s == "Emergency").mean()),
                                    readmitted=("rd_hit", "max"), payer=("payer_category", lambda s: s.mode().iat[0]),
                                    n_depts=("department", "nunique")).reset_index()
PAT = PAT[PAT.revenue > 0].copy()
PAT["revenue"] = PAT.revenue / 100
PAT["pct"] = PAT.revenue.rank(pct=True)
PAT["band"] = np.where(PAT.pct > .9, "Top 10%", np.where(PAT.pct <= .1, "Bottom 10%", "Middle 80%"))
pb = PAT.groupby("band").agg(patients=("patient_id", "size"), revenue=("revenue", "sum"), admissions=("admissions", "mean"), avg_los_days=("los", "mean"),
                             age=("age", "mean"), emergency_share=("emergency_share", "mean"), readmitted_share=("readmitted", "mean"),
                             insured_share=("payer", lambda s: s.isin(["Insurance", "Corporate"]).mean())).reindex(["Top 10%", "Middle 80%", "Bottom 10%"])
pb["revenue_share"] = pb.revenue / pb.revenue.sum()
pb["revenue_per_patient"] = pb.revenue / pb.patients
pb["admissions_per_patient"] = pb.admissions
t_, b_ = pb.loc["Top 10%"], pb.loc["Bottom 10%"]
RES["patient_decomp"] = {"ln_total_ratio": float(np.log(t_.revenue_per_patient / b_.revenue_per_patient)),
                         "share_from_frequency": float(np.log(t_.admissions_per_patient / b_.admissions_per_patient) / np.log(t_.revenue_per_patient / b_.revenue_per_patient))}

# ---- contribution analysis: bottom vs top doctors (case-mix effect vs practice effect)
chain_alos = tot.alos
dd = adm.merge(DOCS[["doctor_id", "band"]], on="doctor_id", how="inner")
dd = dd[dd.los_valid]
def waterfall(g):
    n = len(g)
    return pd.Series({"stays": n, "observed_alos": g.los_days.mean(), "expected_alos_given_hospital_and_case_mix": g.exp_los.mean(),
                      "case_mix_effect_days": g.exp_los.mean() - chain_alos, "practice_effect_days": g.los_days.mean() - g.exp_los.mean(),
                      "emergency_share": (g.admission_type == "Emergency").mean(), "insured_or_govt_share": (g.payer_category != "Self-pay").mean(), "mean_age": g.age.mean()})
WF = dd.groupby("band").apply(waterfall, include_groups=False).reindex(["Top 10%", "Middle 80%", "Bottom 10%"])
bot = dd[dd.band == "Bottom 10%"].copy()
bot["excess"] = bot.los_days - bot.exp_los
excess_dx = bot.groupby("diagnosis_group").excess.sum().sort_values(ascending=False)
excess_payer = bot.groupby("payer_category").excess.sum().sort_values(ascending=False)
excess_hosp = bot.groupby("hospital").excess.sum().sort_values(ascending=False)
RES["bottom_doctor_excess_bed_days"] = float(bot.excess.sum())

# ---- stretch: bottom quartile -> median (doctors, LOS index) + readmission
q25 = DOCS.los_index.quantile(.75)  # worst quartile = highest index
med = DOCS.los_index.median()
bq = DOCS[DOCS.los_index >= q25].copy()
bq["freed_bed_days"] = (bq.los_obs - bq.los_exp * med).clip(lower=0)
freed_total = float(bq.freed_bed_days.sum())
yrs = 2.0
rpd = adm.revenue_p.sum() / 100 / adm.loc[adm.revenue_p.notna(), "los_sum"].sum()
rd_med = DOCS.readmit_oe.median()
rq = DOCS[DOCS.readmit_oe >= DOCS.readmit_oe.quantile(.75)].copy()
rq["avoided_rd"] = (rq.rd_obs - rq.rd_exp * rd_med).clip(lower=0)
avg_rd_bill = adm[adm.rd_hit == 1].revenue_p.mean() / 100  # not used for savings (a readmission that is avoided removes revenue); use cost proxy below
rd_total = float(rq.avoided_rd.sum())
# value of an avoided readmission = the bed-days it would consume (avg LOS of readmitted stays) -> freed capacity, valued at revenue per bed-day
rd_los = adm[(adm.readmit_30d)].los_days.mean()
rd_bed_days = rd_total * rd_los
IMP = {"rev_per_bed_day_inr": float(rpd), "median_doctor_los_index": float(med), "bottom_quartile_doctors": int(len(bq)),
       "freed_bed_days_2yr": freed_total, "freed_bed_days_per_year": freed_total / yrs,
       "avoided_readmissions_2yr": rd_total, "avoided_readmissions_per_year": rd_total / yrs, "avg_los_of_a_readmission_stay": float(rd_los),
       "readmission_bed_days_per_year": rd_bed_days / yrs}
for lab_, share in [("conservative_50pct_refilled", .5), ("full_100pct_refilled", 1.0)]:
    IMP[f"annual_los_impact_inr_{lab_}"] = freed_total / yrs * rpd * share
    IMP[f"annual_readmission_impact_inr_{lab_}"] = rd_bed_days / yrs * rpd * share
    IMP[f"annual_total_impact_inr_{lab_}"] = IMP[f"annual_los_impact_inr_{lab_}"] + IMP[f"annual_readmission_impact_inr_{lab_}"]
RES["impact"] = IMP

# =========================================================== 4. data-quality driver for "which issue matters most"
miss_dis = int(adm.discharge_imputed.sum() + (~adm.los_valid).sum())
alos_if_excluded = adm[adm.los_valid & ~adm.discharge_imputed].los_days.mean()
RES["dq"] = {"alos_after_cleaning": float(tot.alos), "alos_if_missing_discharge_rows_dropped": float(alos_if_excluded),
             "gst_excess_inr": float((bil.total_raw_p - bil.total_clean_p).sum() / 100),
             "gst_excess_share_of_billed": float((bil.total_raw_p - bil.total_clean_p).sum() / bil.total_raw_p.sum()),
             "bills_missing_for_admissions": int(adm.revenue_p.isna().sum())}

# =========================================================== 5. scorecard (RAG)
base24 = yr.set_index("year").loc[2024]
BASE = {"alos": float(base24.alos), "readmit": float(base24.readmit_rate)}
def rag_alos(x): return "G" if x <= BASE["alos"] else ("A" if x <= BASE["alos"] * 1.10 else "R")
def rag_rd(x): return "G" if x <= BASE["readmit"] else ("A" if x <= BASE["readmit"] * 1.15 else "R")
def rag_occ(x): return "G" if .70 <= x <= .85 else ("A" if (.60 <= x < .70) or (.85 < x <= .90) else "R")
sc = loc[loc.year == 2025].set_index("hospital")
chain25 = {"alos": float(yr.set_index("year").loc[2025].alos), "occupancy": float(pd_year.loc[2025].occupancy), "readmit": float(yr.set_index("year").loc[2025].readmit_rate)}
rowsC = [{"unit": h, "alos": sc.loc[h].alos, "occupancy": sc.loc[h].occupancy, "readmit": sc.loc[h].readmit_rate,
          "alos_d": sc.loc[h].alos_vs_prev_year, "occ_d": sc.loc[h].occupancy_vs_prev_year, "rd_d": sc.loc[h].readmit_rate_vs_prev_year} for h in HOSP]
c25 = yr.set_index("year")
rowsC.append({"unit": "Chain", "alos": chain25["alos"], "occupancy": chain25["occupancy"], "readmit": chain25["readmit"],
              "alos_d": chain25["alos"] - float(c25.loc[2024].alos), "occ_d": chain25["occupancy"] - float(pd_year.loc[2024].occupancy),
              "rd_d": chain25["readmit"] - float(c25.loc[2024].readmit_rate)})
SC = pd.DataFrame(rowsC)
SC["alos_rag"], SC["occ_rag"], SC["rd_rag"] = SC.alos.map(rag_alos), SC.occupancy.map(rag_occ), SC.readmit.map(rag_rd)
RES["thresholds"] = {"alos_green_le": BASE["alos"], "alos_amber_le": BASE["alos"] * 1.10, "readmit_green_le": BASE["readmit"], "readmit_amber_le": BASE["readmit"] * 1.15}

# =========================================================== 6. charts
COL = {"Kothrud": TEAL, "Hadapsar": ORANGE, "Wakad": GREY, "Pimpri": NAVY}
mlab = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# 1 ALOS by month, 2024 vs 2025
fig, ax = plt.subplots(figsize=(8, 4.2))
for y, c, ls in [(2024, GREY, "--"), (2025, TEAL, "-")]:
    d = mon[mon.year == y]
    ax.plot(d.mm, d.alos, ls, color=c, marker="o", ms=4, lw=2, label=str(y))
ax.axhline(tot.alos, color=ORANGE, lw=1, ls=":"); ax.text(12.15, tot.alos, f"24-mo avg {tot.alos:.2f}", color=ORANGE, va="center", fontsize=8)
ax.set_xticks(range(1, 13)); ax.set_xticklabels(mlab); ax.set_ylabel("Average length of stay (days)"); ax.set_title("Chain ALOS peaks in the monsoon months (Jul-Sep)")
ax.legend(frameon=False); ax.set_xlim(.7, 13.4); fig.savefig(f"{CH}/chart1_alos_by_month.png"); plt.close()

# 2 occupancy by hospital by month
fig, ax = plt.subplots(figsize=(8.6, 4.2))
x = np.arange(len(months))
for h in HOSP:
    d = lm[lm.hospital == h].sort_values("month")
    ax.plot(x, d.occupancy * 100, color=COL[h], lw=2, label=h)
ax.axhspan(70, 85, color=GREEN, alpha=.08); ax.text(0.2, 85.6, "green band 70-85%", fontsize=8, color=GREEN)
ax.set_xticks(x[::3]); ax.set_xticklabels([str(m) for m in months][::3], rotation=0, fontsize=8); ax.set_ylabel("Bed occupancy (%)")
ax.set_title("Hadapsar runs hot (>85%), Wakad has spare capacity (<65%)"); ax.legend(frameon=False, ncol=4, loc="lower center"); fig.savefig(f"{CH}/chart2_occupancy_by_hospital.png"); plt.close()

# 3 readmission by hospital & year
fig, ax = plt.subplots(figsize=(7.4, 4.2))
w = .36
for i, (y, c) in enumerate([(2024, GREY), (2025, TEAL)]):
    d = loc[loc.year == y].set_index("hospital").reindex(HOSP)
    b = ax.bar(np.arange(4) + (i - .5) * w, d.readmit_rate * 100, w, color=c, label=str(y))
    for r_, v_ in zip(b, d.readmit_rate * 100): ax.text(r_.get_x() + r_.get_width() / 2, v_ + .15, f"{v_:.1f}", ha="center", fontsize=8)
ax.axhline(tot.readmit_rate * 100, color=ORANGE, ls=":", lw=1.2); ax.text(3.45, tot.readmit_rate * 100 + .15, f"chain {tot.readmit_rate * 100:.1f}%", color=ORANGE, fontsize=8, ha="right")
ax.set_xticks(range(4)); ax.set_xticklabels(HOSP); ax.set_ylabel("30-day readmission rate (%)"); ax.set_title("Hadapsar's readmission rate is the outlier"); ax.legend(frameon=False)
fig.savefig(f"{CH}/chart3_readmission_by_hospital.png"); plt.close()

# 4 doctor scatter
fig, ax = plt.subplots(figsize=(7.4, 5))
for band, c, s in [("Middle 80%", "#CBD5E1", 22), ("Top 10%", GREEN, 46), ("Bottom 10%", RED, 46)]:
    d = DOCS[DOCS.band == band]
    ax.scatter(d.los_index, d.readmit_oe, s=s, color=c, label=f"{band} (n={len(d)})", edgecolor="white", lw=.5)
ax.axvline(1, color=GREY, lw=.8); ax.axhline(1, color=GREY, lw=.8)
ax.set_xlabel("LOS index (observed / case-mix-expected; <1 = faster)"); ax.set_ylabel("Readmission O/E (<1 = fewer than expected)")
ax.set_title("Weak doctors are slow AND readmit more"); ax.legend(frameon=False, fontsize=8); fig.savefig(f"{CH}/chart4_doctor_quadrant.png"); plt.close()

# 5 dept-unit heat map (LOS index)
piv = UNITS.pivot(index="department", columns="hospital", values="los_index").reindex(columns=HOSP)
fig, ax = plt.subplots(figsize=(7.2, 4.8))
im = ax.imshow(piv.values, cmap="RdYlGn_r", vmin=.85, vmax=1.2, aspect="auto")
ax.set_xticks(range(4)); ax.set_xticklabels(HOSP); ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=8)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]): ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("Department x hospital LOS index (1.00 = as expected)"); plt.colorbar(im, ax=ax, shrink=.8); fig.savefig(f"{CH}/chart5_dept_heatmap.png"); plt.close()

# 6 excess bed-days by diagnosis (bottom doctors)
fig, ax = plt.subplots(figsize=(7.4, 4.2))
e = excess_dx.head(8)[::-1]
ax.barh(e.index, e.values, color=ORANGE)
for i, v_ in enumerate(e.values): ax.text(v_ + 5, i, f"{v_:,.0f}", va="center", fontsize=8)
ax.set_xlabel("Excess bed-days vs case-mix expectation (24 months)"); ax.set_title("Where bottom-10% doctors lose bed-days"); fig.savefig(f"{CH}/chart6_excess_bed_days_by_diagnosis.png"); plt.close()

# 7 patient revenue concentration
ps = PAT.sort_values("revenue", ascending=False)
cum = ps.revenue.cumsum() / ps.revenue.sum()
fig, ax = plt.subplots(figsize=(6.8, 4.2))
ax.plot(np.arange(1, len(ps) + 1) / len(ps) * 100, cum * 100, color=TEAL, lw=2)
t10 = float(cum.iloc[int(.1 * len(ps)) - 1] * 100)
ax.scatter([10], [t10], color=ORANGE, zorder=5); ax.annotate(f"top 10% of patients = {t10:.0f}% of revenue", (10, t10), (22, t10 - 14), arrowprops=dict(arrowstyle="->", color=ORANGE), fontsize=9)
ax.plot([0, 100], [0, 100], color=GREY, ls=":"); ax.set_xlabel("% of patients (highest revenue first)"); ax.set_ylabel("Cumulative % of revenue"); ax.set_title("Patient revenue is concentrated")
fig.savefig(f"{CH}/chart7_patient_concentration.png"); plt.close()
RES["top10_patient_revenue_share"] = t10

# 8 impact
fig, ax = plt.subplots(figsize=(6.8, 4.2))
cs = IMP["annual_total_impact_inr_conservative_50pct_refilled"] / 1e7; fl = IMP["annual_total_impact_inr_full_100pct_refilled"] / 1e7
l1 = IMP["annual_los_impact_inr_conservative_50pct_refilled"] / 1e7; r1 = IMP["annual_readmission_impact_inr_conservative_50pct_refilled"] / 1e7
l2 = IMP["annual_los_impact_inr_full_100pct_refilled"] / 1e7; r2 = IMP["annual_readmission_impact_inr_full_100pct_refilled"] / 1e7
ax.bar(["50% of freed\nbed-days refilled", "100% refilled"], [l1, l2], color=TEAL, label="Shorter stays")
ax.bar(["50% of freed\nbed-days refilled", "100% refilled"], [r1, r2], bottom=[l1, l2], color=ORANGE, label="Fewer readmission stays")
for i, v_ in enumerate([cs, fl]): ax.text(i, v_ + .1, f"Rs {v_:.1f} cr", ha="center", fontweight="bold")
ax.set_ylabel("Annual impact (Rs crore)"); ax.set_title("Lifting bottom-quartile doctors to the median"); ax.legend(frameon=False, loc="upper left"); fig.savefig(f"{CH}/chart8_impact.png"); plt.close()

# scorecard figure (one page)
fig = plt.figure(figsize=(11, 6.2)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 6.2)
ax.text(.4, 5.85, "Hospital KPI Scorecard - calendar 2025 (Pune diagnostic & hospital chain, simulated data)", fontsize=14, fontweight="bold", color=NAVY)
ax.text(.4, 5.55, "RAG vs chain 2024 baseline. Arrows show change vs 2024 (green = improving, red = worsening; occupancy change shown in grey).", fontsize=9, color="#475569")
hdr = ["Location", "Avg length of stay (days)", "Bed occupancy", "30-day readmission"]
xs = [.4, 2.4, 5.2, 8.0]
for x_, h_ in zip(xs, hdr): ax.text(x_, 5.0, h_, fontsize=10, fontweight="bold", color=NAVY)
RC = {"G": "#BBF7D0", "A": "#FDE68A", "R": "#FECACA"}
for i, r in SC.iterrows():
    y = 4.35 - i * .72
    ax.add_patch(Rectangle((.3, y - .3), 10.4, .66, fc="#F8FAFC" if r.unit != "Chain" else "#E2E8F0", ec="none"))
    ax.text(.4, y, r.unit, fontsize=11, fontweight="bold", va="center", color=NAVY)
    for x_, val, d_, rag, fmt, better in [(2.4, r.alos, r.alos_d, r.alos_rag, "{:.2f}", "down"), (5.2, r.occupancy, r.occ_d, r.occ_rag, "{:.1%}", None), (8.0, r.readmit, r.rd_d, r.rd_rag, "{:.1%}", "down")]:
        ax.add_patch(Rectangle((x_ - .1, y - .26), 2.5, .56, fc=RC[rag], ec="white", lw=2))
        arrow = "▲" if d_ > 0 else "▼"
        ax.text(x_ + .05, y, fmt.format(val), fontsize=12, fontweight="bold", va="center")
        dv = f"{d_:+.2f}" if fmt == "{:.2f}" else f"{d_ * 100:+.1f} pp"
        good = (d_ < 0) if better == "down" else None
        ax.text(x_ + 1.25, y, f"{arrow} {dv}", fontsize=9, va="center", color=("#15803D" if good else "#B91C1C") if good is not None else "#334155", fontweight="bold")
ax.text(.4, .78, f"Thresholds:  ALOS  Green <= {BASE['alos']:.2f}d (2024 chain), Amber <= +10%, Red above  |  Occupancy  Green 70-85%, Amber 60-70% or 85-90%, Red <60% or >90%\n"
        f"                       Readmission  Green <= {BASE['readmit']:.1%} (2024 chain), Amber <= +15%, Red above", fontsize=8.5, color="#334155")
ax.text(.4, .3, "Reconciliation: all KPIs are built from additive counts; location, month and category totals equal the overall totals with zero difference.", fontsize=8.5, color="#334155")
fig.savefig(f"{CH}/chart9_kpi_scorecard.png"); plt.close()

# =========================================================== 7. save everything
for name, df in dict(monthly=mon, by_location=loc, by_location_month=lm, by_category=cat, by_department=dept, recon=RECON, doctors=DOCS, units=UNITS,
                     patients_band=pb.reset_index(), patient_decile=PAT, waterfall=WF.reset_index(), excess_dx=excess_dx.reset_index(), excess_payer=excess_payer.reset_index(),
                     excess_hosp=excess_hosp.reset_index(), scorecard=SC, bottom_q_doctors=bq, readmit_q_doctors=rq).items():
    df.to_pickle(f"{OUT}/{name}.pkl")
json.dump(RES, open(f"{OUT}/results.json", "w"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(json.dumps(RES, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))[:3500])
print(RECON); print(yr[["year", "admissions", "alos", "readmit_rate"]]); print(pd_year)
print(loc[["hospital", "year", "alos", "occupancy", "readmit_rate"]].round(3)); print(cat[["payer_category", "year", "alos", "readmit_rate", "share_of_patient_days"]].round(3))
print(SC.round(3)); print(WF.round(2)); print(pb.round(2)); print(excess_dx.head(6).round(0)); print(excess_payer.round(0)); print(excess_hosp.round(0))
print(DOCS[["doctor_id", "doctor_name", "hosp", "dept", "los_index", "readmit_oe", "score", "band"]].head(4).round(2)); print(DOCS[["doctor_id", "doctor_name", "hosp", "dept", "los_index", "readmit_oe", "score", "band"]].tail(4).round(2))
print(UNITS[["hospital", "department", "los_index", "readmit_oe", "score", "band"]].head(4).round(2)); print(UNITS[["hospital", "department", "los_index", "readmit_oe", "score", "band"]].tail(4).round(2))
