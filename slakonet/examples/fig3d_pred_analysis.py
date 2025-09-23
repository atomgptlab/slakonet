#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.getcwd()


# In[12]:


from jarvis.db.jsonutils import loadjson
import pandas as pd
import plotly.offline as pyo
import plotly.graph_objs as go
# Set notebook mode to work in offline
pyo.init_notebook_mode()
import plotly.express as px
#get_ipython().run_line_magic('matplotlib', 'inline')
# Load dataframe
df = pd.DataFrame(loadjson('pred.json'))

# Remove rows where mbj_gap == 'na'
df = df[df['mbj_gap'] != 'na']

# Convert mbj_gap to numeric
df['mbj_gap'] = pd.to_numeric(df['mbj_gap'], errors='coerce')

# Drop rows with NaN after conversion
df = df.dropna(subset=['mbj_gap'])

# Scatter plot with hover info
fig_color = px.scatter(
    df,
    x="mbj_gap",
    y="bandgap",
    hover_name="formula",       # shows formula as main hover label
    hover_data=["jid", "opt_gap"]  # additional info on hover
)
fig_color.show()


# In[13]:


# === REVIEWER-READY FAILURE ANALYSIS FOR BAND GAPS ===
# Assumes columns: ['jid','bandgap','opt_gap','mbj_gap','formula']
# If your prediction/target are different, just set:
PRED_COL = "bandgap"   # model output (e.g., SlaKoNet)
TRUE_COL = "mbj_gap"   # reference (e.g., MBJ-DFT)

import re
import math
import pandas as pd
import numpy as np
from jarvis.db.jsonutils import loadjson

# ---------- 1) Load & clean ----------
df = pd.DataFrame(loadjson("pred.json"))
# drop rows where TRUE_COL is 'na' or non-numeric
df = df[df[TRUE_COL].astype(str).str.lower() != "na"].copy()
df[TRUE_COL] = pd.to_numeric(df[TRUE_COL], errors="coerce")
df[PRED_COL] = pd.to_numeric(df[PRED_COL], errors="coerce")
df = df.dropna(subset=[TRUE_COL, PRED_COL]).reset_index(drop=True)

# ---------- 2) Errors & basic metrics ----------
df["abs_err"] = (df[PRED_COL] - df[TRUE_COL]).abs()
df["signed_err"] = df[PRED_COL] - df[TRUE_COL]
mae = df["abs_err"].mean()
rmse = math.sqrt((df["signed_err"]**2).mean())
bias = df["signed_err"].mean()  # positive = overprediction, negative = underprediction

print(f"Global MAE = {mae:.3f} eV, RMSE = {rmse:.3f} eV, Bias = {bias:.3f} eV")

# ---------- 3) Define interpretable “classes” to characterize failures ----------
# (a) Target gap regime classes (you can tweak thresholds)
def gap_class(g):
    if g < 0.1: return "metallic (~0 eV)"
    if g < 1.0: return "small (0.1–1 eV)"
    if g < 3.0: return "medium (1–3 eV)"
    if g < 6.0: return "large (3–6 eV)"
    return "ultra-wide (>6 eV)"

df["true_class"] = df[TRUE_COL].apply(gap_class)

# (b) Number of elements in formula (crude parser)
def count_elements(formula):
    # counts capital letter blocks: SiC -> 2, BaNaB9O15 -> 4
    return len(re.findall(r"[A-Z][a-z]?", str(formula)))
df["n_elems"] = df["formula"].apply(count_elements)

# ---------- 4) Slice metrics to find weak spots ----------
def slice_report(groupby_col, top_k=10):
    rep = (df
           .groupby(groupby_col)
           .agg(n=("jid","count"),
                mae=("abs_err","mean"),
                rmse=("signed_err", lambda x: math.sqrt((x**2).mean())),
                bias=("signed_err","mean"),
                med_true=(TRUE_COL,"median"),
                med_pred=(PRED_COL,"median"))
           .reset_index()
           .sort_values("mae", ascending=False))
    print(f"\n=== Slice: {groupby_col} (sorted by MAE) ===")
    print(rep.head(top_k).to_string(index=False))
    return rep

rep_gap = slice_report("true_class")
rep_ne  = slice_report("n_elems")

# ---------- 5) Failure case mining ----------
# Define failures as top 5% absolute error, or abs_err > 1.0 eV (whichever is looser)
thr_pct = df["abs_err"].quantile(0.95)
thr_abs = 1.0  # eV
FAIL_THR = max(thr_pct, thr_abs)

failures = (df[df["abs_err"] >= FAIL_THR]
            .sort_values("abs_err", ascending=False)
            [["jid","formula",TRUE_COL,PRED_COL,"abs_err","signed_err","true_class","n_elems"]])
print(f"\nFailure threshold = {FAIL_THR:.3f} eV")
print(f"Failure cases: {len(failures)} / {len(df)} (={100*len(failures)/len(df):.1f}%)")
print(failures.head(20).to_string(index=False))

# Save full failure table for the supplement
failures.to_csv("failure_cases.csv", index=False)
print("Saved: failure_cases.csv")

# ---------- 6) Optional: diagnostics helpful for the rebuttal/appendix ----------
# (a) Per-class confusion for “metallicity” (misclassify metal vs non-metal)
metal_tol = 0.1
is_metal_true = df[TRUE_COL] < metal_tol
is_metal_pred = df[PRED_COL] < metal_tol
conf = pd.crosstab(is_metal_true.rename("true_metal"), is_metal_pred.rename("pred_metal"))
print("\nMetal vs non-metal confusion (tol 0.1 eV):\n", conf)

# (b) Systematic bias by true-gap bin (under/over-prediction pattern)
bias_by_bin = (df
               .assign(true_bin=pd.cut(df[TRUE_COL], bins=[0,0.1,1,3,6,df[TRUE_COL].max()], include_lowest=True))
               .groupby("true_bin")["signed_err"].agg(["mean","median","count"]).reset_index())
print("\nSigned error by true-gap bin:\n", bias_by_bin)

# (c) Export 2–3 “representative” failure examples for the main text
rep_examples = failures.groupby("true_class").head(3)  # top 3 per class
rep_examples.to_csv("representative_failures_for_main_text.csv", index=False)
print("Saved: representative_failures_for_main_text.csv")


# In[14]:


import plotly.express as px

# Residuals vs true gap (see heteroscedasticity or bias)
px.scatter(df, x=TRUE_COL, y="signed_err",
           labels={TRUE_COL:"True MBJ gap (eV)", "signed_err":"Prediction - True (eV)"},
           title="Residuals vs True Gap").update_traces(marker=dict(size=5)).show()

# Error by true-class (box)
px.box(df, x="true_class", y="abs_err",
       title="Absolute Error by True Gap Class",
       labels={"true_class":"True gap class","abs_err":"|Error| (eV)"}).show()

# Highlight worst failures on y=x plot
topK = failures.head(50)["jid"].tolist()
df["is_outlier"] = df["jid"].isin(topK)
fig = px.scatter(df, x=TRUE_COL, y=PRED_COL, color="is_outlier",
                 hover_data=["jid","formula","abs_err","true_class","n_elems"],
                 labels={TRUE_COL:"True (MBJ) gap (eV)", PRED_COL:"Predicted gap (eV)"},
                 title="True vs Predicted Band Gaps (outliers highlighted)")
fig.add_shape(type="line", x0=0, y0=0, x1=df[TRUE_COL].max(), y1=df[TRUE_COL].max(),
              line=dict(dash="dash"))
fig.show()


# In[15]:


len(df)


# In[16]:


from jarvis.db.figshare import data
dft_3d=pd.DataFrame(data('dft_3d'))


# In[17]:


# === REVIEWER-READY FAILURE ANALYSIS FOR BAND GAPS ===
# Assumes columns: ['jid','bandgap','opt_gap','mbj_gap','formula']
# If your prediction/target are different, just set:
PRED_COL = "optb88vdw_bandgap"   # model output (e.g., SlaKoNet)
TRUE_COL = "mbj_bandgap"   # reference (e.g., MBJ-DFT)

import re
import math
import pandas as pd
import numpy as np
from jarvis.db.jsonutils import loadjson

# ---------- 1) Load & clean ----------
df = dft_3d #pd.DataFrame(loadjson("pred.json"))
# drop rows where TRUE_COL is 'na' or non-numeric
df = df[df[TRUE_COL].astype(str).str.lower() != "na"].copy()
df[TRUE_COL] = pd.to_numeric(df[TRUE_COL], errors="coerce")
df[PRED_COL] = pd.to_numeric(df[PRED_COL], errors="coerce")
df = df.dropna(subset=[TRUE_COL, PRED_COL]).reset_index(drop=True)

# ---------- 2) Errors & basic metrics ----------
df["abs_err"] = (df[PRED_COL] - df[TRUE_COL]).abs()
df["signed_err"] = df[PRED_COL] - df[TRUE_COL]
mae = df["abs_err"].mean()
rmse = math.sqrt((df["signed_err"]**2).mean())
bias = df["signed_err"].mean()  # positive = overprediction, negative = underprediction

print(f"Global MAE = {mae:.3f} eV, RMSE = {rmse:.3f} eV, Bias = {bias:.3f} eV")

# ---------- 3) Define interpretable “classes” to characterize failures ----------
# (a) Target gap regime classes (you can tweak thresholds)
def gap_class(g):
    if g < 0.1: return "metallic (~0 eV)"
    if g < 1.0: return "small (0.1–1 eV)"
    if g < 3.0: return "medium (1–3 eV)"
    if g < 6.0: return "large (3–6 eV)"
    return "ultra-wide (>6 eV)"

df["true_class"] = df[TRUE_COL].apply(gap_class)

# (b) Number of elements in formula (crude parser)
def count_elements(formula):
    # counts capital letter blocks: SiC -> 2, BaNaB9O15 -> 4
    return len(re.findall(r"[A-Z][a-z]?", str(formula)))
df["n_elems"] = df["formula"].apply(count_elements)

# ---------- 4) Slice metrics to find weak spots ----------
def slice_report(groupby_col, top_k=10):
    rep = (df
           .groupby(groupby_col)
           .agg(n=("jid","count"),
                mae=("abs_err","mean"),
                rmse=("signed_err", lambda x: math.sqrt((x**2).mean())),
                bias=("signed_err","mean"),
                med_true=(TRUE_COL,"median"),
                med_pred=(PRED_COL,"median"))
           .reset_index()
           .sort_values("mae", ascending=False))
    print(f"\n=== Slice: {groupby_col} (sorted by MAE) ===")
    print(rep.head(top_k).to_string(index=False))
    return rep

rep_gap = slice_report("true_class")
rep_ne  = slice_report("n_elems")

# ---------- 5) Failure case mining ----------
# Define failures as top 5% absolute error, or abs_err > 1.0 eV (whichever is looser)
thr_pct = df["abs_err"].quantile(0.95)
thr_abs = 1.0  # eV
FAIL_THR = max(thr_pct, thr_abs)

failures = (df[df["abs_err"] >= FAIL_THR]
            .sort_values("abs_err", ascending=False)
            [["jid","formula",TRUE_COL,PRED_COL,"abs_err","signed_err","true_class","n_elems"]])
print(f"\nFailure threshold = {FAIL_THR:.3f} eV")
print(f"Failure cases: {len(failures)} / {len(df)} (={100*len(failures)/len(df):.1f}%)")
print(failures.head(20).to_string(index=False))

# Save full failure table for the supplement
failures.to_csv("failure_cases.csv", index=False)
print("Saved: failure_cases.csv")

# ---------- 6) Optional: diagnostics helpful for the rebuttal/appendix ----------
# (a) Per-class confusion for “metallicity” (misclassify metal vs non-metal)
metal_tol = 0.1
is_metal_true = df[TRUE_COL] < metal_tol
is_metal_pred = df[PRED_COL] < metal_tol
conf = pd.crosstab(is_metal_true.rename("true_metal"), is_metal_pred.rename("pred_metal"))
print("\nMetal vs non-metal confusion (tol 0.1 eV):\n", conf)

# (b) Systematic bias by true-gap bin (under/over-prediction pattern)
bias_by_bin = (df
               .assign(true_bin=pd.cut(df[TRUE_COL], bins=[0,0.1,1,3,6,df[TRUE_COL].max()], include_lowest=True))
               .groupby("true_bin")["signed_err"].agg(["mean","median","count"]).reset_index())
print("\nSigned error by true-gap bin:\n", bias_by_bin)

# (c) Export 2–3 “representative” failure examples for the main text
rep_examples = failures.groupby("true_class").head(3)  # top 3 per class
rep_examples.to_csv("representative_failures_for_main_text.csv", index=False)
print("Saved: representative_failures_for_main_text.csv")


# In[ ]:




