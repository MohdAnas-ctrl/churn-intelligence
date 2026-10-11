"""
Customer Churn Prediction - working end-to-end model
Pipeline: load -> clean -> EDA -> feature engineering -> train 4 models ->
evaluate (accuracy/precision/recall/F1/ROC-AUC) -> select best -> predict.

Usage:
    python churn_model.py                      # uses synthetic Telco-style data
    python churn_model.py --data telco.csv     # uses real IBM Telco CSV (Kaggle)
"""
import argparse, os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report)
warnings.filterwarnings("ignore")
OUT = "churn_outputs"; os.makedirs(OUT, exist_ok=True)
SEED = 42

# ---------------------------------------------------------------- 1. DATA
def make_synthetic(n=7043, seed=SEED):
    """Telco-style data with realistic churn drivers (contract, tenure, charges, add-ons)."""
    r = np.random.default_rng(seed)
    contract = r.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .21, .24])
    tenure = np.where(contract == "Month-to-month", r.integers(1, 40, n),
             np.where(contract == "One year", r.integers(6, 60, n), r.integers(12, 73, n)))
    internet = r.choice(["DSL", "Fiber optic", "No"], n, p=[.34, .44, .22])
    def addon(): return np.where(internet == "No", "No internet service",
                                 r.choice(["Yes", "No"], n, p=[.4, .6]))
    sec, backup, support = addon(), addon(), addon()
    stream = addon()
    base = np.select([internet == "DSL", internet == "Fiber optic"], [45, 80], 20)
    monthly = np.round(base + 5*(sec == "Yes") + 5*(backup == "Yes") + 5*(support == "Yes")
                       + 8*(stream == "Yes") + r.normal(0, 4, n), 2).clip(18.25)
    total = np.round(monthly * tenure * r.uniform(.95, 1.05, n), 2)
    df = pd.DataFrame({
        "CustomerID": [f"CUST-{1000+i}" for i in range(n)],
        "Gender": r.choice(["Male", "Female"], n),
        "SeniorCitizen": r.choice([0, 1], n, p=[.84, .16]),
        "Tenure": tenure, "Contract": contract, "InternetService": internet,
        "OnlineSecurity": sec, "OnlineBackup": backup, "TechSupport": support,
        "StreamingTV": stream,
        "PaymentMethod": r.choice(["Electronic check", "Mailed check", "Bank transfer",
                                   "Credit card"], n, p=[.34, .23, .22, .21]),
        "PaperlessBilling": r.choice(["Yes", "No"], n, p=[.59, .41]),
        "MonthlyCharges": monthly, "TotalCharges": total})
    z = (-0.6 + 1.5*(contract == "Month-to-month") - 1.2*(contract == "Two year")
         - 0.045*tenure + 0.018*(monthly - 65) + 0.5*(internet == "Fiber optic")
         - 0.5*(sec == "Yes") - 0.5*(support == "Yes") + 0.4*(df.SeniorCitizen == 1)
         + 0.4*(df.PaymentMethod == "Electronic check") + r.normal(0, .6, n))
    df["Churn"] = np.where(r.random(n) < 1/(1+np.exp(-z)), "Yes", "No")
    # inject a few blanks like the real Telco TotalCharges column
    df.loc[r.choice(n, 11, replace=False), "TotalCharges"] = np.nan
    return df

ap = argparse.ArgumentParser(); ap.add_argument("--data", default=None)
args = ap.parse_args()
df = pd.read_csv(args.data) if args.data else make_synthetic()
if args.data: df.columns = [c.strip() for c in df.columns]
print(f"Loaded {df.shape[0]} rows x {df.shape[1]} cols "
      f"({'real CSV' if args.data else 'synthetic Telco-style data'})")

# ---------------------------------------------------------------- 2. CLEANING
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.drop_duplicates()
df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["Tenure"])
df["Churn"] = (df["Churn"].astype(str).str.strip() == "Yes").astype(int)
ids = df["CustomerID"] if "CustomerID" in df else pd.Series(range(len(df)))
print("Churn rate: {:.2%}".format(df["Churn"].mean()))

# ---------------------------------------------------------------- 3. EDA PLOTS
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
df.groupby("Contract")["Churn"].mean().mul(100).plot.bar(ax=ax[0], color="#c0392b", title="Churn % by contract")
df.groupby(pd.cut(df.Tenure, [0, 12, 24, 48, 100], labels=["0-1y", "1-2y", "2-4y", "4y+"]))["Churn"] \
  .mean().mul(100).plot.bar(ax=ax[1], color="#e67e22", title="Churn % by tenure")
df.groupby(pd.cut(df.MonthlyCharges, [0, 35, 65, 90, 200]))["Churn"].mean().mul(100) \
  .plot.bar(ax=ax[2], color="#2980b9", title="Churn % by monthly charges")
plt.tight_layout(); plt.savefig(f"{OUT}/eda_churn_drivers.png", dpi=120); plt.close()

num = df.select_dtypes("number")
plt.figure(figsize=(6, 5)); plt.imshow(num.corr(), cmap="coolwarm", vmin=-1, vmax=1); plt.colorbar()
plt.xticks(range(len(num.columns)), num.columns, rotation=45, ha="right")
plt.yticks(range(len(num.columns)), num.columns); plt.title("Correlation heatmap")
plt.tight_layout(); plt.savefig(f"{OUT}/correlation_heatmap.png", dpi=120); plt.close()

# ---------------------------------------------------------------- 4. FEATURES
df["AvgMonthlySpend"] = df["TotalCharges"] / df["Tenure"].clip(lower=1)      # derived
df["AddOnCount"] = sum((df[c] == "Yes").astype(int) for c in
                       ["OnlineSecurity", "OnlineBackup", "TechSupport", "StreamingTV"] if c in df)
X = df.drop(columns=["Churn", "CustomerID"], errors="ignore"); y = df["Churn"]
cat_cols = X.select_dtypes("object").columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]
pre = ColumnTransformer([("num", StandardScaler(), num_cols),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

# ---------------------------------------------------------------- 5. MODELS
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=SEED),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows, fitted, probs = [], {}, {}
for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("clf", m)]).fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]; pred = (p >= .5).astype(int)
    rows.append({"Model": name, "Accuracy": accuracy_score(y_te, pred),
                 "Precision": precision_score(y_te, pred), "Recall": recall_score(y_te, pred),
                 "F1": f1_score(y_te, pred), "ROC-AUC": roc_auc_score(y_te, p),
                 "CV ROC-AUC": cross_val_score(Pipeline([("pre", pre), ("clf", m)]),
                                               X_tr, y_tr, cv=cv, scoring="roc_auc").mean()})
    fitted[name], probs[name] = pipe, p
res = pd.DataFrame(rows).set_index("Model").round(4).sort_values("ROC-AUC", ascending=False)
res.to_csv(f"{OUT}/model_comparison.csv")
print("\n=== Model comparison (test set) ===\n", res.to_string())

# ---------------------------------------------------------------- 6. BEST MODEL
best = res.index[0]; best_pipe = fitted[best]
print(f"\nBest model: {best}")
print(classification_report(y_te, (probs[best] >= .5).astype(int), target_names=["Not Churn", "Churn"]))

plt.figure(figsize=(6, 5))
for n_, p in probs.items():
    f, t, _ = roc_curve(y_te, p); plt.plot(f, t, label=f"{n_} (AUC={roc_auc_score(y_te, p):.3f})")
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curve comparison"); plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT}/roc_curves.png", dpi=120); plt.close()

ConfusionMatrixDisplay(confusion_matrix(y_te, (probs[best] >= .5).astype(int)),
                       display_labels=["Not Churn", "Churn"]).plot(cmap="Blues")
plt.title(f"Confusion matrix - {best}"); plt.savefig(f"{OUT}/confusion_matrix.png", dpi=120); plt.close()

# feature importance (use a tree model for interpretability)
imp_pipe = fitted["Gradient Boosting"]
names = imp_pipe.named_steps["pre"].get_feature_names_out()
imp = pd.Series(imp_pipe.named_steps["clf"].feature_importances_, index=names).sort_values()
imp.tail(12).plot.barh(figsize=(7, 5), color="#27ae60", title="Top churn drivers (Gradient Boosting)")
plt.tight_layout(); plt.savefig(f"{OUT}/feature_importance.png", dpi=120); plt.close()
print("Top drivers:", ", ".join(imp.tail(5).index[::-1]))

# ---------------------------------------------------------------- 7. SAVE + PREDICT
joblib.dump(best_pipe, f"{OUT}/churn_model.joblib")
def risk(p): return "High Risk" if p >= .6 else "Medium Risk" if p >= .3 else "Low Risk"
sample = X_te.head(5).copy(); sample.insert(0, "CustomerID", ids.loc[sample.index].values)
sample["ChurnProbability"] = best_pipe.predict_proba(X_te.head(5))[:, 1].round(2)
sample["Risk"] = sample["ChurnProbability"].map(risk)
sample["Actual"] = y_te.head(5).map({1: "Yes", 0: "No"}).values
sample[["CustomerID", "Contract", "Tenure", "MonthlyCharges", "ChurnProbability", "Risk", "Actual"]] \
    .to_csv(f"{OUT}/sample_predictions.csv", index=False)
print("\n=== Sample prediction output ===\n",
      sample[["CustomerID", "Contract", "Tenure", "MonthlyCharges", "ChurnProbability", "Risk", "Actual"]].to_string(index=False))
print(f"\nSaved model + charts to ./{OUT}/  (load with joblib.load('{OUT}/churn_model.joblib'))")
