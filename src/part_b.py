"""Part B: RFM, K-means segments, PCA, lapse prediction (no leakage), persistence, retention list."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, silhouette_score, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample

os.makedirs("outputs", exist_ok=True)
pd.set_option("display.width", 200)

orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
cust = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
prod = pd.read_csv("data/clean/products.csv")
orders = orders.merge(prod[["product_id", "category"]], on="product_id", how="left")
comp = orders[orders.status == "completed"]

# ------------------------------------------------------------------ 1. RFM
SNAP = pd.Timestamp("2025-10-01 00:00:00")
rfm = comp.groupby("customer_id").agg(last=("order_ts", "max"), frequency=("order_id", "nunique"),
                                      monetary=("order_amount", "sum")).reset_index()
rfm["recency"] = (SNAP - rfm["last"]).dt.days
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]
print("##### RFM #####")
print("customers:", len(rfm))
print(f"recency median {rfm.recency.median()} min {rfm.recency.min()} max {rfm.recency.max()}")
print(f"frequency mean {rfm.frequency.mean():.2f} max {rfm.frequency.max()}")
print(f"monetary mean {rfm.monetary.mean():.2f} max {rfm.monetary.max():.2f}")

# ------------------------------------------------------------------ 2. K-means
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append((k, km.inertia_, silhouette_score(X, km.labels_)))
ktab = pd.DataFrame(rows, columns=["K", "inertia", "silhouette"])
print("\n##### K-MEANS TABLE #####")
print(ktab.round(3).to_string(index=False))
K = 4
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_

# ------------------------------------------------------------------ 3. profile, name, PCA
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"), recency=("recency", "mean"),
                                     frequency=("frequency", "mean"), monetary=("monetary", "mean")).round(2)
print("\n##### SEGMENT PROFILE #####")
print(prof.to_string())
premium_id = prof["monetary"].idxmax()
rest = prof.drop(index=premium_id)
loyal_id = rest["frequency"].idxmax()
rest2 = rest.drop(index=loyal_id)
dormant_id = rest2["recency"].idxmax()
names = {premium_id: "Premium Big Spenders", loyal_id: "Loyal Regulars", dormant_id: "Dormant / At-Risk"}
for sid in prof.index:
    names.setdefault(sid, "Occasional Buyers")
rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium_id
print({int(k): v for k, v in names.items()}, "premium segment id:", premium_id)
pca = PCA(n_components=2).fit(X)
print("PCA explained_variance_ratio_:", pca.explained_variance_ratio_.round(3), "total", pca.explained_variance_ratio_.sum().round(3))
rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]] \
    .to_csv("outputs/customer_segments.csv", index=False)

# ------------------------------------------------------------------ 4. lapse problem
CUT = pd.Timestamp("2025-09-01")
before = orders[orders.order_ts < CUT]                      # ONLY orders before cutoff are used for features
comp_b = before[before.status == "completed"]
cohort_ids = comp_b.customer_id.unique()
g = comp_b.groupby("customer_id")
feat = pd.DataFrame({
    "recency": (CUT - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum(),
    "avg_discount": g["discount_pct"].mean(),
    "avg_delivery": g["delivery_minutes"].mean(),
    "fav_category": g["category"].agg(lambda s: s.value_counts().sort_index().idxmax()),
})
final = before[before.status.isin(["completed", "cancelled"])].groupby("customer_id")["status"] \
    .apply(lambda s: (s == "cancelled").mean())
feat["cancel_rate"] = final.reindex(feat.index).fillna(0)
feat = feat.reset_index().merge(cust[["customer_id", "city", "age", "gender", "signup_date"]], on="customer_id")
feat["tenure_days"] = (CUT - feat["signup_date"]).dt.days
feat = feat.drop(columns="signup_date")
sept = orders[(orders.status == "completed") & (orders.order_ts >= CUT) & (orders.order_ts < pd.Timestamp("2025-10-01"))]
active = set(sept.customer_id)
feat["lapsed"] = (~feat.customer_id.isin(active)).astype(int)

NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
CAT = ["fav_category", "city", "gender"]
assert not feat[NUM + CAT].isna().any().any(), "missing values in features"
print("\n##### LAPSE COHORT #####")
print(f"cohort {len(feat)} | lapsed {feat.lapsed.sum()} ({feat.lapsed.mean()*100:.2f}%)")
y = feat["lapsed"]
Xf = feat[["customer_id"] + NUM + CAT]
Xtr, Xte, ytr, yte = train_test_split(Xf, y, test_size=0.2, stratify=y, random_state=42)
print("train", len(Xtr), "test", len(Xte))
ids_te = Xte["customer_id"].values
Xtr, Xte = Xtr.drop(columns="customer_id"), Xte.drop(columns="customer_id")
base_pred = np.zeros(len(yte), dtype=int)
print(f"baseline (always not-lapsed): accuracy {accuracy_score(yte, base_pred)*100:.1f}% lapsed-F1 {f1_score(yte, base_pred, zero_division=0):.2f}")


def prep():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])


# ------------------------------------------------------------------ 5. preprocess without leakage
chk = prep().fit(Xtr)
print("\n##### SCALER CHECK #####")
fi = NUM.index("frequency")
print(f"scaler mean_ of frequency (train only) {chk.named_transformers_['num'].mean_[fi]:.4f} | "
      f"full-cohort mean {feat.frequency.mean():.4f}")
print("missing values in feature columns:", int(Xf[NUM + CAT].isna().sum().sum()))

# ------------------------------------------------------------------ 6. train, tune, evaluate
lr_pipe = Pipeline([("prep", prep()), ("m", LogisticRegression(max_iter=1000))])
lr_gs = GridSearchCV(lr_pipe, {"m__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(Xtr, ytr)
rf_pipe = Pipeline([("prep", prep()), ("m", RandomForestClassifier(n_estimators=100, random_state=42))])
rf_gs = GridSearchCV(rf_pipe, {"m__max_depth": [3, 5, 8], "m__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1").fit(Xtr, ytr)
print("\n##### TUNING #####")
print("LR best:", lr_gs.best_params_, "cv F1", round(lr_gs.best_score_, 4))
print("RF best:", rf_gs.best_params_, "cv F1", round(rf_gs.best_score_, 4))


def report(name, model):
    p = model.predict(Xte)
    print(f"{name}: confusion [[TN FP],[FN TP]] = {confusion_matrix(yte, p).tolist()} "
          f"precision {precision_score(yte, p, zero_division=0):.3f} recall {recall_score(yte, p, zero_division=0):.3f} "
          f"F1 {f1_score(yte, p, zero_division=0):.3f}")
    return p


print("\n##### TEST EVALUATION #####")
p_lr = report("LogReg", lr_gs.best_estimator_)
p_rf = report("RandForest", rf_gs.best_estimator_)
best_model = rf_gs.best_estimator_ if rf_gs.best_score_ >= lr_gs.best_score_ else lr_gs.best_estimator_
best_name = "RandomForest" if best_model is rf_gs.best_estimator_ else "LogisticRegression"
print("selected (by cross-validated F1):", best_name)
f1_sel = f1_score(yte, best_model.predict(Xte), zero_division=0)
assert f1_sel >= 0.55, f1_sel

# ------------------------------------------------------------------ 7. class balance experiment (train only)
tr = Xtr.copy(); tr["y"] = ytr.values
maj, mino = tr[tr.y == 0], tr[tr.y == 1]
extra = resample(mino, replace=True, n_samples=len(maj) - len(mino), random_state=42)
bal = pd.concat([tr, extra])
print("\n##### BALANCE #####")
print("balanced train counts:", bal.y.value_counts().to_dict())
from sklearn.base import clone
lr_bal = clone(lr_gs.best_estimator_).fit(bal.drop(columns="y"), bal.y)
print("-- un-balanced LR:"); report("LR plain   ", lr_gs.best_estimator_)
print("-- balanced LR:"); report("LR balanced", lr_bal)

# ------------------------------------------------------------------ 8. interpret, persist, use
print("\n##### INTERPRETATION #####")
names_out = rf_gs.best_estimator_.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf_gs.best_estimator_.named_steps["m"].feature_importances_, index=names_out).sort_values(ascending=False)
print("RF top-5 importances:\n", imp.head(5).round(4).to_string())
coefs = pd.Series(lr_gs.best_estimator_.named_steps["m"].coef_[0],
                  index=lr_gs.best_estimator_.named_steps["prep"].get_feature_names_out())
for f in ["num__recency", "num__frequency", "num__monetary", "num__avg_discount", "num__cancel_rate"]:
    print(f"LR coef {f}: {coefs[f]:+.3f}")
joblib.dump(best_model, "outputs/lapse_model.joblib")
re = joblib.load("outputs/lapse_model.joblib")
same = np.array_equal(re.predict(Xte), best_model.predict(Xte))
print("reloaded model reproduces test predictions exactly:", same)
assert same

proba = best_model.predict_proba(Xte)[:, 1]
res = pd.DataFrame({"customer_id": ids_te, "lapse_probability": proba, "actually_lapsed": yte.values})
prem = set(rfm.loc[rfm.is_premium, "customer_id"])
ret = res[res.customer_id.isin(prem)].sort_values("lapse_probability", ascending=False).head(10)
ret["lapse_probability"] = ret["lapse_probability"].round(4)
ret.to_csv("outputs/retention_list.csv", index=False)
print("\n##### RETENTION LIST #####")
print("premium customers in test set:", res.customer_id.isin(prem).sum(), "| rows written:", len(ret),
      "| truly lapsed:", int(ret.actually_lapsed.sum()))
