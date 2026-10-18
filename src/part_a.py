"""Part A: audit, clean, store, validate-ready outputs, SQL analysis, pandas cross-check."""
import os, sqlite3
import numpy as np
import pandas as pd

RAW, CLEAN = "data/raw", "data/clean"
os.makedirs(CLEAN, exist_ok=True)
pd.set_option("display.width", 200)


def audit_table(name, df):
    print(f"\n== {name}: shape={df.shape}")
    rep = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
                        "missing_pct": (df.isna().mean() * 100).round(2)})
    print(rep.to_string())


# ---------------- load raw ----------------
cust_raw = pd.read_csv(f"{RAW}/customers.csv")
prod = pd.read_csv(f"{RAW}/products.csv")
ord_raw = pd.read_csv(f"{RAW}/orders.csv", keep_default_na=False, dtype=str)  # keep "N/A"/"error" visible
ord_na = pd.read_csv(f"{RAW}/orders.csv")                                      # blanks -> NaN for the missing-value report

print("##### AUDIT #####")
audit_table("customers", cust_raw)
audit_table("products", prod)
audit_table("orders", ord_na)

amt_num = pd.to_numeric(ord_raw["order_amount"], errors="coerce")
dm_num = pd.to_numeric(ord_raw["delivery_minutes"], errors="coerce")
age_num = pd.to_numeric(cust_raw["age"], errors="coerce")
genders = cust_raw["gender"].dropna()
audit = {
    "raw orders rows": len(ord_raw),
    "exact duplicate rows": int(ord_raw.duplicated().sum()),
    "pending rows": int((ord_raw["status"] == "pending").sum()),
    "distinct order_id": ord_raw["order_id"].nunique(),
    "non-numeric order_amount": int(amt_num.isna().sum()),
    "negative delivery_minutes": int((dm_num < 0).sum()),
    "blank delivery_minutes": int((ord_raw["delivery_minutes"] == "").sum()),
    "customers: blank age": int(age_num.isna().sum()),
    "customers: age outside 10-100": int(((age_num < 10) | (age_num > 100)).sum()),
    "customers: distinct raw city spellings": cust_raw["city"].nunique(),
    "customers: distinct non-blank gender spellings": genders.nunique(),
    "customers: blank gender": int(cust_raw["gender"].isna().sum()),
}
print("\n-- audit figures")
for k, v in audit.items():
    print(f"{k}: {v}")

# ---------------- clean customers ----------------
cust = cust_raw.copy()
cust["city"] = cust["city"].str.strip().str.title()
gmap = {"male": "male", "m": "male", "female": "female", "f": "female"}
cust["gender"] = cust["gender"].fillna("").astype(str).str.strip().str.lower().map(gmap).fillna("unknown")
cust["age"] = pd.to_numeric(cust["age"], errors="coerce")
n_bad_age = int(((cust["age"] < 10) | (cust["age"] > 100)).sum())
cust.loc[(cust["age"] < 10) | (cust["age"] > 100), "age"] = np.nan
n_age_filled = int(cust["age"].isna().sum())
cust["age"] = cust["age"].fillna(cust.groupby("city")["age"].transform("median"))
cust["age"] = cust["age"].round().astype(int)
cust["signup_date"] = pd.to_datetime(cust["signup_date"])
print("\n##### CLEAN CUSTOMERS #####")
print(cust["city"].value_counts().to_string())
print(cust["gender"].value_counts().to_string())
print("missing ages:", int(cust["age"].isna().sum()), "min/max age:", cust["age"].min(), cust["age"].max())
print("impossible ages set to missing:", n_bad_age, "| total ages imputed:", n_age_filled)

# ---------------- clean orders ----------------
o = pd.read_csv(f"{RAW}/orders.csv", keep_default_na=False, dtype=str)
o["order_ts"] = pd.to_datetime(o["order_ts"])
n0 = len(o)
o = o.drop_duplicates()
n_dup = n0 - len(o)
n1 = len(o)
o = o.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")
n_pending_dropped = n1 - len(o)
o["quantity"] = o["quantity"].astype(int)
o["discount_pct"] = o["discount_pct"].astype(int)
o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")
n_amt_missing = int(o["order_amount"].isna().sum())
o = o.merge(prod[["product_id", "category", "price_inr"]], on="product_id", how="left")
calc = (o["price_inr"] * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)
o["order_amount"] = o["order_amount"].fillna(calc)
o["delivery_minutes"] = pd.to_numeric(o["delivery_minutes"], errors="coerce")
n_neg = int((o["delivery_minutes"] < 0).sum())
o.loc[o["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
comp = o["status"] == "completed"
med = o.loc[comp].groupby("category")["delivery_minutes"].transform("median")
n_dm_fill = int((o.loc[comp, "delivery_minutes"].isna()).sum())
o.loc[comp, "delivery_minutes"] = o.loc[comp, "delivery_minutes"].fillna(med)
o = o.drop(columns=["category", "price_inr"]).sort_values("order_id").reset_index(drop=True)

print("\n##### CLEAN ORDERS #####")
print(f"exact duplicates removed: {n_dup}")
print(f"older pending records removed (keep latest order_ts per order_id): {n_pending_dropped}")
print(f"unparseable amounts filled from product price: {n_amt_missing}")
print(f"negative delivery times set to missing: {n_neg}")
print(f"missing delivery times of completed orders filled with category median: {n_dm_fill} (incl. the {n_neg} negatives)")
print("rows:", len(o), "| status:", o["status"].value_counts().to_dict())
print("missing order_amount:", int(o["order_amount"].isna().sum()))
print("missing delivery completed:", int(o.loc[o.status == "completed", "delivery_minutes"].isna().sum()),
      "| cancelled:", int(o.loc[o.status == "cancelled", "delivery_minutes"].isna().sum()))
print("completed revenue INR:", round(o.loc[o.status == "completed", "order_amount"].sum(), 2))

# ---------------- store ----------------
o_out = o.copy()
o_out["order_ts"] = o_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
c_out = cust.copy()
c_out["signup_date"] = c_out["signup_date"].dt.strftime("%Y-%m-%d")
c_out.to_csv(f"{CLEAN}/customers.csv", index=False)
prod.to_csv(f"{CLEAN}/products.csv", index=False)
o_out.to_csv(f"{CLEAN}/orders.csv", index=False)
if os.path.exists("data/instamart.db"):
    os.remove("data/instamart.db")
con = sqlite3.connect("data/instamart.db")
c_out.to_sql("customers", con, index=False)
prod.to_sql("products", con, index=False)
o_out.to_sql("orders", con, index=False)
con.commit()

# ---------------- SQL analysis ----------------
print("\n##### SQL #####")
sqltext = open("sql/queries.sql").read()
queries = {}
for block in sqltext.split("-- Q")[1:]:
    qid, body = block.split("\n", 1)
    queries["Q" + qid.split()[0].rstrip(":")] = body.strip().rstrip(";")
res = {k: pd.read_sql_query(v, con) for k, v in queries.items()}
q1, q2, q3, q4, q5, q6, q7 = (res[f"Q{i}"] for i in range(1, 8))
print("Q1 customers:", len(q1))
print("Q2 rows:", len(q2), "sum:", round(q2["revenue"].sum(), 2))
print(q2.sort_values("revenue", ascending=False).head(1).to_string(index=False))
print("Q3 products above average price:", len(q3))
print("Q4 customers with no completed order:", len(q4))
print("Q5 customers above average:", len(q5), "| avg:", round(q5["avg_spend"].iloc[0], 2))
top = q6[q6["row_num"] == 1][["city", "customer_id", "total_spend"]]
print("Q6 top customer per city:\n", top.to_string(index=False))
print("Q6 rows with RANK != ROW_NUMBER:", int((q6["rnk"] != q6["row_num"]).sum()))
print("Q7 rows:", len(q7), "| last cumulative:", round(q7["cumulative_revenue"].iloc[-1], 2))

# ---------------- pandas cross-check ----------------
print("\n##### PANDAS #####")
full = pd.read_sql_query("""SELECT o.order_amount, o.discount_pct, o.quantity, o.delivery_minutes, c.city, p.category
 FROM orders o JOIN customers c USING(customer_id) JOIN products p USING(product_id) WHERE o.status='completed'""", con)
pv = pd.pivot_table(full, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True, margins_name="Total")
print(pv.round(2).to_string())
print("pivot grand total:", round(pv.loc["Total", "Total"], 2), "== SQL total:", round(q2["revenue"].sum(), 2))
g = full.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"), n=("order_amount", "count")).round(2)
print(g.to_string())
x = full["order_amount"]
q1_, q3_ = x.quantile(.25), x.quantile(.75)
iqr = q3_ - q1_
lo, hi = q1_ - 1.5 * iqr, q3_ + 1.5 * iqr
print(f"mean {x.mean():.2f} median {x.median():.2f} skew {x.skew():.2f} Q1 {q1_:.2f} Q3 {q3_:.2f} "
      f"fences [{lo:.2f}, {hi:.2f}] below {int((x < lo).sum())} above {int((x > hi).sum())}")
corr = full[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print(corr.round(3).to_string())
c2 = corr.where(~np.eye(len(corr), dtype=bool)).abs().stack()
print("largest |corr| pair:", c2.idxmax(), round(corr.loc[c2.idxmax()], 3))
con.close()
