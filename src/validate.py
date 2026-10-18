"""Validation of Part A reference checkpoints. Prints ALL CHECKPOINTS PASSED when all hold."""
import sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
o = pd.read_sql_query("SELECT * FROM orders", con)
c = pd.read_sql_query("SELECT * FROM customers", con)
comp = o[o.status == "completed"]
checks = {
    "orders rows == 12309": len(o) == 12309,
    "order_id unique": o.order_id.is_unique,
    "completed == 11110": (o.status == "completed").sum() == 11110,
    "cancelled == 1199": (o.status == "cancelled").sum() == 1199,
    "no pending rows": (o.status == "pending").sum() == 0,
    "0 missing order_amount": o.order_amount.isna().sum() == 0,
    "0 missing delivery (completed)": comp.delivery_minutes.isna().sum() == 0,
    "1199 missing delivery (cancelled)": o[o.status == "cancelled"].delivery_minutes.isna().sum() == 1199,
    "completed revenue == 2998668.80": round(comp.order_amount.sum(), 2) == 2998668.80,
    "six cities": c.city.nunique() == 6 and c.city.value_counts().to_dict() ==
        {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184},
    "genders 610/560/30": c.gender.value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30},
    "ages complete and in 18-60": c.age.notna().all() and c.age.between(18, 60).all(),
    "tables exist": {r[0] for r in con.execute("select name from sqlite_master where type='table'")} >= {"customers", "products", "orders"},
}
bad = [k for k, v in checks.items() if not v]
for k, v in checks.items():
    print(("PASS  " if v else "FAIL  ") + k)
assert not bad, f"failed: {bad}"
print("ALL CHECKPOINTS PASSED")
