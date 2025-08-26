import warnings
warnings.filterwarnings("ignore")

import pandas as pd, numpy as np, matplotlib.pyplot as plt
from math import pi
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ---------------- CONFIG ----------------
CSV = "workload-synthetic-bf-patched.csv"
BF_DATE = "2025-08-08"

INTERVAL_MIN = 15           # forecast interval (15-min here)
# REQ_PER_VM_PER_MIN = 120.0  # capacity of ONE VM at 100% util
REQ_PER_VM_PER_MIN = 60.0  # capacity of ONE VM at 100% util
TARGET_UTIL = 0.6           # target utilization (e.g. 60%)
SAFETY_BUFFER = 0.2         # extra headroom (20%)
MIN_VMS, MAX_VMS = 2, 200   # scaling bounds
# ----------------------------------------


# ============ Load & Prepare ============

df = pd.read_csv(CSV, parse_dates=["Timestamp"]).set_index("Timestamp")
df = df[["Requests","BlackFriday_Flag"]]

# Resample to 15-min
df15 = df.resample("15min").mean().ffill().bfill()

# Fourier features (daily seasonality)
period = 96  # 96 intervals per day (15-min)
K = 5
t = np.arange(len(df15))
fourier = {f"sin{k}": np.sin(2*pi*k*t/period) for k in range(1,K+1)}
fourier |= {f"cos{k}": np.cos(2*pi*k*t/period) for k in range(1,K+1)}
F = pd.DataFrame(fourier, index=df15.index)
exog_all = pd.concat([df15[["BlackFriday_Flag"]], F], axis=1)

# Split: train = before BF, test = BF day
bf_day = pd.to_datetime(BF_DATE)
train = df15.loc[: bf_day - pd.Timedelta(minutes=15)]
test  = df15.loc[bf_day : bf_day + pd.Timedelta(hours=23, minutes=45)]

y_train, y_test = train["Requests"], test["Requests"]
X_train, X_test = exog_all.loc[train.index], exog_all.loc[test.index]

print("Train:", y_train.index.min(), "→", y_train.index.max())
print("Test :", y_test.index.min(),  "→", y_test.index.max())
print("TRAIN BF-flag minutes:", int((X_train['BlackFriday_Flag']==1).sum()))
print("TEST  BF-flag minutes:", int((X_test['BlackFriday_Flag']==1).sum()))

# ============ Fit SARIMAX ============
y_train_log = np.log1p(y_train)

model = SARIMAX(
    y_train_log,
    exog=X_train,
    order=(1,1,0),
    seasonal_order=(0,0,0,0),
    simple_differencing=False,  # important!
    concentrate_scale=True,
    enforce_stationarity=True,
    enforce_invertibility=True
)
res = model.fit(method="lbfgs", maxiter=50, disp=False)

print("\nParams snapshot:\n", res.params.head(12))

# ============ Forecast ============
fc = res.get_forecast(steps=len(test), exog=X_test)
pred_log = fc.predicted_mean
ci_log = fc.conf_int()

pred = np.expm1(pred_log)
ci   = np.expm1(ci_log)

# ============ Metrics ============
rmse = float(np.sqrt(np.mean((y_test.values - pred.values)**2)))
mape = float(np.mean(np.abs((y_test.values - pred.values) / np.clip(y_test.values,1e-6,None)))) * 100
print(f"\nRMSE (BF day): {rmse:.2f} | MAPE (BF day): {mape:.2f}%")

# ============ Capacity Planning ============
# Convert forecast (requests per 15-min) to requests/min
# req_per_min = pred / INTERVAL_MIN
req_per_min = pred

print("Peak forecasted req/min:", float(req_per_min.max()))
# Effective capacity per VM
effective_capacity = REQ_PER_VM_PER_MIN * TARGET_UTIL * (1+SAFETY_BUFFER)

print("Effective VM capacity (req/min):", effective_capacity)

# Required VMs
vms_raw = np.ceil(req_per_min / effective_capacity).astype(int)
vms_bounded = np.clip(vms_raw, MIN_VMS, MAX_VMS)

capacity_df = pd.DataFrame({
    "Timestamp": test.index,
    "ForecastReqs": pred.values,
    "ReqPerMin": req_per_min.values,
    "VMs_Required": vms_bounded.values
})
capacity_df.to_csv("capacity_timeseries.csv", index=False)

# Build a compact scale plan (only changes)
plan = capacity_df.loc[capacity_df["VMs_Required"].ne(capacity_df["VMs_Required"].shift(1))]
plan = plan[["Timestamp","VMs_Required"]].reset_index(drop=True)
plan.to_csv("capacity_scale_plan.csv", index=False)

print("\n=== Predictive Scaling Plan ===")
print(plan.head(10))
print("... saved full timeseries and plan to CSV.")

# ============ Plot ============

context_start = test.index.min() - pd.Timedelta(days=1)
context = df15.loc[context_start:test.index.max()]

plt.figure(figsize=(13,6))
plt.plot(context.index, context["Requests"], label="Actual", color="black")
plt.plot(test.index, pred, label="Forecast", color="red")
plt.fill_between(test.index, ci.iloc[:,0], ci.iloc[:,1], alpha=0.25, label="CI")
plt.axvline(test.index[0], color="gray", ls="--", label="BF forecast start")
plt.title("SARIMAX Black Friday Forecast + Predictive Scaling")
plt.xlabel("Time"); plt.ylabel("Requests"); plt.legend()
plt.tight_layout()
plt.show()
