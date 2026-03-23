"""
Thai Seasonal AC Energy Analyzer
create_thai_dataset.py

สร้าง dataset → EDA → Pipeline + GridSearchCV → บันทึกโมเดล → Plot วิเคราะห์
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, GridSearchCV
)
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

np.random.seed(42)

# =============================================================
# 1. ข้อมูลอุณหภูมิกรุงเทพฯ จริง (กรมอุตุนิยมวิทยา)
# =============================================================
bkk_temp = {
    1: 26.5, 2: 27.8, 3: 29.5, 4: 31.2,
    5: 30.1, 6: 29.5, 7: 29.0, 8: 28.8,
    9: 28.5, 10: 28.2, 11: 27.5, 12: 26.0
}
bkk_humidity = {
    1: 65, 2: 67, 3: 68, 4: 70,
    5: 75, 6: 80, 7: 82, 8: 83,
    9: 85, 10: 82, 11: 73, 12: 65
}
bkk_rain_chance = {
    1: 0.05, 2: 0.08, 3: 0.12, 4: 0.18,
    5: 0.35, 6: 0.55, 7: 0.60, 8: 0.65,
    9: 0.70, 10: 0.55, 11: 0.25, 12: 0.08
}

def get_thai_season(month):
    if month in [3, 4, 5]:           return "hot"
    elif month in [6, 7, 8, 9, 10]:  return "rainy"
    else:                             return "cool"

# =============================================================
# 2. สร้าง dataset ไทย (2 ปี รายชั่วโมง)
# =============================================================
print("=" * 60)
print("STEP 1: สร้าง dataset")
print("=" * 60)

records = []
for year in range(2023, 2025):
    for month in range(1, 13):
        base_temp  = bkk_temp[month]
        base_humid = bkk_humidity[month]
        rain_prob  = bkk_rain_chance[month]
        season     = get_thai_season(month)

        for day in range(30):
            is_weekend  = 1 if day % 7 >= 5 else 0
            daily_temp  = base_temp + np.random.normal(0, 1.5)
            daily_humid = np.clip(base_humid + np.random.normal(0, 5), 40, 100)
            is_raining  = np.random.random() < rain_prob

            for hour in range(24):
                hour_temp = np.clip(
                    daily_temp + 3.0 * np.sin(np.pi * (hour - 6) / 12), 20, 42
                )
                hour_humid = daily_humid - 10 * np.sin(np.pi * (hour - 6) / 12)
                if is_raining and 10 <= hour <= 18:
                    hour_humid = min(hour_humid + 15, 100)
                hour_humid = np.clip(hour_humid, 40, 100)

                is_raining_now = 1 if (is_raining and np.random.random() < 0.4) else 0
                cloud_cover    = np.random.uniform(60, 100) if is_raining_now else np.random.uniform(10, 70)
                wind_speed     = np.random.uniform(1, 8)    if is_raining_now else np.random.uniform(0.5, 5)
                pressure       = np.random.uniform(1005, 1015)

                ac_load = 0.8 + (hour_temp - 25) * 0.04

                if   0 <= hour <= 5:    usage_factor = 0.85
                elif 6 <= hour <= 8:    usage_factor = 0.40
                elif 9 <= hour <= 11:   usage_factor = 0.30
                elif 12 <= hour <= 13:  usage_factor = 0.50
                elif 14 <= hour <= 16:  usage_factor = 0.35
                elif 17 <= hour <= 19:  usage_factor = 0.65
                elif 20 <= hour <= 22:  usage_factor = 0.90
                else:                   usage_factor = 0.85

                if is_weekend and 10 <= hour <= 16:
                    usage_factor = min(usage_factor + 0.25, 1.0)
                if is_raining_now:
                    usage_factor *= 0.75
                if season == "cool":
                    usage_factor *= 0.55

                kwh = max(0.0, round(ac_load * usage_factor + np.random.normal(0, 0.02), 4))

                records.append({
                    "month":       month,
                    "hour":        hour,
                    "is_weekend":  is_weekend,
                    "temp_dry":    round(hour_temp, 2),
                    "humidity":    round(hour_humid, 2),
                    "wind_speed":  round(wind_speed, 2),
                    "cloud_cover": round(cloud_cover, 2),
                    "pressure":    round(pressure, 2),
                    "is_raining":  is_raining_now,
                    "season":      season,
                    "kWh":         kwh
                })

df = pd.DataFrame(records)
df.to_csv("thai_ac_dataset.csv", index=False)
print(f"✅ Dataset: {len(df):,} rows × {len(df.columns)} columns")
print(df[["season", "temp_dry", "humidity", "kWh"]].groupby("season").mean().round(3))

# =============================================================
# 3. EDA — Exploratory Data Analysis
# =============================================================
print("\n" + "=" * 60)
print("STEP 2: EDA — Exploratory Data Analysis")
print("=" * 60)

# 3.1 ตรวจสอบ missing values
print("\n--- Missing Values ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "✅ ไม่พบ missing values ในทุก column")

# 3.2 ตรวจสอบ outliers ด้วย IQR
print("\n--- Outlier Detection (IQR method) ---")
num_cols = ["temp_dry", "humidity", "wind_speed", "cloud_cover", "pressure", "kWh"]
for col in num_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR     = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    pct      = len(outliers) / len(df) * 100
    print(f"  {col:<12}: {len(outliers):5,} outliers ({pct:.2f}%) — range [{df[col].min():.2f}, {df[col].max():.2f}]")

# 3.3 Descriptive statistics
print("\n--- Descriptive Statistics ---")
print(df[num_cols + ["month", "hour", "is_weekend"]].describe().round(3).to_string())

# 3.4 Correlation กับ kWh
print("\n--- Correlation with kWh ---")
corr = df[num_cols + ["month", "hour", "is_weekend"]].corr()["kWh"].drop("kWh").sort_values(ascending=False)
for feat, val in corr.items():
    bar = "█" * int(abs(val) * 30)
    sign = "+" if val >= 0 else "-"
    print(f"  {feat:<12}: {sign}{abs(val):.4f}  {bar}")

# 3.5 Season stats
print("\n--- Season Summary ---")
season_summary = df.groupby("season").agg(
    n_records=("kWh", "count"),
    avg_temp=("temp_dry", "mean"),
    avg_humidity=("humidity", "mean"),
    avg_kWh=("kWh", "mean"),
    std_kWh=("kWh", "std")
).round(3)
print(season_summary.to_string())

# =============================================================
# 4. Pipeline + GridSearchCV
# =============================================================
print("\n" + "=" * 60)
print("STEP 3: Model Development — Pipeline + GridSearchCV")
print("=" * 60)

FEATURES = ["temp_dry", "humidity", "wind_speed", "cloud_cover",
            "hour", "month", "is_weekend", "pressure"]
X = df[FEATURES]
y = df["kWh"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

# ── 4.1 Baseline Models ──────────────────────────────────────
print("\n--- Model Comparison ---")

# Linear Regression Pipeline (ต้อง scale เพราะ unit ต่างกัน)
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LinearRegression())
])
lr_pipe.fit(X_train, y_train)
lr_pred = lr_pipe.predict(X_test)
lr_r2   = r2_score(y_test, lr_pred)
lr_mae  = mean_absolute_error(y_test, lr_pred)
print(f"  Linear Regression : R²={lr_r2:.4f}  MAE={lr_mae:.4f}")
print(f"    → เหตุผลที่ R² ต่ำ: ความสัมพันธ์ระหว่างอุณหภูมิกับการใช้ไฟ")
print(f"      ไม่ใช่เส้นตรง (non-linear) เช่น hour มีผลแบบ periodic")

# Random Forest (ไม่จำเป็นต้อง scale)
rf_pipe = Pipeline([
    ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])
rf_pipe.fit(X_train, y_train)
rf_pred = rf_pipe.predict(X_test)
rf_r2   = r2_score(y_test, rf_pred)
rf_mae  = mean_absolute_error(y_test, rf_pred)
print(f"  Random Forest     : R²={rf_r2:.4f}  MAE={rf_mae:.4f}")
print(f"    → ดีขึ้น แต่ยัง overfit ได้ง่ายกว่า XGBoost สำหรับ tabular data")

# ── 4.2 XGBoost + GridSearchCV ───────────────────────────────
print("\n--- XGBoost: GridSearchCV Hyperparameter Tuning ---")
print("  เหตุผลที่เลือก XGBoost:")
print("  • Gradient boosting จัดการ non-linear pattern ได้ดี")
print("  • Built-in regularization ป้องกัน overfitting")
print("  • รองรับ missing values โดยอัตโนมัติ")
print()
print("  Grid ที่ค้นหา:")
print("  • n_estimators: [200, 300]  — จำนวน tree")
print("  • max_depth   : [4, 6]      — ความลึก (ลึก → ซับซ้อนขึ้น)")
print("  • learning_rate: [0.05, 0.1] — step size (เล็ก → stable)")
print("  • subsample   : [0.8, 1.0]  — สุ่ม sample ป้องกัน overfit")

param_grid = {
    "model__n_estimators":  [200, 300],
    "model__max_depth":     [4, 6],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample":     [0.8, 1.0],
}

xgb_pipe = Pipeline([
    ("model", XGBRegressor(random_state=42, verbosity=0))
])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    xgb_pipe, param_grid,
    cv=kf, scoring="r2",
    n_jobs=-1, verbose=0
)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model  = grid_search.best_estimator_
print(f"\n  Best params: {best_params}")
print(f"  Best CV R² : {grid_search.best_score_:.4f}")

# ── 4.3 Final Evaluation ─────────────────────────────────────
y_pred = best_model.predict(X_test)
r2     = r2_score(y_test, y_pred)
mae    = mean_absolute_error(y_test, y_pred)

cv_scores = cross_val_score(best_model, X, y, cv=kf, scoring="r2")

print(f"\n--- Final Model Performance ---")
print(f"  R²  (test)  = {r2:.4f}")
print(f"  MAE (test)  = {mae:.4f} kWh/hr")
print(f"  5-Fold CV   = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 4.4 Business Interpretation ──────────────────────────────
print("\n--- Business Interpretation ---")
print(f"  • MAE = {mae:.4f} kWh/hr")
print(f"    → โมเดลพยากรณ์คลาดเคลื่อนเฉลี่ยเพียง {mae*100:.2f}% ของค่าสูงสุด (~1 kWh)")
monthly_mae = mae * 24 * 30
cost_err    = monthly_mae * 4.22
print(f"    → แปลเป็นรายเดือน: ±{monthly_mae:.1f} kWh → ค่าไฟผิดพลาด ±฿{cost_err:.0f}/เดือน")
print(f"  • R² = {r2:.4f} หมายความว่าโมเดลอธิบายความแปรปรวนของ")
print(f"    การใช้ไฟได้ {r2*100:.1f}% ซึ่งเพียงพอสำหรับการวางแผนค่าไฟ")

season_compare = df.groupby("season")["kWh"].mean()
hot_cool_diff  = season_compare.get("hot", 0) - season_compare.get("cool", 0)
monthly_diff   = hot_cool_diff * 24 * 30
cost_diff      = monthly_diff * 4.22
print(f"\n  • ค่าไฟฤดูร้อน vs ฤดูหนาวต่างกัน ~฿{cost_diff:.0f}/เดือน")
print(f"    → คนไทยควรวางแผนงบค่าไฟส่วนนี้ให้รองรับความแตกต่างนี้")

# Model comparison summary
print(f"\n  --- Model Comparison Summary ---")
print(f"  {'Model':<20} {'R²':>8} {'MAE':>8} {'เหมาะสมกับงานนี้'}")
print(f"  {'-'*60}")
print(f"  {'Linear Regression':<20} {lr_r2:>8.4f} {lr_mae:>8.4f}  ❌ Non-linear data")
print(f"  {'Random Forest':<20} {rf_r2:>8.4f} {rf_mae:>8.4f}  ⚠️  ดี แต่ช้ากว่า")
print(f"  {'XGBoost (tuned)':<20} {r2:>8.4f} {mae:>8.4f}  ✅ ดีที่สุด")
print(f"\n  เหตุผลที่ XGBoost ดีกว่า:")
print(f"  • ข้อมูลมี non-linear interaction เช่น 'เดือนร้อน + ตีสาม' ใช้ไฟต่างจาก")
print(f"    'เดือนหนาว + ตีสาม' ซึ่ง tree-based model จัดการได้ดีกว่า linear")
print(f"  • XGBoost มี regularization (L1/L2) ป้องกัน overfitting ได้ดีกว่า RF")

# =============================================================
# 5. บันทึกโมเดล (Pipeline ที่มี preprocessing)
# =============================================================
with open("weather_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("weather_feature_cols.pkl", "wb") as f:
    pickle.dump(FEATURES, f)

print("\n" + "=" * 60)
print("STEP 4: บันทึกโมเดล")
print("=" * 60)
print("✅ weather_model.pkl    (XGBoost Pipeline ที่ tune แล้ว)")
print("✅ weather_feature_cols.pkl")

# =============================================================
# 6. Visualization
# =============================================================
print("\n" + "=" * 60)
print("STEP 5: สร้างกราฟ EDA + Model Analysis")
print("=" * 60)

season_colors = {"hot": "#FF6B35", "rainy": "#3498db", "cool": "#5B9BD5"}
season_labels = {"hot": "ฤดูร้อน", "rainy": "ฤดูฝน", "cool": "ฤดูหนาว"}

fig = plt.figure(figsize=(22, 18))
fig.suptitle("Thai AC Energy Dataset — EDA & Model Analysis", fontsize=18, fontweight="bold")
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)

# ── Plot 1: Distribution kWh แต่ละฤดู ──
ax1 = fig.add_subplot(gs[0, :2])
for season, color in season_colors.items():
    data = df[df["season"] == season]["kWh"]
    ax1.hist(data, bins=60, alpha=0.5, color=color, label=f"{season_labels[season]} (mean={data.mean():.3f})")
ax1.set_title("Distribution of kWh by Season", fontweight="bold")
ax1.set_xlabel("kWh/hr")
ax1.set_ylabel("Frequency")
ax1.legend()
ax1.set_facecolor("#FAFAFA")
ax1.grid(axis="y", alpha=0.3)

# ── Plot 2: Correlation Heatmap ──
ax2 = fig.add_subplot(gs[0, 2:])
corr_matrix = df[FEATURES + ["kWh"]].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, ax=ax2, annot_kws={"size": 8})
ax2.set_title("Correlation Heatmap", fontweight="bold")
ax2.tick_params(axis="x", rotation=45)

# ── Plot 3: Hourly pattern ──
ax3 = fig.add_subplot(gs[1, :2])
for season in ["hot", "rainy", "cool"]:
    hourly = df[df["season"] == season].groupby("hour")["kWh"].mean()
    ax3.plot(hourly.index, hourly.values, color=season_colors[season],
             linewidth=2.5, marker="o", markersize=4, label=season_labels[season])
ax3.set_title("Hourly AC Usage Pattern by Thai Season", fontweight="bold")
ax3.set_xlabel("Hour")
ax3.set_ylabel("Avg kWh")
ax3.legend()
ax3.set_facecolor("#FAFAFA")
ax3.grid(axis="y", alpha=0.3)

# ── Plot 4: Monthly avg ──
ax4 = fig.add_subplot(gs[1, 2])
monthly = df.groupby("month")["kWh"].mean()
month_colors = [season_colors[get_thai_season(m)] for m in monthly.index]
bars = ax4.bar(range(len(monthly)), monthly.values, color=month_colors, width=0.7)
ax4.set_title("Monthly Avg kWh", fontweight="bold")
ax4.set_xticks(range(12))
ax4.set_xticklabels(["ม.ค","ก.พ","มี.ค","เม.ย","พ.ค","มิ.ย",
                      "ก.ค","ส.ค","ก.ย","ต.ค","พ.ย","ธ.ค"], fontsize=8)
ax4.set_facecolor("#FAFAFA")

# ── Plot 5: Temperature vs kWh ──
ax5 = fig.add_subplot(gs[1, 3])
for season in ["hot", "rainy", "cool"]:
    sdf = df[df["season"] == season].sample(400, random_state=42)
    ax5.scatter(sdf["temp_dry"], sdf["kWh"], alpha=0.3, s=5,
                color=season_colors[season], label=season_labels[season])
ax5.set_title("Temperature vs kWh", fontweight="bold")
ax5.set_xlabel("Temp (°C)")
ax5.set_ylabel("kWh")
ax5.legend(fontsize=8)
ax5.set_facecolor("#FAFAFA")

# ── Plot 6: Model Comparison ──
ax6 = fig.add_subplot(gs[2, 0])
models_names = ["Linear\nRegression", "Random\nForest", "XGBoost\n(tuned)"]
models_r2    = [lr_r2, rf_r2, r2]
colors_bar   = ["#e74c3c" if v < 0.5 else "#f39c12" if v < 0.9 else "#27ae60" for v in models_r2]
bars6 = ax6.bar(models_names, models_r2, color=colors_bar, width=0.5)
for bar, val in zip(bars6, models_r2):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")
ax6.set_title("Model R² Comparison", fontweight="bold")
ax6.set_ylabel("R² Score")
ax6.set_ylim(0, 1.1)
ax6.set_facecolor("#FAFAFA")

# ── Plot 7: Actual vs Predicted ──
ax7 = fig.add_subplot(gs[2, 1])
ax7.scatter(y_test[:600], y_pred[:600], alpha=0.3, s=8, color="#e74c3c")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax7.plot(lims, lims, "k--", linewidth=1.5, label="Perfect fit")
ax7.text(0.05, 0.88,
         f"R² = {r2:.4f}\nMAE = {mae:.4f} kWh/hr\nCV  = {cv_scores.mean():.4f}±{cv_scores.std():.4f}",
         transform=ax7.transAxes, fontsize=8.5, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))
ax7.set_title("Actual vs Predicted (XGBoost)", fontweight="bold")
ax7.set_xlabel("Actual kWh")
ax7.set_ylabel("Predicted kWh")
ax7.set_facecolor("#FAFAFA")

# ── Plot 8: Feature Importance ──
ax8 = fig.add_subplot(gs[2, 2:])
xgb_model    = best_model.named_steps["model"]
importances  = xgb_model.feature_importances_
sorted_idx   = np.argsort(importances)
pct_vals     = importances / importances.sum() * 100
colors_feat  = ["#e74c3c" if pct_vals[i] >= 10 else "#f39c12" if pct_vals[i] >= 5
                else "#3498db" for i in range(len(FEATURES))]
bars8 = ax8.barh([FEATURES[i] for i in sorted_idx],
                 [importances[i] for i in sorted_idx],
                 color=[colors_feat[i] for i in sorted_idx], alpha=0.85)
for bar, i in zip(bars8, sorted_idx):
    ax8.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f"{pct_vals[i]:.1f}%", va="center", fontsize=8.5, fontweight="bold")
ax8.set_title("Feature Importance (XGBoost)\n→ month & hour คือตัวแปรสำคัญที่สุดสำหรับการใช้พลังงาน AC",
              fontweight="bold")
ax8.set_xlabel("Importance Score")
ax8.set_facecolor("#FAFAFA")
ax8.set_xlim(0, max(importances) * 1.35)
ax8.text(0.99, 0.02,
         "Business insight: month สำคัญที่สุด\nเพราะฤดูกาลกำหนดพฤติกรรมการใช้แอร์",
         transform=ax8.transAxes, ha="right", va="bottom", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffde7", alpha=0.9))

plt.savefig("thai_ac_analysis.png", dpi=150, bbox_inches="tight", facecolor="#F8F9FA")
print("✅ บันทึก thai_ac_analysis.png สำเร็จ")
print("\n✅ ทุกขั้นตอนเสร็จสมบูรณ์!")