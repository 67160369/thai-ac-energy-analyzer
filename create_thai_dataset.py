import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)

# =========================
# 1. ข้อมูลอุณหภูมิกรุงเทพฯ จริง (กรมอุตุฯ)
# =========================
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
    else:                            return "cool"

# =========================
# 2. สร้าง dataset ไทย
# =========================
records = []
hours_per_day = 24
days_per_month = 30
n_months = 24  # 2 ปี

for year in range(2023, 2025):
    for month in range(1, 13):
        base_temp  = bkk_temp[month]
        base_humid = bkk_humidity[month]
        rain_prob  = bkk_rain_chance[month]
        season     = get_thai_season(month)

        for day in range(days_per_month):
            is_weekend  = 1 if day % 7 >= 5 else 0
            daily_temp  = base_temp + np.random.normal(0, 1.5)
            daily_humid = base_humid + np.random.normal(0, 5)
            daily_humid = np.clip(daily_humid, 40, 100)
            is_raining  = np.random.random() < rain_prob

            for hour in range(hours_per_day):
                # อุณหภูมิรายชั่วโมง — ร้อนสุดบ่าย 14:00
                hour_temp = daily_temp + 3.0 * np.sin(np.pi * (hour - 6) / 12)
                hour_temp = np.clip(hour_temp, 20, 42)

                # ความชื้นรายชั่วโมง
                hour_humid = daily_humid - 10 * np.sin(np.pi * (hour - 6) / 12)
                if is_raining and 10 <= hour <= 18:
                    hour_humid = min(hour_humid + 15, 100)
                hour_humid = np.clip(hour_humid, 40, 100)

                # ฝน
                is_raining_now = 1 if (is_raining and np.random.random() < 0.4) else 0
                cloud_cover    = np.random.uniform(60, 100) if is_raining_now else np.random.uniform(10, 70)
                wind_speed     = np.random.uniform(1, 8) if is_raining_now else np.random.uniform(0.5, 5)
                pressure       = np.random.uniform(1005, 1015)

                # =========================
                # คำนวณการใช้ไฟแอร์ (kWh)
                # สูตรจากพฤติกรรมคนไทยจริง
                # =========================

                # แอร์ทำงานหนักขึ้นตามอุณหภูมิ
                ac_load = 0.8 + (hour_temp - 25) * 0.04  # 1.2kW ที่ 35°C

                # pattern รายชั่วโมง (คนไทยเปิดแอร์)
                if 0 <= hour <= 5:      usage_factor = 0.85  # นอนหลับ เปิดตลอด
                elif 6 <= hour <= 8:    usage_factor = 0.40  # ตื่นนอน อาบน้ำ
                elif 9 <= hour <= 11:   usage_factor = 0.30  # ทำงาน/โรงเรียน
                elif 12 <= hour <= 13:  usage_factor = 0.50  # กลับบ้านกินข้าว
                elif 14 <= hour <= 16:  usage_factor = 0.35  # ร้อนสุด แต่ไม่อยู่บ้าน
                elif 17 <= hour <= 19:  usage_factor = 0.65  # กลับบ้าน
                elif 20 <= hour <= 22:  usage_factor = 0.90  # อยู่บ้านเต็มที่
                else:                   usage_factor = 0.85  # เตรียมนอน

                # วันหยุด อยู่บ้านมากกว่า
                if is_weekend:
                    if 10 <= hour <= 16:
                        usage_factor = min(usage_factor + 0.25, 1.0)

                # ถ้าฝนตก อากาศเย็นลง ใช้แอร์น้อยลง
                if is_raining_now:
                    usage_factor *= 0.75

                # ฤดูหนาว ใช้แอร์น้อยกว่ามาก
                if season == "cool":
                    usage_factor *= 0.55

                kwh = round(ac_load * usage_factor + np.random.normal(0, 0.02), 4)
                kwh = max(0.0, kwh)

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
print(f"Dataset size: {len(df):,} rows")
print(df.groupby("season")[["temp_dry", "humidity", "kWh"]].mean().round(3))

# =========================
# 3. Train XGBoost
# =========================
features = ["temp_dry", "humidity", "wind_speed", "cloud_cover",
            "hour", "month", "is_weekend", "pressure"]
X = df[features]
y = df["kWh"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(
    n_estimators=300, learning_rate=0.05,
    max_depth=6, subsample=0.8, random_state=42
)
model.fit(X_train, y_train)

y_pred    = model.predict(X_test)
r2        = round(r2_score(y_test, y_pred), 4)
mae       = round(mean_absolute_error(y_test, y_pred), 4)
cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

print(f"\nR²  = {r2}")
print(f"MAE = {mae}")
print(f"CV  = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# =========================
# 4. บันทึกโมเดล
# =========================
with open("weather_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("weather_feature_cols.pkl", "wb") as f:
    pickle.dump(features, f)
print("\nบันทึก weather_model.pkl และ weather_feature_cols.pkl สำเร็จ")

# =========================
# 5. Plot วิเคราะห์
# =========================
fig = plt.figure(figsize=(18, 10))
fig.suptitle("Thai AC Dataset — Analysis", fontsize=16, fontweight="bold")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

season_colors = {"hot": "#FF6B35", "rainy": "#3498db", "cool": "#5B9BD5"}
season_labels = {"hot": "ฤดูร้อน", "rainy": "ฤดูฝน", "cool": "ฤดูหนาว"}

# --- Hourly pattern แต่ละฤดู ---
ax1 = fig.add_subplot(gs[0, :2])
for season in ["hot", "rainy", "cool"]:
    hourly = df[df["season"] == season].groupby("hour")["kWh"].mean()
    ax1.plot(hourly.index, hourly.values, color=season_colors[season],
             linewidth=2.5, marker="o", markersize=4, label=season_labels[season])
ax1.set_title("Hourly AC Usage Pattern by Thai Season", fontweight="bold")
ax1.set_xlabel("Hour")
ax1.set_ylabel("Avg kWh")
ax1.legend()
ax1.set_facecolor("#FAFAFA")
ax1.grid(axis="y", alpha=0.3)

# --- Monthly avg ---
ax2 = fig.add_subplot(gs[0, 2])
monthly = df.groupby("month")["kWh"].mean()
month_colors = [season_colors[get_thai_season(m)] for m in monthly.index]
ax2.bar(range(len(monthly)), monthly.values, color=month_colors, width=0.7)
ax2.set_title("Monthly Avg kWh", fontweight="bold")
ax2.set_xticks(range(12))
ax2.set_xticklabels(["ม.ค","ก.พ","มี.ค","เม.ย","พ.ค","มิ.ย",
                      "ก.ค","ส.ค","ก.ย","ต.ค","พ.ย","ธ.ค"], fontsize=8)
ax2.set_facecolor("#FAFAFA")

# --- Temp vs kWh ---
ax3 = fig.add_subplot(gs[1, 0])
for season in ["hot", "rainy", "cool"]:
    sdf = df[df["season"] == season].sample(500)
    ax3.scatter(sdf["temp_dry"], sdf["kWh"], alpha=0.3, s=5,
                color=season_colors[season], label=season_labels[season])
ax3.set_title("Temperature vs kWh", fontweight="bold")
ax3.set_xlabel("Temp (°C)")
ax3.set_ylabel("kWh")
ax3.legend(fontsize=8)
ax3.set_facecolor("#FAFAFA")

# --- Actual vs Predicted ---
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(y_test[:500], y_pred[:500], alpha=0.3, s=8, color="#e74c3c")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax4.plot(lims, lims, "k--", linewidth=1.5)
ax4.text(0.05, 0.90, f"R² = {r2}\nMAE = {mae}", transform=ax4.transAxes,
         fontsize=9, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))
ax4.set_title("Actual vs Predicted", fontweight="bold")
ax4.set_xlabel("Actual kWh")
ax4.set_ylabel("Predicted kWh")
ax4.set_facecolor("#FAFAFA")

# --- Feature Importance ---
ax5 = fig.add_subplot(gs[1, 2])
importances = model.feature_importances_
sorted_idx  = np.argsort(importances)
pct_vals    = [v / sum(importances) * 100 for v in importances]
bars = ax5.barh([features[i] for i in sorted_idx],
                [importances[i] for i in sorted_idx], color="#e74c3c", alpha=0.8)
for bar, i in zip(bars, sorted_idx):
    ax5.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f"{pct_vals[i]:.1f}%", va="center", fontsize=8, fontweight="bold")
ax5.set_title("Feature Importance", fontweight="bold")
ax5.set_xlabel("Importance Score")
ax5.set_facecolor("#FAFAFA")
ax5.set_xlim(0, max(importances) * 1.3)

plt.savefig("thai_ac_analysis.png", dpi=150, bbox_inches="tight",
            facecolor="#F8F9FA")
print("บันทึก thai_ac_analysis.png สำเร็จ")