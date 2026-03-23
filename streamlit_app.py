"""
Thai Seasonal AC Energy Analyzer — Streamlit App
streamlit_app.py

Features:
- Input validation ป้องกันค่าผิดปกติ
- Confidence range (±MAE) แสดง uncertainty
- Interactive Feature Importance
- Business interpretation
- Disclaimer ครบถ้วน
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Thai Seasonal AC Energy Analyzer",
    page_icon="🌡️",
    layout="wide"
)

# =============================================================
# โหลดโมเดล
# =============================================================
@st.cache_resource
def load_model():
    with open("weather_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("weather_feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, feature_cols

model, feature_cols = load_model()

# =============================================================
# ค่าคงที่
# =============================================================
# MAE จาก training (ใช้เป็น confidence range)
MODEL_MAE = 0.022  # kWh/hr

season_icons      = {"ฤดูร้อน": "🌞", "ฤดูฝน": "🌧️", "ฤดูหนาว": "❄️"}
season_colors_map = {"ฤดูร้อน": "#FF6B35", "ฤดูฝน": "#3498db", "ฤดูหนาว": "#5B9BD5"}
seasons_months    = {"ฤดูร้อน": 4, "ฤดูฝน": 8, "ฤดูหนาว": 1}
season_temp_avg   = {"ฤดูร้อน": 35.0, "ฤดูฝน": 29.0, "ฤดูหนาว": 25.0}
season_hum_avg    = {"ฤดูร้อน": 65.0, "ฤดูฝน": 80.0, "ฤดูหนาว": 60.0}
month_names_th    = ["ม.ค.","ก.พ.","มี.ค.","เม.ย.","พ.ค.","มิ.ย.",
                     "ก.ค.","ส.ค.","ก.ย.","ต.ค.","พ.ย.","ธ.ค."]

# อุณหภูมิปกติของแต่ละเดือนกรุงเทพฯ (กรมอุตุฯ) — ใช้ validate input
NORMAL_TEMP_RANGE = {
    1: (22, 34), 2: (23, 35), 3: (25, 37), 4: (27, 38),
    5: (26, 36), 6: (25, 35), 7: (25, 34), 8: (25, 34),
    9: (24, 33), 10: (23, 33), 11: (23, 34), 12: (21, 33)
}

FEATURE_LABELS = {
    "temp_dry":    "อุณหภูมิ (°C)",
    "humidity":    "ความชื้น (%)",
    "wind_speed":  "ความเร็วลม (m/s)",
    "cloud_cover": "เมฆปกคลุม (%)",
    "hour":        "ชั่วโมง",
    "month":       "เดือน",
    "is_weekend":  "วันหยุด",
    "pressure":    "ความดัน (hPa)"
}

# =============================================================
# Helper functions
# =============================================================
def get_season(month):
    if month in [3, 4, 5]:           return "ฤดูร้อน", "#FF6B35"
    elif month in [6, 7, 8, 9, 10]:  return "ฤดูฝน",   "#3498db"
    else:                             return "ฤดูหนาว", "#5B9BD5"

def hex_to_rgba(hex_color, alpha=0.2):
    h = hex_color.lstrip("#")
    r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"

def calc_cost(kwh):
    """คำนวณค่าไฟตามโครงสร้าง PEA/MEA Tiered Rate 2024-2025"""
    if kwh <= 0:     return 0.0
    if kwh <= 150:   return kwh * 3.24
    elif kwh <= 400: return 150 * 3.24 + (kwh - 150) * 4.22
    else:            return 150 * 3.24 + 250 * 4.22 + (kwh - 400) * 4.62

def predict_kwh(t, h, hum, ws, cc, pr, m, iw):
    row = {
        "temp_dry":    t,
        "humidity":    hum,
        "wind_speed":  ws,
        "cloud_cover": cc,
        "hour":        h,
        "month":       m,
        "is_weekend":  int(iw),
        "pressure":    pr
    }
    val = float(model.predict(pd.DataFrame([row])[feature_cols])[0])
    return max(0.0, round(val, 4))

def validate_inputs(temp, humidity, month, wind_speed, pressure):
    """ตรวจสอบค่า input ว่าอยู่ในช่วงที่สมเหตุสมผล"""
    warnings_list = []
    temp_min, temp_max = NORMAL_TEMP_RANGE[month]

    if temp < temp_min - 5:
        warnings_list.append(
            f"⚠️ อุณหภูมิ {temp}°C ต่ำกว่าปกติของเดือนนี้มาก "
            f"(ปกติ {temp_min}–{temp_max}°C)"
        )
    elif temp > temp_max + 5:
        warnings_list.append(
            f"⚠️ อุณหภูมิ {temp}°C สูงกว่าปกติของเดือนนี้มาก "
            f"(ปกติ {temp_min}–{temp_max}°C)"
        )

    if humidity < 30:
        warnings_list.append("⚠️ ความชื้น < 30% ต่ำผิดปกติสำหรับประเทศไทย")
    if humidity > 98:
        warnings_list.append("⚠️ ความชื้น > 98% สูงผิดปกติ")
    if wind_speed > 20:
        warnings_list.append("⚠️ ความเร็วลม > 20 m/s เกินระดับพายุ")
    if pressure < 990 or pressure > 1030:
        warnings_list.append(f"⚠️ ความดัน {pressure} hPa อยู่นอกช่วงปกติ (990–1030 hPa)")

    return warnings_list

# =============================================================
# Header
# =============================================================
st.title("🌡️ Thai Seasonal AC Energy Analyzer")
st.markdown(
    "พยากรณ์การใช้ไฟฟ้าของเครื่องปรับอากาศตามฤดูกาลไทย (ร้อน/ฝน/หนาว) "
    "ด้วย **XGBoost** (R² = 0.9789, MAE = 0.022 kWh/hr)"
)

# info bar เกี่ยวกับโมเดล
with st.expander("ℹ️ เกี่ยวกับโมเดลนี้", expanded=False):
    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
    col_i1.metric("Algorithm", "XGBoost")
    col_i2.metric("R² Score", "0.9789")
    col_i3.metric("MAE", "0.022 kWh/hr")
    col_i4.metric("Training Data", "~17,280 rows")
    st.markdown("""
    **วิธีการสร้างโมเดล:**
    - Dataset จำลองโดยใช้อุณหภูมิเฉลี่ยกรุงเทพฯ จริง (กรมอุตุนิยมวิทยา 2023–2024)
    - Hyperparameter tuning ด้วย **GridSearchCV** + **5-Fold Cross Validation**
    - เปรียบเทียบ 3 โมเดล: Linear Regression (R²=0.14), Random Forest (R²=0.24), **XGBoost (R²=0.98)**
    - XGBoost ดีที่สุดเพราะจัดการ non-linear relationship ระหว่างฤดูกาล-เวลา-การใช้ไฟได้ดี
    """)

st.divider()

# =============================================================
# Sidebar — Input
# =============================================================
with st.sidebar:
    st.header("📅 เดือนและฤดูกาล")
    month = st.selectbox(
        "เลือกเดือน",
        options=list(range(1, 13)),
        format_func=lambda x: f"{x} — {month_names_th[x-1]}"
    )
    season_name, season_color = get_season(month)
    st.markdown(f"### {season_icons[season_name]} {season_name}")
    temp_min_normal, temp_max_normal = NORMAL_TEMP_RANGE[month]
    st.caption(f"อุณหภูมิปกติเดือนนี้: {temp_min_normal}–{temp_max_normal}°C")

    st.header("🌡️ สภาพอากาศ")
    temp_dry = st.slider(
        "อุณหภูมิ (°C)",
        min_value=15.0, max_value=42.0,
        value=float(season_temp_avg[season_name]),
        step=0.5,
        help=f"อุณหภูมิปกติของเดือนนี้: {temp_min_normal}–{temp_max_normal}°C"
    )
    humidity = st.slider(
        "ความชื้นสัมพัทธ์ (%)",
        min_value=30.0, max_value=100.0,
        value=float(season_hum_avg[season_name]),
        step=1.0,
        help="ความชื้นปกติกรุงเทพฯ: 60–85%"
    )
    wind_speed = st.slider(
        "ความเร็วลม (m/s)",
        min_value=0.0, max_value=20.0,
        value=3.0, step=0.5,
        help="ลมปกติกรุงเทพฯ: 1–5 m/s"
    )
    cloud_cover = st.slider(
        "เมฆปกคลุม (%)",
        min_value=0.0, max_value=100.0,
        value=50.0, step=1.0
    )
    pressure = st.slider(
        "ความดันอากาศ (hPa)",
        min_value=990.0, max_value=1030.0,
        value=1010.0, step=0.5,
        help="ความดันปกติกรุงเทพฯ: 1005–1015 hPa"
    )

    st.header("⏰ เวลา")
    hour = st.slider("ชั่วโมง (0–23)", 0, 23, 14,
                     help="0 = เที่ยงคืน, 14 = บ่ายสองโมง")
    is_weekend = st.checkbox(
        "วันหยุดสุดสัปดาห์",
        value=False,
        help="วันหยุดคนอยู่บ้านมากกว่า → ใช้แอร์มากกว่า"
    )

    st.header("🍂 เปรียบเทียบฤดู")
    st.caption("เลือกฤดูที่ต้องการแสดงในกราฟ")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        show_hot  = st.toggle("🌞 ร้อน",  value=True)
        show_cold = st.toggle("❄️ หนาว", value=True)
    with col_s2:
        show_rainy = st.toggle("🌧️ ฝน", value=True)

    selected_seasons = []
    if show_hot:   selected_seasons.append("ฤดูร้อน")
    if show_rainy: selected_seasons.append("ฤดูฝน")
    if show_cold:  selected_seasons.append("ฤดูหนาว")
    if not selected_seasons:
        selected_seasons = ["ฤดูร้อน"]

# =============================================================
# Input Validation
# =============================================================
input_warnings = validate_inputs(temp_dry, humidity, month, wind_speed, pressure)
if input_warnings:
    for w in input_warnings:
        st.warning(w)

# =============================================================
# Prediction + Confidence
# =============================================================
prediction         = predict_kwh(temp_dry, hour, humidity, wind_speed,
                                  cloud_cover, pressure, month, is_weekend)
pred_low           = max(0.0, round(prediction - MODEL_MAE, 4))
pred_high          = round(prediction + MODEL_MAE, 4)

prediction_monthly = round(prediction * 24 * 30, 2)
monthly_low        = round(pred_low * 24 * 30, 2)
monthly_high       = round(pred_high * 24 * 30, 2)

cost_monthly  = round(calc_cost(prediction_monthly), 2)
cost_low      = round(calc_cost(monthly_low), 2)
cost_high     = round(calc_cost(monthly_high), 2)

# =============================================================
# Metrics Row
# =============================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    f"{season_icons[season_name]} ฤดูกาล",
    season_name
)
c2.metric(
    "การใช้ไฟ (kWh/hr)",
    f"{prediction}",
    help=f"ช่วงความเชื่อมั่น: {pred_low} – {pred_high} kWh/hr (±MAE)"
)
c3.metric(
    "ประมาณการ/เดือน",
    f"{prediction_monthly} kWh",
    help=f"ช่วง: {monthly_low} – {monthly_high} kWh"
)
c4.metric(
    "ค่าไฟ/เดือน (PEA/MEA)",
    f"฿{cost_monthly:,.0f}",
    help=f"ช่วง: ฿{cost_low:,.0f} – ฿{cost_high:,.0f}"
)

# Confidence bar
st.markdown(
    f"**ช่วงความเชื่อมั่น (±MAE = {MODEL_MAE} kWh/hr):** "
    f"{pred_low} – **{prediction}** – {pred_high} kWh/hr &nbsp;|&nbsp; "
    f"ค่าไฟ: ฿{cost_low:,.0f} – **฿{cost_monthly:,.0f}** – ฿{cost_high:,.0f}/เดือน"
)
st.divider()

# =============================================================
# Charts Row 1: Hourly + Temp vs kWh
# =============================================================
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("📈 Pattern การใช้ไฟรายชั่วโมงแต่ละฤดู")
    fig_hourly = go.Figure()
    for s_name in selected_seasons:
        kwh_list = [round(predict_kwh(season_temp_avg[s_name], h, season_hum_avg[s_name],
                                      3, 50, 1010, seasons_months[s_name], False), 4)
                    for h in range(24)]
        fig_hourly.add_trace(go.Scatter(
            x=list(range(24)), y=kwh_list,
            mode="lines+markers",
            name=f"{season_icons[s_name]} {s_name}",
            line=dict(color=season_colors_map[s_name], width=2.5),
            marker=dict(size=4)
        ))
    fig_hourly.add_vline(
        x=hour, line_dash="dash", line_color="gray",
        annotation_text=f"เวลาปัจจุบัน {hour}:00"
    )
    fig_hourly.update_layout(
        xaxis_title="ชั่วโมง",
        yaxis_title="kWh/hr",
        height=350, margin=dict(t=10, b=10),
        legend=dict(orientation="h", y=-0.25)
    )
    st.plotly_chart(fig_hourly, use_container_width=True)
    st.caption("💡 ฤดูร้อนใช้ไฟสูงสุดช่วงดึก (เปิดแอร์นอน) และบ่าย ฤดูหนาวใช้น้อยกว่าอย่างเห็นได้ชัด")

with col_r:
    st.subheader("🌡️ อุณหภูมิ vs การใช้ไฟ (ฤดูกาลปัจจุบัน)")
    temps = np.linspace(15, 42, 60)
    kwh_by_temp = [round(predict_kwh(t, hour, humidity, wind_speed,
                                     cloud_cover, pressure, month, is_weekend), 4)
                   for t in temps]
    kwh_low  = [max(0, v - MODEL_MAE) for v in kwh_by_temp]
    kwh_high = [v + MODEL_MAE for v in kwh_by_temp]

    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=list(temps) + list(temps[::-1]),
        y=kwh_high + kwh_low[::-1],
        fill="toself",
        fillcolor=hex_to_rgba(season_color, 0.15),
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence range (±MAE)"
    ))
    fig_temp.add_trace(go.Scatter(
        x=list(temps), y=kwh_by_temp,
        mode="lines",
        line=dict(color=season_color, width=2.5),
        name="Predicted kWh"
    ))
    fig_temp.add_vline(
        x=temp_dry, line_dash="dash", line_color="red",
        annotation_text=f"{temp_dry}°C"
    )
    fig_temp.update_layout(
        xaxis_title="อุณหภูมิ (°C)", yaxis_title="kWh/hr",
        height=350, margin=dict(t=10, b=10),
        legend=dict(orientation="h", y=-0.25)
    )
    st.plotly_chart(fig_temp, use_container_width=True)
    st.caption("💡 แถบสีคือช่วงความเชื่อมั่น ±MAE แสดง uncertainty ของโมเดล")

# =============================================================
# Charts Row 2: Bar comparison + Cost
# =============================================================
season_avg = {}
for s, m in seasons_months.items():
    vals = [predict_kwh(season_temp_avg[s], h, season_hum_avg[s],
                        3, 50, 1010, m, False) for h in range(24)]
    season_avg[s] = round(np.mean(vals), 4)

col_b1, col_b2 = st.columns(2)

with col_b1:
    st.subheader("📊 เปรียบเทียบการใช้ไฟเฉลี่ยแต่ละฤดู")
    fig_bar = go.Figure(go.Bar(
        x=[f"{season_icons[s]} {s}" for s in season_avg],
        y=list(season_avg.values()),
        marker_color=[season_colors_map[s] for s in season_avg],
        text=[f"{v:.4f} kWh" for v in season_avg.values()],
        textposition="outside",
        error_y=dict(type="constant", value=MODEL_MAE,
                     color="gray", thickness=1.5, width=8)
    ))
    fig_bar.update_layout(
        yaxis_title="Avg kWh/hr",
        height=320, margin=dict(t=20, b=10)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_b2:
    st.subheader("💰 ค่าไฟประมาณการแต่ละฤดู (บาท/เดือน)")
    cost_data = {s: round(calc_cost(round(v * 24 * 30, 2)), 2) for s, v in season_avg.items()}
    fig_cost = go.Figure(go.Bar(
        x=[f"{season_icons[s]} {s}" for s in cost_data],
        y=list(cost_data.values()),
        marker_color=[season_colors_map[s] for s in cost_data],
        text=[f"฿{v:,.0f}" for v in cost_data.values()],
        textposition="outside"
    ))
    fig_cost.update_layout(
        yaxis_title="บาท/เดือน",
        height=320, margin=dict(t=20, b=10)
    )
    st.plotly_chart(fig_cost, use_container_width=True)

# =============================================================
# Feature Importance (Interactive)
# =============================================================
st.divider()
st.subheader("🔍 Feature Importance — ปัจจัยที่ส่งผลต่อการใช้ไฟมากที่สุด")

try:
    xgb_model    = model.named_steps["model"]
    importances  = xgb_model.feature_importances_
except AttributeError:
    importances  = model.feature_importances_

pct_vals   = importances / importances.sum() * 100
feat_df    = pd.DataFrame({
    "Feature":    [FEATURE_LABELS.get(f, f) for f in feature_cols],
    "Importance": importances,
    "Percent":    pct_vals
}).sort_values("Importance", ascending=True)

feat_colors = []
for p in feat_df["Percent"]:
    if p >= 20:    feat_colors.append("#e74c3c")
    elif p >= 10:  feat_colors.append("#f39c12")
    elif p >= 5:   feat_colors.append("#3498db")
    else:          feat_colors.append("#95a5a6")

fig_feat = go.Figure(go.Bar(
    x=feat_df["Importance"],
    y=feat_df["Feature"],
    orientation="h",
    marker_color=feat_colors,
    text=[f"{p:.1f}%" for p in feat_df["Percent"]],
    textposition="outside",
    hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<br>%{text}<extra></extra>"
))
fig_feat.update_layout(
    xaxis_title="Importance Score",
    height=320,
    margin=dict(t=10, b=10, r=60),
    xaxis=dict(range=[0, max(importances) * 1.3])
)
st.plotly_chart(fig_feat, use_container_width=True)

# Business interpretation
top_feats = feat_df.sort_values("Importance", ascending=False)
col_fi1, col_fi2 = st.columns(2)
with col_fi1:
    st.info(
        f"**🏆 ปัจจัยสำคัญที่สุด: {top_feats.iloc[0]['Feature']} ({top_feats.iloc[0]['Percent']:.1f}%)**\n\n"
        f"เดือน/ฤดูกาล คือตัวกำหนดพฤติกรรมการใช้แอร์โดยรวม "
        f"เช่น ฤดูร้อนใช้ไฟเกือบ 3 เท่าของฤดูหนาว"
    )
with col_fi2:
    st.info(
        f"**⏰ อันดับ 2: {top_feats.iloc[1]['Feature']} ({top_feats.iloc[1]['Percent']:.1f}%)**\n\n"
        f"ช่วงเวลากลางดึก (00:00–05:00) และกลางคืน (20:00–23:00) "
        f"ใช้ไฟสูงสุดเพราะคนเปิดแอร์นอน"
    )

# =============================================================
# Summary Table
# =============================================================
st.divider()
st.subheader("📋 สรุปเปรียบเทียบ 3 ฤดูกาลไทย")
summary_data = []
for s in seasons_months:
    mon_kwh  = round(season_avg[s] * 24 * 30, 2)
    mon_cost = round(calc_cost(mon_kwh), 2)
    summary_data.append({
        "ฤดูกาล":              f"{season_icons[s]} {s}",
        "อุณหภูมิเฉลี่ย (°C)": season_temp_avg[s],
        "ความชื้นเฉลี่ย (%)":  season_hum_avg[s],
        "Avg kWh/hr":          season_avg[s],
        "kWh/เดือน":           mon_kwh,
        "ค่าไฟ/เดือน (฿)":    f"฿{mon_cost:,.0f}",
        "เทียบกับฤดูหนาว":     "—"
    })

# เพิ่มคอลัมน์เปรียบเทียบ
base_cost = summary_data[2]["kWh/เดือน"]  # cool season
for row in summary_data:
    diff = row["kWh/เดือน"] - base_cost
    if abs(diff) < 1:
        row["เทียบกับฤดูหนาว"] = "—"
    else:
        row["เทียบกับฤดูหนาว"] = f"+{diff:.0f} kWh (+฿{calc_cost(row['kWh/เดือน'])-calc_cost(base_cost):,.0f})"

st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

# =============================================================
# Disclaimer
# =============================================================
st.divider()
st.warning(
    """
    ⚠️ **Disclaimer**

    ผลการทำนายนี้มาจากโมเดล **XGBoost** ที่ train จาก dataset จำลอง
    โดยใช้อุณหภูมิเฉลี่ยกรุงเทพฯ จริงจากกรมอุตุนิยมวิทยา

    - ค่าไฟคำนวณตามโครงสร้างอัตรา **PEA/MEA ปี 2024–2025** (Tiered Rate)
    - ผลที่แสดงเป็น **การประมาณการเท่านั้น** (MAE ≈ ±0.022 kWh/hr)
    - อาจแตกต่างจากการใช้งานจริงขึ้นอยู่กับ: ประสิทธิภาพของเครื่องปรับอากาศ,
      ขนาดห้อง, จำนวนคน, อุปกรณ์ไฟฟ้าอื่นๆ และพฤติกรรมการใช้งานจริง
    - ไม่ควรใช้ข้อมูลนี้เพื่อการตัดสินใจทางการเงินโดยตรง
    """
)