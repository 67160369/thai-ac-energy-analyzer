import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Thai Seasonal AC Energy Analyzer",
    page_icon="🌡️",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open("weather_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("weather_feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, feature_cols

model, feature_cols = load_model()

def get_season(month):
    if month in [3, 4, 5]:           return "ฤดูร้อน", "#FF6B35"
    elif month in [6, 7, 8, 9, 10]:  return "ฤดูฝน",   "#3498db"
    else:                             return "ฤดูหนาว", "#5B9BD5"

season_icons      = {"ฤดูร้อน": "🌞", "ฤดูฝน": "🌧️", "ฤดูหนาว": "❄️"}
season_colors_map = {"ฤดูร้อน": "#FF6B35", "ฤดูฝน": "#3498db", "ฤดูหนาว": "#5B9BD5"}
seasons_months    = {"ฤดูร้อน": 4, "ฤดูฝน": 8, "ฤดูหนาว": 1}
season_temp_avg   = {"ฤดูร้อน": 35.0, "ฤดูฝน": 29.0, "ฤดูหนาว": 25.0}
season_hum_avg    = {"ฤดูร้อน": 65.0, "ฤดูฝน": 80.0, "ฤดูหนาว": 60.0}
month_names_th    = ["ม.ค.","ก.พ.","มี.ค.","เม.ย.","พ.ค.","มิ.ย.",
                     "ก.ค.","ส.ค.","ก.ย.","ต.ค.","พ.ย.","ธ.ค."]

def hex_to_rgba(hex_color, alpha=0.2):
    h = hex_color.lstrip('#')
    r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"

def calc_cost(kwh):
    if kwh <= 150:   return kwh * 3.24
    elif kwh <= 400: return 150 * 3.24 + (kwh - 150) * 4.22
    else:            return 150 * 3.24 + 250 * 4.22 + (kwh - 400) * 4.62

def predict_kwh(t, h, hum, ws, cc, pr, m, iw):
    row = {"temp_dry": t, "humidity": hum, "wind_speed": ws,
           "cloud_cover": cc, "hour": h, "month": m,
           "is_weekend": int(iw), "pressure": pr}
    return float(model.predict(pd.DataFrame([row])[feature_cols])[0])

st.title("🌡️ Thai Seasonal AC Energy Analyzer")
st.markdown("พยากรณ์การใช้ไฟฟ้าของเครื่องปรับอากาศตามฤดูกาลไทย (ร้อน/ฝน/หนาว) ด้วย XGBoost")
st.divider()

with st.sidebar:
    st.header("📅 เดือนและฤดูกาล")
    month = st.selectbox("เดือน", options=list(range(1, 13)),
                         format_func=lambda x: month_names_th[x-1])
    season_name, season_color = get_season(month)
    st.markdown(f"### {season_icons[season_name]} {season_name}")

    st.header("🌡️ สภาพอากาศ")
    temp_dry    = st.slider("อุณหภูมิ (°C)", 15.0, 42.0, season_temp_avg[season_name], 0.5)
    humidity    = st.slider("ความชื้น (%)", 30.0, 100.0, season_hum_avg[season_name], 1.0)
    wind_speed  = st.slider("ความเร็วลม (m/s)", 0.0, 15.0, 3.0, 0.5)
    cloud_cover = st.slider("เมฆปกคลุม (%)", 0.0, 100.0, 50.0, 1.0)
    pressure    = st.slider("ความดันอากาศ (hPa)", 1000.0, 1025.0, 1010.0, 0.5)

    st.header("⏰ เวลา")
    hour       = st.slider("ชั่วโมง (0-23)", 0, 23, 14)
    is_weekend = st.checkbox("วันหยุดสุดสัปดาห์", value=False)

    st.header("🍂 เปรียบเทียบฤดู")
    st.caption("เลือกฤดูที่ต้องการแสดงในกราฟ")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        show_hot   = st.toggle("🌞 ร้อน",  value=True)
        show_cold  = st.toggle("❄️ หนาว", value=True)
    with col_s2:
        show_rainy = st.toggle("🌧️ ฝน",   value=True)

    selected_seasons = []
    if show_hot:   selected_seasons.append("ฤดูร้อน")
    if show_rainy: selected_seasons.append("ฤดูฝน")
    if show_cold:  selected_seasons.append("ฤดูหนาว")
    if not selected_seasons:
        selected_seasons = ["ฤดูร้อน"]

prediction         = round(predict_kwh(temp_dry, hour, humidity, wind_speed,
                                        cloud_cover, pressure, month, is_weekend), 4)
prediction_monthly = round(prediction * 24 * 30, 2)
cost_monthly       = round(calc_cost(prediction_monthly), 2)

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{season_icons[season_name]} ฤดูกาล", season_name)
c2.metric("การใช้ไฟ (kWh/hr)", f"{prediction}")
c3.metric("ประมาณการ/เดือน", f"{prediction_monthly} kWh")
c4.metric("ค่าไฟ/เดือน (PEA/MEA)", f"฿{cost_monthly:,.2f}")

st.divider()

col_l, col_r = st.columns(2)

with col_l:
    st.subheader("📈 Pattern การใช้ไฟรายชั่วโมงแต่ละฤดู")
    fig_hourly = go.Figure()
    for s_name in selected_seasons:
        kwh_list = [round(predict_kwh(season_temp_avg[s_name], h, season_hum_avg[s_name],
                                      3, 50, 1010, seasons_months[s_name], False), 4)
                    for h in range(24)]
        fig_hourly.add_trace(go.Scatter(
            x=list(range(24)), y=kwh_list, mode="lines+markers",
            name=f"{season_icons[s_name]} {s_name}",
            line=dict(color=season_colors_map[s_name], width=2.5),
            marker=dict(size=4)
        ))
    fig_hourly.add_vline(x=hour, line_dash="dash", line_color="gray",
                         annotation_text=f"{hour}:00")
    fig_hourly.update_layout(xaxis_title="ชั่วโมง", yaxis_title="kWh",
                             height=350, margin=dict(t=10, b=10),
                             legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_hourly, use_container_width=True)

with col_r:
    st.subheader("🌡️ อุณหภูมิ vs การใช้ไฟ (ฤดูปัจจุบัน)")
    temps = np.linspace(15, 42, 60)
    kwh_by_temp = [round(predict_kwh(t, hour, humidity, wind_speed,
                                     cloud_cover, pressure, month, is_weekend), 4)
                   for t in temps]
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=list(temps), y=kwh_by_temp, mode="lines",
        line=dict(color=season_color, width=2.5),
        fill="tozeroy", fillcolor=hex_to_rgba(season_color)
    ))
    fig_temp.add_vline(x=temp_dry, line_dash="dash", line_color="red",
                       annotation_text=f"{temp_dry}°C")
    fig_temp.update_layout(xaxis_title="อุณหภูมิ (°C)", yaxis_title="kWh",
                           height=350, margin=dict(t=10, b=10))
    st.plotly_chart(fig_temp, use_container_width=True)

col_b1, col_b2 = st.columns(2)

season_avg = {}
for s, m in seasons_months.items():
    vals = [predict_kwh(season_temp_avg[s], h, season_hum_avg[s],
                        3, 50, 1010, m, False) for h in range(24)]
    season_avg[s] = round(np.mean(vals), 4)

with col_b1:
    st.subheader("📊 เปรียบเทียบการใช้ไฟเฉลี่ยแต่ละฤดู")
    fig_bar = go.Figure(go.Bar(
        x=[f"{season_icons[s]} {s}" for s in season_avg],
        y=list(season_avg.values()),
        marker_color=[season_colors_map[s] for s in season_avg],
        text=[f"{v:.4f} kWh" for v in season_avg.values()],
        textposition="outside"
    ))
    fig_bar.update_layout(yaxis_title="Avg kWh/hr", height=320, margin=dict(t=20, b=10))
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
    fig_cost.update_layout(yaxis_title="บาท/เดือน", height=320, margin=dict(t=20, b=10))
    st.plotly_chart(fig_cost, use_container_width=True)

st.divider()
st.subheader("📋 สรุปเปรียบเทียบ 3 ฤดูกาลไทย")
summary_data = []
for s in seasons_months:
    mon_kwh  = round(season_avg[s] * 24 * 30, 2)
    mon_cost = round(calc_cost(mon_kwh), 2)
    summary_data.append({
        "ฤดูกาล":            f"{season_icons[s]} {s}",
        "อุณหภูมิเฉลี่ย (°C)": season_temp_avg[s],
        "ความชื้นเฉลี่ย (%)":  season_hum_avg[s],
        "Avg kWh/hr":         season_avg[s],
        "kWh/เดือน":          mon_kwh,
        "ค่าไฟ/เดือน (฿)":    f"฿{mon_cost:,.2f}"
    })
st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

st.divider()
st.caption("""
⚠️ Disclaimer: ผลการทำนายนี้มาจากโมเดล XGBoost ที่ train จาก dataset สภาพอากาศจริง
ปรับฤดูกาลให้ตรงกับบริบทไทย (ร้อน/ฝน/หนาว) โดยใช้อุณหภูมิเฉลี่ยกรุงเทพฯ จริง
ค่าไฟคำนวณตามโครงสร้างอัตรา PEA/MEA ปี 2024-2025
""")