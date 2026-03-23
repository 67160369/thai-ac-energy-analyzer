# 🌡️ Thai Seasonal AC Energy Analyzer

> **พยากรณ์การใช้ไฟฟ้าของเครื่องปรับอากาศตามฤดูกาลไทย ด้วย XGBoost**

🔗 **Live App:** [thai-ac-energy-analyzer.streamlit.app](https://thai-ac-energy-analyzer-36qxmimbmt3gupxhteae2x.streamlit.app)

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?logo=xgboost)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red?logo=streamlit)
![R²](https://img.shields.io/badge/R²-0.9789-brightgreen)

---

## 📌 ปัญหาที่แก้ไข

ประเทศไทยมี **3 ฤดูกาล** ได้แก่ ฤดูร้อน ฤดูฝน และฤดูหนาว ซึ่งแต่ละฤดูมีอุณหภูมิและความชื้นที่แตกต่างกัน ส่งผลโดยตรงต่อการใช้ไฟฟ้าของเครื่องปรับอากาศ

โปรเจกต์นี้ใช้ **Machine Learning** ในการพยากรณ์การใช้ไฟฟ้าและเปรียบเทียบพฤติกรรมการใช้แอร์แต่ละฤดู เพื่อให้ผู้ใช้สามารถประมาณค่าไฟและวางแผนการใช้พลังงานได้อย่างมีประสิทธิภาพ

---

## 📊 Dataset

สร้าง dataset จำลองโดยอ้างอิงจากข้อมูลจริง:

- 🌡️ **อุณหภูมิเฉลี่ยกรุงเทพฯ จริง** จากกรมอุตุนิยมวิทยา
- 🏠 **พฤติกรรมการใช้แอร์ของคนไทย** จำลองตามชั่วโมงในแต่ละวัน
- 📅 **ข้อมูล 2 ปี** (2023–2024) รายชั่วโมง รวม ~17,280 แถว

| Feature | คำอธิบาย | หน่วย |
|---|---|---|
| `temp_dry` | อุณหภูมิอากาศ | °C |
| `humidity` | ความชื้นสัมพัทธ์ | % |
| `wind_speed` | ความเร็วลม | m/s |
| `cloud_cover` | เมฆปกคลุม | % |
| `hour` | ชั่วโมงของวัน | 0–23 |
| `month` | เดือน | 1–12 |
| `is_weekend` | วันหยุดสุดสัปดาห์ | 0 / 1 |
| `pressure` | ความดันอากาศ | hPa |
| **`kWh`** | **การใช้ไฟฟ้าต่อชั่วโมง (Target)** | **kWh** |

---

## 🤖 Model Development

### เปรียบเทียบโมเดล

| Model | R² | MAE |
|---|---|---|
| Linear Regression | 0.14 | 0.118 |
| Random Forest | 0.24 | 0.104 |
| **XGBoost ✅** | **0.9789** | **0.022** |

### ทำไมถึงเลือก XGBoost?

- ✅ **R² = 0.9789** — แม่นยำสูงที่สุดในบรรดาโมเดลที่ทดสอบ
- ✅ จัดการ **non-linear relationship** ระหว่างอุณหภูมิกับการใช้ไฟได้ดี
- ✅ รองรับ **missing values** โดยอัตโนมัติ

### การประเมินผล

- 🔁 **5-Fold Cross Validation**
- 📊 **Feature Importance** — วิเคราะห์ว่า feature ไหนสำคัญที่สุด
- 📈 **Actual vs Predicted** plot — ตรวจสอบความแม่นยำของโมเดล

---

## 🌞 ผลลัพธ์ 3 ฤดูกาลไทย

| ฤดูกาล | อุณหภูมิเฉลี่ย | Avg kWh/hr | ค่าไฟโดยประมาณ/เดือน |
|---|---|---|---|
| 🌞 ฤดูร้อน (มี.ค.–พ.ค.) | 35°C | 0.6894 | ฿1,986 |
| 🌧️ ฤดูฝน (มิ.ย.–ต.ค.) | 29°C | 0.5936 | ฿1,668 |
| ❄️ ฤดูหนาว (พ.ย.–ก.พ.) | 25°C | 0.2711 | ฿677 |

---

## 💰 การคำนวณค่าไฟ

ใช้อัตราค่าไฟแบบ **Tiered Rate ของ PEA/MEA ปี 2024–2025**

| หน่วยที่ใช้ | อัตรา (บาท/หน่วย) |
|---|---|
| 0–150 หน่วย | 3.24 |
| 151–400 หน่วย | 4.22 |
| 400+ หน่วย | 4.62 |

---

## 🚀 วิธีรัน

```bash
# 1. ติดตั้ง dependencies
pip install -r requirements.txt

# 2. สร้าง dataset และ retrain โมเดล
python create_thai_dataset.py

# 3. รัน Streamlit app
streamlit run streamlit_app.py
```

---

## 📁 โครงสร้างไฟล์

```
thai-ac-energy-analyzer/
├── streamlit_app.py           # Streamlit web app
├── create_thai_dataset.py     # สร้าง dataset + train model
├── seasonal_analysis.py       # วิเคราะห์ตามฤดูกาล
├── requirements.txt           # Python dependencies
├── weather_model.pkl          # โมเดล XGBoost ที่ train แล้ว
├── weather_feature_cols.pkl   # Feature columns
├── thai_ac_dataset.csv        # Dataset ไทย
└── thai_ac_analysis.png       # กราฟวิเคราะห์โมเดล
```

---

## ⚠️ Disclaimer

ผลการทำนายนี้มาจากโมเดล XGBoost ที่ train จาก **dataset จำลอง** โดยใช้อุณหภูมิเฉลี่ยกรุงเทพฯ จริงจากกรมอุตุนิยมวิทยา ค่าที่แสดงเป็น **การประมาณการเท่านั้น** อาจแตกต่างจากการใช้งานจริงขึ้นอยู่กับประสิทธิภาพของอุปกรณ์และพฤติกรรมการใช้งาน

---

<div align="center">
  <sub>Made with ❤️ for Thai energy awareness · Data source: กรมอุตุนิยมวิทยา · Rate: PEA/MEA 2024–2025</sub>
</div>
