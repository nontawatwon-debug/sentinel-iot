# Sentinel — IoT Machine Failure Prediction

ระบบทำนายความเสี่ยงความเสียหายของเครื่องจักรแบบ Real-time โดยใช้ Machine Learning
บน AI4I2020 Predictive Maintenance Dataset

---

##  ภาพรวมโปรเจค

Sentinel วิเคราะห์ข้อมูลจาก sensor ของเครื่องจักร แล้วทำนายว่ามีโอกาสเสียหายมากน้อยแค่ไหน
โดยแสดงผลเป็น % ความเสี่ยงแบบ real-time พร้อมแจ้งเตือนเมื่อค่าเกิน threshold

---

##  โครงสร้างไฟล์

```
sentinel-iot/
├── app.py                  # Flask backend + โมเดล
├── requirements.txt        # Python dependencies
├── render.yaml             # Render deployment config
├── ai4i2020.csv            # Dataset
├── predictive_app.py       # Desktop app (Tkinter)
└── templates/
    └── index.html          # Web dashboard UI
```

---

##  Model

| รายละเอียด | ค่า |
|---|---|
| Algorithm | Random Forest Classifier |
| Dataset | AI4I2020 (10,002 rows) |
| AUC Score | **0.97** |
| Features | 6 inputs + 6 engineered features |

**Features ที่ใช้:**
- Air Temperature, Process Temperature, Rotational Speed
- Torque, Tool Wear, Machine Type
- + temp_diff, power_proxy, wear_torque, rpm_excess, torque_excess, wear_excess

---

##  Risk Thresholds

| ค่า | Threshold | Failure Rate |
|---|---|---|
| RPM | > 2,500 | 81.2% |
| Torque | > 60 Nm | 41.9% |
| Tool Wear | > 200 min | 15.5% |

---

##  การใช้งาน

### Desktop App
```bash
python predictive_app.py
```

### Web Dashboard
 [sentinel-iot-1.onrender.com](https://sentinel-iot-1.onrender.com)

---

##  Tech Stack

- Python, Scikit-learn, Pandas, NumPy
- Flask (Web API)
- HTML / CSS / JavaScript (Dashboard)
- Render (Cloud Deployment)
