# 🌾 AgriVision: Precision Agriculture Yield Prediction via Satellite Imagery
### SpatioTemporal Big Data Analytics Course Project

[![PySpark](https://img.shields.io/badge/Apache_Spark-4.2.0-orange.svg)](https://spark.apache.org/)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37-red.svg)](https://streamlit.io/)
[![Geospatial](https://img.shields.io/badge/Remote_Sensing-Sentinel--2-green.svg)](https://sentinel.esa.int/)

---

## 📌 Project Overview
**AgriVision** is an end-to-end SpatioTemporal Big Data pipeline designed to predict agricultural crop yields across Indian agro-climatic zones by coupling multi-spectral satellite imagery (Sentinel-2 / ISRO Resourcesat) with gridded meteorological observations (IMD / ERA5) and official district production records (DES / PMFBY).

### Key Highlights:
- **Distributed Big Data Engine**: Built with **Apache PySpark (Spark SQL & MLlib)** for high-throughput spatiotemporal joins and distributed model training.
- **Indian Agro-Ecological Focus**: Covers 30+ prominent districts across 10 Indian states (Punjab, Haryana, UP, MP, Maharashtra, West Bengal, Telangana, Karnataka, Rajasthan, Gujarat) for both **Kharif** (Rice, Cotton, Soybean, Maize) and **Rabi** (Wheat, Mustard) seasons.
- **Scientific Remote Sensing Metrics**: Computes **NDVI**, **EVI**, **NDRE**, and **NDWI** along with phenological integral and terminal heat stress metrics.
- **Interactive Geospatial Dashboard**: Interactive Leaflet/Folium GIS map, vegetation phenology explorer, and a real-time **What-If Climate Impact Simulator**.

---

## 📂 Project Directory Structure

```text
precision_agri_yield_prediction/
├── app/
│   └── dashboard.py          # Interactive Streamlit Web & Geospatial Dashboard
├── data/
│   ├── raw/                  # Multi-spectral satellite & weather Parquet/CSV files
│   └── processed/            # PySpark-engineered SpatioTemporal Feature Store
├── docs/
│   └── ARCHITECTURE.md       # Comprehensive academic report & math formulations
├── models/
│   ├── model_evaluation_metrics.json
│   └── yield_rf_dashboard_model.joblib
├── src/
│   ├── __init__.py
│   ├── config.py             # Global constants & PySpark session factory
│   ├── data_generator.py     # Indian agricultural benchmark generator
│   ├── spark_etl.py          # Distributed PySpark ETL & phenological window joins
│   └── ml_pipeline.py        # PySpark MLlib regressors & evaluation pipeline
├── run_pipeline.py           # Master one-click execution script
└── README.md
```

---

## ⚡ Quickstart Guide

### 1. Set Active Workspace
In your IDE, open the project folder:
```text
C:\Users\vedan\.gemini\antigravity\scratch\precision_agri_yield_prediction
```

### 2. Run the Full End-to-End Big Data Pipeline
Run the master script to generate benchmark data, execute distributed PySpark ETL, train MLlib models, and compute evaluation metrics:
```bash
python run_pipeline.py
```

### 3. Launch the Interactive Web Dashboard
Run the Streamlit app to explore the interactive geospatial map and what-if simulator:
```bash
python -m streamlit run app/dashboard.py
```
*Or simply double-click / run [`run_dashboard.bat`](file:///C:/Users/vedan/.gemini/antigravity/scratch/precision_agri_yield_prediction/run_dashboard.bat).*
*The dashboard will automatically open in your browser at `http://localhost:8501`.*

---

## 📊 Experimental Results

| Model Architecture | $R^2$ Score | RMSE ($kg/ha$) | RMSE ($q/ha$) | MAE ($kg/ha$) |
| :--- | :---: | :---: | :---: | :---: |
| **Spark MLlib Linear Regression** | **0.9866** | **140.32** | **1.40** | **114.20** |
| **Spark MLlib Random Forest** | **0.9222** | **338.67** | **3.39** | **224.13** |
| **Spark MLlib GBT Regressor** | **0.9178** | **347.94** | **3.48** | **256.85** |

*Evaluated on out-of-time test partitions (2023–2024 seasons).*

---

## 🎓 Academic Viva & Presentation Highlights
- **SpatioTemporal Formulation**: Explains how spatial grids (districts) and temporal phenology (weekly satellite revisits) are joined in parallel via PySpark SQL.
- **Biophysical Meaning**: Peak NDVI represents maximum green canopy photosynthesizing during silking/flowering; terminal heat days simulate heat shocks in March for wheat.
- **Real-World Policy Relevance**: Directly aligns with ISRO FASAL & PMFBY crop insurance claim verification mandates in India.
