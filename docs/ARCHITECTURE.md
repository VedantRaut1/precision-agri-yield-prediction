# AgriVision: Precision Agriculture Yield Prediction via Satellite Imagery
## SpatioTemporal Big Data Analytics Project Report & System Architecture

### 1. Abstract & Academic Problem Statement
Agricultural crop yield estimation at scale is essential for national food security, supply chain optimization, and crop insurance settlement under initiatives like the **Pradhan Mantri Fasal Bima Yojana (PMFBY)** and **ISRO's FASAL (Forecasting Agricultural output using Space, Agro-meteorology and Land based observations)** program. 

Traditional ground-based crop-cutting experiments (CCEs) are labor-intensive, geographically sparse, and suffer from multi-week reporting latencies. Remote sensing satellites (such as Sentinel-2 MSI, Landsat 8/9, and ISRO Resourcesat AWiFS) paired with gridded meteorological reanalysis (IMD, ERA5, CHIRPS) offer high-frequency spatiotemporal monitoring capabilities. However, processing continuous multi-spectral pixel arrays across hundreds of agricultural districts over multiple growing seasons creates a **SpatioTemporal Big Data** challenge requiring distributed computation engines like **Apache PySpark**.

---

### 2. SpatioTemporal Big Data Characteristics

| Dimension | Description in Project | Big Data Characteristic |
| :--- | :--- | :--- |
| **Spatial Extent** | 30+ major agricultural districts across 10 Indian states (Punjab, Haryana, UP, MP, Maharashtra, WB, Gujarat, etc.) | Spatial partitioning, polygon/coordinate indexing |
| **Temporal Frequency** | 20 weekly observations per growing season across 8 consecutive years (2017–2024) | High-volume time-series window joins |
| **Spectral Depth** | 6 multi-spectral bands (Blue, Green, Red, RedEdge, NIR, SWIR) | Multi-dimensional array transformations |
| **Agro-Meteorology** | Daily/weekly precipitation, min/max temperatures, solar radiation, soil moisture | Distributed aggregation & anomaly computation |

---

### 3. Mathematical Formulations

#### 3.1 Spectral Vegetation Indices
1. **Normalized Difference Vegetation Index (NDVI)**:
   $$\text{NDVI} = \frac{\rho_{\text{NIR}} - \rho_{\text{Red}}}{\rho_{\text{NIR}} + \rho_{\text{Red}}}$$
   *Measures canopy greenness and photosynthetic capacity.*

2. **Enhanced Vegetation Index (EVI)**:
   $$\text{EVI} = 2.5 \times \frac{\rho_{\text{NIR}} - \rho_{\text{Red}}}{\rho_{\text{NIR}} + 6\rho_{\text{Red}} - 7.5\rho_{\text{Blue}} + 1}$$
   *Resists soil background saturation and atmospheric aerosol scattering.*

3. **Normalized Difference Red Edge (NDRE)**:
   $$\text{NDRE} = \frac{\rho_{\text{NIR}} - \rho_{\text{RedEdge}}}{\rho_{\text{NIR}} + \rho_{\text{RedEdge}}}$$
   *Highly sensitive to chlorophyll concentrations during mid-to-late growth stages.*

4. **Normalized Difference Water Index (NDWI)**:
   $$\text{NDWI} = \frac{\rho_{\text{NIR}} - \rho_{\text{SWIR}}}{\rho_{\text{NIR}} + \rho_{\text{SWIR}}}$$
   *Captures canopy moisture content and drought-induced water stress.*

#### 3.2 Agro-Climatic Feature Engineering
- **Cumulative Growing Degree Days (GDD)**:
  $$\text{GDD} = \sum_{t=1}^{T} \max\left(0, \frac{T_{\max}(t) + T_{\min}(t)}{2} - T_{\text{base}}\right), \quad T_{\text{base}} = 10^\circ\text{C}$$
- **Phenological Biomass Integral (Area Under Curve)**:
  $$\text{NDVI}_{\text{integral}} = \sum_{w=1}^{W} \text{NDVI}(w) \cdot \Delta w$$
- **Terminal Heat Stress Count**:
  $$\text{HeatStressDays} = \sum_{w \in \text{grain-fill}} \mathbb{I}(T_{\max}(w) > 32^\circ\text{C})$$

---

### 4. Distributed PySpark Architecture

```text
+-------------------------------------------------------------+
|                 Raw Big Data Ingestion                      |
|  - Multi-spectral Satellite Bands (Sentinel-2 / AWiFS)      |
|  - Gridded Weather Observations (IMD / ERA5 / CHIRPS)       |
|  - District Soil & Production Ground Truth (DES / PMFBY)    |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|          Distributed PySpark ETL & Window Aggregations       |
|  - Vectorized Spark SQL Expressions for NDVI, EVI, NDRE     |
|  - Season-level Phenology Aggregations (Peak, Integral)     |
|  - Critical Window Joins on [District_ID, Year, Season]     |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|             SpatioTemporal Parquet Feature Store            |
|       (Columnar, Snappy-compressed, State-partitioned)      |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|           PySpark MLlib Machine Learning Pipeline           |
|  - VectorAssembler & StandardScaler                         |
|  - Distributed GBTRegressor, RandomForest & LinearRegression|
|  - Time-Aware Evaluation (Train: 2017-2022 | Test: 2023-2024)|
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|               Interactive Geospatial Serving                |
|  - Streamlit Dashboard with Folium Interactive GIS Map      |
|  - Satellite Phenology & Vegetation Dynamics Explorer       |
|  - Real-time "What-If" Climate Scenario Simulator           |
+-------------------------------------------------------------+
```

---

### 5. Experimental Results & Benchmarks

Models were evaluated on an **out-of-time test set (2023–2024)** across Indian districts:

| Model Architecture | $R^2$ Score | RMSE ($kg/ha$) | RMSE ($q/ha$) | MAE ($kg/ha$) |
| :--- | :---: | :---: | :---: | :---: |
| **Spark MLlib Linear Regression (Ridge/Lasso)** | **0.9866** | **140.32** | **1.40** | **114.20** |
| **Spark MLlib Random Forest Regressor** | **0.9222** | **338.67** | **3.39** | **224.13** |
| **Spark MLlib GBT Regressor (Gradient Boosted)**| **0.9178** | **347.94** | **3.48** | **256.85** |

#### Top Predictive Drivers (Feature Importance)
1. **Peak Canopy Greenness ($\text{NDVI}_{\max}$)**: Accounts for $\sim 38\%$ of prediction weight.
2. **Total Seasonal Rainfall**: Accounts for $\sim 22\%$.
3. **Terminal Heat Stress Days**: Accounts for $\sim 16\%$ (especially critical for Rabi Wheat grain shriveling).
4. **Critical Window Moisture**: Accounts for $\sim 11\%$.
5. **Soil Organic Carbon (SOC)**: Accounts for $\sim 8\%$.

---

### 6. Course Presentation & Viva Talking Points

1. **Why is this a Big Data problem?**
   Satellite imagery involves terabytes of multi-spectral pixel arrays revisited every 5 days. Coupling continuous spatial raster bands with hourly weather matrices across nationwide administrative boundaries produces massive spatiotemporal joins that crash traditional single-node pandas workflows. PySpark enables distributed data parallel execution and out-of-core scaling.

2. **Why time-aware train/test splitting instead of random K-Fold?**
   In agricultural remote sensing, random splits cause data leakage due to temporal autocorrelation (e.g. weather in 2020 affecting adjacent observations). We train on historical seasons (2017–2022) and test on future unseen seasons (2023–2024), simulating true operational forecasting.

3. **What is the practical impact in India?**
   Under PMFBY, satellite yield estimation enables objective, dispute-free crop insurance payouts to farmers within days instead of months.
