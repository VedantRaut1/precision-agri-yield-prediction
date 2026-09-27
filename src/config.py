import os
import sys
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# Ensure directories exist
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# File Paths
SATELLITE_RAW_FILE = RAW_DATA_DIR / "satellite_vegetation_timeseries.parquet"
WEATHER_RAW_FILE = RAW_DATA_DIR / "weather_climate_timeseries.parquet"
DISTRICT_METADATA_FILE = RAW_DATA_DIR / "district_metadata.csv"
CROP_YIELD_GROUNDTRUTH_FILE = RAW_DATA_DIR / "crop_yield_groundtruth.csv"
PROCESSED_FEATURE_STORE = PROCESSED_DATA_DIR / "spatiotemporal_feature_store.parquet"
SPARK_MODEL_PATH = MODELS_DIR / "spark_gbt_yield_model"
SKLEARN_MODEL_PATH = MODELS_DIR / "yield_rf_dashboard_model.joblib"

# Indian Agricultural Crops and Seasons
CROPS = ["Rice (Paddy)", "Wheat", "Cotton", "Soybean", "Maize", "Mustard"]
SEASONS = ["Kharif", "Rabi"]
BASE_YEARS = list(range(2017, 2025))  # 8 years of spatiotemporal data (2017-2024)

# Remote Sensing & Agro-Meteorological Features
SPECTRAL_INDICES = ["ndvi", "evi", "ndre", "ndwi"]
WEATHER_VARS = [
    "precip_week_mm", "temp_max_c", "temp_min_c", "temp_mean_c",
    "gdd_week", "heat_stress_days", "soil_moisture_pct", "solar_radiation_mj"
]

ENGINEERED_FEATURES = [
    "ndvi_max", "ndvi_mean", "ndvi_integral", "evi_max", "ndre_mean", "ndwi_mean",
    "total_precip_season", "precip_critical_window", "gdd_cumulative",
    "heat_stress_days_total", "avg_temp_season", "solar_radiation_total", "soil_organic_carbon_pct"
]
TARGET_VARIABLE = "yield_kg_per_ha"

def get_spark_session(app_name="PrecisionAgri-YieldPrediction"):
    """
    Initializes a PySpark session configured for Windows and Big Data analytics.
    """
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    
    from pyspark.sql import SparkSession
    
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
