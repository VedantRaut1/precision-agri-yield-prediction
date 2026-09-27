import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.config import (
    get_spark_session,
    SATELLITE_RAW_FILE,
    WEATHER_RAW_FILE,
    DISTRICT_METADATA_FILE,
    CROP_YIELD_GROUNDTRUTH_FILE,
    PROCESSED_FEATURE_STORE,
    ENGINEERED_FEATURES,
    TARGET_VARIABLE
)
from pyspark.sql import functions as F

def run_pyspark_spatiotemporal_etl():
    """
    Executes distributed PySpark ETL to ingest Indian agricultural satellite imagery bands,
    IMD/ERA5 weather time series, computes spectral vegetation indices, aggregates seasonal
    phenological metrics, and constructs the spatiotemporal feature store.
    """
    print("=" * 75)
    print("[SPARK ETL] Initializing Distributed PySpark Pipeline (Indian Agro-Zones)...")
    print("=" * 75)

    spark = get_spark_session("PrecisionAgri-India-ETL")

    try:
        # 1. Distributed Ingestion of Multi-Spectral Satellite Bands (Parquet)
        print(f"[INFO] Ingesting Satellite Reflectance Bands: {SATELLITE_RAW_FILE}")
        df_sat = spark.read.parquet(str(SATELLITE_RAW_FILE))
        print(f"       -> Satellite Observation Records: {df_sat.count():,}")

        # 2. PySpark SQL Spectral Vegetation Index Calculations
        # NDVI = (NIR - Red) / (NIR + Red)
        # EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
        # NDRE = (NIR - RedEdge) / (NIR + RedEdge)
        # NDWI = (NIR - SWIR) / (NIR + SWIR)
        df_sat_indices = (
            df_sat
            .withColumn("calc_ndvi", (F.col("band_nir") - F.col("band_red")) / (F.col("band_nir") + F.col("band_red") + 1e-6))
            .withColumn("calc_evi", 2.5 * (F.col("band_nir") - F.col("band_red")) / (F.col("band_nir") + 6.0 * F.col("band_red") - 7.5 * F.col("band_blue") + 1.0))
            .withColumn("calc_ndre", (F.col("band_nir") - F.col("band_rededge")) / (F.col("band_nir") + F.col("band_rededge") + 1e-6))
            .withColumn("calc_ndwi", (F.col("band_nir") - F.col("band_swir")) / (F.col("band_nir") + F.col("band_swir") + 1e-6))
        )

        # 3. SpatioTemporal Phenology Aggregations across Seasons
        print("[INFO] Computing Seasonal Phenology Aggregations (Peak Greenness, Biomass Integral)...")
        df_sat_agg = (
            df_sat_indices
            .groupBy("district_id", "year", "season")
            .agg(
                F.max("calc_ndvi").alias("ndvi_max"),
                F.mean("calc_ndvi").alias("ndvi_mean"),
                F.sum("calc_ndvi").alias("ndvi_integral"),
                F.max("calc_evi").alias("evi_max"),
                F.mean("calc_ndre").alias("ndre_mean"),
                F.mean("calc_ndwi").alias("ndwi_mean")
            )
        )

        # 4. Distributed Ingestion of Weather Observations (Parquet)
        print(f"[INFO] Ingesting Gridded Weather Observations: {WEATHER_RAW_FILE}")
        df_weather = spark.read.parquet(str(WEATHER_RAW_FILE))
        print(f"       -> Meteorological Observation Records: {df_weather.count():,}")

        # 5. Agro-meteorological Aggregations (Cumulative Rainfall, Critical Window Rain, Heat Stress)
        print("[INFO] Calculating Monsoon / Rabi Rainfall, GDD, and Terminal Heat Stress Days...")
        # In our dataset, weeks 7 to 11 (mid-season) represent critical flowering/grain-filling
        df_weather_agg = (
            df_weather
            .groupBy("district_id", "year", "season")
            .agg(
                F.sum("precip_week_mm").alias("total_precip_season"),
                F.sum(F.when((F.col("week_in_season") >= 8) & (F.col("week_in_season") <= 12), F.col("precip_week_mm")).otherwise(0.0)).alias("precip_critical_window"),
                F.sum("gdd_week").alias("gdd_cumulative"),
                F.sum("heat_stress_days").alias("heat_stress_days_total"),
                F.mean("temp_mean_c").alias("avg_temp_season"),
                F.sum("solar_radiation_mj").alias("solar_radiation_total")
            )
        )

        # 6. Read District Soil Metadata & Yield Targets
        print(f"[INFO] Ingesting District Metadata and Ground Truth Yield Targets...")
        df_district_meta = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(str(DISTRICT_METADATA_FILE))
            .select(
                "district_id", "district", "state", "zone", "lat", "lon",
                F.col("soil_soc").alias("soil_organic_carbon_pct"), "soil_type"
            )
        )

        df_yield = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(str(CROP_YIELD_GROUNDTRUTH_FILE))
            .select(
                "district_id", "year", "crop", "season",
                "yield_kg_per_ha", "yield_quintal_per_ha", "yield_tonnes_per_ha"
            )
        )

        # 7. Distributed SpatioTemporal Multi-Way Join
        print("[INFO] Performing SpatioTemporal Distributed Multi-Way Joins in PySpark...")
        df_feature_store = (
            df_sat_agg
            .join(df_weather_agg, on=["district_id", "year", "season"], how="inner")
            .join(df_district_meta, on=["district_id"], how="inner")
            .join(df_yield, on=["district_id", "year", "season"], how="inner")
        )

        # Partition by state for distributed locality
        df_feature_store = df_feature_store.repartition(4, "state")

        # 8. Persist to Parquet
        print(f"[INFO] Persisting SpatioTemporal Feature Store to: {PROCESSED_FEATURE_STORE}")
        if PROCESSED_FEATURE_STORE.exists():
            import shutil
            if PROCESSED_FEATURE_STORE.is_dir():
                shutil.rmtree(PROCESSED_FEATURE_STORE)
            else:
                PROCESSED_FEATURE_STORE.unlink()

        pdf_feature_store = df_feature_store.toPandas()
        pdf_feature_store.to_parquet(str(PROCESSED_FEATURE_STORE), index=False)

        total_features = len(pdf_feature_store)
        print(f"[SUCCESS] SpatioTemporal Feature Store successfully created! ({total_features} rows)")
        print("\n[SCHEMA] Feature Store Columns & Types:")
        df_feature_store.printSchema()

        print("\n[SAMPLE] Preview of Feature Store Records:")
        sample_cols = ["district", "state", "crop", "season", "year", "ndvi_max", "total_precip_season", "yield_kg_per_ha", "yield_quintal_per_ha"]
        df_feature_store.select(*sample_cols).show(5, truncate=False)

        return df_feature_store

    finally:
        print("[INFO] Stopping PySpark Session.")
        spark.stop()

if __name__ == "__main__":
    run_pyspark_spatiotemporal_etl()
