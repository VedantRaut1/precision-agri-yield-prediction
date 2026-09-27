import sys
import os
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.data_generator import generate_indian_spatiotemporal_dataset
from src.spark_etl import run_pyspark_spatiotemporal_etl
from src.ml_pipeline import run_ml_training_pipeline

def main():
    print("=" * 80)
    print("🌾 AGRIVISION: PRECISION AGRICULTURE YIELD PREDICTION VIA SATELLITE IMAGERY")
    print("   SpatioTemporal Big Data Analytics Project Pipeline")
    print("   Indian Agricultural Benchmark (ISRO FASAL, PMFBY & DES Standards)")
    print("=" * 80)

    start_total = time.time()

    # Step 1: Benchmark Data Generation
    print("\n[PHASE 1/3] Generating SpatioTemporal Remote Sensing & Weather Dataset...")
    t0 = time.time()
    generate_indian_spatiotemporal_dataset()
    print(f"-> Phase 1 completed in {time.time() - t0:.2f} seconds.")

    # Step 2: Distributed PySpark ETL
    print("\n[PHASE 2/3] Executing Distributed PySpark SpatioTemporal ETL Pipeline...")
    t1 = time.time()
    run_pyspark_spatiotemporal_etl()
    print(f"-> Phase 2 completed in {time.time() - t1:.2f} seconds.")

    # Step 3: PySpark MLlib Machine Learning Pipeline
    print("\n[PHASE 3/3] Training Distributed MLlib Regressors & Serializing Models...")
    t2 = time.time()
    metadata = run_ml_training_pipeline()
    print(f"-> Phase 3 completed in {time.time() - t2:.2f} seconds.")

    print("\n" + "=" * 80)
    print("🎉 FULL END-TO-END BIG DATA PIPELINE EXECUTED SUCCESSFULLY!")
    print(f"⏱️ Total Execution Time: {time.time() - start_total:.2f} seconds")
    print("=" * 80)
    print("\n📌 Quick Summary of Results:")
    print(f"   • Best Model: {metadata.get('best_model_name')}")
    print(f"   • Out-of-Time Test R² Score: {metadata.get('test_r2')}")
    print(f"   • Test RMSE: {metadata.get('test_rmse_kg_per_ha')} kg/ha ({metadata.get('test_rmse_quintal_per_ha')} q/ha)")
    print(f"   • Test MAE:  {metadata.get('test_mae_kg_per_ha')} kg/ha")
    print("\n🚀 To launch the Interactive Geospatial Dashboard, run:")
    print("   streamlit run app/dashboard.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
