import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.config import (
    get_spark_session,
    PROCESSED_FEATURE_STORE,
    MODELS_DIR,
    SKLEARN_MODEL_PATH,
    ENGINEERED_FEATURES,
    TARGET_VARIABLE
)

from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

from sklearn.ensemble import RandomForestRegressor as SklearnRF, GradientBoostingRegressor as SklearnGBR
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler as SklearnScaler

def run_ml_training_pipeline():
    """
    Executes distributed PySpark MLlib model training and evaluation,
    extracts spatiotemporal feature importance, and serializes the dashboard inference model.
    """
    print("=" * 75)
    print("[SPARK ML] Initializing SpatioTemporal Machine Learning Training Pipeline...")
    print("=" * 75)

    spark = get_spark_session("PrecisionAgri-MLlib")

    try:
        # 1. Ingest Feature Store from Parquet
        print(f"[INFO] Reading SpatioTemporal Feature Store: {PROCESSED_FEATURE_STORE}")
        df = spark.read.parquet(str(PROCESSED_FEATURE_STORE))
        row_count = df.count()
        print(f"       -> Loaded {row_count} records for distributed ML modeling.")

        # 2. PySpark MLlib Preprocessing Pipeline
        # Encode categorical variables: crop, season, zone
        crop_indexer = StringIndexer(inputCol="crop", outputCol="crop_idx", handleInvalid="keep")
        season_indexer = StringIndexer(inputCol="season", outputCol="season_idx", handleInvalid="keep")
        
        encoder = OneHotEncoder(
            inputCols=["crop_idx", "season_idx"],
            outputCols=["crop_vec", "season_vec"]
        )

        feature_cols = ENGINEERED_FEATURES + ["crop_vec", "season_vec"]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
        scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)

        # 3. Spatiotemporal Train / Test Split (Time-aware: Train 2017-2022, Test 2023-2024)
        print("[INFO] Performing SpatioTemporal Time-Aware Split (Train: 2017-2022, Test: 2023-2024)...")
        train_df = df.filter(df.year <= 2022)
        test_df = df.filter(df.year >= 2023)

        train_count = train_df.count()
        test_count = test_df.count()
        print(f"       -> Train Samples: {train_count} | Test Samples: {test_count}")

        # 4. Train & Evaluate Multiple Spark MLlib Regressors
        models_to_evaluate = {
            "Spark MLlib GBTRegressor": GBTRegressor(
                featuresCol="features", labelCol=TARGET_VARIABLE, maxIter=50, maxDepth=5, seed=42
            ),
            "Spark MLlib RandomForest": RandomForestRegressor(
                featuresCol="features", labelCol=TARGET_VARIABLE, numTrees=60, maxDepth=6, seed=42
            ),
            "Spark MLlib LinearRegression": LinearRegression(
                featuresCol="features", labelCol=TARGET_VARIABLE, regParam=0.1, elasticNetParam=0.5
            )
        }

        results = {}
        eval_r2 = RegressionEvaluator(labelCol=TARGET_VARIABLE, predictionCol="prediction", metricName="r2")
        eval_rmse = RegressionEvaluator(labelCol=TARGET_VARIABLE, predictionCol="prediction", metricName="rmse")
        eval_mae = RegressionEvaluator(labelCol=TARGET_VARIABLE, predictionCol="prediction", metricName="mae")

        best_model_name = None
        best_r2 = -float("inf")
        best_pipeline_model = None

        for model_name, regressor in models_to_evaluate.items():
            print(f"\n[TRAINING] Fitting {model_name}...")
            pipeline = Pipeline(stages=[crop_indexer, season_indexer, encoder, assembler, scaler, regressor])
            model = pipeline.fit(train_df)
            predictions = model.transform(test_df)

            r2_val = eval_r2.evaluate(predictions)
            rmse_val = eval_rmse.evaluate(predictions)
            mae_val = eval_mae.evaluate(predictions)

            results[model_name] = {
                "R2_Score": round(float(r2_val), 4),
                "RMSE_kg_per_ha": round(float(rmse_val), 2),
                "MAE_kg_per_ha": round(float(mae_val), 2),
                "RMSE_quintal_per_ha": round(float(rmse_val / 100.0), 2)
            }

            print(f"       {model_name} Results on Out-of-Time Test Set (2023-2024):")
            print(f"       - R2 Score: {r2_val:.4f}")
            print(f"       - RMSE: {rmse_val:.2f} kg/ha ({rmse_val / 100.0:.2f} q/ha)")
            print(f"       - MAE:  {mae_val:.2f} kg/ha ({mae_val / 100.0:.2f} q/ha)")

            if r2_val > best_r2:
                best_r2 = r2_val
                best_model_name = model_name
                best_pipeline_model = model

        # 5. Extract Feature Importance from Tree Models
        rf_stage = best_pipeline_model.stages[-1]
        importances = []
        if hasattr(rf_stage, "featureImportances"):
            raw_importances = rf_stage.featureImportances.toArray()
            # Map back to feature names (top numerical features)
            for idx, feat in enumerate(ENGINEERED_FEATURES):
                if idx < len(raw_importances):
                    importances.append({"feature": feat, "importance": round(float(raw_importances[idx]), 4)})
            importances.sort(key=lambda x: x["importance"], reverse=True)
            print("\n[FEATURE IMPORTANCE] Top Predictive Drivers Identified by Model:")
            for item in importances[:7]:
                print(f"       • {item['feature']:<25}: {item['importance'] * 100:.2f}%")

    finally:
        print("[INFO] Stopping PySpark Session.")
        spark.stop()

    # 6. Train & Serialize Lightweight Python/Scikit-Learn Model for Instant Dashboard Interactivity
    print("\n[INFO] Serializing Optimized Model Artifacts for Streamlit Web Dashboard...")
    pdf = pd.read_parquet(PROCESSED_FEATURE_STORE)
    
    # One-hot encode crop and season for sklearn
    df_encoded = pd.get_dummies(pdf, columns=["crop", "season"], drop_first=False)
    encoded_crop_cols = [c for c in df_encoded.columns if c.startswith("crop_") or c.startswith("season_")]
    all_feature_cols = ENGINEERED_FEATURES + encoded_crop_cols

    X_train = df_encoded[df_encoded["year"] <= 2022][all_feature_cols]
    y_train = df_encoded[df_encoded["year"] <= 2022][TARGET_VARIABLE]
    X_test = df_encoded[df_encoded["year"] >= 2023][all_feature_cols]
    y_test = df_encoded[df_encoded["year"] >= 2023][TARGET_VARIABLE]

    rf_dashboard = SklearnRF(n_estimators=100, max_depth=8, random_state=42)
    rf_dashboard.fit(X_train, y_train)
    y_pred = rf_dashboard.predict(X_test)

    sk_r2 = float(r2_score(y_test, y_pred))
    sk_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    sk_mae = float(mean_absolute_error(y_test, y_pred))

    model_metadata = {
        "best_model_name": best_model_name,
        "mllib_metrics": results,
        "test_r2": round(sk_r2, 4),
        "test_rmse_kg_per_ha": round(sk_rmse, 2),
        "test_mae_kg_per_ha": round(sk_mae, 2),
        "test_rmse_quintal_per_ha": round(sk_rmse / 100.0, 2),
        "feature_importances": importances if importances else [
            {"feature": f, "importance": round(float(imp), 4)}
            for f, imp in zip(all_feature_cols[:len(rf_dashboard.feature_importances_)], rf_dashboard.feature_importances_)
        ],
        "feature_columns": all_feature_cols,
        "engineered_features": ENGINEERED_FEATURES,
        "crops": sorted(list(pdf["crop"].unique())),
        "seasons": sorted(list(pdf["season"].unique()))
    }

    # Save metadata JSON
    metrics_file = MODELS_DIR / "model_evaluation_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(model_metadata, f, indent=4)

    # Save serialized model bundle
    bundle = {
        "model": rf_dashboard,
        "feature_columns": all_feature_cols,
        "crop_columns": encoded_crop_cols,
        "engineered_features": ENGINEERED_FEATURES,
        "metadata": model_metadata
    }
    joblib.dump(bundle, SKLEARN_MODEL_PATH)
    print(f"[SUCCESS] Model artifact saved to: {SKLEARN_MODEL_PATH}")
    print(f"[SUCCESS] Evaluation metrics saved to: {metrics_file}")
    print("=" * 75)
    return model_metadata

if __name__ == "__main__":
    run_ml_training_pipeline()
