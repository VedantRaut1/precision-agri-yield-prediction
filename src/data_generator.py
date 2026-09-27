import numpy as np
import pandas as pd
import datetime
from pathlib import Path
import os
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.config import (
    SATELLITE_RAW_FILE, WEATHER_RAW_FILE, DISTRICT_METADATA_FILE,
    CROP_YIELD_GROUNDTRUTH_FILE, CROPS, SEASONS, BASE_YEARS
)

# Prominent Indian Agricultural Districts across Agro-Climatic Zones
# (Aligned with ISRO FASAL, MNCFC, and DES Ministry of Agriculture classification)
INDIAN_DISTRICT_PROFILES = [
    # Indo-Gangetic Plains - Breadbasket (Punjab & Haryana)
    {"district_id": "PB_LUD", "district": "Ludhiana", "state": "Punjab", "zone": "Trans-Gangetic Plains", "lat": 30.90, "lon": 75.85, "soil_soc": 0.85, "soil_type": "Alluvial Loam", "primary_crop": "Wheat", "season": "Rabi"},
    {"district_id": "PB_SAN", "district": "Sangrur", "state": "Punjab", "zone": "Trans-Gangetic Plains", "lat": 30.24, "lon": 75.84, "soil_soc": 0.82, "soil_type": "Alluvial Clay Loam", "primary_crop": "Rice (Paddy)", "season": "Kharif"},
    {"district_id": "PB_PAT", "district": "Patiala", "state": "Punjab", "zone": "Trans-Gangetic Plains", "lat": 30.33, "lon": 76.38, "soil_soc": 0.80, "soil_type": "Alluvial Loam", "primary_crop": "Wheat", "season": "Rabi"},
    {"district_id": "PB_BAT", "district": "Bathinda", "state": "Punjab", "zone": "Trans-Gangetic Plains", "lat": 30.21, "lon": 74.94, "soil_soc": 0.70, "soil_type": "Sandy Loam", "primary_crop": "Cotton", "season": "Kharif"},
    {"district_id": "HR_KAR", "district": "Karnal", "state": "Haryana", "zone": "Trans-Gangetic Plains", "lat": 29.68, "lon": 76.99, "soil_soc": 0.88, "soil_type": "Alluvial Loam", "primary_crop": "Rice (Paddy)", "season": "Kharif"},
    {"district_id": "HR_KUR", "district": "Kurukshetra", "state": "Haryana", "zone": "Trans-Gangetic Plains", "lat": 29.96, "lon": 76.87, "soil_soc": 0.84, "soil_type": "Alluvial Loam", "primary_crop": "Wheat", "season": "Rabi"},
    {"district_id": "HR_SIR", "district": "Sirsa", "state": "Haryana", "zone": "Trans-Gangetic Plains", "lat": 29.53, "lon": 75.02, "soil_soc": 0.65, "soil_type": "Arid Sandy Loam", "primary_crop": "Mustard", "season": "Rabi"},

    # Upper & Middle Gangetic Plains (Uttar Pradesh)
    {"district_id": "UP_MEE", "district": "Meerut", "state": "Uttar Pradesh", "zone": "Upper Gangetic Plains", "lat": 28.98, "lon": 77.70, "soil_soc": 0.78, "soil_type": "Alluvial Deep Loam", "primary_crop": "Wheat", "season": "Rabi"},
    {"district_id": "UP_ALI", "district": "Aligarh", "state": "Uttar Pradesh", "zone": "Upper Gangetic Plains", "lat": 27.89, "lon": 78.08, "soil_soc": 0.72, "soil_type": "Sandy Clay Loam", "primary_crop": "Mustard", "season": "Rabi"},
    {"district_id": "UP_VAR", "district": "Varanasi", "state": "Uttar Pradesh", "zone": "Middle Gangetic Plains", "lat": 25.31, "lon": 82.97, "soil_soc": 0.75, "soil_type": "Older Alluvial (Bangar)", "primary_crop": "Rice (Paddy)", "season": "Kharif"},
    {"district_id": "UP_BAR", "district": "Bareilly", "state": "Uttar Pradesh", "zone": "Upper Gangetic Plains", "lat": 28.36, "lon": 79.43, "soil_soc": 0.76, "soil_type": "Tarai Silt Loam", "primary_crop": "Wheat", "season": "Rabi"},
    {"district_id": "UP_GOR", "district": "Gorakhpur", "state": "Uttar Pradesh", "zone": "Middle Gangetic Plains", "lat": 26.76, "lon": 83.37, "soil_soc": 0.79, "soil_type": "Alluvial Clay Loam", "primary_crop": "Rice (Paddy)", "season": "Kharif"},

    # Central Plateau & Soybean/Wheat Heart (Madhya Pradesh)
    {"district_id": "MP_IND", "district": "Indore", "state": "Madhya Pradesh", "zone": "Central Plateau", "lat": 22.71, "lon": 75.85, "soil_soc": 0.68, "soil_type": "Medium Black (Vertisol)", "primary_crop": "Soybean", "season": "Kharif"},
    {"district_id": "MP_UJJ", "district": "Ujjain", "state": "Madhya Pradesh", "zone": "Central Plateau", "lat": 23.17, "lon": 75.78, "soil_soc": 0.70, "soil_type": "Deep Black Soil", "primary_crop": "Soybean", "season": "Kharif"},
    {"district_id": "MP_HOS", "district": "Narmadapuram", "state": "Madhya Pradesh", "zone": "Central Plateau", "lat": 22.75, "lon": 77.72, "soil_soc": 0.82, "soil_type": "Deep Clay Alluvial", "primary_crop": "Wheat", "season": "Rabi"},
    {"district_id": "MP_SEH", "district": "Sehore", "state": "Madhya Pradesh", "zone": "Central Plateau", "lat": 23.20, "lon": 77.08, "soil_soc": 0.69, "soil_type": "Medium Black Soil", "primary_crop": "Soybean", "season": "Kharif"},

    # Western Black Soil & Cotton Belt (Maharashtra & Gujarat)
    {"district_id": "MH_NAS", "district": "Nashik", "state": "Maharashtra", "zone": "Western Plateau", "lat": 19.99, "lon": 73.78, "soil_soc": 0.65, "soil_type": "Black Basaltic Soil", "primary_crop": "Maize", "season": "Kharif"},
    {"district_id": "MH_AUR", "district": "Chhatrapati Sambhajinagar", "state": "Maharashtra", "zone": "Western Plateau", "lat": 19.87, "lon": 75.34, "soil_soc": 0.58, "soil_type": "Medium Black Soil", "primary_crop": "Cotton", "season": "Kharif"},
    {"district_id": "MH_NAG", "district": "Nagpur", "state": "Maharashtra", "zone": "Eastern Plateau", "lat": 21.14, "lon": 79.08, "soil_soc": 0.66, "soil_type": "Deep Black Soil", "primary_crop": "Cotton", "season": "Kharif"},
    {"district_id": "MH_KOL", "district": "Kolhapur", "state": "Maharashtra", "zone": "Western Ghats & Coastal", "lat": 16.70, "lon": 74.24, "soil_soc": 0.85, "soil_type": "Lateritic & Red Loam", "primary_crop": "Rice (Paddy)", "season": "Kharif"},
    {"district_id": "GJ_RAJ", "district": "Rajkot", "state": "Gujarat", "zone": "Gujarat Plains & Hills", "lat": 22.30, "lon": 70.80, "soil_soc": 0.52, "soil_type": "Medium Black & Sandy", "primary_crop": "Cotton", "season": "Kharif"},

    # Lower Gangetic Rice Bowl (West Bengal)
    {"district_id": "WB_BUR", "district": "Purba Bardhaman", "state": "West Bengal", "zone": "Lower Gangetic Plains", "lat": 23.23, "lon": 87.86, "soil_soc": 0.95, "soil_type": "Fertile Delta Alluvium", "primary_crop": "Rice (Paddy)", "season": "Kharif"},
    {"district_id": "WB_HOO", "district": "Hooghly", "state": "West Bengal", "zone": "Lower Gangetic Plains", "lat": 22.90, "lon": 88.39, "soil_soc": 0.92, "soil_type": "Gangetic Alluvium", "primary_crop": "Rice (Paddy)", "season": "Kharif"},

    # Southern Agro-Climatic Zone (Telangana, Andhra Pradesh, Karnataka)
    {"district_id": "TS_WAR", "district": "Warangal", "state": "Telangana", "zone": "Southern Plateau", "lat": 17.96, "lon": 79.59, "soil_soc": 0.58, "soil_type": "Red Sandy Loam (Chalka)", "primary_crop": "Cotton", "season": "Kharif"},
    {"district_id": "AP_GUN", "district": "Guntur", "state": "Andhra Pradesh", "zone": "East Coast Plains", "lat": 16.30, "lon": 80.43, "soil_soc": 0.74, "soil_type": "Krishna Delta Alluvial", "primary_crop": "Rice (Paddy)", "season": "Kharif"},
    {"district_id": "KA_BEL", "district": "Belagavi", "state": "Karnataka", "zone": "Southern Plateau", "lat": 15.84, "lon": 74.49, "soil_soc": 0.72, "soil_type": "Mixed Red & Black", "primary_crop": "Maize", "season": "Kharif"},
    {"district_id": "KA_MAN", "district": "Mandya", "state": "Karnataka", "zone": "Southern Plateau", "lat": 12.52, "lon": 76.89, "soil_soc": 0.70, "soil_type": "Red Sandy Loam", "primary_crop": "Rice (Paddy)", "season": "Kharif"},

    # Arid & Semi-Arid Zone (Rajasthan)
    {"district_id": "RJ_GAN", "district": "Sri Ganganagar", "state": "Rajasthan", "zone": "Western Dry Region", "lat": 29.90, "lon": 73.87, "soil_soc": 0.45, "soil_type": "Canal Irrigated Sandy Loam", "primary_crop": "Wheat", "season": "Rabi"},
    {"district_id": "RJ_KOT", "district": "Kota", "state": "Rajasthan", "zone": "Central Plateau", "lat": 25.18, "lon": 75.83, "soil_soc": 0.64, "soil_type": "Black Vertisol", "primary_crop": "Mustard", "season": "Rabi"},
    {"district_id": "RJ_BHA", "district": "Bharatpur", "state": "Rajasthan", "zone": "Eastern Plains", "lat": 27.21, "lon": 77.48, "soil_soc": 0.60, "soil_type": "Alluvial Sandy Loam", "primary_crop": "Mustard", "season": "Rabi"},
]

def generate_indian_spatiotemporal_dataset(seed=42):
    """
    Generates synthetic Indian agricultural satellite imagery time-series (Sentinel-2 / Resourcesat AWiFS emulation),
    IMD/ERA5 agro-meteorology, and district-level crop yield targets (DES / PMFBY standards).
    """
    np.random.seed(seed)
    print("[INFO] Generating Indian Spatiotemporal Agricultural Benchmark Dataset (ISRO/DES Standards)...")

    # 1. District Metadata
    df_districts = pd.DataFrame(INDIAN_DISTRICT_PROFILES)
    df_districts.to_csv(DISTRICT_METADATA_FILE, index=False)
    print(f"[SUCCESS] District Metadata saved: {len(df_districts)} districts across 10 Indian states.")

    satellite_records = []
    weather_records = []
    groundtruth_records = []

    # Historical Indian Monsoon & Temperature Anomalies (2017 - 2024)
    # Reflects actual Indian meteorological patterns:
    # 2018: Sub-par monsoon in West/Central India
    # 2019: Extended excess monsoon (floods in Maharashtra/Karnataka)
    # 2020: Above normal monsoon (La Niña year)
    # 2021: Normal monsoon, late withdrawal
    # 2022: Severe early March terminal heatwave impacting Rabi Wheat!
    # 2023: El Niño year, deficit August rainfall (driest August in 122 years in India)
    # 2024: Above normal southwest monsoon with good spatial distribution
    year_monsoon_profile = {
        2017: {"monsoon_idx": 0.95, "terminal_heat": 1.0, "anomaly": "Normal Monsoon"},
        2018: {"monsoon_idx": 0.88, "terminal_heat": 1.2, "anomaly": "Mild Deficit Monsoon"},
        2019: {"monsoon_idx": 1.20, "terminal_heat": 0.8, "anomaly": "Excess Monsoon & Late Floods"},
        2020: {"monsoon_idx": 1.09, "terminal_heat": 0.9, "anomaly": "La Nina Favorable Monsoon"},
        2021: {"monsoon_idx": 1.02, "terminal_heat": 1.0, "anomaly": "Normal Monsoon with Late Rains"},
        2022: {"monsoon_idx": 1.06, "terminal_heat": 2.4, "anomaly": "Record March Terminal Heatwave"},
        2023: {"monsoon_idx": 0.86, "terminal_heat": 1.8, "anomaly": "El Nino Deficit August Dry Spell"},
        2024: {"monsoon_idx": 1.08, "terminal_heat": 1.1, "anomaly": "Bountiful Monsoon Distribution"}
    }

    weeks_in_season = 20  # 20 weeks per season (covers sowing to harvest)

    for year in BASE_YEARS:
        climate_meta = year_monsoon_profile[year]
        monsoon_mult = climate_meta["monsoon_idx"]
        heatwave_mult = climate_meta["terminal_heat"]

        for _, district in df_districts.iterrows():
            d_id = district["district_id"]
            d_name = district["district"]
            d_state = district["state"]
            lat = district["lat"]
            lon = district["lon"]
            soc = district["soil_soc"]
            primary_crop = district["primary_crop"]
            season = district["season"]

            d_seed = abs(hash(d_id)) % 100000 + year
            rng = np.random.RandomState(d_seed)

            # Local meteorological variation
            local_rain_mult = rng.normal(1.0, 0.14)
            local_temp_mult = rng.normal(0.0, 0.5)

            # Cumulative season trackers
            season_rain_total = 0.0
            critical_window_rain = 0.0
            cumulative_gdd = 0.0
            heat_stress_days_total = 0
            temp_mean_sum = 0.0
            solar_rad_total = 0.0

            ndvi_series = []
            evi_series = []
            ndre_series = []
            ndwi_series = []

            # Determine season timing
            # Kharif: Calendar week 24 (mid June) to week 43 (late October)
            # Rabi: Calendar week 44 (early November) to week 13 (late March)
            start_week = 24 if season == "Kharif" else 44
            critical_window_start = 32 if season == "Kharif" else 7  # Tillering/Flowering or Heading
            critical_window_end = 36 if season == "Kharif" else 10

            for w in range(weeks_in_season):
                week_num = (start_week + w) % 52
                if week_num == 0:
                    week_num = 52

                # Phenology peak week: mid-season (week 9-11 from sowing)
                pheno_progress = np.exp(-0.5 * ((w - 9.5) / 3.8) ** 2)

                # Temperature modeling for Indian seasons
                if season == "Kharif":
                    # Monsoon: warm and humid, 28C - 36C
                    t_mean = 29.5 + rng.normal(0, 1.2) + local_temp_mult
                    t_max = t_mean + rng.uniform(4.0, 7.5)
                    t_min = t_mean - rng.uniform(3.5, 6.0)
                    # Southwest Monsoon rainfall
                    base_rain = rng.exponential(scale=38.0) * monsoon_mult * local_rain_mult
                    if year == 2023 and (w in [7, 8, 9, 10]):  # August 2023 deficit
                        base_rain *= 0.28
                    heat_days = int(np.clip((t_max - 35.0) * 1.5, 0, 7))
                else:
                    # Rabi: cool winters (Nov-Jan) warming up in Feb-March
                    winter_cool = 8.0 * np.sin(np.pi * w / 19.0)
                    t_mean = 17.5 + winter_cool + local_temp_mult
                    t_max = t_mean + rng.uniform(5.0, 9.0)
                    t_min = t_mean - rng.uniform(4.0, 7.0)
                    # Terminal heat in March (end of Rabi season, weeks 14-19)
                    if w >= 14:
                        terminal_heat_boost = 3.5 * heatwave_mult
                        t_max += terminal_heat_boost
                        t_mean += terminal_heat_boost * 0.7
                    # Rabi rainfall (Western disturbances / winter showers)
                    base_rain = rng.exponential(scale=6.5) * local_rain_mult
                    # Terminal heat stress days (t_max > 32°C during grain fill)
                    heat_days = int(np.clip((t_max - 31.5) * 1.8, 0, 7)) if w >= 13 else 0

                # Accumulations
                season_rain_total += base_rain
                if critical_window_start <= (w + start_week) <= critical_window_end:
                    critical_window_rain += base_rain

                gdd_week = max(0.0, t_mean - 10.0) * 7.0
                cumulative_gdd += gdd_week
                heat_stress_days_total += heat_days
                temp_mean_sum += t_mean

                solar_rad = (18.0 + rng.uniform(-2.5, 3.5)) * 7.0
                solar_rad_total += solar_rad

                # Soil moisture estimation
                soil_moisture = np.clip(18.0 + 0.35 * base_rain - 0.3 * t_max + 12.0 * soc, 6.0, 48.0)

                weather_records.append({
                    "district_id": d_id,
                    "year": year,
                    "week_in_season": w + 1,
                    "calendar_week": week_num,
                    "season": season,
                    "precip_week_mm": round(base_rain, 2),
                    "temp_max_c": round(t_max, 2),
                    "temp_min_c": round(t_min, 2),
                    "temp_mean_c": round(t_mean, 2),
                    "gdd_week": round(gdd_week, 2),
                    "heat_stress_days": heat_days,
                    "soil_moisture_pct": round(soil_moisture, 2),
                    "solar_radiation_mj": round(solar_rad, 2)
                })

                # Satellite Spectral Indices (Sentinel-2 / Resourcesat emulation)
                crop_max_ndvi = {
                    "Rice (Paddy)": 0.86,
                    "Wheat": 0.85,
                    "Cotton": 0.78,
                    "Soybean": 0.82,
                    "Maize": 0.84,
                    "Mustard": 0.79
                }[primary_crop]

                soil_base_ndvi = 0.16
                water_stress = np.clip(soil_moisture / 22.0, 0.45, 1.0)
                current_ndvi = soil_base_ndvi + (crop_max_ndvi - soil_base_ndvi) * pheno_progress * water_stress
                current_ndvi += rng.normal(0, 0.015)
                current_ndvi = float(np.clip(current_ndvi, 0.10, 0.90))

                # Synthetic multi-spectral bands
                red_val = float(np.clip(0.15 - 0.11 * pheno_progress * water_stress + rng.uniform(-0.01, 0.01), 0.02, 0.22))
                nir_val = float(np.clip(red_val * (1.0 + current_ndvi) / (1.0 - current_ndvi + 1e-5), 0.12, 0.68))
                blue_val = float(np.clip(red_val * 0.65 + rng.uniform(-0.005, 0.005), 0.01, 0.14))
                green_val = float(np.clip(red_val * 1.15 + 0.04 * pheno_progress, 0.03, 0.19))
                rededge_val = float(np.clip(red_val + 0.60 * (nir_val - red_val), 0.07, 0.46))
                swir_val = float(np.clip(nir_val * (1.05 - 0.45 * (soil_moisture / 45.0)), 0.04, 0.38))

                evi_val = float(2.5 * (nir_val - red_val) / (nir_val + 6.0 * red_val - 7.5 * blue_val + 1.0))
                ndre_val = float((nir_val - rededge_val) / (nir_val + rededge_val + 1e-5))
                ndwi_val = float((nir_val - swir_val) / (nir_val + swir_val + 1e-5))

                ndvi_series.append(current_ndvi)
                evi_series.append(evi_val)
                ndre_series.append(ndre_val)
                ndwi_series.append(ndwi_val)

                satellite_records.append({
                    "district_id": d_id,
                    "year": year,
                    "week_in_season": w + 1,
                    "calendar_week": week_num,
                    "season": season,
                    "band_blue": round(blue_val, 4),
                    "band_green": round(green_val, 4),
                    "band_red": round(red_val, 4),
                    "band_rededge": round(rededge_val, 4),
                    "band_nir": round(nir_val, 4),
                    "band_swir": round(swir_val, 4),
                    "ndvi": round(current_ndvi, 4),
                    "evi": round(np.clip(evi_val, 0.0, 1.0), 4),
                    "ndre": round(np.clip(ndre_val, -0.2, 0.8), 4),
                    "ndwi": round(np.clip(ndwi_val, -0.5, 0.8), 4)
                })

            # --- Target Crop Yield Computation (Aligned with Indian DES Yield Averages) ---
            # Yield ranges in India:
            # - Wheat: 3500 - 5400 kg/ha (Punjab/Haryana at top, 5000+ kg/ha)
            # - Rice: 2800 - 4800 kg/ha (Punjab/WB highest)
            # - Cotton: 1600 - 2800 kg/ha (seed cotton)
            # - Soybean: 1200 - 2400 kg/ha
            # - Maize: 3000 - 5200 kg/ha
            # - Mustard: 1100 - 2100 kg/ha
            peak_ndvi = max(ndvi_series)
            integral_ndvi = sum(ndvi_series)

            base_params = {
                "Wheat": {"base": 4200.0, "ndvi_factor": 2200.0, "ndvi_thresh": 0.50, "heat_pen": -42.0, "soc_factor": 800.0, "noise": 120.0},
                "Rice (Paddy)": {"base": 3600.0, "ndvi_factor": 1900.0, "ndvi_thresh": 0.48, "heat_pen": -28.0, "soc_factor": 700.0, "noise": 110.0},
                "Cotton": {"base": 1900.0, "ndvi_factor": 1100.0, "ndvi_thresh": 0.42, "heat_pen": -18.0, "soc_factor": 450.0, "noise": 85.0},
                "Soybean": {"base": 1600.0, "ndvi_factor": 950.0, "ndvi_thresh": 0.44, "heat_pen": -22.0, "soc_factor": 400.0, "noise": 75.0},
                "Maize": {"base": 3800.0, "ndvi_factor": 1800.0, "ndvi_thresh": 0.46, "heat_pen": -30.0, "soc_factor": 650.0, "noise": 115.0},
                "Mustard": {"base": 1400.0, "ndvi_factor": 800.0, "ndvi_thresh": 0.40, "heat_pen": -25.0, "soc_factor": 350.0, "noise": 70.0}
            }[primary_crop]

            y_ndvi = base_params["ndvi_factor"] * (peak_ndvi - base_params["ndvi_thresh"])
            y_heat = base_params["heat_pen"] * heat_stress_days_total
            y_soc = base_params["soc_factor"] * (soc - 0.65)
            y_noise = rng.normal(0, base_params["noise"])

            # Water response
            if season == "Kharif":
                # Rainfed sensitivity
                water_resp = 500.0 * np.tanh((season_rain_total - 650.0) / 250.0)
            else:
                # Rabi irrigated sensitivity + winter rain
                water_resp = 250.0 * np.tanh((critical_window_rain - 35.0) / 20.0)

            yield_kg = base_params["base"] + y_ndvi + y_heat + y_soc + water_resp + y_noise
            yield_kg = float(np.clip(yield_kg, 800.0, 6200.0))

            groundtruth_records.append({
                "district_id": d_id,
                "district": d_name,
                "state": d_state,
                "zone": district["zone"],
                "crop": primary_crop,
                "season": season,
                "year": year,
                "yield_kg_per_ha": round(yield_kg, 1),
                "yield_quintal_per_ha": round(yield_kg / 100.0, 2),
                "yield_tonnes_per_ha": round(yield_kg / 1000.0, 3)
            })

    # Save to Parquet and CSV
    df_sat = pd.DataFrame(satellite_records)
    df_weather = pd.DataFrame(weather_records)
    df_yield = pd.DataFrame(groundtruth_records)

    df_sat.to_parquet(SATELLITE_RAW_FILE, index=False)
    df_weather.to_parquet(WEATHER_RAW_FILE, index=False)
    df_yield.to_csv(CROP_YIELD_GROUNDTRUTH_FILE, index=False)

    print(f"[SUCCESS] Indian Satellite Time-Series: {SATELLITE_RAW_FILE} ({len(df_sat):,} records)")
    print(f"[SUCCESS] Indian Weather Time-Series: {WEATHER_RAW_FILE} ({len(df_weather):,} records)")
    print(f"[SUCCESS] Indian Yield Targets: {CROP_YIELD_GROUNDTRUTH_FILE} ({len(df_yield):,} district-year targets)")
    return df_sat, df_weather, df_yield

if __name__ == "__main__":
    generate_indian_spatiotemporal_dataset()
