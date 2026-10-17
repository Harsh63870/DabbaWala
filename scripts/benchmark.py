"""
scripts/benchmark.py
Benchmarks TabPFN vs XGBoost vs Linear Regression on delivery ETA prediction:
1. Full dataset 80/20 train/test evaluation.
2. Small-data experiments (50, 100, 200 training rows).
Saves results to docs/benchmark.json and creates a matplotlib plot docs/benchmark.png.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Ensure root directory is in sys.path
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.predictor import TabPFNPredictor
from app.baselines import XGBoostPredictor, LinearRegressionPredictor

DATA_FILE = os.path.join(ROOT_DIR, "data", "orders.csv")
DOCS_DIR = os.path.join(ROOT_DIR, "docs")
BENCHMARK_JSON = os.path.join(DOCS_DIR, "benchmark.json")
BENCHMARK_PNG = os.path.join(DOCS_DIR, "benchmark.png")

def evaluate_model(model, X_train, y_train, X_test, y_test) -> dict:
    fit_time = model.fit(X_train, y_train)

    t0 = time.time()
    preds = model.predict(X_test)
    predict_time = time.time() - t0

    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
        "fit_time_sec": round(float(fit_time), 4),
        "predict_time_sec": round(float(predict_time), 4),
        "sample_size": len(X_train),
    }

def run_benchmarks(df: pd.DataFrame = None):
    os.makedirs(DOCS_DIR, exist_ok=True)

    if df is None:
        if not os.path.exists(DATA_FILE):
            from scripts.make_dataset import main as make_data
            make_data()
        df = pd.read_csv(DATA_FILE)

    X = df.drop(columns=["Delivery_Time_min", "is_late", "Customer_Name", "Customer_Phone", "Delivery_Location"], errors="ignore")
    y = df["Delivery_Time_min"]

    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    models_to_test = {
        "TabPFN": lambda: TabPFNPredictor(),
        "XGBoost": lambda: XGBoostPredictor(),
        "LinearRegression": lambda: LinearRegressionPredictor(),
    }

    full_results = {}
    print(f"--- Running 80/20 Full Benchmark (Train={len(X_train)}, Test={len(X_test)}) ---")
    for name, factory in models_to_test.items():
        print(f"Training {name}...")
        m = factory()
        res = evaluate_model(m, X_train, y_train, X_test, y_test)
        full_results[name] = res
        print(f"  {name}: MAE={res['mae']:.2f} min, RMSE={res['rmse']:.2f}, R2={res['r2']:.3f}, Fit={res['fit_time_sec']:.3f}s")

    # Small-data experiments: 50, 100, 200 rows
    small_sizes = [50, 100, 200]
    small_results = {}

    print("\n--- Running Small-Data Experiments (50, 100, 200 rows) ---")
    for size in small_sizes:
        small_results[f"n_{size}"] = {}
        # Sample deterministic subset
        X_sub = X_train.iloc[:size]
        y_sub = y_train.iloc[:size]

        for name, factory in models_to_test.items():
            m = factory()
            res = evaluate_model(m, X_sub, y_sub, X_test, y_test)
            small_results[f"n_{size}"][name] = res
            print(f"  N={size} | {name}: MAE={res['mae']:.2f} min, R2={res['r2']:.3f}")

    benchmark_output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": len(df),
        "test_rows": len(X_test),
        "full_dataset": full_results,
        "small_data_experiments": small_results,
    }

    with open(BENCHMARK_JSON, "w") as f:
        json.dump(benchmark_output, f, indent=2)
    print(f"\nSaved benchmark metrics to {BENCHMARK_JSON}")

    # Generate Visualization
    generate_benchmark_chart(benchmark_output, BENCHMARK_PNG)
    print(f"Saved benchmark visualization to {BENCHMARK_PNG}")

    return benchmark_output


def generate_benchmark_chart(benchmark_data: dict, output_png_path: str):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    fig.patch.set_facecolor("#0f172a")

    # Chart 1: Full dataset 80/20 comparison (MAE vs Model)
    full = benchmark_data["full_dataset"]
    models = list(full.keys())
    maes = [full[m]["mae"] for m in models]
    fit_times = [full[m]["fit_time_sec"] for m in models]

    colors = ["#38bdf8", "#34d399", "#f87171"]
    bars = ax1.bar(models, maes, color=colors, width=0.5, edgecolor="#ffffff", linewidth=1.2)
    ax1.set_facecolor("#1e293b")
    ax1.set_title("Full Dataset (80/20) - Mean Absolute Error (lower is better)", color="#f8fafc", fontsize=11, fontweight="bold", pad=12)
    ax1.set_ylabel("MAE (Minutes)", color="#94a3b8", fontsize=10)
    ax1.tick_params(colors="#cbd5e1")
    ax1.grid(color="#334155", linestyle="--", alpha=0.7)

    for bar, mae, fit_t in zip(bars, maes, fit_times):
        height = bar.get_height()
        ax1.annotate(f"{mae:.2f} min\n({fit_t:.3f}s fit)",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 6), textcoords="offset points",
                     ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9)

    # Chart 2: Small-Data Regime (50, 100, 200 samples)
    small_exp = benchmark_data["small_data_experiments"]
    sample_sizes = [50, 100, 200]

    line_styles = {
        "TabPFN": {"color": "#38bdf8", "marker": "o", "linewidth": 2.5},
        "XGBoost": {"color": "#34d399", "marker": "s", "linewidth": 2.0},
        "LinearRegression": {"color": "#f87171", "marker": "^", "linewidth": 2.0},
    }

    ax2.set_facecolor("#1e293b")
    ax2.set_title("Small-Data Regime (50 to 200 rows) - TabPFN Advantage", color="#f8fafc", fontsize=11, fontweight="bold", pad=12)
    ax2.set_xlabel("Training Sample Size (N)", color="#94a3b8", fontsize=10)
    ax2.set_ylabel("MAE on Test Set (Minutes)", color="#94a3b8", fontsize=10)
    ax2.tick_params(colors="#cbd5e1")
    ax2.grid(color="#334155", linestyle="--", alpha=0.7)

    for model_name, style in line_styles.items():
        mae_curve = [small_exp[f"n_{n}"][model_name]["mae"] for n in sample_sizes]
        ax2.plot(sample_sizes, mae_curve, label=model_name, **style)

    ax2.set_xticks(sample_sizes)
    legend = ax2.legend(loc="upper right", frameon=True, facecolor="#0f172a", edgecolor="#334155")
    for text in legend.get_texts():
        text.set_color("#f8fafc")

    plt.tight_layout()
    plt.savefig(output_png_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()

if __name__ == "__main__":
    run_benchmarks()
