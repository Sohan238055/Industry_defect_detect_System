
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# PATHS
# ==========================================

RESULTS = Path("results")

PATCHCORE_CSV = RESULTS / "patchcore_fast_results.csv"
DINOMALY_CSV = RESULTS / "dinomaly_all_categories.csv"

OUTPUT_DIR = RESULTS / "comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "Image_AUROC",
    "Image_F1Score",
    "Pixel_AUROC",
    "Pixel_F1Score",
]

# ==========================================
# LOAD CSV FILES
# ==========================================

patchcore = pd.read_csv(PATCHCORE_CSV)
dinomaly = pd.read_csv(DINOMALY_CSV)

print("\nPATCHCORE COLUMNS:")
print(patchcore.columns.tolist())

print("\nDINOMALY COLUMNS:")
print(dinomaly.columns.tolist())

# Remove accidental spaces from column names
patchcore.columns = patchcore.columns.str.strip()
dinomaly.columns = dinomaly.columns.str.strip()

# Check required columns
required = ["Category"] + METRICS

for name, df in [
    ("PatchCore Fast", patchcore),
    ("Dinomaly", dinomaly),
]:
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"{name} CSV is missing columns: {missing}"
        )

# ==========================================
# PREPARE DATA
# ==========================================

patchcore = patchcore[required].copy()
dinomaly = dinomaly[required].copy()

patchcore = patchcore.rename(
    columns={
        metric: f"PatchCore_{metric}"
        for metric in METRICS
    }
)

dinomaly = dinomaly.rename(
    columns={
        metric: f"Dinomaly_{metric}"
        for metric in METRICS
    }
)

comparison = pd.merge(
    patchcore,
    dinomaly,
    on="Category",
    how="inner"
)

if comparison.empty:
    raise ValueError(
        "No matching categories found in the two CSV files."
    )

# ==========================================
# SAVE CATEGORY-WISE COMPARISON
# ==========================================

comparison.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

# ==========================================
# CALCULATE AVERAGES
# ==========================================

summary_rows = []

for metric in METRICS:
    patch_col = f"PatchCore_{metric}"
    dinomaly_col = f"Dinomaly_{metric}"

    summary_rows.append({
        "Metric": metric,
        "PatchCore_Fast_Mean": comparison[patch_col].mean(),
        "Dinomaly_Mean": comparison[dinomaly_col].mean(),
        "Difference": (
            comparison[patch_col].mean()
            - comparison[dinomaly_col].mean()
        ),
    })

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUTPUT_DIR / "model_summary.csv",
    index=False
)

# ==========================================
# GENERATE COMPARISON GRAPHS
# ==========================================

for metric in METRICS:

    patch_col = f"PatchCore_{metric}"
    dinomaly_col = f"Dinomaly_{metric}"

    ax = comparison.set_index("Category")[
        [patch_col, dinomaly_col]
    ].plot(
        kind="bar",
        figsize=(14, 6),
        width=0.8
    )

    ax.set_title(f"PatchCore Fast vs Dinomaly - {metric}")
    ax.set_xlabel("Category")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.legend(["PatchCore Fast", "Dinomaly"])
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{metric}_comparison.png",
        dpi=120
    )

    plt.close()

# ==========================================
# FINAL OUTPUT
# ==========================================

print("\n" + "=" * 55)
print("MODEL COMPARISON COMPLETED")
print("=" * 55)

print("\nCATEGORY-WISE RESULTS:")
print(comparison.to_string(index=False))

print("\nAVERAGE RESULTS:")
print(summary.to_string(index=False))

print("\nFiles saved inside:", OUTPUT_DIR)
print("- model_comparison.csv")
print("- model_summary.csv")
print("- Four comparison graphs")