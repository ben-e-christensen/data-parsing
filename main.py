#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, os

# === CONFIG ===
BASE_DIR = "/media/ben/Extreme SSD/particle-data"
DERIVATIVE_THRESHOLD = -1
BIN_SIZE = 2000  # number of index points per bin

# --- Parse argument like "Acrylic/Acrylic-200" ---
if len(sys.argv) < 2:
    print("Usage: python3 main.py <Material/RunFolder>")
    print("Example: python3 main.py Acrylic/Acrylic-200")
    sys.exit(1)

rel_path = sys.argv[1]
run_dir = os.path.join(BASE_DIR, rel_path)
input_csv = os.path.join(run_dir, "experiment_log.csv")

if not os.path.isfile(input_csv):
    print(f"❌ Error: {input_csv} not found")
    sys.exit(1)

# --- Output paths ---
peaks_csv = os.path.join(run_dir, "fall_local_maxima.csv")
summary_csv = os.path.join(run_dir, "speed_run_summary.csv")
plot_dir = run_dir

run_name = os.path.basename(run_dir)
material_name = run_name.replace("-", " ")

# --- Load CSV ---
cols = [
    "index", "timestamp", "seq", "ms",
    "motor_angle_deg", "motor_speed", "CH0_volts", "CH2_volts", "CH3_volts",
    "ellipse_angle_deg", "ellipse_area_px2", "frame_name",
    "ch2_dv/dt", "ch3_dv/dt", "ch2_flag", "ch3_flag"
]
df = pd.read_csv(input_csv, names=cols, header=0, on_bad_lines="skip", engine="python")
df["ellipse_angle_derivative"] = df["ellipse_angle_deg"].diff()

# === STEP 1: Detect local maxima ===
peaks = []
i = 1
while i < len(df):
    if df.loc[i, "ellipse_angle_derivative"] <= DERIVATIVE_THRESHOLD:
        j = i - 1
        while j > 0 and df.loc[j - 1, "ellipse_angle_deg"] >= df.loc[j, "ellipse_angle_deg"]:
            j -= 1
        peaks.append(df.loc[j])
        while i < len(df) - 1 and df.loc[i + 1, "ellipse_angle_deg"] <= df.loc[i, "ellipse_angle_deg"]:
            i += 1
    i += 1

result_df = pd.DataFrame(peaks).drop_duplicates(subset="index")
result_df = result_df[result_df["ellipse_angle_deg"] <= 70]

result_df[["index", "ellipse_angle_deg", "ellipse_angle_derivative", "motor_speed"]].to_csv(peaks_csv, index=False)
print(f"✅ Saved → {peaks_csv} ({len(result_df)} local maxima)")

# === STEP 2: Summarize runs ===
summary = None
if len(result_df) > 0:
    result_df["speed_change"] = result_df["motor_speed"].ne(result_df["motor_speed"].shift())
    result_df["run_id"] = result_df["speed_change"].cumsum()

    summary = (
        result_df.groupby("run_id")
        .agg(
            motor_speed=("motor_speed", "first"),
            num_points=("ellipse_angle_deg", "size"),
            angle_max=("ellipse_angle_deg", "max"),
            angle_min=("ellipse_angle_deg", "min"),
            angle_median=("ellipse_angle_deg", "median"),
            angle_mean=("ellipse_angle_deg", "mean"),
            angle_std=("ellipse_angle_deg", "std"),
            start_index=("index", "min"),
            end_index=("index", "max"),
        )
        .reset_index(drop=True)
    )

    summary.to_csv(summary_csv, index=False)
    print(f"✅ Saved → {summary_csv} ({len(summary)} speed runs summarized)")
else:
    print("⚠️ No valid peaks found — skipping summary.")

# === STEP 3: Plots ===
if len(result_df) > 0:
    # 1️⃣ Scatter: angle vs speed
    plt.figure(figsize=(8, 5))
    plt.scatter(result_df["motor_speed"], result_df["ellipse_angle_deg"], alpha=0.7)
    plt.xlabel("Motor Speed (RPM)")
    plt.ylabel("Ellipse Angle (°)")
    plt.title(f"{material_name} — Angle of Repose vs Motor Speed")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{run_name}_angle_vs_speed.png"))
    plt.close()

    # 2️⃣ Trace: index vs angle (filtered ≤70°) + slip bins overlay
    filtered_df = df[df["ellipse_angle_deg"] <= 70].copy()
    filtered_df["is_slip"] = filtered_df["ellipse_angle_derivative"] < 0

    # --- Compute slip counts in bins of 2000 indices ---
    filtered_df["index_bin"] = (filtered_df["index"] // BIN_SIZE) * BIN_SIZE
    slip_bins = (
        filtered_df.groupby("index_bin")["is_slip"]
        .sum()
        .reset_index(name="num_slips")
    )

    # --- Plot setup ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    # --- Angle trace and peaks ---
    ax1.plot(
        filtered_df["index"], filtered_df["ellipse_angle_deg"],
        color="tab:gray", lw=0.8, label="Raw Angle (≤70°)"
    )
    ax1.scatter(
        result_df["index"], result_df["ellipse_angle_deg"],
        color="red", s=25, label="Detected Peaks"
    )

    # --- Slip markers ---
    slip_points = filtered_df[filtered_df["is_slip"]]
    ax1.scatter(
        slip_points["index"], slip_points["ellipse_angle_deg"],
        color="tab:blue", s=8, alpha=0.5, label="Slip (neg. derivative)"
    )

    # --- Slip bin line overlay ---
    ax2.plot(
        slip_bins["index_bin"], slip_bins["num_slips"],
        color="tab:green", marker="s", markersize=6,
        lw=1.8, label=f"Slip Count per {BIN_SIZE:,} indices"
    )

    # --- Labels, legend, grid ---
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Ellipse Angle (°)")
    ax2.set_ylabel(f"Slip Count / {BIN_SIZE} indices", color="tab:green")
    ax2.tick_params(axis='y', labelcolor="tab:green")
    ax1.set_title(f"{material_name} — Angle Trace with Slip Bins ({BIN_SIZE} index window)")
    ax1.grid(True)

    lns1, labs1 = ax1.get_legend_handles_labels()
    lns2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lns1 + lns2, labs1 + labs2, loc="best")

    plt.tight_layout()
    trace_path = os.path.join(plot_dir, f"{run_name}_angle_trace.png")
    plt.savefig(trace_path)
    plt.close()
    print(f"📊 Saved → {trace_path}")

# === STEP 4: Sweep Comparison (simplified) ===
if summary is not None and len(summary) > 0:
    half = len(summary) // 2
    first_half = summary.iloc[:half].copy()
    second_half = summary.iloc[half:].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    if len(first_half) > 0:
        ax.errorbar(
            first_half["motor_speed"], first_half["angle_mean"],
            yerr=first_half["angle_std"],
            fmt="o-", lw=2, capsize=5, color="tab:blue", label="First Sweep (1→25)"
        )
    if len(second_half) > 0:
        ax.errorbar(
            second_half["motor_speed"], second_half["angle_mean"],
            yerr=second_half["angle_std"],
            fmt="o-", lw=2, capsize=5, color="tab:orange", label="Second Sweep (25→1)"
        )

    ax.set_xlabel("Motor Speed (RPM)")
    ax.set_ylabel("Mean Angle of Repose (°)")
    ax.set_title(f"{material_name} — Mean ± Std Dev")
    ax.grid(True)
    ax.legend(loc="best")
    plt.tight_layout()

    overlay_path = os.path.join(plot_dir, f"{run_name}_sweep_comparison.png")
    plt.savefig(overlay_path)
    plt.close()
    print(f"📊 Saved → {overlay_path}")

print("✅ Processing complete.")
