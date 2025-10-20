import pandas as pd

input_csv = "experiment_log.csv"
output_csv = "fall_local_maxima.csv"
DERIVATIVE_THRESHOLD = -1

cols = [
    "index", "timestamp", "seq", "ms",
    "motor_angle_deg", "motor_speed", "CH0_volts", "CH2_volts", "CH3_volts",
    "ellipse_angle_deg", "ellipse_area_px2", "frame_name",
    "ch2_dv/dt", "ch3_dv/dt", "ch2_flag", "ch3_flag"
]

df = pd.read_csv(input_csv, names=cols, header=0, on_bad_lines="skip", engine="python")
df["ellipse_angle_derivative"] = df["ellipse_angle_deg"].diff()

peaks = []
i = 1

while i < len(df):
    if df.loc[i, "ellipse_angle_derivative"] <= DERIVATIVE_THRESHOLD:
        # --- Backtrack to local max ---
        j = i - 1
        while j > 0 and df.loc[j - 1, "ellipse_angle_deg"] >= df.loc[j, "ellipse_angle_deg"]:
            j -= 1

        peaks.append(df.loc[j])

        # --- Skip forward until fall ends ---
        while i < len(df) - 1 and df.loc[i + 1, "ellipse_angle_deg"] <= df.loc[i, "ellipse_angle_deg"]:
            i += 1
    i += 1

# --- Remove potential duplicates (same index) ---
result_df = pd.DataFrame(peaks).drop_duplicates(subset="index")
# Filter out any peaks where the angle exceeds 70°
result_df = result_df[result_df["ellipse_angle_deg"] <= 70]

result_df[["index", "ellipse_angle_deg", "ellipse_angle_derivative", "motor_speed"]].to_csv(output_csv, index=False)



print(f"Saved → {output_csv} ({len(result_df)} unique local maxima found)")
