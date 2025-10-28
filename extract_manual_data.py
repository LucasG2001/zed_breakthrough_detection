import os
import json
import pandas as pd
from tabulate import tabulate  # for pretty printing

base_dir = "results_manual"
output_rows = []

for participant_id in sorted(os.listdir(base_dir), key=lambda x: int(x) if x.isdigit() else x):
    participant_path = os.path.join(base_dir, participant_id)
    if not os.path.isdir(participant_path):
        continue

    SP, SP_E, OE, OE_E = [], [], [], []

    for i in range(1, 4):  # always 1.json, 2.json, 3.json
        file_path = os.path.join(participant_path, f"{i}.json")
        if not os.path.exists(file_path):
            print(f"⚠️ Missing file: {file_path}")
            SP.append(None)
            SP_E.append(None)
            OE.append(None)
            OE_E.append(None)
            continue

        with open(file_path, "r") as f:
            data = json.load(f)[0]

        SP.append(data["soft_tissue_penetration_mm_constrained"])
        SP_E.append(abs(122 - data["distance_weighted_mm"]))
        OE.append(data["orientation_error_constrained"])
        OE_E.append(abs(data["orientation_error_constrained"] - data["orientation_error_weighted"]))

    # Compute averages safely
    def avg(lst):
        valid = [v for v in lst if v is not None]
        return sum(valid) / len(valid) if valid else None

    row = {
        "participant_id": int(participant_id) if participant_id.isdigit() else participant_id,
        "SP1": SP[0], "SP2": SP[1], "SP3": SP[2],
        "SP_E1": SP_E[0], "SP_E2": SP_E[1], "SP_E3": SP_E[2],
        "OE1": OE[0], "OE2": OE[1], "OE3": OE[2],
        "OE_E1": OE_E[0], "OE_E2": OE_E[1], "OE_E3": OE_E[2],
        "AVG_SP": avg(SP),
        "AVG_SP_E": avg(SP_E),
        "AVG_OE": avg(OE),
        "AVG_OE_E": avg(OE_E),
    }

    output_rows.append(row)

# Create DataFrame and sort
df = pd.DataFrame(output_rows)
df = df.sort_values(by="participant_id").reset_index(drop=True)

# Round numeric columns for cleaner formatting
numeric_cols = df.select_dtypes(include=["float", "int"]).columns
df[numeric_cols] = df[numeric_cols].round(3)

# Save to Excel
output_xlsx = "results_robotic_summary.xlsx"
df.to_excel(output_xlsx, index=False)

# Pretty print
print("\n📊 Summary of Results (sorted by participant):\n")
print(tabulate(df, headers="keys", tablefmt="grid", showindex=False, numalign="right"))
print(f"\n✅ Excel file saved as: {output_xlsx}")
