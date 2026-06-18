"""Generate knowledge encoding CSV files from alert data.

This script generates two knowledge files used for edge feature construction:

1. encoded_signature.csv — maps each unique Snort rule ID to a one-hot vector.
   Generated from the alert CSV files in data/alert_logs/correct_data/.

2. encoded_port_service.csv — maps port ranges to one-hot vectors.
   Generated from a hand-crafted port-to-service mapping CSV.

Usage:
    python data/knowledge/generate_encodings.py

This is extracted from the original research notebook:
    dataset/match_knowledge/knowledge_summary.ipynb

Prerequisites:
    - Alert CSV files in data/alert_logs/correct_data/
    - For port_service: a CSV with columns [Port, Service, Protocol, Description]
      at data/knowledge/port_service.csv (hand-crafted for your network)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
from config import ALERT_CORRECTED_DIR, KNOWLEDGE_DIR


# ── Part 1: Generate encoded_signature.csv ──────────────────────────────────
# (Original code from knowledge_summary.ipynb)

def generate_signature_encoding():
    """Extract unique (rule, class, msg) tuples from all alert CSV files and
    one-hot encode them."""
    print("=" * 60)
    print("Generating encoded_signature.csv")
    print("=" * 60)

    # Collect all alert CSV files
    alert_files = sorted(ALERT_CORRECTED_DIR.glob('*.csv'))
    if not alert_files:
        print(f"ERROR: No CSV files found in {ALERT_CORRECTED_DIR}")
        print("Place your alert CSV files there first.")
        return

    print(f"Found {len(alert_files)} alert files:")
    for f in alert_files:
        print(f"  {f.name}")

    # Merge all alert files
    merge_df = pd.DataFrame()
    for file_path in alert_files:
        df = pd.read_csv(file_path)
        merge_df = pd.concat([merge_df, df], ignore_index=True)

    print(f"Total alerts: {len(merge_df)}")
    print(f"Unique rules: {merge_df['rule'].nunique()}")

    # Extract unique (rule, class, msg) tuples
    # (Original columns: 'rule', 'class', 'msg' — may need renaming if your
    #  alert CSVs use different column names)
    if 'rule' not in merge_df.columns:
        print("NOTE: Column 'rule' not found. Trying 'category_standard_id'...")
        if 'category_standard_id' in merge_df.columns:
            merge_df.rename(columns={'category_standard_id': 'rule'}, inplace=True)
        else:
            print("ERROR: Cannot find signature/rule column. Check alert CSV format.")
            return

    signature_df = merge_df[['rule', 'class', 'msg']].copy()
    signature_df.reset_index(drop=True, inplace=True)
    signature_df.drop_duplicates(inplace=True)
    signature_df.reset_index(drop=True, inplace=True)

    # Assign integer IDs starting from 1
    signature_df['ID'] = range(1, len(signature_df) + 1)
    print(f"Unique signature entries: {len(signature_df)}")

    # One-hot encode IDs
    signature_df['ID'] = signature_df['ID'].astype(str)
    one_hot_encoded = pd.get_dummies(signature_df['ID'])
    columns_sorted = sorted(one_hot_encoded.columns,
                            key=lambda x: int(x.split('_')[-1]))
    one_hot_encoded_sorted = one_hot_encoded[columns_sorted]
    signature_df['one_hot'] = one_hot_encoded_sorted.apply(
        lambda row: row.values.tolist(), axis=1
    )

    # Save
    output_path = KNOWLEDGE_DIR / 'encoded_signature.csv'
    signature_df.to_csv(output_path, index=False)
    print(f"Saved {len(signature_df)} entries to {output_path}")
    print(f"Signature encoding dimension: {len(signature_df)}")


# ── Part 2: Generate encoded_port_service.csv ───────────────────────────────
# (Original code from knowledge_summary.ipynb)
#
# NOTE: This requires a hand-crafted port_service.csv with columns:
#   Port, Service, Protocol, Description
# The port_service.csv maps port numbers/ranges to human-readable service names.
# This file is dataset-specific and must be created manually.

def generate_port_encoding():
    """One-hot encode the port-to-service mapping."""
    print("\n" + "=" * 60)
    print("Generating encoded_port_service.csv")
    print("=" * 60)

    port_path = KNOWLEDGE_DIR / 'port_service.csv'
    if not port_path.exists():
        print(f"NOTE: {port_path} not found.")
        print("This is a hand-crafted file mapping port numbers/ranges to")
        print("service names. Create it with columns: Port, Service, Protocol, Description")
        print("Example row: 80, HTTP, TCP, Hypertext Transfer Protocol")
        print("\nIf you already have encoded_port_service.csv, you can skip this step.")
        return

    port_df = pd.read_csv(port_path)
    port_df.reset_index(inplace=True)
    port_df.rename(columns={'index': 'id'}, inplace=True)

    port_df['id'] = port_df['id'].astype(str)
    one_hot_encoded = pd.get_dummies(port_df['id'])
    columns_sorted = sorted(one_hot_encoded.columns,
                            key=lambda x: int(x.split('_')[-1]))
    one_hot_encoded_sorted = one_hot_encoded[columns_sorted]
    port_df['one_hot'] = one_hot_encoded_sorted.apply(
        lambda row: row.values.tolist(), axis=1
    )

    output_path = KNOWLEDGE_DIR / 'encoded_port_service.csv'
    port_df.to_csv(output_path, index=False)
    print(f"Saved {len(port_df)} entries to {output_path}")
    print(f"Port encoding dimension: {len(port_df)}")


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    generate_signature_encoding()
    generate_port_encoding()
    print("\nDone. The encoded files are used by data_process/edge_attribute_tools.py")
    print("for edge feature construction during graph building.")
