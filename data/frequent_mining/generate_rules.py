"""Generate frequent pattern rules from mined community edges.

Reads frequent_mining_edges.txt (produced by graph_partition.py), applies
Apriori frequent itemset mining and association rule generation, and writes
frequent_combinations_output.txt for use by log2graph.py during graph
partition refinement.

Usage:
    python generate_rules.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import ast
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from config import FREQUENT_MINING_DIR

# ── Parameters ──────────────────────────────────────────────────────────────
MIN_SUPPORT = 0.10
MIN_CONFIDENCE = 0.70

if __name__ == '__main__':
    edges_file = FREQUENT_MINING_DIR / 'frequent_mining_edges.txt'

    # Read transactions
    transactions = []
    with open(edges_file, 'r') as f:
        for line in f:
            transaction = ast.literal_eval(line.strip())
            transactions.append(transaction)
    print(f"Loaded {len(transactions)} transactions from {edges_file}")

    # Transaction encode
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Apriori mining
    frequent_itemsets = apriori(df, min_support=MIN_SUPPORT, use_colnames=True)
    print(f"Found {len(frequent_itemsets)} frequent itemsets (min_support={MIN_SUPPORT})")

    # Association rules
    rules = association_rules(frequent_itemsets, metric="confidence",
                              min_threshold=MIN_CONFIDENCE)
    print(f"Found {len(rules)} association rules (min_confidence={MIN_CONFIDENCE})")

    # Format output
    output = []
    for _, row in rules.iterrows():
        output.append((frozenset(row['antecedents']),
                       frozenset(row['consequents']),
                       row['confidence']))

    # Write output
    output_file = FREQUENT_MINING_DIR / 'frequent_combinations_output.txt'
    with open(output_file, 'w') as f:
        f.write(str(output))
    print(f"Written {len(output)} rules to {output_file}")
