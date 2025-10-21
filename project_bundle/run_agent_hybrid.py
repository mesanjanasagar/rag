# run_agent_hybrid.py
import json
import argparse
import os
from agent.graph_hybrid import HybridGraph

def ensure_docs_present():
    os.makedirs("docs", exist_ok=True)
    files = {
        "marketing_calendar.md": "# Northwind Marketing Calendar (1997)\n\n## Summer Beverages 1997\n- Dates: 1997-06-01 to 1997-06-30\n- Notes: Focus on Beverages and Condiments.\n\n## Winter Classics 1997\n- Dates: 1997-12-01 to 1997-12-31\n- Notes: Push Dairy Products and Confections for holiday gifting.\n",
        "kpi_definitions.md": "# KPI Definitions\n## Average Order Value (AOV)\n\n- AOV = SUM(UnitPrice * Quantity * (1 - Discount)) / COUNT(DISTINCT OrderID)\n\n## Gross Margin\n- GM = SUM((UnitPrice - CostOfGoods) * Quantity * (1 - Discount))\n- If cost is missing, approximate with category-level average (document your approach).\n",
        "catalog.md": "# Catalog Snapshot\n- Categories include Beverages, Condiments, Confections, Dairy Products, Grains/Cereals, Meat/Poultry, Produce, Seafood.\n- Products map to categories as in the Northwind DB.\n",
        "product_policy.md": "# Returns & Policy\n- Perishables (Produce, Seafood, Dairy): 3–7 days.\n- Beverages unopened: 14 days; opened: no returns.\n- Non-perishables: 30 days.\n"
    }
    for fn, text in files.items():
        path = os.path.join("docs", fn)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)

def main(batch, out):
    ensure_docs_present()
    agent = HybridGraph(db_path="data/northwind.sqlite", docs_dir="docs")
    with open(batch, "r", encoding="utf-8") as f:
        qlines = [json.loads(l) for l in f.readlines() if l.strip()]
    outputs = []
    for q in qlines:
        result = agent.run_single(q)
        # keep full out but remove _trace from final outputs file if you prefer
        _trace = result.pop("_trace", None)
        outputs.append(result)
    with open(out, "w", encoding="utf-8") as f:
        for o in outputs:
            f.write(json.dumps(o) + "\n")
    print(f"Wrote {len(outputs)} answers to {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, help="input jsonl")
    parser.add_argument("--out", required=True, help="output jsonl")
    args = parser.parse_args()
    main(args.batch, args.out)
