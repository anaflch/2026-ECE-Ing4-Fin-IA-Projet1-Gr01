import json

with open("FCC/src/data/processed/results.json") as f:
    d = json.load(f)

print("MODEL,accuracy,auc,dp_diff,eo_diff")
for k in ["baseline", "fair_dp", "fair_eo"]:
    r = d[k]
    print(
        f"{r['name']},"
        f"{r.get('accuracy','')},"
        f"{r.get('auc','')},"
        f"{r.get('dp_diff','')},"
        f"{r.get('eo_diff','')}"
    )