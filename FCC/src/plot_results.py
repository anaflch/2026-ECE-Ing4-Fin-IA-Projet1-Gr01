import json
from pathlib import Path
import matplotlib.pyplot as plt

RESULTS_PATH = Path("FCC/src/data/processed/results.json")
OUT_DIR = Path("FCC/src/data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _safe_get(d, key, default=None):
    return d[key] if key in d else default

def main():
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    baseline = data["baseline"]
    fair_dp = data["fair_dp"]
    fair_eo = data["fair_eo"]
    sweep = data["sweep_dp"]

    # ---- 1) Trade-off: AUC vs eps (if auc exists)
    eps_vals = []
    auc_vals = []
    dp_vals = []

    for item in sweep:
        name = item["name"]
        # name like "fair_dp_eps0.02"
        eps = float(name.split("eps")[-1])
        eps_vals.append(eps)
        auc_vals.append(_safe_get(item, "auc", None))
        dp_vals.append(item["dp_diff"])

    # sort by eps
    zipped = sorted(zip(eps_vals, auc_vals, dp_vals), key=lambda x: x[0])
    eps_vals, auc_vals, dp_vals = zip(*zipped)

    # Plot AUC vs eps (if auc available)
    if all(v is not None for v in auc_vals):
        plt.figure()
        plt.plot(eps_vals, auc_vals, marker="o")
        plt.xlabel("epsilon (contrainte DP)")
        plt.ylabel("AUC")
        plt.title("Trade-off performance : AUC vs epsilon")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "tradeoff_auc_vs_eps.png", dpi=200)
        plt.close()

    # Plot dp_diff vs eps
    plt.figure()
    plt.plot(eps_vals, dp_vals, marker="o")
    plt.xlabel("epsilon (contrainte DP)")
    plt.ylabel("Demographic parity difference (dp_diff)")
    plt.title("Trade-off équité : dp_diff vs epsilon")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "tradeoff_dp_vs_eps.png", dpi=200)
    plt.close()

    # ---- 2) Selection rate by group (baseline vs fair_dp vs fair_eo)
    # by_group structure: {"selection_rate": {"group_value": value}, ...}
    def selection_by_group(res):
        bg = res["by_group"]
        sel = bg["selection_rate"]
        return sel

    sel_base = selection_by_group(baseline)
    sel_dp = selection_by_group(fair_dp)
    sel_eo = selection_by_group(fair_eo)

    # union of group keys
    groups = sorted(set(sel_base.keys()) | set(sel_dp.keys()) | set(sel_eo.keys()))
    x = range(len(groups))

    base_vals = [sel_base.get(g, 0.0) for g in groups]
    dp_vals_b = [sel_dp.get(g, 0.0) for g in groups]
    eo_vals_b = [sel_eo.get(g, 0.0) for g in groups]

    width = 0.25
    plt.figure()
    plt.bar([i - width for i in x], base_vals, width=width, label="baseline")
    plt.bar([i for i in x], dp_vals_b, width=width, label="fair DP")
    plt.bar([i + width for i in x], eo_vals_b, width=width, label="fair EO")
    plt.xticks(list(x), [str(g) for g in groups])
    plt.xlabel("Groupe sensible (sex)")
    plt.ylabel("Selection rate")
    plt.title("Taux d'acceptation par groupe (baseline vs fair)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "selection_rate_by_group.png", dpi=200)
    plt.close()

    print(f"Saved plots to: {OUT_DIR}")

if __name__ == "__main__":
    main()