"""
explain_shap_lime.py  --  global + local explanations for the phishing-URL XGBoost model.

Run (after train_model.py):   python explain_shap_lime.py
Everything is saved to ./explain_out/

What it does
  1. SHAP (TreeExplainer) on a stratified sample of the test set  -> summary + bar plots, CSV
  2. SHAP waterfall + LIME explanation for a few example URLs
  3. SHAP-vs-LIME agreement on N_COMPARE test URLs (Jaccard of top-K, sign agreement, same #1)

Reusable in Streamlit:   from explain_shap_lime import load_all, explain_url
SHAP values are in log-odds: positive = pushes toward PHISHING, negative = toward benign.
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")                      # save figures without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split

from url_features import canonical_url, extract_features

# ------------------------------------------------------------------ settings
N_SHAP = 5000          # test rows for global SHAP (~seconds for XGBoost)
N_COMPARE = 100        # URLs for SHAP-vs-LIME comparison
TOP_K = 5
LIME_SAMPLES = 1000    # 1000 is plenty; lower it if the app feels slow
SEED = 42
OUT = "explain_out"

EXAMPLE_URLS = [
    "https://www.google.com/",
    "https://github.com/pallets/flask",
    "http://paypal.com.secure-login-verify.xyz/webscr?cmd=_login",
    "https://microsoft-office365-verify.weebly.com/signin",
]


# ------------------------------------------------------------------ loading
def load_all():
    """Load model, feature names, SHAP explainer and LIME explainer (do this ONCE in the app)."""
    model = xgb.XGBClassifier()
    model.load_model("models/xgb_phish.json")
    with open("models/meta.json") as f:
        meta = json.load(f)
    names = meta["feature_names"]

    bg_path = "cache/X_bg.pkl" if os.path.exists("cache/X_bg.pkl") else "cache/X_test.pkl"
    bg = pd.read_pickle(bg_path)[names]
    if len(bg) > 20000:
        bg = bg.sample(20000, random_state=SEED)

    shap_exp = shap.TreeExplainer(model)
    # Binary 0/1 flags (is_http, free_hosting, ...) must be treated as categorical.
    # Otherwise LIME quantile-bins them into meaningless "<= 0.00" buckets and its
    # explanations drift away from SHAP.
    bin_idx = [i for i, c in enumerate(names) if set(np.unique(bg[c].values)) <= {0.0, 1.0}]
    lime_exp = LimeTabularExplainer(
        training_data=bg.values,
        feature_names=names,
        class_names=["benign", "phishing"],
        mode="classification",
        categorical_features=bin_idx,
        discretize_continuous=True,
        random_state=SEED,
    )
    return model, meta, shap_exp, lime_exp


def _base_value(shap_exp):
    return float(np.ravel(shap_exp.expected_value)[0])


def _predict_fn(model, names):
    return lambda arr: model.predict_proba(pd.DataFrame(arr, columns=names))


def lime_explain(lime_exp, model, names, x_row, num_features=10):
    return lime_exp.explain_instance(
        np.asarray(x_row, dtype=float), _predict_fn(model, names),
        num_features=num_features, num_samples=LIME_SAMPLES, labels=(1,))


# ------------------------------------------------------------------ app helper
def explain_url(url, model, meta, shap_exp, lime_exp, num_features=10):
    """One URL -> dict with probability, verdict, SHAP and LIME contributions (for Streamlit)."""
    names = meta["feature_names"]
    url = canonical_url(url if "://" in url else "https://" + url.strip())
    X = pd.DataFrame([extract_features(url)])[names].astype("float32")
    p = float(model.predict_proba(X)[0, 1])

    sv = shap_exp.shap_values(X)[0]
    shap_df = (pd.DataFrame({"feature": names, "value": X.iloc[0].values, "shap": sv})
                 .assign(abs_shap=lambda d: d.shap.abs())
                 .sort_values("abs_shap", ascending=False).head(num_features)
                 .drop(columns="abs_shap").reset_index(drop=True))

    lx = lime_explain(lime_exp, model, names, X.iloc[0].values, num_features)
    lime_df = pd.DataFrame(lx.as_list(label=1), columns=["rule", "weight"])

    return {"url": url, "prob_phishing": p, "features": X.iloc[0],
            "verdict": "PHISHING" if p >= meta["threshold"] else "benign",
            "shap": shap_df, "lime": lime_df, "lime_obj": lx,
            "shap_explanation": shap.Explanation(
                values=sv, base_values=_base_value(shap_exp),
                data=X.iloc[0].values, feature_names=names)}


# ------------------------------------------------------------------ main
def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    model, meta, shap_exp, lime_exp = load_all()
    names = meta["feature_names"]

    X_test = pd.read_pickle("cache/X_test.pkl")[names]
    y_test = np.load("cache/y_test.npy")

    # ---------- 1) global SHAP on a stratified sample
    _, X_s, _, y_s = train_test_split(X_test, y_test, test_size=min(N_SHAP, len(X_test) // 2),
                                      stratify=y_test, random_state=SEED)
    print(f"SHAP on {len(X_s):,} test rows ...")
    sv = shap_exp.shap_values(X_s)

    plt.figure()
    shap.summary_plot(sv, X_s, show=False, max_display=20)
    plt.tight_layout(); plt.savefig(f"{OUT}/shap_summary_beeswarm.png", dpi=200); plt.close()

    plt.figure()
    shap.summary_plot(sv, X_s, plot_type="bar", show=False, max_display=20)
    plt.tight_layout(); plt.savefig(f"{OUT}/shap_summary_bar.png", dpi=200); plt.close()

    imp = (pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(axis=0)})
             .sort_values("mean_abs_shap", ascending=False))
    imp.to_csv(f"{OUT}/shap_global_importance.csv", index=False)
    print("Top 10 features by mean |SHAP|:")
    print(imp.head(10).to_string(index=False))
    print(f"  ({time.time() - t0:.0f}s)")

    # ---------- 2) local explanations for example URLs
    for i, url in enumerate(EXAMPLE_URLS):
        r = explain_url(url, model, meta, shap_exp, lime_exp)
        print(f"\n{r['url']}\n  P(phishing)={r['prob_phishing']:.3f} -> {r['verdict']}")
        print("  SHAP top:", ", ".join(f"{a}({b:+.2f})" for a, b in zip(r['shap'].feature, r['shap'].shap)))
        print("  LIME top:", ", ".join(f"{a}({b:+.2f})" for a, b in zip(r['lime'].rule, r['lime'].weight)))

        plt.figure()
        shap.plots.waterfall(r["shap_explanation"], max_display=12, show=False)
        plt.tight_layout(); plt.savefig(f"{OUT}/example{i + 1}_shap_waterfall.png", dpi=200); plt.close()

        fig = r["lime_obj"].as_pyplot_figure(label=1)
        fig.tight_layout(); fig.savefig(f"{OUT}/example{i + 1}_lime.png", dpi=200); plt.close(fig)
        r["lime_obj"].save_to_file(f"{OUT}/example{i + 1}_lime.html")

    # ---------- 3) SHAP vs LIME agreement
    print(f"\nComparing SHAP vs LIME on {N_COMPARE} test URLs ...")
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(X_test), size=N_COMPARE, replace=False)
    jac, sign_agree, same_top1 = [], [], []
    for i in idx:
        x = X_test.iloc[i]
        s = shap_exp.shap_values(x.to_frame().T)[0]
        s_top = list(np.argsort(-np.abs(s))[:TOP_K])
        lx = lime_explain(lime_exp, model, names, x.values, num_features=TOP_K)
        l_map = dict(lx.as_map()[1])                       # {feature_idx: weight}
        l_top = sorted(l_map, key=lambda k: -abs(l_map[k]))[:TOP_K]

        jac.append(len(set(s_top) & set(l_top)) / len(set(s_top) | set(l_top)))
        same_top1.append(s_top[0] == l_top[0])
        both = set(s_top) & set(l_top)
        if both:
            sign_agree.append(np.mean([np.sign(s[j]) == np.sign(l_map[j]) for j in both]))

    summary = {
        "n_urls": N_COMPARE, "top_k": TOP_K,
        "mean_jaccard_topk": float(np.mean(jac)),
        "mean_sign_agreement": float(np.mean(sign_agree)) if sign_agree else None,
        "same_top1_rate": float(np.mean(same_top1)),
    }
    print(json.dumps(summary, indent=2))
    with open(f"{OUT}/shap_vs_lime_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs saved in ./{OUT}/   (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()