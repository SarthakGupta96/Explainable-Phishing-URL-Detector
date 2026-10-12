"""
train_model.py  --  train XGBoost phishing-URL detector on PhreshPhish (url + label CSVs).

Run:  python train_model.py
Outputs (in ./models and ./cache):
    models/xgb_phish.json      trained model
    models/meta.json           feature names + threshold
    cache/X_test.pkl, y_test.pkl   test features (reuse these for SHAP / LIME later)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from url_features import FEATURE_NAMES, extract_matrix, homepage_of, registered_domains

# ------------------------------------------------------------------ settings
TRAIN_CSV = "data/phreshphish_train.csv"
TEST_CSV = "data/phreshphish_test.csv"
N_TRAIN = 400_000      # stratified sample from train; set None to use all ~498k
AUG_FRAC = 0.20        # fraction of rows that also get a "homepage only" copy (see note below)
THRESHOLD = 0.50
SEED = 42

os.makedirs("models", exist_ok=True)
os.makedirs("cache", exist_ok=True)


def load(path):
    df = pd.read_csv(path, usecols=["url", "label"])
    df = df.dropna(subset=["url", "label"])
    df["url"] = df["url"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    df = df[df["url"].str.len() > 3].drop_duplicates(subset="url")
    return df.reset_index(drop=True)


def main():
    t0 = time.time()
    train = load(TRAIN_CSV)
    test = load(TEST_CSV)
    print(f"train rows: {len(train):,}  (phish={train.label.sum():,})")
    print(f"test  rows: {len(test):,}  (phish={test.label.sum():,})")

    # ---- stratified sample for speed
    if N_TRAIN and len(train) > N_TRAIN:
        train, _ = train_test_split(train, train_size=N_TRAIN, stratify=train["label"],
                                    random_state=SEED)
        train = train.reset_index(drop=True)
        print(f"sampled train: {len(train):,}")

    # ---- homepage augmentation (THE fix for "known sites get flagged")
    # In PhreshPhish almost every benign URL has a deep path
    # (…/blog/some-article/). A bare "https://google.com/" has path length 0, which
    # in the raw data only ever looks like phishing -> the model learns
    # "short / no path = phishing". We add homepage-only copies for BOTH classes
    # (same label as the source URL) so path length can no longer be a shortcut and
    # the model has to judge the *domain* itself.
    aug = train.sample(frac=AUG_FRAC, random_state=SEED).copy()
    aug["url"] = aug["url"].map(homepage_of)
    train = (pd.concat([train, aug], ignore_index=True)
               .drop_duplicates(subset="url").reset_index(drop=True))
    print(f"train after homepage augmentation: {len(train):,}")

    # ---- features
    print("extracting train features ...")
    X = extract_matrix(train["url"])
    y = train["label"].values
    groups = registered_domains(train["url"])
    print(f"  done in {time.time() - t0:.0f}s, {X.shape[1]} features")
    X.sample(n=min(20000, len(X)), random_state=SEED).to_pickle("cache/X_bg.pkl")  # for LIME

    print("extracting test features ...")
    X_test = extract_matrix(test["url"])
    y_test = test["label"].values
    X_test.to_pickle("cache/X_test.pkl")
    np.save("cache/y_test.npy", y_test)

    # ---- validation split grouped by registered domain (no domain in both sets)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
    tr_idx, va_idx = next(gss.split(X, y, groups))
    X_tr, y_tr, X_va, y_va = X.iloc[tr_idx], y[tr_idx], X.iloc[va_idx], y[va_idx]

    # ---- model
    model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.08,
        max_depth=8,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        tree_method="hist",
        eval_metric="auc",
        early_stopping_rounds=60,
        n_jobs=-1,
        random_state=SEED,
    )
    print("training XGBoost ...")
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=100)
    print(f"best iteration: {model.best_iteration}")

    # ---- evaluation on the official (time-aware) test split
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    print("\n================ TEST RESULTS (official PhreshPhish test) ================")
    print(f"accuracy : {accuracy_score(y_test, pred):.4f}")
    print(f"precision: {precision_score(y_test, pred):.4f}")
    print(f"recall   : {recall_score(y_test, pred):.4f}")
    print(f"F1       : {f1_score(y_test, pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, proba):.4f}")
    print("confusion matrix [rows=true 0/1, cols=pred 0/1]:")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=["benign", "phishing"], digits=4))

    # ---- robustness check: strip every test URL down to its homepage
    # Directly measures your complaint ("known URLs flagged as phishing").
    rng = np.random.RandomState(SEED)
    sub = test.iloc[rng.choice(len(test), size=min(20000, len(test)), replace=False)]
    Xh = extract_matrix(sub["url"].map(homepage_of))
    ph = model.predict_proba(Xh)[:, 1]
    yh = sub["label"].values
    fpr = (ph[yh == 0] >= THRESHOLD).mean()
    tpr = (ph[yh == 1] >= THRESHOLD).mean()
    print("---- homepage-only robustness (test URLs cut to scheme://host/) ----")
    print(f"benign sites wrongly flagged as phishing: {fpr:.2%}")
    print(f"phishing sites still caught             : {tpr:.2%}")

    # ---- top features
    imp = pd.Series(model.get_booster().get_score(importance_type="gain"))
    print("\nTop 15 features by gain:")
    print(imp.sort_values(ascending=False).head(15).round(1).to_string())

    # ---- save
    model.save_model("models/xgb_phish.json")
    with open("models/meta.json", "w") as f:
        json.dump({"feature_names": FEATURE_NAMES, "threshold": THRESHOLD,
                   "best_iteration": int(model.best_iteration)}, f, indent=2)
    print(f"\nsaved models/xgb_phish.json  (total time {time.time() - t0:.0f}s)")
    print("Next: python check_urls.py")


if __name__ == "__main__":
    sys.exit(main())