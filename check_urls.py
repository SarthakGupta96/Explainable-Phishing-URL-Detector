"""
check_urls.py  --  load the trained model and score URLs.

Run:
    python check_urls.py                       # built-in sample list
    python check_urls.py https://example.com   # specific URL(s)
    python check_urls.py -i                    # interactive prompt

For every URL it prints:
    model probability of phishing, the model verdict, and the 3 features
    that pushed the score the most (XGBoost's built-in contributions - a
    quick preview before you run full SHAP/LIME).
"""
import json
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

from url_features import extract_features, registered_domain, split_url

# Optional "reputation layer": a tiny allowlist of very well-known domains.
# The raw model probability is ALWAYS shown; this only changes the FINAL verdict.
# Set to False to see pure model behaviour (use False when you run SHAP/LIME).
USE_ALLOWLIST = True
TRUSTED_DOMAINS = {
    "google.com", "youtube.com", "microsoft.com", "live.com", "office.com",
    "apple.com", "amazon.com", "facebook.com", "instagram.com", "twitter.com",
    "x.com", "linkedin.com", "wikipedia.org", "github.com", "stackoverflow.com",
    "netflix.com", "paypal.com", "ebay.com", "reddit.com", "whatsapp.com",
    "yahoo.com", "bing.com", "adobe.com", "dropbox.com", "zoom.us", "spotify.com",
    "medium.com", "nytimes.com", "bbc.com", "bbc.co.uk", "cnn.com", "irs.gov",
    "usa.gov", "mit.edu", "stanford.edu", "vit.ac.in", "anthropic.com", "openai.com",
}
# Hosts under a trusted domain where ANYONE can publish content -> never auto-trust
USER_CONTENT_HOSTS = {
    "sites.google.com", "docs.google.com", "drive.google.com", "forms.gle",
    "storage.googleapis.com", "script.google.com", "github.io", "medium.com",
}

SAMPLE_URLS = [
    # --- should be BENIGN
    "https://www.google.com/",
    "google.com",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://github.com/pallets/flask",
    "https://en.wikipedia.org/wiki/Phishing",
    "https://learn.microsoft.com/en-us/azure/defender-for-cloud/secure-score-security-controls",
    "https://www.amazon.in/",
    "https://vtop.vit.ac.in/",
    # --- should be PHISHING
    "http://paypal.com.secure-login-verify.xyz/webscr?cmd=_login",
    "http://192.168.10.5:8080/login/update-account.php",
    "https://microsoft-office365-verify.weebly.com/signin",
    "http://amazon.account-suspended-support.top/confirm/9f8a7c6b5d4e3f2a1b0c",
    "https://user-confirmation.fanpages-improve365823.com/4764056672",
    "http://secure-bankofamerica-login.tk/",
]


def load_model():
    model = xgb.XGBClassifier()
    model.load_model("models/xgb_phish.json")
    with open("models/meta.json") as f:
        meta = json.load(f)
    return model, meta


def normalise(url: str) -> str:
    """Same cleanup for every URL typed by a user."""
    url = url.strip().strip("\"'")
    if "://" not in url:
        url = "https://" + url      # most real sites are https; avoids 'no scheme' oddities
    return url


def score(urls, model, meta):
    urls = [normalise(u) for u in urls]
    X = pd.DataFrame([extract_features(u) for u in urls])[meta["feature_names"]].astype("float32")
    proba = model.predict_proba(X)[:, 1]
    contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)  # last col = bias
    return urls, X, proba, contrib


def final_verdict(url, p, thr):
    model_says = "PHISHING" if p >= thr else "benign"
    if not USE_ALLOWLIST:
        return model_says, ""
    _, host, *_ = split_url(url)
    reg = registered_domain(url)
    if reg in TRUSTED_DOMAINS and host not in USER_CONTENT_HOSTS and p >= thr:
        return "benign", "(trusted domain overrides model)"
    return model_says, ""


def report(urls, model, meta):
    thr = meta["threshold"]
    names = meta["feature_names"]
    urls, X, proba, contrib = score(urls, model, meta)
    print(f"\n{'URL':70s} {'P(phish)':>9s}  VERDICT")
    print("-" * 105)
    for i, u in enumerate(urls):
        verdict, note = final_verdict(u, proba[i], thr)
        shown = u if len(u) <= 68 else u[:65] + "..."
        print(f"{shown:70s} {proba[i]:9.3f}  {verdict} {note}")
        c = contrib[i][:-1]
        top = np.argsort(-np.abs(c))[:3]
        why = ", ".join(f"{names[j]}={X.iloc[i, j]:.2f} ({'+' if c[j] > 0 else '-'}{abs(c[j]):.2f})"
                        for j in top)
        print(f"     why: {why}")


if __name__ == "__main__":
    model, meta = load_model()
    args = sys.argv[1:]
    if args and args[0] == "-i":
        print("Enter a URL (blank line to quit)")
        while True:
            u = input("> ").strip()
            if not u:
                break
            report([u], model, meta)
    else:
        report(args or SAMPLE_URLS, model, meta)