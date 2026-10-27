"""
app.py  --  Explainable Phishing URL Detector (XGBoost + SHAP + LIME)

Run:  streamlit run app.py
Needs: models/xgb_phish.json, models/meta.json, cache/ (from train_model.py)
Optional: explain_out/ (from explain_shap_lime.py) for the "Global insights" tab.
"""
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from check_urls import SAMPLE_URLS, TRUSTED_DOMAINS, USER_CONTENT_HOSTS, score
from explain_shap_lime import explain_url, load_all
from url_features import registered_domain, split_url

st.set_page_config(page_title="Explainable Phishing URL Detector", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Loading model and explainers ...")
def get_assets():
    return load_all()


model, meta, shap_exp, lime_exp = get_assets()
THR = meta["threshold"]

EXAMPLES = {
    "google.com": "https://google.com",
    "GitHub repo": "https://github.com/pallets/flask",
    "PayPal spoof": "http://paypal.com.secure-login-verify.xyz/webscr?cmd=_login",
    "Weebly phish": "https://microsoft-office365-verify.weebly.com/signin",
    "IP + .php": "http://192.168.10.5:8080/login/update-account.php",
}


def final_verdict(url, p, use_override):
    model_says = "PHISHING" if p >= THR else "BENIGN"
    if use_override and p >= THR:
        _, host, *_ = split_url(url)
        if registered_domain(url) in TRUSTED_DOMAINS and host not in USER_CONTENT_HOSTS:
            return "BENIGN", True
    return model_says, False


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    use_override = st.checkbox(
        "Trusted-domain override", value=False,
        help="Reputation layer: if the model flags a very well-known domain, show BENIGN. "
             "Leave OFF to see the raw model (this is what SHAP/LIME explain).")
    st.caption(f"Decision threshold: {THR:.2f}")
    st.divider()
    st.caption(
        "Binary classifier: **phishing vs benign**, trained on PhreshPhish URLs only. "
        "It judges the URL string, not the page content, so treat it as a screening aid, "
        "not a guarantee.")

# ------------------------------------------------------------------ header
st.title("🛡️ Explainable Phishing URL Detector")
st.write("XGBoost on 72 URL features, explained with **SHAP** and **LIME**.")

tab_check, tab_batch, tab_global = st.tabs(["Check a URL", "Batch check", "Global insights"])

# ================================================================== single URL
with tab_check:
    st.session_state.setdefault("url_input", "")

    def _set(u):
        st.session_state["url_input"] = u

    st.text_input("Enter a URL", key="url_input", placeholder="https://example.com/login")
    cols = st.columns(len(EXAMPLES))
    for col, (label, u) in zip(cols, EXAMPLES.items()):
        col.button(label, on_click=_set, args=(u,), width="stretch")

    if st.button("Analyze", type="primary") and st.session_state["url_input"].strip():
        with st.spinner("Scoring and explaining ..."):
            st.session_state["result"] = explain_url(
                st.session_state["url_input"], model, meta, shap_exp, lime_exp)

    res = st.session_state.get("result")
    if res:
        p = res["prob_phishing"]
        verdict, overridden = final_verdict(res["url"], p, use_override)

        (st.error if verdict == "PHISHING" else st.success)(f"**{verdict}**  —  {res['url']}")
        if overridden:
            st.info("The model flagged this URL, but it belongs to a trusted domain, so the "
                    "override shows BENIGN. Raw model score is shown below.")
        c1, c2 = st.columns([1, 3])
        c1.metric("P(phishing)", f"{p:.1%}")
        c2.progress(float(min(max(p, 0.0), 1.0)))

        t_shap, t_lime, t_feat = st.tabs(["SHAP — why this score", "LIME — local surrogate",
                                          "Extracted features"])
        with t_shap:
            st.caption("Red pushes toward phishing, blue toward benign. Units are log-odds; "
                       "the bars add up from the base value to the model output f(x).")
            # plt.figure()
            # shap.plots.waterfall(res["shap_explanation"], max_display=12, show=False)
            # st.pyplot(plt.gcf(), clear_figure=True)
            
            try:
                plt.figure()
                shap.plots.waterfall(
                    res["shap_explanation"],
                    max_display=12,
                    show=False
                )
                st.pyplot(plt.gcf(), clear_figure=True)

            except IndexError:
                # Fallback if SHAP's waterfall plot fails with a Matplotlib error.
                plt.close("all")

                st.warning(
                    "The SHAP waterfall plot could not be rendered. "
                    "Showing feature contributions in a bar chart instead."
                )

                shap_df = res["shap"].copy()
                shap_df = shap_df.sort_values(
                    "shap",
                    key=lambda values: values.abs(),
                    ascending=True
                )

                fig, ax = plt.subplots(figsize=(9, 6))
                ax.barh(shap_df["feature"], shap_df["shap"])
                ax.set_xlabel("SHAP contribution (log-odds)")
                ax.set_title("Feature contributions toward the prediction")
                fig.tight_layout()

                st.pyplot(fig, clear_figure=True)
                plt.close(fig)

            st.dataframe(res["shap"].rename(columns={"value": "feature value"}),
                         width="stretch", hide_index=True)
        with t_lime:
            st.caption("LIME fits a simple model on perturbed copies of this URL's features. "
                       "Positive weight = toward phishing. It can rank features differently "
                       "from SHAP; sign agreement is high, rank agreement is lower.")
            fig = res["lime_obj"].as_pyplot_figure(label=1)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
            st.dataframe(res["lime"], width="stretch", hide_index=True)
        with t_feat:
            st.dataframe(res["features"].rename("value").to_frame(), width="stretch")

# ================================================================== batch
with tab_batch:
    st.write("One URL per line.")
    txt = st.text_area("URLs", value="\n".join(SAMPLE_URLS[:6]), height=160)
    if st.button("Score all"):
        urls = [u.strip() for u in txt.splitlines() if u.strip()]
        if urls:
            urls, X, proba, _ = score(urls, model, meta)
            out = pd.DataFrame({"url": urls, "P(phishing)": proba.round(3)})
            out["verdict"] = [final_verdict(u, p, use_override)[0] for u, p in zip(urls, proba)]
            st.dataframe(out, width="stretch", hide_index=True)

# ================================================================== global
with tab_global:
    d = "explain_out"
    if not os.path.isdir(d):
        st.info("Run `python explain_shap_lime.py` to generate the global plots.")
    else:
        a, b = st.columns(2)
        for col, f, cap in [(a, "shap_summary_beeswarm.png", "SHAP beeswarm (5,000 test URLs)"),
                            (b, "shap_summary_bar.png", "Mean |SHAP| per feature")]:
            if os.path.exists(f"{d}/{f}"):
                col.image(f"{d}/{f}", caption=cap)
        sp = f"{d}/shap_vs_lime_summary.json"
        if os.path.exists(sp):
            with open(sp) as fh:
                s = json.load(fh)
            st.subheader("SHAP vs LIME agreement")
            m1, m2, m3 = st.columns(3)
            m1.metric(f"Top-{s['top_k']} overlap (Jaccard)", f"{s['mean_jaccard_topk']:.2f}")
            m2.metric("Sign agreement", f"{(s['mean_sign_agreement'] or 0):.0%}")
            m3.metric("Same #1 feature", f"{s['same_top1_rate']:.0%}")