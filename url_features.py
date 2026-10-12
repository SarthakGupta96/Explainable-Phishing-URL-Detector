"""
url_features.py  --  ONE feature extractor shared by training AND inference.

Using the exact same code in both places is the single most important thing for
avoiding "model works on the test set but fails on real URLs".

Install:  pip install tldextract xgboost scikit-learn pandas joblib
"""
import math
import re
from urllib.parse import urlparse, parse_qsl

import tldextract

# Offline extractor: uses the public-suffix snapshot bundled with the package
# (no network calls, so it is fast and deterministic).
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

# ----------------------------------------------------------------- word lists
SUSPICIOUS_WORDS = [
    "login", "signin", "sign-in", "logon", "verify", "verification", "secure",
    "security", "account", "update", "confirm", "banking", "bank", "password",
    "passwd", "credential", "wallet", "support", "billing", "invoice", "payment",
    "suspend", "unlock", "recover", "validate", "authenticate", "webscr",
    "alert", "limited", "refund", "claim", "reward", "prize", "free", "bonus",
    "survey", "helpdesk", "webmail", "office365", "onedrive", "sharepoint",
]
BRANDS = [
    "paypal", "apple", "icloud", "microsoft", "outlook", "office", "google",
    "gmail", "amazon", "facebook", "instagram", "whatsapp", "netflix", "spotify",
    "linkedin", "twitter", "dropbox", "docusign", "adobe", "dhl", "fedex", "ups",
    "usps", "chase", "wellsfargo", "bankofamerica", "citibank", "hsbc", "coinbase",
    "binance", "metamask", "blockchain", "steam", "roblox", "att", "verizon",
]
SUSPICIOUS_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "icu", "cyou", "click", "buzz",
    "monster", "rest", "work", "support", "live", "site", "online", "shop",
    "club", "vip", "cam", "link", "info", "bid", "loan", "men", "country",
    "stream", "download", "zip", "mov", "sbs", "cfd", "lol", "fun", "pw",
}
COMMON_TLDS = {"com", "org", "net", "edu", "gov", "co", "io", "uk", "de", "in"}
SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "rb.gy", "tiny.cc", "t.ly",
}
# Free hosting / site-builders / user-content platforms that phishers abuse
FREE_HOSTING = {
    "weebly.com", "wixsite.com", "blogspot.com", "github.io", "netlify.app",
    "vercel.app", "pages.dev", "web.app", "firebaseapp.com", "herokuapp.com",
    "000webhostapp.com", "webflow.io", "godaddysites.com", "square.site",
    "glitch.me", "repl.co", "framer.app", "carrd.co", "filesusr.com",
    "workers.dev", "surge.sh", "myshopify.com", "wordpress.com", "sites.google.com",
    "ipfs.io", "cloudfront.net", "r2.dev", "appspot.com", "azurewebsites.net",
    "fanpages.page", "page.link", "typedream.app", "notion.site",
}

_IP_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
_HEX_RE = re.compile(r"[0-9a-f]{16,}", re.I)
_B64ISH_RE = re.compile(r"[A-Za-z0-9+/_-]{30,}")
_DIGIT_RUN_RE = re.compile(r"\d+")
_CONSONANT_RUN_RE = re.compile(r"[bcdfghjklmnpqrstvwxyz]+", re.I)


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _max_run(regex, s: str) -> int:
    runs = [len(m) for m in regex.findall(s)]
    return max(runs) if runs else 0


_CANON_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.\-]*://)?([^/?#]*)(.*)$", re.S)


def canonical_url(url: str) -> str:
    """Strip quotes/space and make sure there is always a path: 'https://a.com' -> 'https://a.com/'.
    Without this, 'google.com' and 'google.com/' give different features (path_len 0 vs 1)."""
    url = (url or "").strip().strip("\"'")
    m = _CANON_RE.match(url)
    if not m:
        return url
    scheme, host, rest = m.group(1) or "", m.group(2), m.group(3)
    if rest == "" or rest[0] in "?#":
        rest = "/" + rest
    return scheme + host + rest


def split_url(url: str):
    """Return (scheme, host, port, path, query, fragment). Never raises."""
    url = (url or "").strip().strip("\"'")
    scheme = ""
    low = url.lower()
    if low.startswith("https://"):
        scheme = "https"
    elif low.startswith("http://"):
        scheme = "http"
    try:
        p = urlparse(url if "://" in url else "//" + url)
        host = (p.hostname or "").lower()
        port = p.port
        path, query, frag = p.path or "", p.query or "", p.fragment or ""
        if not scheme and p.scheme:  # e.g. ftp://
            scheme = p.scheme.lower()
    except ValueError:
        # malformed (e.g. bad IPv6 bracket) -> crude fallback
        rest = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", url)
        host_part, _, tail = rest.partition("/")
        host = host_part.split(":")[0].lower()
        port = None
        path, _, query = ("/" + tail).partition("?")
        query, _, frag = query.partition("#")
    return scheme, host, port, path, query, frag


def registered_domain(url: str) -> str:
    _, host, *_ = split_url(url)
    ext = _EXTRACT(host)
    return ext.registered_domain or host


def homepage_of(url: str) -> str:
    """scheme://host/  -- used for training augmentation & robustness checks."""
    scheme, host, port, *_ = split_url(url)
    scheme = scheme or "https"
    return f"{scheme}://{host}" + (f":{port}" if port else "") + "/"


def extract_features(url: str) -> dict:
    url = canonical_url(url)
    scheme, host, port, path, query, frag = split_url(url)
    ext = _EXTRACT(host)
    sub, dom, suf = ext.subdomain, ext.domain, ext.suffix
    reg = ext.registered_domain or host
    tld = suf.split(".")[-1] if suf else ""

    low = url.lower()
    low_path_q = (path + "?" + query).lower() if query else path.lower()
    host_no_www = host[4:] if host.startswith("www.") else host
    segs = [s for s in path.split("/") if s]
    last_seg = segs[-1] if segs else ""
    sub_labels = [s for s in sub.split(".") if s] if sub else []
    host_labels = [s for s in host.split(".") if s]

    # keyword / brand logic
    kw_url = sum(low.count(w) for w in SUSPICIOUS_WORDS)
    kw_host = sum(host.count(w) for w in SUSPICIOUS_WORDS)
    kw_path = sum(low_path_q.count(w) for w in SUSPICIOUS_WORDS)
    brand_in_host = any(b in host for b in BRANDS)
    # brand appears in the host but is NOT the real registered domain -> classic spoof
    brand_spoof = any(b in host and b not in dom for b in BRANDS)
    brand_in_path = any(b in low_path_q for b in BRANDS)

    n = max(len(url), 1)
    nh = max(len(host), 1)
    digits_url = sum(c.isdigit() for c in url)
    letters_url = sum(c.isalpha() for c in url)
    digits_host = sum(c.isdigit() for c in host)
    specials_url = sum(not c.isalnum() for c in url)

    try:
        n_params = len(parse_qsl(query, keep_blank_values=True))
    except Exception:
        n_params = query.count("&") + (1 if query else 0)

    f = {
        # ---- lengths
        "url_len": len(url),
        "host_len": len(host),
        "path_len": len(path),
        "query_len": len(query),
        "fragment_len": len(frag),
        "domain_len": len(dom),
        "subdomain_len": len(sub),
        "tld_len": len(suf),
        "longest_host_label": max((len(x) for x in host_labels), default=0),
        "last_path_seg_len": len(last_seg),
        # ---- structure counts
        "num_subdomains": len(sub_labels),
        "num_dots_host": host.count("."),
        "num_dots_url": url.count("."),
        "path_depth": len(segs),
        "num_params": n_params,
        "num_slashes": url.count("/"),
        "num_hyphens_host": host.count("-"),
        "num_hyphens_domain": dom.count("-"),
        "num_hyphens_url": url.count("-"),
        "num_underscores": url.count("_"),
        "num_digits_host": digits_host,
        "num_digits_domain": sum(c.isdigit() for c in dom),
        "num_digits_url": digits_url,
        "num_at": url.count("@"),
        "num_percent": url.count("%"),
        "num_equals": url.count("="),
        "num_ampersand": url.count("&"),
        "num_question": url.count("?"),
        "num_special_url": specials_url,
        # ---- ratios
        "digit_ratio_url": digits_url / n,
        "letter_ratio_url": letters_url / n,
        "digit_ratio_host": digits_host / nh,
        "special_ratio_url": specials_url / n,
        "host_to_url_ratio": len(host) / n,
        # ---- randomness / entropy
        "entropy_url": _entropy(low),
        "entropy_host": _entropy(host),
        "entropy_domain": _entropy(dom),
        "entropy_path": _entropy(path.lower()),
        "max_digit_run": _max_run(_DIGIT_RUN_RE, url),
        "max_digit_run_domain": _max_run(_DIGIT_RUN_RE, dom),
        "max_consonant_run_domain": _max_run(_CONSONANT_RUN_RE, dom),
        # ---- flags
        "is_https": int(scheme == "https"),
        "is_http": int(scheme == "http"),
        "has_ip_host": int(bool(_IP_RE.match(host))),
        "has_port": int(port is not None and port not in (80, 443)),
        "has_at": int("@" in url),
        "has_punycode": int("xn--" in host),
        "starts_with_www": int(host.startswith("www.")),
        "has_double_slash_path": int("//" in path),
        "has_hex_encoding": int(bool(re.search(r"%[0-9a-fA-F]{2}", url))),
        "has_long_hex_token": int(bool(_HEX_RE.search(low_path_q))),
        "has_long_b64_token": int(bool(_B64ISH_RE.search(path + query))),
        "trailing_slash": int(path.endswith("/")),
        "is_root_path": int(path in ("", "/") and not query),
        "has_query": int(bool(query)),
        "tld_common": int(tld in COMMON_TLDS),
        "tld_suspicious": int(tld in SUSPICIOUS_TLDS),
        "is_shortener": int(reg in SHORTENERS),
        "free_hosting": int(reg in FREE_HOSTING or host in FREE_HOSTING),
        "ext_php": int(low_path_q.split("?")[0].endswith(".php")),
        "ext_html": int(low_path_q.split("?")[0].endswith((".html", ".htm"))),
        "ext_exec": int(low_path_q.split("?")[0].endswith((".exe", ".apk", ".zip", ".scr", ".js", ".jar"))),
        "has_index_php": int("index.php" in low),
        "has_wp_path": int("/wp-" in low),
        "domain_has_digit_and_hyphen": int(any(c.isdigit() for c in dom) and "-" in dom),
        # ---- keywords / brands
        "kw_url": kw_url,
        "kw_host": kw_host,
        "kw_path": kw_path,
        "brand_in_host": int(brand_in_host),
        "brand_spoof": int(brand_spoof),
        "brand_in_path": int(brand_in_path),
        "brand_in_path_not_host": int(brand_in_path and not brand_in_host),
    }
    return f


FEATURE_NAMES = list(extract_features("https://example.com/").keys())


# ------------------------------------------------------- batch helpers (parallel)
def _feat_chunk(urls):
    return [list(extract_features(u).values()) for u in urls]


def _domain_chunk(urls):
    return [registered_domain(u) for u in urls]


def _run_parallel(fn, items, n_jobs=-1, chunk=5000):
    from joblib import Parallel, delayed
    chunks = [items[i:i + chunk] for i in range(0, len(items), chunk)]
    out = Parallel(n_jobs=n_jobs)(delayed(fn)(c) for c in chunks)
    return [row for part in out for row in part]


def extract_matrix(urls, n_jobs=-1):
    """list[str] -> pandas DataFrame with columns FEATURE_NAMES (float32)."""
    import pandas as pd
    rows = _run_parallel(_feat_chunk, list(urls), n_jobs)
    return pd.DataFrame(rows, columns=FEATURE_NAMES).astype("float32")


def registered_domains(urls, n_jobs=-1):
    return _run_parallel(_domain_chunk, list(urls), n_jobs)