"""Trend Sniffer - core engine: signal simulator, feature builder, model, backtest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

CATEGORIES = [
    "Millet snacks", "Sunscreen sticks", "Matcha lattes", "Korean skincare", "Pickleball gear",
    "Protein coffee", "Thrift fashion", "Air fryers", "Cold-brew cans", "Mushroom coffee",
    "Weighted blankets", "Magnetic wallets", "Barefoot shoes", "Kombucha", "Rosemary hair oil",
    "Sleep gummies", "Portable blenders", "Oversized tees", "Cargo pants", "Gua sha tools",
    "Ragi pasta", "Smart rings", "Mechanical keyboards", "Desk treadmills", "Lip oils",
    "Dopamine decor", "Creatine gummies", "Electrolyte sachets", "A2 ghee", "Phone tripods",
    "Mini fridges", "Vegan leather bags", "Crochet kits", "Journalling kits", "Tote bags",
    "Retro sneakers", "Peptide serums", "Cold plunge tubs", "Sauna blankets", "Under-desk pads",
    "Tinned fish", "Sourdough kits", "Bento boxes", "Neck fans", "Beard serums",
    "Scalp massagers", "Jade rollers", "Cork yoga mats", "Bamboo cutlery", "Refillable perfume",
    "Nasal strips", "Massage guns", "Fragrance layering", "Hydrogen water", "Baby-led weaning kits",
    "Pet cooling mats", "Glass straws", "Solar lanterns", "Ceramic cookware", "Lotus seed snacks",
]
assert len(CATEGORIES) == 60

T = 104                       # weeks simulated
HORIZON = 10                  # brand needs ~10 weeks lead for sourcing / creative / media
TRAIN_END, TEST_START = 50, 62  # time split with a 12-week gap (labels look 10 weeks ahead)
SIGNALS = ["niche", "creator", "search", "sales"]
STAGES = ["Quiet", "Whisper", "Spark", "Ignition", "Mainstream"]

FEATURES = [
    "niche_z", "niche_g", "niche_acc", "creator_z", "creator_g", "search_z", "search_g",
    "intent", "intent_d", "breadth_r", "breadth", "divergence", "creator_gap",
]
FEATURE_LABELS = {
    "niche_z": "Niche-community heat", "niche_g": "Niche growth rate", "niche_acc": "Niche acceleration",
    "creator_z": "Creator heat", "creator_g": "Creator growth rate", "search_z": "Search heat",
    "search_g": "Search growth rate", "intent": "Buy-intent share", "intent_d": "Buy-intent change",
    "breadth_r": "Creator-breadth expansion", "breadth": "Distinct creators", "divergence": "Upstream/downstream gap",
    "creator_gap": "Creator-vs-search gap",
}


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate(seed: int = 7):
    """Simulate weekly multi-source signals for 60 micro-categories.

    Stylised fact used: real demand diffuses niche communities -> creators -> search -> retail sales,
    while hype-only fads spike upstream but with low buy-intent, narrow creator base and no sales.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    frames, events = [], []
    for name in CATEGORIES:
        bn, bc, bs = rng.uniform(20, 45), rng.uniform(15, 40), rng.uniform(25, 55)
        bsales, bint, bbr = rng.uniform(80, 140), rng.uniform(0.07, 0.13), rng.uniform(3, 7)
        season = rng.uniform(0, 0.15) * np.sin(2 * np.pi * (t / 52 + rng.uniform()))
        niche, creator = bn * (1 + season), bc * (1 + season)
        search, sales = bs * (1 + season), bsales * (1 + 0.8 * season)
        intent, breadth = np.full(T, bint), np.full(T, bbr)
        for slot, (lo, hi) in enumerate([(10, 45), (60, 88)]):
            kind = rng.choice(["breakout", "fad", "none"], p=[0.42, 0.36, 0.22])
            if kind == "none":
                continue
            t0 = int(rng.integers(lo, hi))
            events.append(dict(category=name, kind=kind, t0=t0, slot=slot))
            ramp = lambda lag, k=1.6: _sig((t - t0 - lag - 5) / k)
            bump = lambda c, sd=2.0: np.exp(-0.5 * ((t - t0 - c) / sd) ** 2)
            if kind == "breakout":
                amp = rng.uniform(0.9, 2.1)
                lc, ls, lsl = rng.integers(2, 5), rng.integers(4, 8), rng.integers(7, 11)
                niche = niche + bn * amp * 1.2 * ramp(0)
                creator = creator + bc * amp * ramp(lc)
                search = search + bs * amp * ramp(ls)
                sales = sales + bsales * rng.uniform(0.6, 1.4) * ramp(lsl)
                intent = intent + rng.uniform(0.10, 0.22) * ramp(1)
                breadth = breadth + bbr * rng.uniform(2.0, 4.5) * ramp(2)
            else:  # fad: loud upstream, thin underneath
                amp = rng.uniform(0.8, 2.0)
                niche = niche + bn * amp * 1.2 * bump(3)
                creator = creator + bc * amp * 0.9 * bump(4)
                search = search + bs * rng.uniform(0.15, 0.6) * bump(6)
                sales = sales + bsales * rng.uniform(0.0, 0.06) * bump(8)
                intent = intent - rng.uniform(0.0, 0.03) * bump(3)
                breadth = breadth + rng.uniform(1.0, 3.0) * bump(4)
        noise = lambda s, sd: s * (1 + rng.normal(0, sd, T))
        df = pd.DataFrame({
            "week": t, "category": name,
            "niche": noise(niche, 0.09), "creator": noise(creator, 0.09),
            "search": noise(search, 0.07), "sales": noise(sales, 0.045),
            "intent": np.clip(intent + rng.normal(0, 0.02, T), 0.01, 0.6),
            "breadth": np.clip(breadth + rng.normal(0, 0.9, T), 1, None),
        })
        frames.append(df)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(events)


def _zg(s: pd.Series):
    recent = s.rolling(2).mean()
    base, sd = s.shift(2).rolling(24).mean(), s.shift(2).rolling(24).std()
    z = (recent - base) / (sd + 1e-6)
    g = s.rolling(3).mean() / s.shift(3).rolling(4).mean() - 1
    return z, g


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, d in raw.groupby("category", sort=False):
        d = d.sort_values("week").reset_index(drop=True)
        f = d[["week", "category"]].copy()
        for c in SIGNALS:
            f[f"{c}_z"], f[f"{c}_g"] = _zg(d[c])
        f["niche_acc"] = f["niche_g"] - f["niche_g"].shift(2)
        f["intent"] = d["intent"].rolling(2).mean()
        f["intent_d"] = f["intent"] - d["intent"].shift(2).rolling(12).mean()
        f["breadth"] = d["breadth"].rolling(2).mean()
        f["breadth_r"] = f["breadth"] / d["breadth"].shift(2).rolling(12).mean()
        f["divergence"] = f["niche_z"] - f["search_z"]
        f["creator_gap"] = f["creator_z"] - f["search_z"]
        base = d["sales"].rolling(8).mean()
        fut = pd.concat([d["sales"].shift(-k) for k in range(1, HORIZON + 1)], axis=1).max(axis=1)
        f["y"] = (fut / base >= 1.35).astype(float)
        f.loc[fut.isna(), "y"] = np.nan
        out.append(f)
    return pd.concat(out, ignore_index=True)


def classify_stage(row) -> str:
    if row["sales_z"] > 1.5:
        return "Mainstream"
    if row["search_z"] > 1.2:
        return "Ignition"
    if row["creator_z"] > 1.2:
        return "Spark"
    if row["niche_z"] > 1.2:
        return "Whisper"
    return "Quiet"


def hype_trap(row) -> bool:
    """Loud upstream but thin underneath: high niche heat, low buy-intent change, narrow creator base."""
    return bool(row["niche_z"] > 1.5 and row["intent_d"] < 0.03 and row["breadth_r"] < 1.35)


def fit(feats: pd.DataFrame):
    d = feats.dropna(subset=FEATURES + ["y"])
    tr = d[d.week <= TRAIN_END]
    model = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.06,
                                       subsample=0.8, random_state=11)
    model.fit(tr[FEATURES], tr["y"])
    return model


def score(feats: pd.DataFrame, model) -> pd.DataFrame:
    f = feats.copy()
    ok = f[FEATURES].notna().all(axis=1)
    f["p"] = np.nan
    f.loc[ok, "p"] = model.predict_proba(f.loc[ok, FEATURES])[:, 1]
    return f


def backtest(scored: pd.DataFrame, events: pd.DataFrame, thr: float = 0.5):
    d = scored.dropna(subset=["p", "y"])
    te = d[(d.week >= TEST_START)]
    res = {
        "n_test": int(len(te)), "base_rate": float(te.y.mean()),
        "auc_model": roc_auc_score(te.y, te.p),
        "auc_naive": roc_auc_score(te.y, te.search_g.fillna(0)),
        "ap_model": average_precision_score(te.y, te.p),
        "ap_naive": average_precision_score(te.y, te.search_g.fillna(0)),
    }
    alerts = te[te.p >= thr]
    res["precision_at_thr"] = float(alerts.y.mean()) if len(alerts) else float("nan")
    # weekly top-5 hit rate
    def top5(col):
        hits = []
        for _, g in te.groupby("week"):
            if g.y.sum() == 0:
                continue
            hits.append(g.nlargest(5, col).y.mean())
        return float(np.mean(hits))
    res["top5_model"], res["top5_naive"] = top5("p"), top5("search_g")
    # event-level lead time on held-out breakouts
    rows = []
    for _, e in events[(events.kind == "breakout") & (events.slot == 1)].iterrows():
        g = scored[(scored.category == e.category) & (scored.week >= max(e.t0, TEST_START))].sort_values("week")
        a = g[g.p >= thr]
        n = g[g.search_z > 1.5]
        s = g[(g.sales_z > 2.0) & (g.sales_g > 0.15)]
        if len(s) == 0:
            continue
        sw = int(s.week.iloc[0])
        rows.append(dict(category=e.category, t0=e.t0, sniffer=int(a.week.iloc[0]) if len(a) else None,
                         naive=int(n.week.iloc[0]) if len(n) else None, sales_visible=sw))
    lt = pd.DataFrame(rows)
    res["events"] = lt
    if len(lt):
        det = lt.dropna(subset=["sniffer"])
        res["recall"] = len(det) / len(lt)
        res["lead_sniffer"] = float((det.sales_visible - det.sniffer).mean())
        dn = lt.dropna(subset=["naive"])
        res["lead_naive"] = float((dn.sales_visible - dn.naive).mean()) if len(dn) else float("nan")
        res["n_events"] = len(lt)
    # false alarms: fad events in test slot flagged
    fads = events[(events.kind == "fad") & (events.slot == 1)]
    fa = 0
    for _, e in fads.iterrows():
        g = scored[(scored.category == e.category) & (scored.week >= max(e.t0 - 2, TEST_START)) & (scored.week <= e.t0 + 12)]
        fa += int((g.p >= thr).any())
    res["fads_total"], res["fads_flagged"] = len(fads), fa
    return res
