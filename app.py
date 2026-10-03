import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine as E

st.set_page_config(page_title="Trend Sniffer", page_icon="🔥", layout="wide")
st.markdown("""
<style>
h1, h2, h3 {letter-spacing:-0.5px}
.stage {display:inline-block;white-space:nowrap;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600}
.card {background:#171B22;border:1px solid #262C36;border-radius:12px;padding:16px 18px;margin-bottom:10px}
.small {color:#8B94A3;font-size:13px}
</style>""", unsafe_allow_html=True)

STAGE_COL = {"Quiet": "#5B6472", "Whisper": "#7C9CFF", "Spark": "#F5B841", "Ignition": "#FF6B35", "Mainstream": "#3DDC97"}
THR = 0.6


@st.cache_resource
def load():
    raw, events = E.simulate()
    feats = E.build_features(raw)
    model = E.fit(feats)
    scored = E.score(feats, model)
    return raw, events, scored, model, E.backtest(scored, events, THR)


raw, events, scored, model, bt = load()

st.title("🔥 Trend Sniffer")
st.caption("MarketMind AI · Finds demand while it is still a whisper, and tells real ignition from hype.")

tab1, tab2, tab3, tab4 = st.tabs(["Radar (time machine)", "Deep dive", "Proof: backtest", "Action brief"])

# ---------------------------------------------------------------- Radar
with tab1:
    c1, c2 = st.columns([3, 1])
    week = c1.slider("Replay the market as of week", E.TEST_START, E.T - 11, 78,
                     help="Weeks after training cut-off, so the model has never seen them.")
    show_hype = c2.toggle("Show hype traps", True)
    snap = scored[scored.week == week].dropna(subset=["p"]).copy()
    snap["stage"] = snap.apply(E.classify_stage, axis=1)
    snap["trap"] = snap.apply(E.hype_trap, axis=1)
    snap["outcome"] = snap.y.map({1.0: "Sales spiked within 10 wks", 0.0: "No spike"})
    early = snap[snap.stage != "Mainstream"].sort_values("p", ascending=False)
    n_main = int((snap.stage == "Mainstream").sum())
    flagged = early[early.p >= THR]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Early flags", len(flagged), help="Flagged before sales have visibly moved")
    k2.metric("Of those, did spike", f"{int(flagged.y.sum())} / {len(flagged)}" if len(flagged) else "0")
    k3.metric("Hype traps caught", int(snap.trap.sum()))
    k4.metric("Already mainstream", n_main, help="Too late for a head start; excluded from the list")

    st.markdown("##### Ignition leaderboard (pre-sales only)")
    for _, r in early.head(8).iterrows():
        col = STAGE_COL[r.stage]
        trap = " &nbsp;⚠️ <b>hype trap</b>" if (r.trap and show_hype) else ""
        bar = f"<div style='height:6px;border-radius:3px;background:#262C36;margin-top:8px'><div style='height:6px;border-radius:3px;width:{min(r.p,1)*100:.0f}%;background:#FF6B35'></div></div>"
        st.markdown(
            f"<div class='card'><b>{r.category}</b> &nbsp;<span class='stage' style='background:{col}22;color:{col}'>{r.stage}</span>{trap}"
            f"<div class='small'>Ignition probability <b>{r.p:.0%}</b> · buy-intent {r.intent:.0%} · creators {r.breadth:.0f} · "
            f"<i>what actually happened: {r.outcome}</i></div>{bar}</div>", unsafe_allow_html=True)
    st.markdown("##### Signal ladder")
    st.caption("Where every category sits on the niche → creator → search → sales ladder right now.")
    lad = snap.groupby("stage").size().reindex(E.STAGES).fillna(0)
    fig = go.Figure(go.Bar(x=lad.index, y=lad.values, marker_color=[STAGE_COL[s] for s in lad.index]))
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), template="plotly_dark", plot_bgcolor="#0E1116", paper_bgcolor="#0E1116")
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- Deep dive
with tab2:
    cat = st.selectbox("Category", E.CATEGORIES, index=E.CATEGORIES.index("Matcha lattes"))
    d = raw[raw.category == cat]
    s = scored[scored.category == cat]
    wk = st.slider("As-of week", E.TEST_START, E.T - 11, 70, key="dd")
    fig = go.Figure()
    names = {"niche": "Niche communities", "creator": "Creators", "search": "Search", "sales": "Retail sales"}
    cols = {"niche": "#7C9CFF", "creator": "#F5B841", "search": "#FF6B35", "sales": "#3DDC97"}
    for c in E.SIGNALS:
        y = d[c] / d[c].iloc[:24].mean() * 100
        fig.add_trace(go.Scatter(x=d.week, y=y, name=names[c], line=dict(color=cols[c], width=2.4 if c in ("niche", "sales") else 1.6)))
    fig.add_vline(x=wk, line_dash="dash", line_color="#fff")
    fig.update_layout(height=340, template="plotly_dark", plot_bgcolor="#0E1116", paper_bgcolor="#0E1116",
                      margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Index (first 24 wks = 100)",
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")
    row = s[s.week == wk].iloc[0]
    stage = E.classify_stage(row)
    a, b, c = st.columns(3)
    a.metric("Ignition probability", f"{row.p:.0%}")
    b.metric("Stage", stage)
    c.metric("Hype trap?", "Yes ⚠️" if E.hype_trap(row) else "No")
    st.markdown("**Why the model says this** (this category vs. all categories in quiet weeks)")
    quiet = scored[(scored.week <= E.TRAIN_END)].dropna(subset=E.FEATURES)
    pct = {f: float((quiet[f] < row[f]).mean()) for f in ["intent_d", "breadth_r", "niche_z", "creator_z", "search_z", "divergence"]}
    fig2 = go.Figure(go.Bar(x=list(pct.values()), y=[E.FEATURE_LABELS[k] for k in pct], orientation="h", marker_color="#FF6B35"))
    fig2.update_layout(height=260, template="plotly_dark", plot_bgcolor="#0E1116", paper_bgcolor="#0E1116",
                       margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(range=[0, 1], tickformat=".0%", title="Percentile vs. normal weeks"))
    st.plotly_chart(fig2, width="stretch")
    imp = pd.Series(model.feature_importances_, index=[E.FEATURE_LABELS[f] for f in E.FEATURES]).sort_values(ascending=False).head(4)
    st.caption("Model leans most on: " + ", ".join(f"{k} ({v:.0%})" for k, v in imp.items()))

# ---------------------------------------------------------------- Backtest
with tab3:
    st.markdown("Model trained on weeks 0–50 only. Everything below is from weeks 62+ that it never saw.")
    a, b, c, d_ = st.columns(4)
    a.metric("AUC · Sniffer", f"{bt['auc_model']:.2f}", f"{bt['auc_model'] - bt['auc_naive']:+.2f} vs search-only")
    b.metric("Avg precision", f"{bt['ap_model']:.2f}", f"{bt['ap_model'] - bt['ap_naive']:+.2f} vs search-only")
    c.metric("Lead over sales", f"{bt['lead_sniffer']:.1f} wks", f"{bt['lead_sniffer'] - bt['lead_naive']:+.1f} wks vs search-only")
    d_.metric("Breakouts caught", f"{bt['recall']:.0%}", f"{bt['n_events']} held-out events")
    st.markdown("##### When each method fires, relative to the week sales became visible")
    ev = bt["events"].dropna(subset=["sniffer"]).copy()
    ev["Sniffer lead"] = ev.sales_visible - ev.sniffer
    ev["Search-only lead"] = ev.sales_visible - ev.naive
    fig = go.Figure()
    fig.add_bar(x=ev.category, y=ev["Sniffer lead"], name="Sniffer", marker_color="#FF6B35")
    fig.add_bar(x=ev.category, y=ev["Search-only lead"], name="Search-only rule", marker_color="#5B6472")
    fig.update_layout(barmode="group", height=320, template="plotly_dark", plot_bgcolor="#0E1116", paper_bgcolor="#0E1116",
                      margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Weeks of warning")
    st.plotly_chart(fig, width="stretch")
    st.info(f"At threshold {THR:.0%}: precision of alerts {bt['precision_at_thr']:.0%}; hype-only fads wrongly flagged "
            f"{bt['fads_flagged']} of {bt['fads_total']}. Weekly top-5 hit rate {bt['top5_model']:.0%} vs {bt['top5_naive']:.0%} for search-only.")
    st.warning("Data note: signals are simulated from the niche→creator→search→sales diffusion pattern so the method can be "
               "tested end-to-end. Live connectors (Reddit, YouTube, Google Trends, marketplace sales) are the next step.")

# ---------------------------------------------------------------- Brief
PLAYS = {
    "Whisper": ("Watch and prepare", ["Start a 2-week watch; do not spend media yet.", "Ask sourcing for lead times and MOQ quotes.", "Brief one small creator for organic content."]),
    "Spark": ("Prepare supply", ["Place a small test order (10–15% of expected peak).", "Line up 5–8 mid-size creators, not one celebrity.", "Draft 3 creative angles built on the language buyers use."]),
    "Ignition": ("Move now", ["Release inventory and lock the price band.", "Buy search terms before CPCs rise.", "Launch hero creative in the next 7 days."]),
    "Mainstream": ("Defend margin", ["Focus on availability and bundles.", "Expect competitors and price cuts; differentiate."]),
    "Quiet": ("No action", ["Nothing to do this week."]),
}
with tab4:
    cat = st.selectbox("Category", E.CATEGORIES, index=E.CATEGORIES.index("Air fryers"), key="ab")
    wk = st.slider("As-of week", E.TEST_START, E.T - 11, 72, key="ab2")
    row = scored[(scored.category == cat) & (scored.week == wk)].iloc[0]
    stage = E.classify_stage(row)
    trap = E.hype_trap(row)
    head, steps = PLAYS[stage]
    if trap:
        head, steps = "Hold: likely hype", ["Chatter is loud but buy-intent is flat and few creators are involved.", "Set a 2-week re-check instead of committing stock.", "Test one cheap creative only."]
    st.markdown(f"### {cat}: {head}")
    st.markdown(f"Ignition probability **{row.p:.0%}** · stage **{stage}** · buy-intent **{row.intent:.0%}** · distinct creators **{row.breadth:.0f}**")
    for s_ in steps:
        st.markdown(f"- {s_}")
    outcome = "Sales did spike within 10 weeks." if row.y == 1 else "Sales did not spike within 10 weeks."
    st.caption("Hindsight check: " + outcome)
    st.markdown("---")
    st.markdown("**Optional: LLM-written brief.** In production, an LLM reads the raw comments to score buy-intent and writes this brief in the team's voice.")
    key = st.text_input("Anthropic API key (optional, not stored)", type="password")
    if key and st.button("Write brief with Claude"):
        try:
            import anthropic
            msg = anthropic.Anthropic(api_key=key).messages.create(
                model="claude-sonnet-4-5", max_tokens=500,
                messages=[{"role": "user", "content": f"Write a 120-word brand brief for a D2C marketing lead. Category: {cat}. "
                           f"Stage: {stage}. Ignition probability {row.p:.0%}. Buy-intent {row.intent:.0%}. Creators {row.breadth:.0f}. Hype trap: {trap}. "
                           "Be concrete about the next 7 days. No fluff."}])
            st.write(msg.content[0].text)
        except Exception as e:
            st.error(f"Could not call the API: {e}")
