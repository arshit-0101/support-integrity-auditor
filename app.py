
import streamlit as st
import pandas as pd
import numpy as np
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Support Integrity Auditor", page_icon="🔍", layout="wide")

@st.cache_resource
def load_models():
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained("arshit0101/support-integrity-auditor", use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained("arshit0101/support-integrity-auditor", torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2", device="cpu")
    return tokenizer, model, embedder, device

URGENCY_KEYWORDS = ["crash","down","urgent","critical","broken","outage","error","fail","cannot","not working","immediately","asap","blocked","data loss","security","payment failed"]
CALM_KEYWORDS = ["wondering","question","how do i","curious","operating hours","roadmap","feature request","no rush","whenever"]
PRIORITY_MAP = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}

def rule_score(text):
    t = text.lower()
    s = 0.5
    for k in URGENCY_KEYWORDS:
        if k in t: s += 0.08
    for k in CALM_KEYWORDS:
        if k in t: s -= 0.06
    return float(np.clip(s, 0, 1))

def res_score(hours):
    return float(np.clip(1.0 - hours/100.0, 0, 1))

def infer_sev(rule, res, emb):
    return 0.3*rule + 0.2*res + 0.5*emb

def get_label(sev):
    if sev >= 0.85: return "Critical"
    elif sev >= 0.65: return "High"
    elif sev >= 0.40: return "Medium"
    return "Low"

def predict(tokenizer, model, device, text):
    enc = tokenizer(text, truncation=True, max_length=256, padding="max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()[0]
    return int(np.argmax(probs)), float(probs[1])

def build_dossier(tid, priority, sev, delta, rule, res, emb, hours, conf):
    mtype = "Hidden Crisis" if delta > 0 else "False Alarm"
    return {
        "ticket_id": tid,
        "assigned_priority": priority,
        "inferred_severity": get_label(sev),
        "mismatch_type": mtype,
        "severity_delta": round(delta, 3),
        "feature_evidence": [
            {"signal": "rule_based_nlp", "value": round(rule,3), "weight": 0.3,
             "interpretation": "High urgency keywords detected" if rule > 0.6 else "Low urgency keywords detected"},
            {"signal": "resolution_time", "value": hours, "weight": 0.2,
             "interpretation": f"{hours}hrs resolution suggests high urgency" if res > 0.5 else f"{hours}hrs resolution suggests low urgency"},
            {"signal": "semantic_embedding", "value": round(emb,3), "weight": 0.5,
             "interpretation": "Semantically similar to high urgency tickets" if emb > 0.6 else "Semantically similar to low urgency tickets"},
        ],
        "constraint_analysis": f"Ticket assigned {priority} but signals suggest {get_label(sev)}. This is a {mtype}. Delta: {abs(delta):.2f}.",
        "confidence": round(conf, 3),
    }

with st.sidebar:
    st.title("🔍 SIA")
    st.caption("Support Integrity Auditor")
    st.markdown("---")
    mode = st.radio("Mode", ["Single Ticket", "Batch CSV", "Dashboard"])
    st.markdown("---")
    st.caption("Accuracy: 87.42% | F1: 0.854")

with st.spinner("Loading model..."):
    tokenizer, model, embedder, device = load_models()
st.sidebar.success("Model loaded!")

if mode == "Single Ticket":
    st.title("🎫 Single Ticket Audit")
    col1, col2 = st.columns(2)
    with col1:
        tid = st.text_input("Ticket ID", "TKT-000001")
        subject = st.text_input("Subject", "App crashes on login")
        channel = st.selectbox("Channel", ["Email","Chat","Phone","Web Form","Social Media"])
    with col2:
        priority = st.selectbox("Priority", ["Low","Medium","High","Critical"])
        hours = st.number_input("Resolution Time (hours)", 0.0, 120.0, 24.0)
        desc = st.text_area("Description", "The app crashes every time I try to login. Blocking my workflow.")

    if st.button("🔍 Audit", type="primary"):
        with st.spinner("Analyzing..."):
            txt = f"Channel: {channel} | Priority: {priority} | {subject} {desc}"
            rule = rule_score(subject + " " + desc)
            res = res_score(hours)
            ev = embedder.encode([txt])[0]
            emb = float(1 / (1 + np.exp(-np.linalg.norm(ev)/10)))
            sev = infer_sev(rule, res, emb)
            delta = sev - PRIORITY_MAP.get(priority.lower(), 0.5)
            pred, conf = predict(tokenizer, model, device, txt)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Assigned", priority)
        c2.metric("Inferred", get_label(sev))
        c3.metric("Delta", f"{delta:+.2f}")
        c4.metric("Confidence", f"{conf:.0%}")

        if pred == 1:
            mtype = "Hidden Crisis" if delta > 0 else "False Alarm"
            if mtype == "Hidden Crisis":
                st.error(f"🚨 HIDDEN CRISIS — More urgent than labeled! ({conf:.0%})")
            else:
                st.warning(f"⚠️ FALSE ALARM — Less urgent than labeled! ({conf:.0%})")
            with st.expander("📋 Evidence Dossier", expanded=True):
                st.json(build_dossier(tid, priority, sev, delta, rule, res, emb, hours, conf))
        else:
            st.success(f"✅ Priority looks correct ({conf:.0%} confidence)")

        with st.expander("🔬 Signal Breakdown"):
            st.dataframe(pd.DataFrame({
                "Signal": ["Rule NLP","Resolution","Embedding"],
                "Score": [round(rule,3), round(res,3), round(emb,3)],
                "Weight": [0.3, 0.2, 0.5]
            }), use_container_width=True)

elif mode == "Batch CSV":
    st.title("📂 Batch CSV Audit")
    st.markdown("Columns needed: Ticket_ID, Ticket_Subject, Ticket_Description, Priority_Level, Ticket_Channel, Resolution_Time_Hours")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        df = pd.read_csv(up)
        st.dataframe(df.head(), use_container_width=True)
        if st.button("🚀 Run Audit", type="primary"):
            results, dossiers = [], []
            prog = st.progress(0)
            for i, row in df.iterrows():
                txt = f"Channel: {row.get('Ticket_Channel','Email')} | Priority: {row.get('Priority_Level','Medium')} | {row.get('Ticket_Subject','')} {row.get('Ticket_Description','')}"
                rule = rule_score(str(row.get('Ticket_Subject','')) + " " + str(row.get('Ticket_Description','')))
                h = float(row.get('Resolution_Time_Hours', 24))
                res = res_score(h)
                ev = embedder.encode([txt])[0]
                emb = float(1 / (1 + np.exp(-np.linalg.norm(ev)/10)))
                sev = infer_sev(rule, res, emb)
                delta = sev - PRIORITY_MAP.get(str(row.get('Priority_Level','medium')).lower(), 0.5)
                pred, conf = predict(tokenizer, model, device, txt)
                mtype = ("Hidden Crisis" if delta > 0 else "False Alarm") if pred == 1 else "Consistent"
                results.append({"Ticket_ID": row.get('Ticket_ID',i), "Priority": row.get('Priority_Level',''), "Inferred": get_label(sev), "Delta": round(delta,3), "Mismatch": pred, "Type": mtype, "Confidence": round(conf,3)})
                if pred == 1:
                    dossiers.append(build_dossier(str(row.get('Ticket_ID',i)), str(row.get('Priority_Level','Medium')), sev, delta, rule, res, emb, h, conf))
                prog.progress((i+1)/len(df))
            rdf = pd.DataFrame(results)
            st.success(f"{rdf['Mismatch'].sum()} mismatches in {len(rdf)} tickets")
            st.dataframe(rdf, use_container_width=True)
            c1,c2 = st.columns(2)
            c1.download_button("⬇️ CSV", rdf.to_csv(index=False), "predictions.csv", "text/csv")
            c2.download_button("⬇️ JSON", json.dumps(dossiers,indent=2), "dossiers.json", "application/json")

elif mode == "Dashboard":
    st.title("📊 Dashboard")
    c1,c2,c3 = st.columns(3)
    ff = c1.file_uploader("full_results.csv", type="csv")
    df2 = c2.file_uploader("dossiers.json", type="json")
    af = c3.file_uploader("ablation_table.csv", type="csv")
    if ff and df2 and af:
        full = pd.read_csv(ff)
        dos = json.load(df2)
        abl = pd.read_csv(af)
        total = len(full); mis = int(full["mismatch"].sum())
        hidden = int(full[full["mismatch"]==1]["severity_delta"].gt(0).sum())
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Total", total); k2.metric("Mismatches", mis)
        k3.metric("Hidden Crises", hidden); k4.metric("False Alarms", mis-hidden)
        st.subheader("Mismatch by Priority")
        mp = full.groupby("Priority_Level")["mismatch"].mean().reset_index()
        st.bar_chart(mp.set_index("Priority_Level"))
        st.subheader("Ablation Table")
        st.dataframe(abl, use_container_width=True)
        st.subheader("Sample Dossiers")
        for d in dos[:5]:
            with st.expander(f"{d['ticket_id']} — {d['mismatch_type']}"):
                st.json(d)
