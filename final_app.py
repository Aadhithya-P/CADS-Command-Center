import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, f1_score

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CADS COMMAND CENTER",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CYBER SECURITY STYLING (Merged from Reference)
# =========================================================
st.markdown("""
    <style>
    .stApp {
        background-color: #0A0F1C;
        color: #E0F2FE;
    }
    section[data-testid="stSidebar"] {
        background-color: #0F1629;
        border-right: 1px solid #1E2A44;
    }
    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #22D3EE;
        letter-spacing: 2px;
        text-shadow: 0 0 15px rgba(34, 211, 238, 0.4);
    }
    .sub-title {
        font-size: 14px;
        color: #94A3B8;
        font-family: 'Courier New', monospace;
        margin-bottom: 20px;
    }
    .terminal {
        background-color: #0B0F1A;
        border: 1px solid #22D3EE;
        border-radius: 8px;
        padding: 15px;
        font-family: 'Courier New', monospace;
        color: #67E8F9;
        height: 450px;
        overflow-y: auto;
        white-space: pre-wrap;
        box-shadow: inset 0 0 10px #000;
    }
    .status-active {
        color: #4ADE80;
        font-weight: bold;
    }
    .log-id { color: #FBBF24; }
    .log-threat { color: #F87171; font-weight: bold; }
    .log-clear { color: #4ADE80; }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_cads_components():
    # Loading your ENN-balanced models
    dt_data = joblib.load("dt_enn.pkl")
    rf_data = joblib.load("rf_enn.pkl")
    ada_data = joblib.load("ada_enn.pkl")
    return dt_data, rf_data, ada_data, dt_data["scaler"], dt_data["features"]

try:
    dt_comp, rf_comp, ada_comp, scaler, features = load_cads_components()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# =========================================================
# HEADER & STATUS
# =========================================================
st.markdown('<div class="main-title">CADS: COMMAND CENTER</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">CONFIDENCE-AWARE DECISION SYSTEM | INTERPRETABLE ENSEMBLE LAYER</div>', unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("🔐 Security Console")
    uploaded_file = st.file_uploader("Ingest Activity Logs (CSV)", type=["csv"])
    st.divider()
    alpha = st.slider("Alpha (Agreement vs Prob)", 0.0, 1.0, 0.6)
    st.info("System Engine: ENN-Optimized")
    
    if st.button("🗑️ Clear Analysis Cache"):
        st.cache_resource.clear()
        st.rerun()

# =========================================================
# MAIN LOGIC
# =========================================================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # 1. Preprocessing
    X_input = df[features]
    X_scaled = scaler.transform(X_input)
    
    # 2. Parallel Model Inference
    # DT Inference
    p_dt = dt_comp["model"].predict(dt_comp["pca"].transform(X_scaled))
    prob_dt = dt_comp["model"].predict_proba(dt_comp["pca"].transform(X_scaled))[:, 1]
    
    # RF Inference
    p_rf = rf_comp["model"].predict(rf_comp["pca"].transform(X_scaled))
    prob_rf = rf_comp["model"].predict_proba(rf_comp["pca"].transform(X_scaled))[:, 1]
    
    # ADA Inference
    p_ada = ada_comp["model"].predict(ada_comp["pca"].transform(X_scaled))
    prob_ada = ada_comp["model"].predict_proba(ada_comp["pca"].transform(X_scaled))[:, 1]

    # 3. Dynamic F1 Weight Calculation
    if "insider" in df.columns:
        y_true = df["insider"]
        w_dt = f1_score(y_true, p_dt)
        w_rf = f1_score(y_true, p_rf)
        w_ada = f1_score(y_true, p_ada)
        weights = np.array([w_dt, w_rf, w_ada]) + 1e-6
        weights /= weights.sum()
    else:
        # Static fallback if labels aren't present
        weights = np.array([0.33, 0.33, 0.34])
        y_true = None

    # 4. CADS Pipeline & Terminal Log Generation
    final_preds, conf_scores, risk_tiers, terminal_html = [], [], [], ""
    
    for i in range(len(p_dt)):
        preds = np.array([p_dt[i], p_rf[i], p_ada[i]])
        probs = np.array([prob_dt[i], prob_rf[i], prob_ada[i]])
        
        # Weighted Decision
        score = np.sum(weights * preds)
        f_pred = 1 if score >= 0.5 else 0
        
        # Agreement & Probability Components
        agree = np.sum(preds == f_pred) / 3
        m_prob = np.mean(probs)
        
        # Composite Confidence Formula (C)
        conf = (alpha * agree) + ((1 - alpha) * m_prob)
        
        # Risk Tiering
        tier = "LOW RISK" if conf > 0.9 else "MODERATE RISK" if conf > 0.7 else "ESCALATE"
        
        final_preds.append(f_pred)
        conf_scores.append(conf)
        risk_tiers.append(tier)
        
        # Terminal Formatting
        status_label = '<span class="log-threat">THREAT_DET</span>' if f_pred == 1 else '<span class="log-clear">CLEAN_ACT</span>'
        terminal_html += f'<span class="log-id">[{i:03}]</span> {status_label} | C:{conf:.4f} | AGREE:{int(agree*100)}% | TIER:{tier}<br>'

    # =====================================================
    # UI LAYOUT: TERMINAL AND DATA
    # =====================================================
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.subheader("🖥️ CADS System Terminal")
        st.markdown(f'<div class="terminal">{terminal_html}</div>', unsafe_allow_html=True)
        
        st.subheader("📋 Ingested Dataset Overview")
        st.dataframe(df.head(10), use_container_width=True)

    with col_right:
        st.subheader("📊 Threat Analytics")
        
        # Metrics
        m1, m2 = st.columns(2)
        m1.metric("Total Scanned", len(df))
        m2.metric("Escalations", risk_tiers.count("ESCALATE"))
        
        # Chart
        res_df = pd.DataFrame({"Risk Tier": risk_tiers})
        fig = px.pie(res_df, names="Risk Tier", hole=0.5,
                     color="Risk Tier", 
                     color_discrete_map={'LOW RISK':'#22C55E', 'MODERATE RISK':'#EAB308', 'ESCALATE':'#EF4444'})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#E0F2FE")
        st.plotly_chart(fig, use_container_width=True)
        
        # Priority Table
        st.subheader("⚠️ Priority Action Items")
        full_res = pd.DataFrame({
            "ID": df.index,
            "Decision": ["Insider" if p == 1 else "Normal" for p in final_preds],
            "Conf": np.round(conf_scores, 3),
            "Tier": risk_tiers
        })
        
        # Display only the ESCALATE tier in the UI table
        escalate_df = full_res[full_res["Tier"] == "ESCALATE"].head(10)
        st.table(escalate_df)

        # --- DOWNLOAD LOGIC ---
        # 1. Convert dataframe to CSV
        @st.cache_data
        def convert_df(df_to_convert):
            return df_to_convert.to_csv(index=False).encode('utf-8')

        csv_data = convert_df(full_res)

        # 2. Add the download button
        st.download_button(
            label="📥 Export Full Forensic Report (CSV)",
            data=csv_data,
            file_name=f"CADS_Forensic_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # =====================================================
    # PERFORMANCE AUDIT
    # =====================================================
    if y_true is not None:
        st.divider()
        with st.expander("🔍 System Performance Audit"):
            acc = accuracy_score(y_true, final_preds)
            st.write(f"**CADS Combined Accuracy:** {acc:.4%}")
            report_dict = classification_report(y_true, final_preds, output_dict=True)

            # Convert to DataFrame and transpose
            temp_df = pd.DataFrame(report_dict).transpose()

            # Remove the average and accuracy rows
            # These keys are standard in the scikit-learn output
            to_drop = ['accuracy', 'macro avg', 'weighted avg']
            df_filtered = temp_df.drop(index=[row for row in to_drop if row in temp_df.index])
            st.text(df_filtered)

else:
    st.warning("SYSTEM STANDBY: Please upload a CERT dataset CSV to initiate the CADS decision layer.")