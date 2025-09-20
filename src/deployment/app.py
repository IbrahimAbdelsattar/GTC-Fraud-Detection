import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import json
import logging
from typing import Dict, Any

from src.utils.model_utils import ModelPredictor
from src.utils.data_preprocessing import validate_input_data
from src.utils.monitoring import ModelMonitor
from src.config.parameters import ARTIFACT_PATHS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Advanced Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found: {file_name}. Please ensure 'style.css' is in the 'src/deployment/' directory.")

load_css("src/deployment/style.css")


if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'monitor' not in st.session_state:
    st.session_state.monitor = ModelMonitor()

def load_model():
    """Loads the real model using your ModelPredictor class."""
    try:
        if st.session_state.predictor is None:
            with st.spinner("🔄 Loading AI model..."):
                st.session_state.predictor = ModelPredictor()
                st.session_state.predictor.load_model()
                st.session_state.predictor.load_threshold()
        return True
    except Exception as e:
        st.error(f"🚨 Error loading model: {str(e)}")
        return False

def create_sample_transaction() -> Dict:
    """Creates a random sample transaction for demonstration."""
    np.random.seed(int(time.time()))
    sample = {
        'Time': np.random.randint(0, 172800),
        'Amount': round(np.random.uniform(1, 1000), 2),
    }
    for i in range(1, 29):
        sample[f'V{i}'] = np.random.normal(0, 1)
    return sample

def main():
    """Main function to run the Streamlit app."""
    st.markdown('<div class="main-header"><h1>🛡️ Advanced Fraud Detection System</h1><p>AI-Powered Transaction Security Analysis</p></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="sidebar-nav"><h3 style="color: white; text-align: center;">🧭 Navigation</h3></div>', unsafe_allow_html=True)
        
        page = st.selectbox(
            "",
            ["🔍 Single Prediction", "📊 Batch Processing", "📈 Model Monitoring"],
            label_visibility="collapsed"
        )

    if not load_model():
        st.stop()
    
    if "Single Prediction" in page:
        single_prediction_page()
    elif "Batch Processing" in page:
        batch_processing_page()
    elif "Model Monitoring" in page:
        monitoring_page()


def single_prediction_page():
    """Page for analyzing a single transaction."""
    st.markdown('<div class="feature-card"><h2>🔍 Real-Time Transaction Analysis</h2><p>Analyze individual transactions for fraud patterns using advanced machine learning</p></div>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("💳 Transaction Input")

        input_method = st.radio(
            "Select your preferred input method:",
            ["✍️ Manual Entry", "🎲 Sample Transaction", "📋 JSON Import"],
            horizontal=True
        )
        
        transaction_data = {}
        submitted = False
        
        if "Manual Entry" in input_method:
            with st.form("transaction_form"):
                st.markdown("**⏰ Transaction Timing & Amount**")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    transaction_data['Time'] = st.number_input("⏱️ Time (seconds)", min_value=0, value=3600, help="Time elapsed from the first transaction in the dataset")
                with col_b:
                    transaction_data['Amount'] = st.number_input("💰 Amount ($)", min_value=0.01, value=100.0, format="%.2f")
                
                st.markdown("---")
                st.markdown("**🔢 PCA Feature Components**")
                st.caption("These are anonymized features derived from the original transaction data")
                
                tabs = st.tabs(["V1-V10", "V11-V20", "V21-V28"])
                
                with tabs[0]:
                    cols = st.columns(5)
                    for i in range(1, 11):
                        with cols[(i-1) % 5]:
                            transaction_data[f'V{i}'] = st.number_input(f"V{i}", value=0.0, format="%.6f", key=f"v{i}_1")
                
                with tabs[1]:
                    cols = st.columns(5)
                    for i in range(11, 21):
                        with cols[(i-11) % 5]:
                            transaction_data[f'V{i}'] = st.number_input(f"V{i}", value=0.0, format="%.6f", key=f"v{i}_2")
                
                with tabs[2]:
                    cols = st.columns(4)
                    for i in range(21, 29):
                        with cols[(i-21) % 4]:
                            transaction_data[f'V{i}'] = st.number_input(f"V{i}", value=0.0, format="%.6f", key=f"v{i}_3")
                
                st.markdown("---")
                submitted = st.form_submit_button("🚀 Analyze Transaction", use_container_width=True, type="primary")
        
        elif "Sample Transaction" in input_method:
            sample_data = create_sample_transaction()
            st.markdown("**📋 Generated Sample Data**")
            st.dataframe(pd.DataFrame([sample_data]), use_container_width=True)
            transaction_data = sample_data
            submitted = st.button("🎯 Analyze Sample", use_container_width=True, type="primary")
            
        elif "JSON Import" in input_method:
            st.markdown("**📋 Paste JSON Transaction Data**")
            json_input = st.text_area("", height=200, placeholder='{"Time": 3600, "Amount": 100.0, "V1": 0.1, ...}', label_visibility="collapsed")
            submitted = st.button("🔍 Process JSON", use_container_width=True, type="primary")
            
            if json_input and submitted:
                try:
                    transaction_data = json.loads(json_input)
                except json.JSONDecodeError:
                    st.error("❌ Invalid JSON format. Please check your input.")
                    submitted = False # Prevent processing invalid data
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("🎯 Analysis Results")
        
        if submitted and transaction_data:
            if not validate_input_data(transaction_data):
                st.error("❌ Invalid input data. Please check all required fields.")
                return
            
            try:
                with st.spinner("🧠 AI analyzing transaction..."):
                    start_time = time.time()
                    prediction, probability = st.session_state.predictor.predict_single(transaction_data, use_optimal_threshold=True)
                    processing_time = time.time() - start_time
                
                st.session_state.monitor.log_prediction(transaction_data, prediction, probability, processing_time)

                if prediction == 1:
                    st.markdown('<div class="fraud-alert">🚨 FRAUD DETECTED 🚨</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="safe-alert">✅ LEGITIMATE TRANSACTION</div>', unsafe_allow_html=True)
                
                col_prob, col_risk = st.columns(2)
                with col_prob:
                    st.metric("🎯 Fraud Probability", f"{probability:.6%}")
                with col_risk:
                    if probability > 0.8:
                        risk_level, risk_color = "🔴 High", "red"
                    elif probability > 0.5:
                        risk_level, risk_color = "🟡 Medium", "orange"
                    else:
                        risk_level, risk_color = "🟢 Low", "green"
                    st.metric("⚠️ Risk Level", risk_level)

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=probability,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Fraud Risk Score", 'font': {'size': 18}},
                    gauge={
                        'axis': {'range': [None, 1], 'tickformat': '.0%'},
                        'bar': {'color': risk_color, 'thickness': 0.8},
                        'steps': [
                            {'range': [0, 0.5], 'color': "#e8f5e8"},
                            {'range': [0.5, 0.8], 'color': "#fff3cd"},
                            {'range': [0.8, 1], 'color': "#f8d7da"}],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': st.session_state.predictor.optimal_threshold}
                    }
                ))
                fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.success(f"⚡ Processed in {processing_time*1000:.1f}ms")
                
            except Exception as e:
                st.error(f"💥 Analysis error: {str(e)}")
        
        else:
            st.info("👆 Enter transaction details and click analyze to see results")
            
        st.markdown('</div>', unsafe_allow_html=True)

def batch_processing_page():
    """Page for processing a batch of transactions from a CSV file."""
    st.markdown('<div class="feature-card"><h2>📊 Bulk Transaction Processing</h2><p>Process multiple transactions simultaneously with comprehensive reporting</p></div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        st.markdown("### 📁 Upload Your Data")
        uploaded_file = st.file_uploader("", type=['csv'], help="CSV file should contain Time, Amount, and V1-V28 columns", label_visibility="collapsed")
        st.markdown("**Supported format:** CSV with Time, Amount, and V1-V28 columns")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="processing-card">', unsafe_allow_html=True)
        st.markdown("**📋 Processing Info**")
        st.markdown("• Upload CSV file\n• Preview your data\n• Run batch analysis\n• Download results")
        st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df.columns = df.columns.str.strip()
            df = df.loc[:, ~df.columns.duplicated()]

            st.success(f"✅ Successfully loaded {len(df):,} transactions")

            with st.expander("📋 Data Preview", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Rows", f"{len(df):,}")
                c2.metric("Columns", len(df.columns))
                c3.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")

            if not validate_input_data(df):
                st.error("❌ Invalid data format. Please ensure all required columns are present.")
                return

            if st.button("🚀 Start Batch Processing", use_container_width=True, type="primary"):
                progress_container = st.container()
                progress_bar = progress_container.progress(0)
                status_text = progress_container.empty()
                
                start_time = time.time()
                results = [st.session_state.predictor.predict_single(row.to_dict(), use_optimal_threshold=True) for i, row in df.iterrows()]
                
                df['Prediction'] = [res[0] for res in results]
                df['Fraud_Probability'] = [res[1] for res in results]
                df['Risk_Level'] = pd.cut(df['Fraud_Probability'], bins=[0, 0.5, 0.8, 1.0], labels=['Low', 'Medium', 'High'], include_lowest=True)
                
                progress_bar.progress(100)
                status_text.success("✅ Batch processing complete!")
                time.sleep(1)
                progress_container.empty()

                total_time = time.time() - start_time
                
                st.markdown('<div class="feature-card">', unsafe_allow_html=True)
                st.subheader("📊 Processing Results")
                c1, c2, c3, c4 = st.columns(4)
                fraud_count = df['Prediction'].sum()
                c1.metric("📈 Total Processed", f"{len(df):,}")
                c2.metric("🚨 Fraud Detected", f"{fraud_count:,}")
                c3.metric("⚠️ Fraud Rate", f"{(fraud_count / len(df) * 100) if len(df) > 0 else 0:.1f}%")
                c4.metric("🔴 High Risk", f"{len(df[df['Risk_Level'] == 'High']):,}")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("💰 Total Amount", f"${df['Amount'].sum():,.2f}")
                c6.metric("💸 Fraud Amount", f"${df[df['Prediction'] == 1]['Amount'].sum():,.2f}")
                c7.metric("⚡ Avg Speed", f"{(total_time / len(df) * 1000) if len(df) > 0 else 0:.1f}ms")
                c8.metric("⏱️ Total Time", f"{total_time:.1f}s")
                st.markdown('</div>', unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    fraud_dist = df['Prediction'].value_counts().rename({0: "Legitimate", 1: "Fraud"})
                    fig_pie = px.pie(values=fraud_dist.values, names=fraud_dist.index, title="🎯 Transaction Classification", color_discrete_map={'Legitimate': '#51cf66', 'Fraud': '#ff6b6b'})
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(height=400, title_x=0.5, paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_pie, use_container_width=True)

                with c2:
                    risk_dist = df['Risk_Level'].value_counts().reindex(['Low', 'Medium', 'High'])
                    fig_bar = px.bar(x=risk_dist.index, y=risk_dist.values, title="⚠️ Risk Level Distribution", color=risk_dist.index, color_discrete_map={'Low': '#51cf66', 'Medium': '#ffa726', 'High': '#ff6b6b'})
                    fig_bar.update_layout(showlegend=False, height=400, title_x=0.5, xaxis_title="Risk Level", yaxis_title="Number of Transactions", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_bar, use_container_width=True)

                st.markdown("### 📥 Download Results")
                c1, c2 = st.columns(2)
                c1.download_button("📊 Download Full Results", df.to_csv(index=False).encode('utf-8'), f"fraud_analysis_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv", use_container_width=True)
                c2.download_button("🚨 Download Fraud Only", df[df['Prediction'] == 1].to_csv(index=False).encode('utf-8'), f"fraud_only_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv", use_container_width=True)

                with st.expander("📋 Detailed Results Table"):
                    st.dataframe(df[['Time', 'Amount', 'Prediction', 'Fraud_Probability', 'Risk_Level']].sort_values('Fraud_Probability', ascending=False), use_container_width=True)
        except Exception as e:
            st.error(f"💥 Error processing file: {str(e)}")


def monitoring_page():
    """Page for monitoring model performance."""
    st.markdown('<div class="feature-card"><h2>📈 Real-Time Model Monitoring</h2><p>Track model performance, prediction patterns, and system health metrics</p></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 1, 1])
    days_back = c1.selectbox("📅 Select monitoring period:", [1, 7, 30, 90], index=1)
    if c2.button("🔄 Refresh Data", use_container_width=True): st.rerun()
    if c3.checkbox("⚡ Auto-refresh"): time.sleep(5); st.rerun()

    try:
        report = st.session_state.monitor.generate_monitoring_report(days_back)
        if "error" in report:
            st.warning(f"⚠️ {report['error']}")
            return

        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("🎯 Key Performance Indicators")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔢 Total Predictions", f"{report['total_predictions']:,}", delta=f"+{report['total_predictions']} new" if report['total_predictions'] > 0 else None)
        c2.metric("🚨 Fraud Detected", f"{report['fraud_predictions']:,}", delta=f"{report['fraud_rate']:.1%} rate")
        c3.metric("⚡ Avg Response Time", f"{report['avg_processing_time']:.1f}ms", delta="Fast" if report['avg_processing_time'] < 100 else "Slow", delta_color="inverse")
        c4.metric("🎯 Model Confidence", f"{report['avg_fraud_probability']:.1%}", delta="High" if report.get('avg_fraud_probability', 0) > 0.7 else "Moderate")
        st.markdown('</div>', unsafe_allow_html=True)

        df_logs = st.session_state.monitor.load_prediction_logs(days_back)
        if not df_logs.empty:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.subheader("📊 Performance Analytics")
            c1, c2 = st.columns(2)
            with c1:
                daily_stats = df_logs.resample('D', on='timestamp').size()
                fig = px.line(x=daily_stats.index, y=daily_stats.values, title="📈 Daily Prediction Volume", labels={'x': 'Date', 'y': 'Predictions'}, markers=True)
                fig.update_traces(line_color='#667eea')
                fig.update_layout(title_x=0.5, paper_bgcolor="rgba(0,0,0,0)", height=350)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.histogram(df_logs, x='fraud_probability', nbins=25, title="📊 Fraud Probability Distribution", color_discrete_sequence=['#667eea'])
                fig.update_layout(title_x=0.5, paper_bgcolor="rgba(0,0,0,0)", height=350, xaxis_title="Fraud Probability", yaxis_title="Frequency")
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.subheader("📋 Recent Activity Log")
            display_logs = df_logs.tail(50).copy()
            display_logs['Status'] = display_logs['prediction'].map({0: '✅ Safe', 1: '🚨 Fraud'})
            display_logs['Risk Score'] = display_logs['fraud_probability'].map('{:.1%}'.format)
            display_logs['Response Time'] = display_logs['processing_time_ms'].map('{:.1f}ms'.format)
            display_logs['Timestamp'] = display_logs['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(display_logs[['Timestamp', 'Status', 'Risk Score', 'Response Time']].sort_values('Timestamp', ascending=False), use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("📊 No prediction data available for the selected time period.")
    except Exception as e:
        st.error(f"💥 Error generating monitoring report: {str(e)}")

    st.markdown('<div class="footer">🛡️ Advanced Fraud Detection System • Powered by AI • Real-time Protection</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
