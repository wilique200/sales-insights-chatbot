import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import anthropic
import json
import io

# --- Page Config ---
st.set_page_config(
    page_title="Sales Insights Chatbot",
    page_icon="📊",
    layout="wide"
)

# --- Styling ---
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    color: #1f77b4;
    margin-bottom: 0.5rem;
}
.sub-header {
    font-size: 1rem;
    text-align: center;
    color: #666;
    margin-bottom: 2rem;
}
.chat-message-user {
    background: #e3f2fd;
    border-radius: 10px;
    padding: 10px 15px;
    margin: 5px 0;
    border-left: 4px solid #1f77b4;
}
.chat-message-ai {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 10px 15px;
    margin: 5px 0;
    border-left: 4px solid #28a745;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-header">📊 Sales Insights Chatbot</div>',
            unsafe_allow_html=True)
st.markdown('''<div class="sub-header">Upload your sales CSV and chat with
            your data using AI</div>''', unsafe_allow_html=True)

# --- Initialize Session State ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'data_summary' not in st.session_state:
    st.session_state.data_summary = None

# --- Helper: Generate Data Summary ---
def generate_data_summary(df):
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'numeric_summary': {},
        'sample_data': df.head(3).to_string()
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_summary'][col] = {
            'min': round(df[col].min(), 2),
            'max': round(df[col].max(), 2),
            'mean': round(df[col].mean(), 2),
            'sum': round(df[col].sum(), 2)
        }

    cat_cols = df.select_dtypes(include=['object']).columns
    summary['categorical_info'] = {}
    for col in cat_cols[:5]:
        summary['categorical_info'][col] = df[col].value_counts().head(5).to_dict()

    return summary

# --- Helper: Build Chart ---
def build_chart(df, question):
    question_lower = question.lower()
    fig = None

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    try:
        if any(word in question_lower for word in
               ['trend', 'over time', 'monthly', 'daily', 'yearly']):
            date_cols = [c for c in df.columns if 'date' in c.lower()
                        or 'time' in c.lower()]
            if date_cols and numeric_cols:
                df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
                df_time = df.groupby(date_cols[0])[numeric_cols[0]].sum().reset_index()
                fig = px.line(df_time, x=date_cols[0], y=numeric_cols[0],
                             title=f'{numeric_cols[0]} Over Time')

        elif any(word in question_lower for word in
                 ['top', 'best', 'highest', 'most', 'ranking']):
            if cat_cols and numeric_cols:
                top = df.groupby(cat_cols[0])[numeric_cols[0]].sum(
                ).sort_values(ascending=False).head(10).reset_index()
                fig = px.bar(top, x=numeric_cols[0], y=cat_cols[0],
                            orientation='h',
                            title=f'Top {cat_cols[0]} by {numeric_cols[0]}',
                            color=numeric_cols[0],
                            color_continuous_scale='Blues')

        elif any(word in question_lower for word in
                 ['distribution', 'breakdown', 'share', 'proportion']):
            if cat_cols and numeric_cols:
                dist = df.groupby(cat_cols[0])[numeric_cols[0]].sum().reset_index()
                fig = px.pie(dist, names=cat_cols[0], values=numeric_cols[0],
                            title=f'{numeric_cols[0]} Distribution by {cat_cols[0]}',
                            hole=0.4)

        elif any(word in question_lower for word in
                 ['compare', 'vs', 'versus', 'difference']):
            if cat_cols and numeric_cols:
                comp = df.groupby(cat_cols[0])[numeric_cols[0]].sum().reset_index()
                fig = px.bar(comp, x=cat_cols[0], y=numeric_cols[0],
                            title=f'{numeric_cols[0]} by {cat_cols[0]}',
                            color=cat_cols[0])

        elif any(word in question_lower for word in
                 ['profit', 'margin', 'performance']):
            if len(numeric_cols) >= 2:
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                                title=f'{numeric_cols[0]} vs {numeric_cols[1]}',
                                opacity=0.6)

    except Exception:
        pass

    return fig

# --- Helper: Ask Claude ---
def ask_claude(question, data_summary, api_key):
    client = anthropic.Anthropic(api_key=api_key)

    system_prompt = f"""You are an expert data analyst and business intelligence assistant.
You have been given access to a sales dataset with the following properties:

Dataset Shape: {data_summary['shape'][0]} rows, {data_summary['shape'][1]} columns
Columns: {', '.join(data_summary['columns'])}

Numeric Column Statistics:
{json.dumps(data_summary['numeric_summary'], indent=2)}

Top Categories:
{json.dumps(data_summary['categorical_info'], indent=2)}

Sample Data:
{data_summary['sample_data']}

Your job is to:
1. Answer questions about this data clearly and concisely
2. Provide specific numbers and insights from the statistics above
3. Give actionable business recommendations
4. Be conversational but professional
5. If asked about something not in the data, say so clearly

Always structure your response with:
- Direct answer to the question
- Key numbers/metrics
- Business insight or recommendation
"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": question}
        ],
        system=system_prompt
    )

    return message.content[0].text

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    api_key = st.text_input("Enter Anthropic API Key:",
                            type="password",
                            placeholder="sk-ant-...")

    st.markdown("---")
    st.markdown("### 📁 Upload Data")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.data_summary = generate_data_summary(df)
            st.success(f"✅ Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")

            st.markdown("### 📋 Dataset Info")
            st.write(f"**Rows:** {df.shape[0]:,}")
            st.write(f"**Columns:** {df.shape[1]}")
            st.write(f"**Columns:** {', '.join(df.columns.tolist())}")

        except Exception as e:
            st.error(f"Error loading file: {e}")

    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    st.markdown("""
    - What are my total sales?
    - Which product has highest profit?
    - Show me the top performing regions
    - What is my overall profit margin?
    - Which category drives most revenue?
    - What are the trends in my sales?
    """)

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- Main Chat Interface ---
if st.session_state.df is None:
    st.info("👈 Please upload a CSV file from the sidebar to get started!")

    st.markdown("### 📊 What can this app do?")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **🤖 AI Analysis**
        Ask any question about your sales data in plain English
        """)
    with col2:
        st.markdown("""
        **📈 Auto Charts**
        Automatically generates relevant visualizations
        """)
    with col3:
        st.markdown("""
        **💡 Insights**
        Get actionable business recommendations instantly
        """)

else:
    df = st.session_state.df

    # Data Preview
    with st.expander("👀 Preview Your Data", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 1:
            col1.metric("Total " + numeric_cols[0],
                       f"{df[numeric_cols[0]].sum():,.0f}")
        if len(numeric_cols) >= 2:
            col2.metric("Total " + numeric_cols[1],
                       f"{df[numeric_cols[1]].sum():,.0f}")
        col3.metric("Total Rows", f"{df.shape[0]:,}")
        col4.metric("Total Columns", f"{df.shape[1]}")

    st.markdown("---")
    st.markdown("### 💬 Chat with Your Data")

    # Display chat history
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message-user">
            👤 <strong>You:</strong> {message['content']}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message-ai">
            🤖 <strong>AI Analyst:</strong> {message['content']}
            </div>""", unsafe_allow_html=True)
            if 'chart' in message and message['chart'] is not None:
                st.plotly_chart(message['chart'],
                               use_container_width=True)

    # Chat Input
    question = st.chat_input("Ask anything about your sales data...")

    if question:
        if not api_key:
            st.warning("Please enter your Anthropic API key in the sidebar!")
        else:
            st.session_state.messages.append({
                'role': 'user',
                'content': question
            })

            with st.spinner("🤔 Analyzing your data..."):
                try:
                    response = ask_claude(
                        question,
                        st.session_state.data_summary,
                        api_key
                    )
                    chart = build_chart(df, question)

                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': response,
                        'chart': chart
                    })

                except Exception as e:
                    st.error(f"Error: {str(e)}")

            st.rerun()

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.8rem'>
Built with ❤️ using Streamlit & Claude AI | Sales Insights Chatbot v1.0
</div>""", unsafe_allow_html=True)
