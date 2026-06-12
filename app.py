import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re

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
st.markdown('''<div class="sub-header">Upload your sales CSV and chat
            with your data instantly</div>''', unsafe_allow_html=True)

# --- Session State ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'df' not in st.session_state:
    st.session_state.df = None

# --- Smart Analytics Engine ---
def detect_columns(df):
    cols = {
        'sales': None, 'profit': None, 'quantity': None,
        'date': None, 'category': None, 'product': None,
        'region': None, 'customer': None, 'discount': None
    }
    for col in df.columns:
        col_lower = col.lower()
        if any(x in col_lower for x in ['sale', 'revenue', 'amount']):
            cols['sales'] = col
        elif any(x in col_lower for x in ['profit', 'margin']):
            cols['profit'] = col
        elif any(x in col_lower for x in ['qty', 'quantity', 'units']):
            cols['quantity'] = col
        elif any(x in col_lower for x in ['date', 'time', 'order']):
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col])
                    cols['date'] = col
                except Exception:
                    pass
        elif any(x in col_lower for x in ['category', 'type', 'segment']):
            cols['category'] = col
        elif any(x in col_lower for x in ['product', 'item', 'name']):
            cols['product'] = col
        elif any(x in col_lower for x in ['region', 'area', 'zone', 'state']):
            cols['region'] = col
        elif any(x in col_lower for x in ['customer', 'client', 'buyer']):
            cols['customer'] = col
        elif any(x in col_lower for x in ['discount', 'promo']):
            cols['discount'] = col
    return cols

def analyze_question(question, df):
    q = question.lower()
    cols = detect_columns(df)
    response = ""
    chart = None
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # --- Total Sales ---
    if any(x in q for x in ['total sales', 'overall sales',
                              'how much sales', 'revenue']):
        if cols['sales']:
            total = df[cols['sales']].sum()
            avg = df[cols['sales']].mean()
            max_val = df[cols['sales']].max()
            response = f"""💰 **Sales Overview**

- **Total Sales:** ${total:,.2f}
- **Average Transaction:** ${avg:,.2f}
- **Highest Single Sale:** ${max_val:,.2f}
- **Total Transactions:** {len(df):,}

📌 **Insight:** Your average transaction value is ${avg:,.2f}.
{'Consider upselling strategies to increase this.' if avg < total/len(df)*1.2 else 'Strong average transaction value!'}"""
            if cols['category']:
                cat_sales = df.groupby(cols['category'])[cols['sales']].sum(
                ).sort_values(ascending=False).reset_index()
                chart = px.bar(cat_sales, x=cols['sales'],
                              y=cols['category'], orientation='h',
                              title='Total Sales by Category',
                              color=cols['sales'],
                              color_continuous_scale='Blues')
        else:
            response = summarize_numeric(df)

    # --- Profit ---
    elif any(x in q for x in ['profit', 'margin', 'earning']):
        if cols['profit']:
            total_profit = df[cols['profit']].sum()
            avg_profit = df[cols['profit']].mean()
            loss_count = (df[cols['profit']] < 0).sum()
            response = f"""📈 **Profit Analysis**

- **Total Profit:** ${total_profit:,.2f}
- **Average Profit per Transaction:** ${avg_profit:,.2f}
- **Loss-making Transactions:** {loss_count:,}
"""
            if cols['sales']:
                margin = (total_profit / df[cols['sales']].sum()) * 100
                response += f"- **Overall Profit Margin:** {margin:.2f}%\n"
                response += f"\n📌 **Insight:** {'Your margin is healthy above 10%!' if margin > 10 else 'Margin below 10% — review pricing and costs.'}"

            if cols['category']:
                cat_profit = df.groupby(cols['category'])[cols['profit']].sum(
                ).sort_values(ascending=True).reset_index()
                chart = go.Figure(go.Bar(
                    y=cat_profit[cols['category']],
                    x=cat_profit[cols['profit']],
                    orientation='h',
                    marker_color=['coral' if x < 0 else 'steelblue'
                                 for x in cat_profit[cols['profit']]]))
                chart.update_layout(
                    title='Profit by Category (Red = Loss)',
                    xaxis_title='Profit ($)')
        else:
            response = "I couldn't find a profit column in your data."

    # --- Top Products ---
    elif any(x in q for x in ['top', 'best', 'highest',
                                'most', 'leading', 'ranking']):
        target_col = cols['product'] or cols['category'] or cat_cols[0] if cat_cols else None
        value_col = cols['sales'] or cols['profit'] or numeric_cols[0] if numeric_cols else None

        if target_col and value_col:
            top = df.groupby(target_col)[value_col].sum(
            ).sort_values(ascending=False).head(10).reset_index()
            response = f"""🏆 **Top 10 {target_col} by {value_col}**\n\n"""
            for i, row in top.iterrows():
                response += f"{i+1}. **{row[target_col]}** — ${row[value_col]:,.2f}\n"
            response += f"\n📌 **Insight:** {top.iloc[0][target_col]} leads with ${top.iloc[0][value_col]:,.2f}"

            chart = px.bar(top, x=value_col, y=target_col,
                          orientation='h',
                          title=f'Top 10 {target_col} by {value_col}',
                          color=value_col,
                          color_continuous_scale='Viridis')
        else:
            response = "I couldn't identify the right columns for ranking."

    # --- Regional Analysis ---
    elif any(x in q for x in ['region', 'area', 'location',
                                'state', 'geographic', 'where']):
        target_col = cols['region'] or next(
            (c for c in cat_cols if any(x in c.lower()
             for x in ['region', 'state', 'city'])), None)
        value_col = cols['sales'] or numeric_cols[0] if numeric_cols else None

        if target_col and value_col:
            regional = df.groupby(target_col).agg({
                value_col: 'sum'
            }).sort_values(value_col, ascending=False).reset_index()

            response = f"""🗺️ **Regional Performance**\n\n"""
            for i, row in regional.iterrows():
                response += f"- **{row[target_col]}:** ${row[value_col]:,.2f}\n"

            top_region = regional.iloc[0][target_col]
            response += f"\n📌 **Insight:** {top_region} is your strongest market. Consider expanding successful strategies from here to other regions."

            chart = px.bar(regional, x=target_col, y=value_col,
                          title=f'{value_col} by {target_col}',
                          color=value_col,
                          color_continuous_scale='Blues')
        else:
            response = "I couldn't find regional data in your dataset."

    # --- Trend Analysis ---
    elif any(x in q for x in ['trend', 'over time', 'monthly',
                                'growth', 'yearly', 'time series']):
        date_col = cols['date'] or next(
            (c for c in df.columns if 'date' in c.lower()), None)
        value_col = cols['sales'] or numeric_cols[0] if numeric_cols else None

        if date_col and value_col:
            df_temp = df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col])
            df_temp['Month'] = df_temp[date_col].dt.to_period('M').astype(str)
            monthly = df_temp.groupby('Month')[value_col].sum().reset_index()

            first = monthly[value_col].iloc[0]
            last = monthly[value_col].iloc[-1]
            growth = ((last - first) / first) * 100

            response = f"""📅 **Sales Trend Analysis**

- **First Period:** ${first:,.2f}
- **Latest Period:** ${last:,.2f}
- **Overall Growth:** {growth:+.1f}%
- **Best Month:** {monthly.loc[monthly[value_col].idxmax(), 'Month']} (${monthly[value_col].max():,.2f})
- **Worst Month:** {monthly.loc[monthly[value_col].idxmin(), 'Month']} (${monthly[value_col].min():,.2f})

📌 **Insight:** {'Positive growth trend!' if growth > 0 else 'Declining trend — needs attention!'}"""

            chart = px.line(monthly, x='Month', y=value_col,
                           title=f'{value_col} Trend Over Time',
                           markers=True)
            chart.update_traces(line_color='steelblue', line_width=2)
        else:
            response = "I couldn't find date/time data for trend analysis."

    # --- Loss Making ---
    elif any(x in q for x in ['loss', 'losing', 'negative',
                                'worst', 'poor', 'underperform']):
        if cols['profit']:
            loss_items = df[df[cols['profit']] < 0]
            if len(loss_items) > 0:
                target_col = cols['product'] or cols['category'] or cat_cols[0] if cat_cols else None
                response = f"""🚨 **Loss-Making Analysis**

- **Total Loss Transactions:** {len(loss_items):,}
- **Total Loss Amount:** ${loss_items[cols['profit']].sum():,.2f}
- **Percentage of Transactions:** {len(loss_items)/len(df)*100:.1f}%\n\n"""

                if target_col:
                    loss_by = loss_items.groupby(target_col)[cols['profit']].sum(
                    ).sort_values().head(5).reset_index()
                    response += "**Top Loss-Making Items:**\n"
                    for _, row in loss_by.iterrows():
                        response += f"- {row[target_col]}: ${row[cols['profit']]:,.2f}\n"
                    response += "\n📌 **Recommendation:** Review pricing strategy for loss-making items immediately."

                    chart = go.Figure(go.Bar(
                        y=loss_by[target_col],
                        x=loss_by[cols['profit']],
                        orientation='h',
                        marker_color='coral'))
                    chart.update_layout(title='Top Loss-Making Items')
            else:
                response = "Great news! No loss-making transactions found in your data."
        else:
            response = "I couldn't find profit data to identify losses."

    # --- Distribution ---
    elif any(x in q for x in ['distribution', 'breakdown',
                                'share', 'proportion', 'percentage']):
        target_col = cols['category'] or cat_cols[0] if cat_cols else None
        value_col = cols['sales'] or numeric_cols[0] if numeric_cols else None

        if target_col and value_col:
            dist = df.groupby(target_col)[value_col].sum().reset_index()
            response = f"""📊 **{value_col} Distribution by {target_col}**\n\n"""
            total = dist[value_col].sum()
            for _, row in dist.iterrows():
                pct = row[value_col] / total * 100
                response += f"- **{row[target_col]}:** ${row[value_col]:,.2f} ({pct:.1f}%)\n"
            response += f"\n📌 **Insight:** {dist.loc[dist[value_col].idxmax(), target_col]} dominates with {dist[value_col].max()/total*100:.1f}% share."

            chart = px.pie(dist, names=target_col, values=value_col,
                          title=f'{value_col} Distribution by {target_col}',
                          hole=0.4)
        else:
            response = "I couldn't find the right columns for distribution analysis."

    # --- Summary / General ---
    elif any(x in q for x in ['summary', 'overview', 'tell me',
                                'about', 'describe', 'what']):
        response = summarize_numeric(df)

    # --- Discount Analysis ---
    elif any(x in q for x in ['discount', 'promo', 'offer']):
        if cols['discount']:
            avg_disc = df[cols['discount']].mean() * 100
            high_disc = (df[cols['discount']] > 0.3).sum()
            response = f"""🏷️ **Discount Analysis**

- **Average Discount:** {avg_disc:.1f}%
- **High Discount Transactions (>30%):** {high_disc:,}
- **No Discount Transactions:** {(df[cols['discount']] == 0).sum():,}

📌 **Insight:** {'High discounting detected — may be hurting margins.' if avg_disc > 20 else 'Discount levels appear reasonable.'}"""

            chart = px.histogram(df, x=cols['discount'],
                               title='Discount Distribution',
                               color_discrete_sequence=['steelblue'])
        else:
            response = "I couldn't find discount data in your dataset."

    # --- Default Fallback ---
    else:
        response = summarize_numeric(df)
        response = f"""🤔 I'm not sure exactly what you're asking, but here's a quick overview of your data:

{response}

💡 **Try asking:**
- "What are my total sales?"
- "Show me the top performing products"
- "What is my profit margin?"
- "Show sales trend over time"
- "Which region performs best?"
- "What items are losing money?"
"""

    return response, chart

def summarize_numeric(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = "📋 **Dataset Overview**\n\n"
    summary += f"- **Total Rows:** {len(df):,}\n"
    summary += f"- **Total Columns:** {df.shape[1]}\n\n"
    summary += "**Key Metrics:**\n"
    for col in numeric_cols[:5]:
        summary += f"- **{col}:** Total={df[col].sum():,.2f} | Avg={df[col].mean():,.2f}\n"
    return summary

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📁 Upload Your Data")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, encoding='latin-1')
            st.session_state.df = df
            st.success(f"✅ Loaded {df.shape[0]:,} rows!")
            st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    sample_questions = [
        "What are my total sales?",
        "Show me profit analysis",
        "Which are the top products?",
        "Show regional performance",
        "What is the sales trend?",
        "Which items are losing money?",
        "Show sales distribution",
        "Discount analysis"
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Main Interface ---
if st.session_state.df is None:
    st.info("👈 Upload a CSV file from the sidebar to get started!")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🧠 Smart Analysis**\nUnderstands natural language questions")
    with col2:
        st.markdown("**📈 Auto Charts**\nGenerates relevant visualizations automatically")
    with col3:
        st.markdown("**💡 Insights**\nActionable business recommendations")
else:
    df = st.session_state.df

    with st.expander("👀 Preview Data", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### 💬 Chat with Your Data")

    # Display messages
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message-user">
            👤 <strong>You:</strong> {message['content']}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message-ai">
            🤖 <strong>AI Analyst:</strong><br>{message['content']}
            </div>""", unsafe_allow_html=True)
            if message.get('chart') is not None:
                st.plotly_chart(message['chart'], use_container_width=True)

    # Handle sidebar button questions
    if hasattr(st.session_state, 'pending_question'):
        question = st.session_state.pending_question
        del st.session_state.pending_question
        st.session_state.messages.append({
            'role': 'user',
            'content': question
        })
        response, chart = analyze_question(question, df)
        st.session_state.messages.append({
            'role': 'assistant',
            'content': response,
            'chart': chart
        })
        st.rerun()

    # Chat input
    question = st.chat_input("Ask anything about your sales data...")
    if question:
        st.session_state.messages.append({
            'role': 'user',
            'content': question
        })
        with st.spinner("Analyzing..."):
            response, chart = analyze_question(question, df)
        st.session_state.messages.append({
            'role': 'assistant',
            'content': response,
            'chart': chart
        })
        st.rerun()
# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.8rem'>
Built with ❤️ using Streamlit | Sales Insights Chatbot v2.0
</div>""", unsafe_allow_html=True)           
            })

            
