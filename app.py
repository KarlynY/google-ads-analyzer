import streamlit as st
import pandas as pd
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
import gspread
from datetime import datetime, timedelta
import plotly.express as px
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Google Ads Campaign Analyzer",
    page_icon="📊",
    layout="wide"
)

def apply_custom_style():
    # Custom CSS for modern styling
    st.markdown("""
        <style>
        /* Main styling */
        .stApp {
            background-color: white;
            color: black;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #1032cf;
            font-weight: 600;
        }
        
        /* Metrics cards */
        div.metric-container {
            background-color: white;
            border: 1px solid #e6e6e6;
            border-radius: 8px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Buttons */
        .stButton > button {
            background-color: #1032cf;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 4px 12px;
            font-weight: 500;
        }
        
        .stButton > button:hover {
            background-color: #0a2599;
            border: none;
        }
        
        /* Selectbox */
        .stSelectbox > div > div {
            background-color: white;
            border: 1px solid #e6e6e6;
            border-radius: 4px;
        }
        
        /* Text input */
        .stTextInput > div > div > input {
            background-color: white;
            border: 1px solid #e6e6e6;
            border-radius: 4px;
        }
        
        /* Dataframe styling */
        .dataframe {
            border: 1px solid #e6e6e6;
            border-radius: 8px;
            overflow: hidden;
        }
        
        /* Info boxes */
        .stAlert {
            background-color: #f0f3ff;
            border: 1px solid #1032cf;
            color: black;
        }
        
        /* Metrics */
        .metric-value {
            color: #1032cf;
            font-size: 24px;
            font-weight: bold;
        }
        
        /* Custom metric container */
        div.custom-metric {
            background-color: white;
            border: 1px solid #e6e6e6;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Custom metric label */
        .metric-label {
            color: #666;
            font-size: 14px;
            margin-bottom: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

def load_google_sheet(sheet_url):
    try:
        # Instead of reading from JSON file, use st.secrets
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        
        client = gspread.authorize(credentials)
        
        # Extract sheet ID from URL
        sheet_id = sheet_url.split('/')[5]
        sheet = client.open_by_key(sheet_id).sheet1
        
        # Get all values and convert to DataFrame
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        st.error(f"Error loading sheet: {str(e)}")
        return None

def load_csv_file(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None

def analyze_campaign(df):
    try:
        # Initialize variables
        summary = ""
        recommendations = []
        monthly_performance = None
        best_month = None
        worst_month = None
        campaign_performance = None
        
        # Get currency at the beginning
        currency = df['Currency code'].iloc[0] if 'Currency code' in df.columns else 'CHF'
        
        # Check for different types of date columns
        date_columns = [col for col in df.columns if any(term in col.lower() for term in ['date', 'month', 'year'])]
        
        if date_columns:
            # Let user select which date column to use
            selected_date_col = st.selectbox(
                "Select the date column to use for analysis:",
                date_columns,
                index=0
            )
            
            # Convert the selected column to datetime
            try:
                if 'year' in selected_date_col.lower() and 'month' not in selected_date_col.lower():
                    # Handle year only
                    df['Analysis_Date'] = pd.to_datetime(df[selected_date_col].astype(str) + '-01-01')
                elif 'month' in selected_date_col.lower() and 'year' not in selected_date_col.lower():
                    # Handle month only (assume current year)
                    current_year = datetime.now().year
                    df['Analysis_Date'] = pd.to_datetime(str(current_year) + '-' + df[selected_date_col].astype(str) + '-01')
                else:
                    # Handle full dates or year-month combinations
                    df['Analysis_Date'] = pd.to_datetime(df[selected_date_col], errors='coerce')
                
                # Get date range for selection
                min_date = df['Analysis_Date'].min()
                max_date = df['Analysis_Date'].max()
                
                # Display time frame information
                st.subheader("📅 Available Data Time Frame")
                
                # Adjust display format based on the type of date column
                if 'year' in selected_date_col.lower() and 'month' not in selected_date_col.lower():
                    st.write(f"Data ranges from Year {min_date.year} to Year {max_date.year}")
                elif 'month' in selected_date_col.lower():
                    st.write(f"Data ranges from {min_date.strftime('%B %Y')} to {max_date.strftime('%B %Y')}")
                else:
                    st.write(f"Data ranges from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
                
                # Add date range selector
                st.subheader("Select Analysis Period")
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "Start Date",
                        min_value=min_date.date(),
                        max_value=max_date.date(),
                        value=min_date.date()
                    )
                with col2:
                    end_date = st.date_input(
                        "End Date",
                        min_value=min_date.date(),
                        max_value=max_date.date(),
                        value=max_date.date()
                    )
                
                # Filter dataframe based on selected dates
                mask = (df['Analysis_Date'].dt.date >= start_date) & (df['Analysis_Date'].dt.date <= end_date)
                df = df.loc[mask]
                
                # Show selected period
                if 'year' in selected_date_col.lower() and 'month' not in selected_date_col.lower():
                    period_text = f"Years {start_date.year} to {end_date.year}"
                elif 'month' in selected_date_col.lower():
                    period_text = f"{start_date.strftime('%B %Y')} to {end_date.strftime('%B %Y')}"
                else:
                    period_text = f"{start_date} to {end_date} ({(end_date - start_date).days + 1} days)"
                
                st.info(f"Analyzing data from {period_text}")
                
                if df.empty:
                    st.warning("No data available for the selected date range")
                    return None, None
                
            except Exception as e:
                st.error(f"Error processing dates: {str(e)}")
                return None, None
        else:
            st.warning("No date-related columns found. The analysis will proceed without time-based filtering.")
        
        # Initialize session state for selections if they don't exist
        if 'selections_made' not in st.session_state:
            st.session_state.selections_made = False
            
        if 'cost_col' not in st.session_state:
            st.session_state.cost_col = 'Cost' if 'Cost' in df.columns else df.columns[0]
        if 'conv_col' not in st.session_state:
            st.session_state.conv_col = 'All conv.' if 'All conv.' in df.columns else 'Conversions' if 'Conversions' in df.columns else df.columns[0]
        if 'clicks_col' not in st.session_state:
            st.session_state.clicks_col = 'Clicks' if 'Clicks' in df.columns else df.columns[0]
        if 'imp_col' not in st.session_state:
            st.session_state.imp_col = 'Impressions' if 'Impressions' in df.columns else df.columns[0]
            
        # Column selection section
        st.subheader("1. Map Your Data Columns")
        col1, col2 = st.columns(2)
        
        with col1:
            cost_col = st.selectbox(
                "Column for Cost:",
                df.columns,
                index=df.columns.get_loc(st.session_state.cost_col)
            )
            conversions_col = st.selectbox(
                "Column for Conversions:",
                df.columns,
                index=df.columns.get_loc(st.session_state.conv_col)
            )
        
        with col2:
            clicks_col = st.selectbox(
                "Column for Clicks:",
                df.columns,
                index=df.columns.get_loc(st.session_state.clicks_col)
            )
            impressions_col = st.selectbox(
                "Column for Impressions:",
                df.columns,
                index=df.columns.get_loc(st.session_state.imp_col)
            )

        # Update session state with selected values
        st.session_state.cost_col = cost_col
        st.session_state.conv_col = conversions_col
        st.session_state.clicks_col = clicks_col
        st.session_state.imp_col = impressions_col

        # Add a button to confirm selections
        if st.button("Confirm Selections and Analyze"):
            st.session_state.selections_made = True

        # Only proceed with analysis if selections are confirmed
        if st.session_state.selections_made:
            try:
                # Data preprocessing
                df[cost_col] = pd.to_numeric(df[cost_col], errors='coerce').fillna(0)
                df[conversions_col] = pd.to_numeric(df[conversions_col], errors='coerce').fillna(0)
                df[clicks_col] = pd.to_numeric(df[clicks_col], errors='coerce').fillna(0)
                df[impressions_col] = pd.to_numeric(df[impressions_col], errors='coerce').fillna(0)
                
                # Add conversion value check
                conv_value_cols = [col for col in df.columns if any(term in col.lower() for term in ['conv. value', 'conversion value', 'conv value'])]
                conv_value_col = None
                if conv_value_cols:
                    conv_value_col = conv_value_cols[0]
                    
                    # Calculate ROAS if conversion value exists
                    df['ROAS'] = df[conv_value_col] / df['Cost'] * 100 if conv_value_col else 0
                    
                    # Add conversion value metrics to display
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        display_metric("Total Spend", f"{df[cost_col].sum():,.2f}", prefix=f"{currency} ")
                    
                    with col2:
                        display_metric("Total Conv. Value", f"{df[conv_value_col].sum():,.2f}", prefix=f"{currency} ")
                    
                    with col3:
                        display_metric("ROAS", f"{(df[conv_value_col].sum() / df[cost_col].sum() * 100):.2f}", suffix="%")
                    
                    with col4:
                        display_metric("Avg. Conv. Value", f"{(df[conv_value_col].sum() / df[conversions_col].sum()):.2f}", prefix=f"{currency} ")
                    
                    with col5:
                        display_metric("Total Conversions", f"{int(df[conversions_col].sum())}")
                
                # Campaign level analysis first (required for summary)
                campaign_performance = df.groupby('Campaign').agg({
                    cost_col: 'sum',
                    conversions_col: 'sum',
                    clicks_col: 'sum',
                    impressions_col: 'sum'
                }).reset_index()
                
                campaign_performance['CTR'] = (campaign_performance[clicks_col] / campaign_performance[impressions_col] * 100)
                campaign_performance['CPA'] = campaign_performance[cost_col] / campaign_performance[conversions_col].replace(0, np.inf)
                campaign_performance['Conv_Rate'] = (campaign_performance[conversions_col] / campaign_performance[clicks_col] * 100)
                
                # Time-based analysis
                if 'Month' in df.columns:
                    monthly_performance = df.groupby('Month').agg({
                        cost_col: 'sum',
                        conversions_col: 'sum',
                        clicks_col: 'sum',
                        impressions_col: 'sum'
                    }).reset_index()
                    
                    monthly_performance['CTR'] = (monthly_performance[clicks_col] / monthly_performance[impressions_col] * 100)
                    monthly_performance['CPA'] = monthly_performance[cost_col] / monthly_performance[conversions_col].replace(0, np.inf)
                    monthly_performance['Conv_Rate'] = (monthly_performance[conversions_col] / monthly_performance[clicks_col] * 100)
                    
                    best_month = monthly_performance.loc[monthly_performance['Conv_Rate'].idxmax()]
                    worst_month = monthly_performance.loc[monthly_performance['Conv_Rate'].idxmin()]

                    # Visualization section
                    st.subheader("1. Performance Trends Visualization")
                    
                    # Sort monthly_performance by Month
                    monthly_performance['Month'] = pd.to_datetime(monthly_performance['Month'], format='%b-%y')
                    monthly_performance = monthly_performance.sort_values('Month')
                    
                    # Convert back to original format for display
                    monthly_performance['Month'] = monthly_performance['Month'].dt.strftime('%b-%y')
                    
                    # Create trend chart for Clicks and Cost
                    fig1 = px.line(monthly_performance, x='Month', y=[clicks_col, cost_col],
                                 title='Clicks and Cost Trends Over Time',
                                 labels={
                                     'Month': 'Time Period',
                                     'value': 'Count/Cost',
                                     'variable': 'Metric'
                                 })
                    
                    # Update legend labels
                    fig1.update_traces(name='Clicks', selector=dict(name=clicks_col))
                    fig1.update_traces(name='Cost', selector=dict(name=cost_col))
                    
                    # Add secondary y-axis for cost
                    fig1.update_layout(
                        yaxis2=dict(
                            title='Cost',
                            overlaying='y',
                            side='right'
                        ),
                        yaxis_title='Clicks',
                        legend_title='Metrics',
                        height=500
                    )
                    
                    # Show the plot
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Create trend chart for CTR and Conversion Rate
                    fig2 = px.line(monthly_performance, x='Month', 
                                 y=['CTR', 'Conv_Rate'],
                                 title='CTR and Conversion Rate Trends',
                                 labels={
                                     'Month': 'Time Period',
                                     'value': 'Rate (%)',
                                     'variable': 'Metric'
                                 })
                    
                    fig2.update_layout(
                        yaxis_title='Rate (%)',
                        legend_title='Metrics',
                        height=500
                    )
                    
                    # Show the plot
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Add insights about trends
                    st.subheader("Insights")
                    
                    # Filter out inactive months (zero cost months)
                    active_months = monthly_performance[monthly_performance[cost_col] > 0]
                    
                    if not active_months.empty:
                        # Calculate month-over-month changes for active months
                        active_months['Clicks_Change'] = active_months[clicks_col].pct_change() * 100
                        active_months['Cost_Change'] = active_months[cost_col].pct_change() * 100
                        active_months['CTR_Change'] = active_months['CTR'].pct_change() * 100
                        active_months['Conv_Rate_Change'] = active_months['Conv_Rate'].pct_change() * 100
                        
                        latest_month = active_months.iloc[-1]
                        previous_month = active_months.iloc[-2] if len(active_months) > 1 else None
                        
                        # Performance Analysis Section
                        st.write("### 📊 Campaign Performance Analysis")
                        
                        # Current Performance
                        st.write(f"""
                        **Current Month Performance ({latest_month['Month']}):**
                        - Clicks: {latest_month[clicks_col]:.0f}
                        - Cost: {currency} {latest_month[cost_col]:.2f}
                        - CTR: {latest_month['CTR']:.2f}%
                        - Conversion Rate: {latest_month['Conv_Rate']:.2f}%
                        """)
                        
                        # Trend Analysis
                        if previous_month is not None:
                            st.write("###")
                            
                            # Performance Changes
                            changes = {
                                'Clicks': latest_month['Clicks_Change'],
                                'Cost': latest_month['Cost_Change'],
                                'CTR': latest_month['CTR_Change'],
                                'Conv_Rate': latest_month['Conv_Rate_Change']
                            }
                            
                            # Analyze significant changes and provide possible explanations
                            insights = []
                            
                            # CTR Analysis
                            if abs(changes['CTR']) > 20:  # Significant CTR change
                                if changes['CTR'] > 0:
                                    insights.append(f"📈 CTR increased by {changes['CTR']:.1f}%. Possible factors:\n"
                                                 "- Improved ad relevance or copy\n"
                                                 "- Better keyword targeting\n"
                                                 "- Seasonal interest increase")
                                else:
                                    insights.append(f"📉 CTR decreased by {abs(changes['CTR']):.1f}%. Consider:\n"
                                                 "- Reviewing ad copy freshness\n"
                                                 "- Checking for increased competition\n"
                                                 "- Analyzing search term relevance")
                            
                            # Cost Efficiency Analysis
                            if changes['Cost'] > changes['Clicks']:
                                insights.append("💡 Cost is growing faster than clicks. Consider:\n"
                                             "- Reviewing bid strategies\n"
                                             "- Analyzing competitor activity\n"
                                             "- Checking for seasonal CPC increases")
                            
                            # Conversion Rate Analysis
                            if abs(changes['Conv_Rate']) > 20:
                                if changes['Conv_Rate'] > 0:
                                    insights.append(f"🎯 Conversion rate improved by {changes['Conv_Rate']:.1f}%. Potential factors:\n"
                                                 "- Better landing page performance\n"
                                                 "- More qualified traffic\n"
                                                 "- Improved offer/messaging alignment")
                                else:
                                    insights.append(f"⚠️ Conversion rate dropped by {abs(changes['Conv_Rate']):.1f}%. Consider:\n"
                                                 "- Checking landing page performance\n"
                                                 "- Reviewing targeting settings\n"
                                                 "- Analyzing user journey for friction points")
                            
                            # Seasonal Patterns
                            st.write("### 🗓️ Seasonal Patterns")
                            best_performing_month = active_months.loc[active_months['Conv_Rate'].idxmax()]
                            st.write(f"""
                            Best performing month was {best_performing_month['Month']} with:
                            - Conversion Rate: {best_performing_month['Conv_Rate']:.2f}%
                            - CTR: {best_performing_month['CTR']:.2f}%
                            - Cost per Click: {currency} {(best_performing_month[cost_col]/best_performing_month[clicks_col]):.2f}
                            
                            This could indicate:
                            - Optimal seasonal timing for your offerings
                            - Strong market demand period
                            - Effective campaign optimizations
                            """)
                            
                            # Display all insights
                            st.write("### 🔍 Key Insights")
                            for insight in insights:
                                st.write(insight)
                            
                            # Recommendations based on analysis
                            st.write("### 🎯 Strategic Recommendations")
                            recommendations = []
                            
                            # Budget Recommendations
                            if best_performing_month['Conv_Rate'] > latest_month['Conv_Rate'] * 1.2:
                                recommendations.append(f"Consider increasing budget allocation for {best_performing_month['Month']} seasonal period")
                            
                            # Performance Optimization
                            if latest_month['CTR'] < active_months['CTR'].mean():
                                recommendations.append("Review and refresh ad copy to improve CTR")
                            
                            # Cost Efficiency
                            if latest_month['CPA'] > active_months['CPA'].mean():
                                recommendations.append("Optimize targeting and bidding strategies to improve cost efficiency")
                            
                            for rec in recommendations:
                                st.write(f"- {rec}")
                    else:
                        st.write("No active campaigns found in the data (all costs are zero)")
                
                # Generate comprehensive summary
                st.subheader("2. Campaign Performance Analysis")
                
                summary = f"""
                📊 Campaign Performance Deep Dive:

                💰 Overall Performance:
                - Total Spend: {currency} {df[cost_col].sum():,.2f}
                - Total Conversions: {int(df[conversions_col].sum())}
                - Overall CTR: {(df[clicks_col].sum() / df[impressions_col].sum() * 100):.2f}%
                - Average CPA: {currency} {(df[cost_col].sum() / df[conversions_col].sum()):.2f}
                
                🏆 Top Performing Campaign:
                - {campaign_performance.loc[campaign_performance['Conv_Rate'].idxmax()]['Campaign']}
                - Conversion Rate: {campaign_performance['Conv_Rate'].max():.2f}%
                """
                
                if 'Month' in df.columns and best_month is not None and worst_month is not None:
                    summary += f"""
                    📈 Monthly Trend:
                    - Best Month: {best_month['Month']} (Conv. Rate: {best_month['Conv_Rate']:.2f}%)
                    - Worst Month: {worst_month['Month']} (Conv. Rate: {worst_month['Conv_Rate']:.2f}%)
                    """
                
                st.write(summary)
                
                # Generate recommendations
                st.subheader("3. Recommendations")
                
                # Budget recommendations
                top_campaign = campaign_performance.loc[campaign_performance['Conv_Rate'].idxmax()]
                recommendations.append(f"💰 Budget Optimization: Increase budget allocation to '{top_campaign['Campaign']}' which shows the highest conversion rate of {top_campaign['Conv_Rate']:.2f}%")
                
                # Performance recommendations
                low_performing = campaign_performance[campaign_performance['CTR'] < 1]
                if not low_performing.empty:
                    recommendations.append(f"📉 Performance Alert: Campaigns {', '.join(low_performing['Campaign'].tolist())} have CTR below 1%. Review ad copy and targeting.")
                
                # Display recommendations
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
                
                # Show detailed performance tables
                if st.checkbox("Show Detailed Performance Data"):
                    st.subheader("Campaign Performance Details")
                    st.dataframe(campaign_performance.sort_values('Conv_Rate', ascending=False))
                    
                    if monthly_performance is not None:
                        st.subheader("Monthly Performance Trends")
                        st.dataframe(monthly_performance.sort_values('Month'))
                        
                # Add Ad Group Analysis Section
                st.write("---")
                st.subheader("4. Ad Group Performance Analysis 📊")
                
                ad_group_data, ad_group_insights = analyze_ad_groups(df)
                
                if isinstance(ad_group_insights, dict):
                    # Overview metrics
                    st.write(f"### Ad Group Overview")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        display_metric(
                            "Total Ad Groups",
                            ad_group_insights['total_ad_groups']
                        )
                    with col2:
                        display_metric(
                            "Average Conversion Rate",
                            f"{ad_group_insights['avg_conv_rate']:.2f}%"
                        )
                    with col3:
                        display_metric(
                            "Average CPA",
                            f"{currency} {ad_group_insights['avg_cpa']:.2f}"
                        )
                    
                    # Top performing ad groups
                    st.write("### 🏆 Top Performing Ad Groups")
                    st.dataframe(
                        ad_group_insights['top_performers'][[
                            'Campaign', 'Ad Group', 'Conversions', 'Conv_Rate', 'CPA', 'Cost'
                        ]].style.format({
                            'Conv_Rate': '{:.2f}%',
                            'CPA': '{:.2f}',
                            'Cost': '{:.2f}'
                        })
                    )
                    
                    # Create performance matrix visualization
                    fig = px.scatter(
                        ad_group_data,
                        x='Conv_Rate',
                        y='CPA',
                        size='Clicks',
                        color='Campaign',
                        hover_data=['Ad Group', 'CTR', 'Conversions'],
                        title='Ad Group Performance Matrix'
                    )
                    
                    # Apply styling
                    fig = create_styled_plot(fig)
                    fig.update_layout(
                        xaxis_title="Conversion Rate (%)",
                        yaxis_title=f"Cost per Acquisition ({currency})"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Ad Group Recommendations
                    st.write("### 📈 Ad Group Optimization Recommendations")
                    
                    # High CPA Ad Groups
                    high_cpa_groups = ad_group_data[
                        ad_group_data['CPA'] > ad_group_data['CPA'].mean() * 1.5
                    ]
                    if not high_cpa_groups.empty:
                        st.write("#### High CPA Ad Groups to Optimize:")
                        st.dataframe(
                            high_cpa_groups[[
                                'Campaign', 'Ad Group', 'CPA', 'Conv_Rate', 'Cost'
                            ]].head()
                        )
                    
                    # Low CTR Ad Groups
                    low_ctr_groups = ad_group_data[
                        ad_group_data['CTR'] < ad_group_data['CTR'].mean() * 0.5
                    ]
                    if not low_ctr_groups.empty:
                        st.write("#### Low CTR Ad Groups to Review:")
                        st.dataframe(
                            low_ctr_groups[[
                                'Campaign', 'Ad Group', 'CTR', 'Impressions', 'Clicks'
                            ]].head()
                        )
                    
                    # Add optimization recommendations
                    st.write("### 🎯 Action Items")
                    recommendations = []
                    
                    # High performing recommendations
                    top_conv_groups = ad_group_data[
                        ad_group_data['Conv_Rate'] > ad_group_data['Conv_Rate'].mean() * 1.2
                    ]
                    if not top_conv_groups.empty:
                        recommendations.append(
                            f"✅ Increase budget for top converting ad groups: "
                            f"{', '.join(top_conv_groups['Ad Group'].head().tolist())}"
                        )
                    
                    # Poor performing recommendations
                    if not high_cpa_groups.empty:
                        recommendations.append(
                            f"⚠️ Review and optimize high CPA ad groups: "
                            f"{', '.join(high_cpa_groups['Ad Group'].head().tolist())}"
                        )
                    
                    for rec in recommendations:
                        st.write(rec)
                else:
                    st.write(ad_group_insights)
                
                # Add Q&A Section
                st.write("---")
                st.subheader("💬 Ask Questions About Your Data")
                st.write("Ask questions about your campaign performance, trends, or metrics.")
                
                user_question = st.text_input("Enter your question:", placeholder="Example: What was the total spend?")
                
                if user_question:
                    with st.spinner("Analyzing your question..."):
                        answer = answer_data_question(df, user_question)
                        st.write("**Answer:**", answer)
                        
                        # Add follow-up suggestions
                        st.write("**You might also want to ask:**")
                        suggestions = [
                            "What was the best performing campaign?",
                            "What's the average cost per conversion?",
                            "What's the conversion trend?",
                            "Which day had the highest conversions?",
                            "What's the total number of conversions?"
                        ]
                        for suggestion in suggestions:
                            if st.button(suggestion, key=suggestion):
                                answer = answer_data_question(df, suggestion)
                                st.write("**Answer:**", answer)
                
            except Exception as e:
                st.error(f"Error in analysis calculations: {str(e)}")
                return "Error in calculations", ["An error occurred during analysis"]

        return summary, recommendations
            
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return "Error in analysis", ["An unexpected error occurred"]

def analyze_keywords(df):
    """Analyzes keyword performance and returns insights"""
    try:
        # Check if keyword data exists
        keyword_cols = [col for col in df.columns if any(term in col.lower() for term in ['keyword', 'search term'])]
        if not keyword_cols:
            return None, "No keyword data found in the dataset."
        
        # Let user select which keyword column to use
        keyword_col = st.selectbox(
            "Select the keyword column to analyze:",
            keyword_cols,
            index=0
        )
        
        # Add conversion value to aggregation if it exists
        conv_value_cols = [col for col in df.columns if any(term in col.lower() for term in ['conv. value', 'conversion value', 'conv value'])]
        agg_dict = {
            'Cost': 'sum',
            'Clicks': 'sum',
            'Impressions': 'sum',
            'Conversions': 'sum'
        }
        
        if conv_value_cols:
            conv_value_col = conv_value_cols[0]
            agg_dict[conv_value_col] = 'sum'
            
        keyword_performance = df.groupby(keyword_col).agg(agg_dict).reset_index()
        
        # Add ROAS calculation if conversion value exists
        if conv_value_cols:
            keyword_performance['ROAS'] = (keyword_performance[conv_value_col] / keyword_performance['Cost'] * 100)
            keyword_performance['Value_per_Conv'] = keyword_performance[conv_value_col] / keyword_performance['Conversions']
        
        # Calculate metrics
        keyword_performance['CTR'] = (keyword_performance['Clicks'] / keyword_performance['Impressions'] * 100)
        keyword_performance['Conv_Rate'] = (keyword_performance['Conversions'] / keyword_performance['Clicks'] * 100)
        keyword_performance['CPC'] = keyword_performance['Cost'] / keyword_performance['Clicks']
        keyword_performance['CPA'] = keyword_performance['Cost'] / keyword_performance['Conversions'].replace(0, np.inf)
        
        # Sort by conversions
        keyword_performance = keyword_performance.sort_values('Conversions', ascending=False)
        
        # Generate insights
        top_keywords = keyword_performance.head(5)
        poor_performers = keyword_performance[
            (keyword_performance['Cost'] > keyword_performance['Cost'].mean()) & 
            (keyword_performance['Conv_Rate'] < keyword_performance['Conv_Rate'].mean())
        ]
        
        return keyword_performance, {
            'top_keywords': top_keywords,
            'poor_performers': poor_performers,
            'total_keywords': len(keyword_performance)
        }
        
    except Exception as e:
        return None, f"Error analyzing keywords: {str(e)}"

def analyze_ad_groups(df):
    """Analyzes ad group performance and returns insights"""
    try:
        # Check if ad group data exists
        ad_group_cols = [col for col in df.columns if 'ad group' in col.lower()]
        if not ad_group_cols:
            return None, "No Ad Group data found in the dataset."
        
        # Let user select which ad group column to use
        ad_group_col = st.selectbox(
            "Select the Ad Group column to analyze:",
            ad_group_cols,
            index=0
        )
        
        # Add conversion value to aggregation if it exists
        conv_value_cols = [col for col in df.columns if any(term in col.lower() for term in ['conv. value', 'conversion value', 'conv value'])]
        agg_dict = {
            'Cost': 'sum',
            'Clicks': 'sum',
            'Impressions': 'sum',
            'Conversions': 'sum'
        }
        
        if conv_value_cols:
            conv_value_col = conv_value_cols[0]
            agg_dict[conv_value_col] = 'sum'
        
        ad_group_performance = df.groupby([ad_group_col, 'Campaign']).agg(agg_dict).reset_index()
        
        # Add ROAS calculation if conversion value exists
        if conv_value_cols:
            ad_group_performance['ROAS'] = (ad_group_performance[conv_value_col] / ad_group_performance['Cost'] * 100)
            ad_group_performance['Value_per_Conv'] = ad_group_performance[conv_value_col] / ad_group_performance['Conversions']
        
        # Calculate metrics
        ad_group_performance['CTR'] = (ad_group_performance['Clicks'] / ad_group_performance['Impressions'] * 100)
        ad_group_performance['Conv_Rate'] = (ad_group_performance['Conversions'] / ad_group_performance['Clicks'] * 100)
        ad_group_performance['CPC'] = ad_group_performance['Cost'] / ad_group_performance['Clicks']
        ad_group_performance['CPA'] = ad_group_performance['Cost'] / ad_group_performance['Conversions'].replace(0, np.inf)
        
        # Sort by conversions
        ad_group_performance = ad_group_performance.sort_values('Conversions', ascending=False)
        
        # Generate insights
        insights = {
            'top_performers': ad_group_performance.head(5),
            'poor_performers': ad_group_performance[
                (ad_group_performance['Cost'] > ad_group_performance['Cost'].mean()) & 
                (ad_group_performance['Conv_Rate'] < ad_group_performance['Conv_Rate'].mean())
            ],
            'total_ad_groups': len(ad_group_performance),
            'avg_conv_rate': ad_group_performance['Conv_Rate'].mean(),
            'avg_cpa': ad_group_performance['CPA'].mean(),
            'campaign_distribution': ad_group_performance.groupby('Campaign')[ad_group_col].count()
        }
        
        return ad_group_performance, insights
        
    except Exception as e:
        return None, f"Error analyzing ad groups: {str(e)}"

def answer_data_question(df, question):
    """
    Provides answers to questions about the data based on the analysis
    """
    try:
        # Add conversion value handling
        conv_value_cols = [col for col in df.columns if any(term in col.lower() for term in ['conv. value', 'conversion value', 'conv value'])]
        conv_value_col = conv_value_cols[0] if conv_value_cols else None
        
        # Basic data stats
        total_spend = df['Cost'].sum() if 'Cost' in df.columns else 0
        total_conversions = df['Conversions'].sum() if 'Conversions' in df.columns else 0
        total_clicks = df['Clicks'].sum() if 'Clicks' in df.columns else 0
        
        # Convert question to lowercase for easier matching
        question = question.lower()
        
        # Common question patterns
        if 'spend' in question or 'cost' in question:
            return f"The total spend is {total_spend:,.2f}. The average daily spend is {total_spend/len(df):,.2f}."
        
        elif 'conversion' in question:
            conv_rate = (total_conversions/total_clicks*100) if total_clicks > 0 else 0
            return f"There were {total_conversions:,} total conversions with a conversion rate of {conv_rate:.2f}%."
        
        elif 'best' in question or 'top' in question:
            if 'campaign' in question:
                best_campaign = df.groupby('Campaign')['Conversions'].sum().sort_values(ascending=False).index[0]
                return f"The best performing campaign is '{best_campaign}' based on total conversions."
            elif 'day' in question or 'date' in question:
                best_day = df.groupby('Date')['Conversions'].sum().sort_values(ascending=False).index[0]
                return f"The best performing date was {best_day} based on conversions."
        
        elif 'worst' in question or 'poorest' in question:
            if 'campaign' in question:
                worst_campaign = df.groupby('Campaign')['Conversions'].sum().sort_values().index[0]
                return f"The poorest performing campaign is '{worst_campaign}' based on total conversions."
        
        elif 'average' in question or 'avg' in question:
            if 'cpc' in question or 'cost per click' in question:
                avg_cpc = total_spend/total_clicks if total_clicks > 0 else 0
                return f"The average cost per click is {avg_cpc:.2f}."
            elif 'cpa' in question or 'cost per acquisition' in question:
                avg_cpa = total_spend/total_conversions if total_conversions > 0 else 0
                return f"The average cost per acquisition is {avg_cpa:.2f}."
        
        elif 'trend' in question:
            if 'conversion' in question:
                recent_trend = "increasing" if df['Conversions'].tail().is_monotonic_increasing else "decreasing" if df['Conversions'].tail().is_monotonic_decreasing else "fluctuating"
                return f"The conversion trend is {recent_trend} in the most recent period."
        
        elif 'keyword' in question:
            if 'best' in question or 'top' in question:
                best_keyword = df.groupby('Keyword')['Conversions'].sum().sort_values(ascending=False).index[0]
                conv_count = df[df['Keyword'] == best_keyword]['Conversions'].sum()
                return f"The best performing keyword is '{best_keyword}' with {int(conv_count)} conversions."
            elif 'worst' in question or 'poor' in question:
                # Find keywords with high cost but low conversions
                keyword_metrics = df.groupby('Keyword').agg({
                    'Cost': 'sum',
                    'Conversions': 'sum'
                }).reset_index()
                keyword_metrics['CPA'] = keyword_metrics['Cost'] / keyword_metrics['Conversions'].replace(0, np.inf)
                worst_keyword = keyword_metrics.nlargest(1, 'CPA')['Keyword'].iloc[0]
                return f"The poorest performing keyword is '{worst_keyword}' based on cost per acquisition."
            elif 'expensive' in question or 'highest cpc' in question:
                keyword_metrics = df.groupby('Keyword').agg({
                    'Cost': 'sum',
                    'Clicks': 'sum'
                }).reset_index()
                keyword_metrics['CPC'] = keyword_metrics['Cost'] / keyword_metrics['Clicks']
                expensive_keyword = keyword_metrics.nlargest(1, 'CPC')['Keyword'].iloc[0]
                cpc = keyword_metrics.nlargest(1, 'CPC')['CPC'].iloc[0]
                return f"The most expensive keyword is '{expensive_keyword}' with a CPC of {cpc:.2f}."
        
        elif 'ad group' in question:
            if 'best' in question or 'top' in question:
                best_adgroup = df.groupby('Ad Group')['Conversions'].sum().sort_values(ascending=False).index[0]
                conv_count = df[df['Ad Group'] == best_adgroup]['Conversions'].sum()
                return f"The best performing ad group is '{best_adgroup}' with {int(conv_count)} conversions."
            elif 'worst' in question or 'poor' in question:
                adgroup_metrics = df.groupby('Ad Group').agg({
                    'Cost': 'sum',
                    'Conversions': 'sum'
                }).reset_index()
                adgroup_metrics['CPA'] = adgroup_metrics['Cost'] / adgroup_metrics['Conversions'].replace(0, np.inf)
                worst_adgroup = adgroup_metrics.nlargest(1, 'CPA')['Ad Group'].iloc[0]
                return f"The poorest performing ad group is '{worst_adgroup}' based on cost per acquisition."
            elif 'expensive' in question:
                adgroup_metrics = df.groupby('Ad Group').agg({
                    'Cost': 'sum',
                    'Clicks': 'sum'
                }).reset_index()
                adgroup_metrics['CPC'] = adgroup_metrics['Cost'] / adgroup_metrics['Clicks']
                expensive_adgroup = adgroup_metrics.nlargest(1, 'CPC')['Ad Group'].iloc[0]
                cpc = adgroup_metrics.nlargest(1, 'CPC')['CPC'].iloc[0]
                return f"The most expensive ad group is '{expensive_adgroup}' with a CPC of {cpc:.2f}."
        
        # Add new question patterns
        if 'roas' in question or 'return on ad spend' in question:
            if conv_value_col:
                roas = (df[conv_value_col].sum() / df['Cost'].sum() * 100)
                return f"The overall ROAS is {roas:.2f}%"
            else:
                return "Conversion value data is not available to calculate ROAS."
                
        elif 'conversion value' in question or 'revenue' in question:
            if conv_value_col:
                total_value = df[conv_value_col].sum()
                avg_value = total_value / df['Conversions'].sum()
                return f"Total conversion value is {currency} {total_value:,.2f}, with an average of {currency} {avg_value:.2f} per conversion."
            else:
                return "Conversion value data is not available."
                
        elif 'best' in question or 'top' in question:
            if 'roas' in question and conv_value_col:
                metrics = df.groupby('Campaign').agg({
                    conv_value_col: 'sum',
                    'Cost': 'sum'
                }).reset_index()
                metrics['ROAS'] = metrics[conv_value_col] / metrics['Cost'] * 100
                best_roas_campaign = metrics.nlargest(1, 'ROAS')
                return f"The campaign with highest ROAS is '{best_roas_campaign['Campaign'].iloc[0]}' with {best_roas_campaign['ROAS'].iloc[0]:.2f}% ROAS."
        
        else:
            return "I'm not sure about that. Try asking about spend, conversions, best/worst campaigns, average metrics, or trends."
            
    except Exception as e:
        return f"Sorry, I couldn't analyze that question due to an error: {str(e)}"

def display_metric(label, value, prefix="", suffix=""):
    st.markdown(f"""
        <div class='custom-metric'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{prefix}{value}{suffix}</div>
        </div>
    """, unsafe_allow_html=True)

def create_styled_plot(fig):
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='black',
        title_font_color='#1032cf',
        legend_title_font_color='#1032cf',
        legend_bgcolor='rgba(255,255,255,0.8)',
        xaxis=dict(
            gridcolor='#f0f0f0',
            tickfont=dict(color='black')
        ),
        yaxis=dict(
            gridcolor='#f0f0f0',
            tickfont=dict(color='black')
        )
    )
    return fig

def main():
    # Apply custom styling
    apply_custom_style()
    
    # Update the title with custom formatting
    st.markdown("""
        <h1 style='color: #1032cf; margin-bottom: 30px;'>
            📊 Google Ads Campaign Analyzer
        </h1>
    """, unsafe_allow_html=True)
    
    # Update instructions with custom styling
    st.markdown("""
        <div style='background-color: #f0f3ff; padding: 20px; border-radius: 8px; margin-bottom: 30px;'>
            <h3 style='color: #1032cf; margin-top: 0;'>Instructions</h3>
            <ol style='color: black; margin-bottom: 0;'>
                <li>Choose your data source: Google Sheets URL or CSV file upload</li>
                <li>Select your date/month/year column and analysis period</li>
                <li>Map your columns to the required metrics</li>
                <li>The analysis will provide a campaign summary and optimization recommendations</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)
    
    # Add data source selection
    data_source = st.radio(
        "Choose your data source:",
        ["Google Sheets URL", "CSV File Upload"],
        horizontal=True
    )
    
    df = None
    
    if data_source == "Google Sheets URL":
        sheet_url = st.text_input("Enter Google Sheets URL:")
        if sheet_url:
            df = load_google_sheet(sheet_url)
    else:
        uploaded_file = st.file_uploader(
            "Upload your Google Ads CSV file",
            type=['csv'],
            help="Upload a CSV file containing your Google Ads data"
        )
        if uploaded_file is not None:
            df = load_csv_file(uploaded_file)
    
    if df is not None:
        analyze_campaign(df)

if __name__ == "__main__":
    main() 