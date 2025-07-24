import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai
import io
import base64
from typing import Dict, List, Any
import json
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Auto Report Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

print("Running script: AutoReport Generator")

def load_data(self, uploaded_file):
        """Load data from uploaded file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                self.df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                self.df = pd.read_json(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload CSV, Excel, or JSON files.")
                return False
            
            st.success(f"Data loaded successfully! Shape: {self.df.shape}")
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False

def get_data_summary(self) -> Dict[str, Any]:
        """Generate comprehensive data summary"""
        if self.df is None:
            return {}
        
        summary = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "numeric_columns": list(self.df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(self.df.select_dtypes(include=['object']).columns),
            "memory_usage": self.df.memory_usage(deep=True).sum(),
            "duplicate_rows": self.df.duplicated().sum()
        }
        
        # Add basic statistics for numeric columns
        if summary["numeric_columns"]:
            summary["numeric_stats"] = self.df[summary["numeric_columns"]].describe().to_dict()
        
        return summary

def perform_eda(self, user_prompt: str = None) -> Dict[str, Any]:
        """Perform comprehensive EDA based on user prompt or default analysis"""
        if self.df is None:
            return {}
        
        eda_results = {}
        
        # Basic data info
        eda_results["data_summary"] = self.get_data_summary()
        
        # Correlation analysis for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            eda_results["correlation_matrix"] = self.df[numeric_cols].corr().to_dict()
        
        # Distribution analysis
        eda_results["distributions"] = {}
        for col in numeric_cols:
            eda_results["distributions"][col] = {
                "mean": float(self.df[col].mean()),
                "median": float(self.df[col].median()),
                "std": float(self.df[col].std()),
                "skewness": float(self.df[col].skew()),
                "kurtosis": float(self.df[col].kurtosis())
            }
        
        # Categorical analysis
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        eda_results["categorical_analysis"] = {}
        for col in categorical_cols:
            value_counts = self.df[col].value_counts().head(10)
            eda_results["categorical_analysis"][col] = {
                "unique_values": int(self.df[col].nunique()),
                "top_values": value_counts.to_dict(),
                "most_frequent": str(value_counts.index[0]) if len(value_counts) > 0 else None
            }
        
        self.analysis_results = eda_results
        return eda_results

def generate_visualizations(self):
        """Generate various visualizations based on the data"""
        if self.df is None:
            return []
        
        figures = []
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        # 1. Correlation heatmap
        if len(numeric_cols) > 1:
            fig_corr = px.imshow(
                self.df[numeric_cols].corr(),
                text_auto=True,
                aspect="auto",
                title="Correlation Matrix",
                color_continuous_scale="RdBu"
            )
            figures.append(("Correlation Matrix", fig_corr))
        
        # 2. Distribution plots for numeric columns
        if len(numeric_cols) > 0:
            for col in numeric_cols[:4]:  # Limit to first 4 columns
                fig_dist = px.histogram(
                    self.df, 
                    x=col, 
                    title=f"Distribution of {col}",
                    marginal="box"
                )
                figures.append((f"Distribution of {col}", fig_dist))
        
        # 3. Box plots for numeric columns
        if len(numeric_cols) > 1:
            fig_box = go.Figure()
            for col in numeric_cols[:5]:  # Limit to first 5 columns
                fig_box.add_trace(go.Box(y=self.df[col], name=col))
            fig_box.update_layout(title="Box Plots Comparison")
            figures.append(("Box Plots Comparison", fig_box))
        
        # 4. Bar plots for categorical columns
        for col in categorical_cols[:3]:  # Limit to first 3 columns
            value_counts = self.df[col].value_counts().head(10)
            fig_bar = px.bar(
                x=value_counts.index, 
                y=value_counts.values,
                title=f"Top 10 Values in {col}",
                labels={'x': col, 'y': 'Count'}
            )
            figures.append((f"Top Values in {col}", fig_bar))
        
        # 5. Scatter plot matrix (if enough numeric columns)
        if len(numeric_cols) >= 2:
            sample_size = min(1000, len(self.df))  # Sample for performance
            df_sample = self.df.sample(n=sample_size) if len(self.df) > sample_size else self.df
            fig_scatter = px.scatter_matrix(
                df_sample[numeric_cols[:4]], 
                title="Scatter Plot Matrix"
            )
            figures.append(("Scatter Plot Matrix", fig_scatter))
        
        return figures

def get_ai_insights(self, user_prompt: str = None) -> str:
        """Get AI-powered insights from OpenAI"""
        if not self.analysis_results:
            return "No analysis results available. Please perform EDA first."
        
        # Prepare data summary for AI
        data_summary = json.dumps(self.analysis_results, indent=2, default=str)
        
        default_prompt = """
        As a data analyst, provide comprehensive insights about this dataset. 
        Focus on:
        1. Key patterns and trends
        2. Notable correlations
        3. Data quality issues
        4. Anomalies or outliers
        5. Business recommendations
        """
        
        prompt = user_prompt if user_prompt else default_prompt
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert data analyst. Analyze the provided dataset summary and provide actionable insights."
                    },
                    {
                        "role": "user", 
                        "content": f"{prompt}\n\nDataset Summary:\n{data_summary}"
                    }
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating AI insights: {str(e)}"

def generate_report(self, user_prompt: str = None) -> str:
        """Generate a comprehensive report"""
        if self.df is None:
            return "No data loaded. Please upload a dataset first."
        
        # Perform EDA
        self.perform_eda(user_prompt)
        
        # Get AI insights
        insights = self.get_ai_insights(user_prompt)
        
        # Create report structure
        report = f"""
        # Auto-Generated Data Analysis Report
        
        ## Dataset Overview
        - **Shape**: {self.analysis_results['data_summary']['shape']}
        - **Columns**: {len(self.analysis_results['data_summary']['columns'])}
        - **Missing Values**: {sum(self.analysis_results['data_summary']['missing_values'].values())}
        - **Duplicate Rows**: {self.analysis_results['data_summary']['duplicate_rows']}
        
        ## AI-Powered Insights
        {insights}
        
        ## Technical Summary
        - **Numeric Columns**: {len(self.analysis_results['data_summary']['numeric_columns'])}
        - **Categorical Columns**: {len(self.analysis_results['data_summary']['categorical_columns'])}
        - **Memory Usage**: {self.analysis_results['data_summary']['memory_usage'] / 1024 / 1024:.2f} MB
        """
        
        return report

def main():
    st.title("📊 Auto Report Generator")
    st.markdown("Upload your dataset and get AI-powered insights with automated EDA!")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # OpenAI API Key input
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
    
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to proceed.")
        return
    
    # Initialize the generator
    generator = AutoReportGenerator(api_key)
    
    # File upload
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file", 
        type=['csv', 'xlsx', 'xls', 'json']
    )
    
    if uploaded_file is not None:
        if generator.load_data(uploaded_file):
            # Display basic info about the dataset
            st.subheader("Dataset Preview")
            st.dataframe(generator.df.head())
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", generator.df.shape[0])
            with col2:
                st.metric("Columns", generator.df.shape[1])
            with col3:
                st.metric("Missing Values", generator.df.isnull().sum().sum())
            with col4:
                st.metric("Memory Usage", f"{generator.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            # Custom prompt input
            st.subheader("Custom Analysis Prompt")
            user_prompt = st.text_area(
                "Enter your specific analysis requirements (optional):",
                placeholder="e.g., Focus on sales trends, identify customer segments, analyze seasonal patterns..."
            )
            
            # Generate report button
            if st.button("🚀 Generate Auto Report", type="primary"):
                with st.spinner("Performing EDA and generating insights..."):
                    # Generate comprehensive report
                    report = generator.generate_report(user_prompt if user_prompt else None)
                    
                    # Display report
                    st.markdown(report)
                    
                    # Generate and display visualizations
                    st.subheader("📈 Visualizations")
                    figures = generator.generate_visualizations()
                    
                    for title, fig in figures:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display raw analysis results
                    with st.expander("🔍 Detailed Analysis Results"):
                        st.json(generator.analysis_results)
    
    # Chatbot interface
    st.subheader("💬 Ask Questions About Your Data")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your dataset..."):
        if uploaded_file is None:
            st.error("Please upload a dataset first!")
        else:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    response = generator.get_ai_insights(prompt)
                    st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
