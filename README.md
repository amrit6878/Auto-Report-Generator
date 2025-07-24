# Auto-Report-Generator
Created a python based chatbot which provides the insights of EDA using openAI's API intergtation

# 📊 Auto Report Generator with GenAI

A Streamlit-powered app that:
- Accepts dataset via upload or file path
- Performs automatic EDA with pandas + visualizations
- Uses OpenAI (GPT) to write executive-level summaries
- Offers a chatbot interface to ask data-specific questions

## 🔧 Features
- Supports `.csv`, `.xlsx`, `.json`
- Chat-like assistant for data Q&A
- Auto PDF report export (WIP)
- Visuals: correlation heatmap, histograms, boxplots, bar charts

## 📦 Setup

```bash
pip install -r requirements.txt
streamlit run app.py

🧠 Powered By
	•	OpenAI GPT-3.5 / GPT-4
	•	Streamlit
	•	Pandas, Seaborn, Plotly
