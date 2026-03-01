import streamlit as st

st.set_page_config(page_title="NYC EMS Predictor", page_icon="🚑", layout="wide")
st.markdown("""
<style>
.big-title {font-size: 40px; font-weight: 600; margin-bottom: 0.2rem;}
.sub {font-size: 16px; color: #6b7280; margin-top: 0;}
.card {padding: 18px; border: 1px solid #e5e7eb; border-radius: 14px; background: white;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='big-title'>🚑 NYC EMS Call & Response Time Predictor</div>", unsafe_allow_html=True)
st.markdown("<p class='sub'>Forecasting call volume and response time by borough (9/1/2024 00:00:00 - 8/31/2025 23:00:00).</p>", unsafe_allow_html=True)
st.divider()
st.markdown("""     
            🚑 Project Overview

This project was developed by Ayman Tabidi, Sarah Oasier, and Utku Seyithanoğlu.

We analyzed 2024–2025 New York City EMS data using nearly 2 million real records from Data.gov.

Our machine learning models answer two key questions for each borough:
• How many emergency calls are expected?
• How long will it take for help to arrive?

            🎯 Our Goal
Our goal is to support smarter emergency resource planning by:

• Reducing unnecessary waiting time  
• Saving operational costs  
• Improving response preparedness  

By predicting both call volume and response time, the city can allocate resources more efficiently and make faster, data-driven decisions.

        🚑 Why This Matters
• Better ambulance allocation  
• Reduced waiting times  
• Improved emergency readiness""")



