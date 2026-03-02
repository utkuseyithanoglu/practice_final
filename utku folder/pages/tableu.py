import streamlit as st
st.set_page_config(page_title="NYC EMS Predictor", page_icon="🚑", layout="wide")
st.markdown("## 📊 Demand & Response Intelligence Dashboard")
st.markdown("This dashboard helps forecast emergency demand and response performance across NYC boroughs.")
st.divider()
st.sidebar.header("NYC EMS Call & Response Time Predictor")
tableau_url = "https://public.tableau.com/views/predictcallsbyborough/Dashboard2?:showVizHome=no"

st.components.v1.iframe(tableau_url, height=1200, scrolling=True)
