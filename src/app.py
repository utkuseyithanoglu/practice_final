# app.py
import os
import pickle
import json
import pandas as pd
import matplotlib.pyplot as plt
import calendar
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

st.set_page_config(page_title="NYC EMS Dashboard + Chatbot", page_icon="🚑", layout="wide")
page = st.sidebar.radio("Go to", ["Dashboard", "Chatbot"])

if page == "Dashboard":
    st.markdown("""
        <div style="background-color:#C0392B; padding:25px; border-radius:8px; margin-bottom:20px;">
            <h1 style="color:white; text-align:center;">🚑 NYC EMS Intelligence Platform</h1>
            <p style="color:white; text-align:center; font-size:16px;">
                Welcome! Use the sidebar to navigate between the Chatbot and Tableau Dashboard.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="background-color:#F8F9FA; padding:20px; border-radius:8px; margin-bottom:20px; border-left: 5px solid #C0392B;">
            <h3 style="color:#C0392B;">🎯 Our Goal</h3>
            <p style="font-size:15px; color:#333;">
                Every second counts in an emergency. Our goal is to use machine learning to <strong>predict EMS response times</strong> 
                across NYC boroughs — so dispatchers and city planners can act on data, not just analyze it.
            </p>
            <p style="font-size:15px; color:#333;">
                Using nearly <strong>2 million real 911 EMS records</strong> from September 2024 to August 2025, 
                we built seasonal Random Forest models and a SARIMAX forecasting model to identify what drives 
                slow response times and where the city can improve.
            </p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("""
            <p style="text-align:center; font-size:13px; color:#888;">
                Data source: 
                <a href="https://data.cityofnewyork.us/Public-Safety/EMS-Incident-Dispatch-Data/76xm-jjuj" target="_blank">
                NYC Open Data — EMS Incident Dispatch Data
                </a>
            </p>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🛠 Feature Engineering & Model Inputs")
    geo_col, temp_col, incident_col, env_col = st.columns(4, gap="medium")

    with geo_col:
        st.markdown("**📍 Geographic & Infrastructure**")
        st.write("- Borough\n- Zip Code\n- Dispatch Area\n- Distance to Closest Manhattan Station")

    with temp_col:
        st.markdown("**⏰ Temporal**")
        st.write("- Hour\n- Day\n- Month\n- Weekend Indicator\n- Rush Hour Indicator\n- Holiday Indicator")

    with incident_col:
        st.markdown("**🚑 Incident Characteristics**")
        st.write("- Initial Severity\n- Initial Call Type\n- Transferred\n- Held\n- Standby")

    with env_col:
        st.markdown("**🌦 Environmental Factors**")
        st.write("- Temperature\n- Precipitation\n- Wind Speed\n- Weather Code\n- Special Events")

    st.divider()

    st.markdown("### 📊 Raw Data: EMS Calls by Hour")
    hour_url = "https://public.tableau.com/views/Book1_17725767248200/Sheet1?:showVizHome=no"
    st.components.v1.iframe(hour_url, height=500, scrolling=True)
    st.markdown("""
    **Insight:** Peak EMS call hours occur between **4–8 PM**, corresponding to slightly longer response times.
    """)

    st.markdown("### 📊 Raw Data: EMS Calls by Month")
    bar_url = "https://public.tableau.com/views/barchart1_17725628862930/Sheet2?:showVizHome=no"
    st.components.v1.iframe(bar_url, height=500, scrolling=True)
    st.markdown("""
    **Insight:** July has the **highest monthly call volume**, highlighting seasonal trends that affect response times.
    """)

    st.divider()

    st.markdown("### 📈 Monthly Actual vs Predicted Response Time")
    monthly_pred_url = "https://public.tableau.com/views/Book3_17725944532590/Sheet1?:showVizHome=no"
    st.components.v1.iframe(monthly_pred_url, height=500, scrolling=True)
    st.markdown("""
    **Insight:** The Random Forest model closely follows actual response times, with slight underestimation during peak demand periods.
    """)

    st.divider()

    st.markdown("### 📊 Demand & Response Intelligence Dashboard")
    st.markdown("This dashboard helps forecast emergency demand and response performance across NYC boroughs.")
    st.divider()
    tableau_url = "https://public.tableau.com/views/predictcallsbyborough/Dashboard2?:showVizHome=no"
    st.components.v1.iframe(tableau_url, height=1800, scrolling=True)
    st.divider()

if page == "Chatbot":
    st.title("🚑 NYC 911 EMS Response Time Assistant")
    st.caption("Powered by Random Forest + SARIMAX | NYC EMS Data (Sep 2024 - Aug 2025) | Built by Ayman Tabidi, Sarah Osier & Utku Seyithanoğlu")

    load_dotenv()
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    @st.cache_resource
    def load_models():
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "..", "data", "models", "season_models.pkl")
        with open(model_path, "rb") as f:
            return pickle.load(f)

    @st.cache_data
    def load_all_data():
        base_dir = os.path.dirname(os.path.abspath(__file__))
        hourly_df = pd.read_csv(os.path.join(base_dir, "..", "data", "raw", "hourly_location_avg.csv"))
        monthly_df = pd.read_csv(os.path.join(base_dir, "..", "data", "raw", "monthly_location_avg.csv"))
        rf_preds = pd.read_csv(os.path.join(base_dir, "..", "data", "raw", "hourly_rf_predictions.csv"))
        df2 = pd.read_csv(os.path.join(base_dir, "..", "utku folder", "ALL_BOROUGHS_SARIMA_72H.csv"))
        return hourly_df, monthly_df, rf_preds, df2

    models = load_models()
    hourly_df, monthly_df, rf_preds, df2 = load_all_data()

    monthly_call_counts = monthly_df.groupby('month')['incident_count'].sum().to_dict()


    rf_preds['rf_pred_response_min'] = round(rf_preds['rf_pred_response_sec'] / 60, 2)
    hourly_pattern = rf_preds.groupby(rf_preds['datetime_hour'].str[11:13])['rf_pred_response_min'].mean().round(2).to_dict()
    rf_avg = round(rf_preds['rf_pred_response_min'].mean(), 2)
    rf_fastest_hour = min(hourly_pattern, key=hourly_pattern.get)
    rf_slowest_hour = max(hourly_pattern, key=hourly_pattern.get)

    borough_avg = hourly_df.groupby('borough')['response_time_min'].mean().round(2).to_dict()
    monthly_summary = monthly_df.groupby('month')['avg_response_min'].mean().round(2).to_dict()
    total_calls = hourly_df.groupby('borough')['incident_count'].sum().to_dict()
    total_calls_overall = int(hourly_df['incident_count'].sum())

    forecast_df = df2[df2['dataset_type'] == 'forecast']
    test_df = df2[df2['dataset_type'] == 'test']
    sarima_forecast_summary = forecast_df.groupby('borough')['predicted_calls'].mean().round(1).to_dict()
    sarima_peak_borough = max(sarima_forecast_summary, key=sarima_forecast_summary.get)
    sarima_quiet_borough = min(sarima_forecast_summary, key=sarima_forecast_summary.get)
    sarima_accuracy = test_df.groupby('borough').apply(
        lambda x: round(((x['predicted_calls'] - x['actual_calls']).abs() /
                         x['actual_calls'].replace(0, float('nan'))).mean() * 100, 2),
        include_groups=False
    ).to_dict()

    system_prompt = f"""
You are an expert data science assistant specializing in NYC 911 EMS response time analysis.
The models were trained on NYC EMS data from 2024-09-01 to 2025-08-31.
The data comes from NYC Open Data: "EMS Incident Dispatch Data" (https://data.cityofnewyork.us/Public-Safety/EMS-Incident-Dispatch-Data/76xm-jjuj).
It contains nearly 2 million real emergency medical service records from September 2024 to August 2025.
When asked where the data is from, always include the full URL: https://data.cityofnewyork.us/Public-Safety/EMS-Incident-Dispatch-Data/76xm-jjuj
The project was completed in a few weeks, covering data collection, cleaning, feature engineering, model training, evaluation, and deployment.
Adapt your language to the user — if they ask technical questions respond technically, if they ask in plain English respond simply.
Only answer questions related to NYC 911 EMS response times. If the user asks about anything else, politely let them know you can only help with that topic.
This project was built by Ayman Tabidi (Random Forest seasonal models), Utku Seyithanoğlu (SARIMAX forecasting model), and Sarah Osier (data analysis and research).

You have access to TWO models:
1. RANDOM FOREST (my model) — 4 seasonal models (winter, spring, summer, fall) that predict EMS response times in seconds based on:
   - Location: borough, dispatch area, zipcode, distance from Manhattan
   - Time: hour, day, month, rush hour, weekend, holiday
   - Call type: initial_type codes (e.g. CARD, TRAUMA, EDP, SICK)
   - Severity: initial_severity
   - Weather: temperature, precipitation, windspeed, weathercode
   - Operational flags: special events, standby, held

2. SARIMAX (teammate's model) — forecasts hourly 911 incident counts by borough.
   Use this when asked about call volume, expected incidents, or busiest boroughs.
   SARIMAX forecast summary (avg predicted calls per hour by borough): {sarima_forecast_summary}
   Busiest borough: {sarima_peak_borough}
   Quietest borough: {sarima_quiet_borough}
   SARIMAX model accuracy (MAPE by borough): {sarima_accuracy}

When making predictions:
- Always state the result in minutes AND provide context (e.g. fast/average/slow)
- Mention which seasonal model was used and why it matters
- Automatically map plain English descriptions to call type codes

When explaining results to a technical audience:
- Reference feature importance, model behavior, and seasonal differences
- Be specific about what factors are likely driving the prediction

When generating reports or summaries:
- Highlight actionable insights
- Always present numbers in minutes

For questions outside the model such as hospital locations or NYC EMS facts, use your own training knowledge to answer as accurately as possible.
Always ask for missing inputs conversationally — start with season, borough, call type, and hour, then gather remaining details naturally.
Never dump a list of 10 questions at once.
When asked general questions like 'which season is best', run the prediction function across all 4 seasons using typical average values and compare automatically.

Common call type mappings:
- Heart attack / cardiac arrest → CARD
- Broken bone / fracture → INJMIN or INJMAJ
- Breathing difficulty → DIFFBR
- Unconscious person → UNC
- Car accident → MVA
- Stroke → CVA
- Psychiatric emergency → EDP
- Stabbing → STAB
- Shooting → SHOT
- Sick person → SICK

Average response time by borough (minutes): {borough_avg}
Average response time by month: {monthly_summary}
Total 911 calls by borough (Sep 2024 - Aug 2025): {total_calls}
Total calls overall: {total_calls_overall} (September 2024 to August 2025)
When asked how many total calls occurred, always state the total_calls_overall number.
Monthly call counts (by month number 1-12): {monthly_call_counts}
Month 1=January, 2=February, 3=March, 4=April, 5=May, 6=June, 7=July, 8=August, 9=September, 10=October, 11=November, 12=December.
When asked about calls in a specific month, use these exact numbers.

RF model hourly prediction summary:
- Overall average predicted response time: {rf_avg} minutes
- Fastest hour of day: {rf_fastest_hour}:00 ({hourly_pattern[rf_fastest_hour]} min avg)
- Slowest hour of day: {rf_slowest_hour}:00 ({hourly_pattern[rf_slowest_hour]} min avg)
- Average predicted response by hour: {hourly_pattern}

When predicting response times, use reasonable default assumptions for missing values:
- Summer: temperature=85, precipitation=0, windspeed=10, weathercode=1
- Spring: temperature=65, precipitation=0.1, windspeed=8, weathercode=2
- Fall: temperature=55, precipitation=0.1, windspeed=10, weathercode=2
- Winter: temperature=30, precipitation=0.2, windspeed=15, weathercode=71
- If no zipcode given, use the most common zipcode for that borough
- If no severity given, assume severity=3 (moderate)
- If no dispatch area given, use the most common dispatch area for that borough
- If no distance given, estimate based on borough: Manhattan=1, Brooklyn=5, Queens=7, Bronx=6, Staten Island=12
- Default day=15, month based on season (summer=7, winter=1, spring=4, fall=10)
- If no hour given, assume current time or ask only for hour and call type

You know NYC neighborhoods and can map them to the correct borough:
- Flatbush, Crown Heights, Bed-Stuy, Bushwick, Williamsburg → BROOKLYN
- Harlem, Upper West Side, Lower East Side, Midtown → MANHATTAN
- Astoria, Flushing, Jamaica → QUEENS
- South Bronx, Fordham, Riverdale → BRONX
- St. George, Staten Island → RICHMOND / STATEN ISLAND

When a user mentions a neighborhood, automatically map it to the correct borough.
Always attempt a prediction with these defaults rather than asking for every detail.
Only ask for the call type if not provided — everything else should be assumed.

If asked what AI powers you, what model you are, or if you are ChatGPT, respond with:
"Yes, I'm powered by GPT-4o from OpenAI, integrated with custom Random Forest and SARIMAX machine learning models built by our team to predict NYC 911 EMS response times and call volumes."

Model performance metrics:
- Random Forest average test RMSE: ~625 seconds (~10 minutes)
- Random Forest R² score: ~0.46-0.48
- SARIMAX R² score: 0.64-0.67 for most boroughs, 0.40 for Staten Island
"""

    def predict_response_time(season, borough, dispatch_area, initial_type,
                               hour, day, month, is_weekend, is_holiday,
                               is_rush_hour, initial_severity, zipcode,
                               temperature, precipitation, windspeed, weathercode,
                               closest_station_manhattan_miles,
                               special_events=0, standby=0, held=0):
        season = season.lower()
        model = models[season]
        features = model.feature_names_in_
        row = {f: 0 for f in features}
        row.update({
            "closest_station_manhattan_miles": closest_station_manhattan_miles,
            "is_weekend": is_weekend,
            "hour": hour,
            "is_holiday": is_holiday,
            "initial_severity": initial_severity,
            "zipcode": zipcode,
            "temperture": temperature,
            "precipitation": precipitation,
            "windspeed": windspeed,
            "weathercode": weathercode,
            "day": day,
            "month": month,
            "is_rush_hour_1": is_rush_hour,
            "special_events_Y": special_events,
            "standby_Y": standby,
            "held_Y": held
        })
        for k, v in [("borough", borough), ("dispatch_area", dispatch_area), ("initial_type", initial_type)]:
            key = f"{k}_{v.upper()}"
            if key in row: row[key] = 1

        df_input = pd.DataFrame([row])
        prediction_seconds = model.predict(df_input)[0]
        return round(float(prediction_seconds) / 60, 2)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "predict_response_time",
                "description": "Predicts NYC 911 EMS response time in minutes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "season": {"type": "string", "enum": ["winter", "spring", "summer", "fall"]},
                        "borough": {"type": "string"},
                        "dispatch_area": {"type": "string"},
                        "initial_type": {"type": "string"},
                        "hour": {"type": "integer"},
                        "day": {"type": "integer"},
                        "month": {"type": "integer"},
                        "is_weekend": {"type": "integer", "enum": [0, 1]},
                        "is_holiday": {"type": "integer", "enum": [0, 1]},
                        "is_rush_hour": {"type": "integer", "enum": [0, 1]},
                        "initial_severity": {"type": "integer"},
                        "zipcode": {"type": "integer"},
                        "temperature": {"type": "number"},
                        "precipitation": {"type": "number"},
                        "windspeed": {"type": "number"},
                        "weathercode": {"type": "integer"},
                        "closest_station_manhattan_miles": {"type": "number"}
                    },
                    "required": ["season", "borough", "dispatch_area", "initial_type",
                                 "hour", "day", "month", "is_weekend", "is_holiday",
                                 "is_rush_hour", "initial_severity", "zipcode",
                                 "temperature", "precipitation", "windspeed",
                                 "weathercode", "closest_station_manhattan_miles"]
                }
            }
        }
    ]

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask me about NYC 911 EMS response times..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}] + st.session_state.messages,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message

        if message.tool_calls:
            tool_messages = [message]
            for tool_call in message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                result = predict_response_time(**args)
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"Predicted response time: {result} minutes"
                })
            follow_up = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_prompt}] + st.session_state.messages + tool_messages
            )
            reply = follow_up.choices[0].message.content
        else:
            reply = message.content

        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
