import os
import pickle
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

# Streamlit UI
st.set_page_config(page_title="NYC 911 EMS Chatbot", page_icon="🚑", layout="centered")

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

# RF summaries
rf_preds['rf_pred_response_min'] = round(rf_preds['rf_pred_response_sec'] / 60, 2)
hourly_pattern = rf_preds.groupby(rf_preds['datetime_hour'].str[11:13])['rf_pred_response_min'].mean().round(2).to_dict()
rf_avg = round(rf_preds['rf_pred_response_min'].mean(), 2)
rf_fastest_hour = min(hourly_pattern, key=hourly_pattern.get)
rf_slowest_hour = max(hourly_pattern, key=hourly_pattern.get)

# Hourly and monthly summaries
borough_avg = hourly_df.groupby('borough')['response_time_min'].mean().round(2).to_dict()
monthly_summary = monthly_df.groupby('month')['avg_response_min'].mean().round(2).to_dict()

# Total calls
total_calls = hourly_df.groupby('borough')['incident_count'].sum().to_dict()
total_calls_overall = int(hourly_df['incident_count'].sum())

# SARIMAX summaries
forecast_df = df2[df2['dataset_type'] == 'forecast']
test_df = df2[df2['dataset_type'] == 'test']
sarima_forecast_summary = forecast_df.groupby('borough')['predicted_calls'].mean().round(1).to_dict()
sarima_peak_borough = max(sarima_forecast_summary, key=sarima_forecast_summary.get)
sarima_quiet_borough = min(sarima_forecast_summary, key=sarima_forecast_summary.get)
sarima_accuracy = test_df.groupby('borough').apply(
    lambda x: round(((x['predicted_calls'] - x['actual_calls']).abs() / x['actual_calls']).mean() * 100, 2),
    include_groups=False
).to_dict()

# Predict function
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
    row["closest_station_manhattan_miles"] = closest_station_manhattan_miles
    row["is_weekend"] = is_weekend
    row["hour"] = hour
    row["is_holiday"] = is_holiday
    row["initial_severity"] = initial_severity
    row["zipcode"] = zipcode
    row["temperture"] = temperature
    row["precipitation"] = precipitation
    row["windspeed"] = windspeed
    row["weathercode"] = weathercode
    row["day"] = day
    row["month"] = month
    row["is_rush_hour_1"] = is_rush_hour
    row["special_events_Y"] = special_events
    row["standby_Y"] = standby
    row["held_Y"] = held
    borough_key = f"borough_{borough.upper()}"
    if borough_key in row: row[borough_key] = 1
    dispatch_key = f"dispatch_area_{dispatch_area.upper()}"
    if dispatch_key in row: row[dispatch_key] = 1
    type_key = f"initial_type_{initial_type.upper()}"
    if type_key in row: row[type_key] = 1
    df_input = pd.DataFrame([row])
    prediction_seconds = model.predict(df_input)[0]
    return round(float(prediction_seconds) / 60, 2)

# Tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "predict_response_time",
            "description": "Predicts NYC 911 EMS response time in minutes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "season":       {"type": "string", "enum": ["winter", "spring", "summer", "fall"]},
                    "borough":      {"type": "string", "description": "BROOKLYN, MANHATTAN, QUEENS, RICHMOND / STATEN ISLAND"},
                    "dispatch_area":{"type": "string", "description": "e.g. M1, K3, Q2, B4"},
                    "initial_type": {"type": "string", "description": "Call type code e.g. CARD, TRAUMA, EDP, SICK"},
                    "hour":         {"type": "integer", "description": "Hour of day 0-23"},
                    "day":          {"type": "integer", "description": "Day of month"},
                    "month":        {"type": "integer", "description": "Month 1-12"},
                    "is_weekend":   {"type": "integer", "enum": [0, 1]},
                    "is_holiday":   {"type": "integer", "enum": [0, 1]},
                    "is_rush_hour": {"type": "integer", "enum": [0, 1]},
                    "initial_severity": {"type": "integer"},
                    "zipcode":      {"type": "integer"},
                    "temperature":  {"type": "number"},
                    "precipitation":{"type": "number"},
                    "windspeed":    {"type": "number"},
                    "weathercode":  {"type": "integer"},
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

# System prompt
system_prompt = f"""
You are an expert data science assistant specializing in NYC 911 EMS response time analysis.
The models were trained on NYC EMS data from 2024-09-01 to 2025-08-31.

The data comes from NYC Open Data: "EMS Incident Dispatch Data" 
(https://data.cityofnewyork.us/Public-Safety/EMS-Incident-Dispatch-Data/76xm-jjuj).
It contains nearly 2 million real emergency medical service records from September 2024 to August 2025.

Adapt your language to the user — if they ask technical questions respond technically,
if they ask in plain English respond simply.

Only answer questions related to NYC 911 EMS response times. If the user asks about
anything else, politely let them know you can only help with that topic.

This project was built by Ayman Tabidi (Random Forest seasonal models), 
Utku Seyithanoğlu (SARIMAX forecasting model), and Sarah Oasier (data analysis and research).

You have access to TWO models:

1. RANDOM FOREST (my model) — 4 seasonal models (winter, spring, summer, fall) that predict 
EMS response times in seconds based on:
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

For questions outside the model such as hospital locations or NYC EMS facts,
use your own training knowledge to answer as accurately as possible.

Always ask for missing inputs conversationally — start with season, borough, call type, and hour,
then gather remaining details naturally. Never dump a list of 10 questions at once.
When asked general questions like 'which season is best', run the prediction function
across all 4 seasons using typical average values and compare automatically.

Common call type mappings:
- Heart attack / cardiac arrest  CARD
- Broken bone / fracture  INJMIN or INJMAJ
- Breathing difficulty  DIFFBR
- Unconscious person  UNC
- Car accident MVA
- Stroke  CVA
- Psychiatric emergency  EDP
- Stabbing STAB
- Shooting  SHOT
- Sick person SICK

Average response time by borough (minutes): {borough_avg}
Average response time by month: {monthly_summary}
Total 911 calls by borough (Sep 2024 - Aug 2025): {total_calls}
Total calls overall: {total_calls_overall}

RF model hourly prediction summary:
- Overall average predicted response time: {rf_avg} minutes
- Fastest hour of day: {rf_fastest_hour}:00 ({hourly_pattern[rf_fastest_hour]} min avg)
- Slowest hour of day: {rf_slowest_hour}:00 ({hourly_pattern[rf_slowest_hour]} min avg)
- Average predicted response by hour: {hourly_pattern}
"""

st.markdown("""
    <style>
        .block-container {
            max-width: 800px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stChatMessage {
            border-radius: 10px;
            padding: 10px;
        }
        h1 {
            text-align: center;
        }
        .stChatInputContainer {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🚑 NYC 911 EMS Response Time Assistant")
st.caption("Powered by Random Forest + SARIMAX | NYC EMS Data (Sep 2024 - Aug 2025) | Built by Ayman Tabidi, Sarah Oasier & Utku Seyithanoğlu")

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
