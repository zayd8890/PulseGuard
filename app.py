import streamlit as st
from datetime import timedelta
import datetime
import pandas as pd 
import numpy
import toml
from deepseek import ChatModel
import time
from send_email import ai_email_agent
import json
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pymongo
from pymongo import MongoClient


print((f"Streamlit version: {st.__version__}"))

config = toml.load('config.toml')
MONGODB_CONNECTION_STRING = config['secrets']['MONGOBD_CONNECTION_STRING']
DEEPSEEK_API = config['secrets']['DEEPSEEK_API']

# Set page configuration
st.set_page_config(page_title="Medical Monitoring System", layout="wide")


deepseek_model = ChatModel(
    api_key=DEEPSEEK_API,  # or directly use your variable
    model="deepseek-chat",               # or another model like "deepseek-coder" if you want
)

# MongoDB Connection
# Initialize MongoDB connection
def get_mongodb_connection():
    # Replace with your MongoDB connection string
    connection_string =MONGODB_CONNECTION_STRING
    client = MongoClient(connection_string)
    db = client["medical_monitoring"]
    return db

# Initialize MongoDB collections
def initialize_collections(db):
    # Create collections if they don't exist
    if "vital_signs" not in db.list_collection_names():
        db.create_collection("vital_signs")
    
    if "chat_history" not in db.list_collection_names():
        db.create_collection("chat_history")
    
    # Create indexes
    db.vital_signs.create_index("timestamp")
    db.chat_history.create_index("timestamp")
    
    return db.vital_signs, db.chat_history

# Get MongoDB connection and collections
db = get_mongodb_connection()
vital_signs_collection, chat_history_collection = initialize_collections(db)


def handle_alert(contact, situation_description):
    """Immediately send an alert email without LLM."""
    alert_message = f"🚨 Immediate Alert: {situation_description}. Please take action!"
    ai_email_agent(contact, alert_message, type="Alerts")


def handle_report(llm, contact, situation_description):
    """Use LLM to generate a report and send."""
    prompt = f"""
You are a medical assistant. Summarize this medical situation into a professional short report email.

Situation:
{ situation_description }

Output ONLY the email body as plain text.
"""
    response = llm(prompt)
    report_message = response.strip()
    ai_email_agent(contact, report_message, type="Report")

# Load chat history from MongoDB
def filter_history_by_period(start_time=None, end_time=None, last_n=120):
    filtered_data = {
        "ppg_history": [],
        "abp_history": [],
        "spo2_history": [],
        "sbp_history": [],
        "dbp_history": [],
        "time_history": []
    }
    
    # If time range is provided, filter by start and end time
    if start_time and end_time:
        for i, timestamp in enumerate(st.session_state.time_history):
            if start_time <= timestamp <= end_time:
                filtered_data["ppg_history"].append(st.session_state.ppg_history[i])
                filtered_data["abp_history"].append(st.session_state.abp_history[i])
                filtered_data["spo2_history"].append(st.session_state.spo2_history[i])
                filtered_data["sbp_history"].append(st.session_state.sbp_history[i])
                filtered_data["dbp_history"].append(st.session_state.dbp_history[i])
                filtered_data["time_history"].append(timestamp)
    else:
        # Default to the last 120 entries if no time range is provided
        if len(st.session_state.time_history) > last_n:
            filtered_data["ppg_history"] = st.session_state.ppg_history[-last_n:]
            filtered_data["abp_history"] = st.session_state.abp_history[-last_n:]
            filtered_data["spo2_history"] = st.session_state.spo2_history[-last_n:]
            filtered_data["sbp_history"] = st.session_state.sbp_history[-last_n:]
            filtered_data["dbp_history"] = st.session_state.dbp_history[-last_n:]
            filtered_data["time_history"] = st.session_state.time_history[-last_n:]
        else:
            # If there are fewer than 120 entries, use all available
            filtered_data["ppg_history"] = st.session_state.ppg_history
            filtered_data["abp_history"] = st.session_state.abp_history
            filtered_data["spo2_history"] = st.session_state.spo2_history
            filtered_data["sbp_history"] = st.session_state.sbp_history
            filtered_data["dbp_history"] = st.session_state.dbp_history
            filtered_data["time_history"] = st.session_state.time_history

    return filtered_data
def load_chat_history():
    try:
        # Get the last 10 messages sorted by timestamp (oldest to newest)
        chat_documents = (
            chat_history_collection.find()
            .sort("timestamp", -1)  # Sort newest first
            .limit(5)              # Take 10 most recent
        )
        chat_documents = list(chat_documents)[::-1]  # Reverse to oldest first
        
        chat_history = []
        for doc in chat_documents:
            chat_history.append({
                "role": doc["role"],
                "content": doc["content"]
            })
        
        return chat_history
    except Exception as e:
        st.error(f"Error loading chat history: {e}")
        return []


# Save chat message to MongoDB
def save_chat_message(role, content):
    try:
        # Create document for chat message
        chat_doc = {
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now()
        }
        
        # Insert the document
        chat_history_collection.insert_one(chat_doc)
    except Exception as e:
        st.error(f"Error saving chat message: {e}")

# Save vital signs data to MongoDB
def save_vital_signs(data_dict):
    try:
        # Add timestamp if not already present
        if "timestamp" not in data_dict:
            data_dict["timestamp"] = datetime.datetime.now()
            
        # Insert the document
        vital_signs_collection.insert_one(data_dict)
    except Exception as e:
        st.error(f"Error saving vital signs: {e}")

# Define thresholds for blood pressure and heart rate
thresholds_data = {
    "blood_pressure": [
        {
            "condition": "Normal",
            "data_type": "BP",
            "min": {"SBP": 0, "DBP": 0},
            "max": {"SBP": 119, "DBP": 79},
            "activity": "Resting",
            "action_alert": "No action required"
        },
        {
            "condition": "Elevated",
            "data_type": "BP",
            "min": {"SBP": 120, "DBP": 0},
            "max": {"SBP": 129, "DBP": 79},
            "activity": "Resting",
            "action_alert": "Monitor, provide lifestyle tips"
        },
        {
            "condition": "Hypertension Stage 1",
            "data_type": "BP",
            "min": {"SBP": 130, "DBP": 80},
            "max": {"SBP": 139, "DBP": 89},
            "activity": "Resting",
            "action_alert": "Alert user, recommend rest"
        },
        {
            "condition": "Hypertension Stage 2",
            "data_type": "BP",
            "min": {"SBP": 140, "DBP": 90},
            "max": {"SBP": 180, "DBP": 120},
            "activity": "Resting",
            "action_alert": "Alert user, suggest medical consultation"
        },
        {
            "condition": "Hypertensive Crisis",
            "data_type": "BP",
            "min": {"SBP": 181, "DBP": 121},
            "max": {"SBP": 300, "DBP": 200},
            "activity": "Resting",
            "action_alert": "Immediate medical help required"
        }
    ],
    "heart_rate": [
        {
            "condition": "Normal",
            "data_type": "HR",
            "min": 60,
            "max": 100,
            "activity": "Resting",
            "action_alert": "No action required"
        },
        {
            "condition": "Elevated HR",
            "data_type": "HR",
            "min": 101,
            "max": 190,
            "activity": "Resting",
            "action_alert": "Alert user to check for stress or dehydration"
        },
        {
            "condition": "Tachycardia (High)",
            "data_type": "HR",
            "min": 191,
            "max": 300,
            "activity": "Exercise/Intense Activity",
            "action_alert": "Alert user to slow down and monitor"
        },
        {
            "condition": "Bradycardia (Low)",
            "data_type": "HR",
            "min": 0,
            "max": 59,
            "activity": "Resting",
            "action_alert": "Alert user and suggest consulting a healthcare provider"
        }
    ]
}

# Function to get color for HR status
def get_hr_status_color(status):
    if status == "Normal":
        return "green"
    elif status == "Tachycardia (High)":
        return "red"
    elif status == "Elevated HR":
        return "orange"
    elif status == "Bradycardia (Low)":
        return "red"
    else:
        return "orange"

# Function to get color for BP status
def get_bp_status_color(status):
    if status == "Normal":
        return "green"
    elif status == "Low":
        return "orange"
    elif status == "Elevated":
        return "orange"
    elif status == "Hypertension Stage 1":
        return "orange"
    elif status == "Hypertension Stage 2":
        return "orange"
    elif status == "Hypertensive Crisis":
        return "red"
    else:
        return "red"

# Function to predict blood pressure condition based on thresholds
def predict_bp_condition(sbp, dbp):
    # Default values
    condition = "Normal"
    action_alert = "No action required"
    confidence = 1.0
    
    # Check the blood pressure thresholds
    if sbp == 0:
        status == 'low'
    else:
        for bp_threshold in thresholds_data["blood_pressure"]:
            if (sbp >= bp_threshold["min"]["SBP"] and sbp <= bp_threshold["max"]["SBP"] and
                dbp >= bp_threshold["min"]["DBP"] and dbp <= bp_threshold["max"]["DBP"]):
                condition = bp_threshold["condition"]
                action_alert = bp_threshold["action_alert"]
                break
        
        # If SBP and DBP fall into different categories, use the more severe one
        if condition == "Normal":
            for bp_threshold in thresholds_data["blood_pressure"]:
                if (sbp >= bp_threshold["min"]["SBP"] and sbp <= bp_threshold["max"]["SBP"]) or \
                (dbp >= bp_threshold["min"]["DBP"] and dbp <= bp_threshold["max"]["DBP"]):
                    if bp_threshold["condition"] != "Normal":
                        condition = bp_threshold["condition"]
                        action_alert = bp_threshold["action_alert"]
                        break
    
    return condition, confidence, action_alert

# Function to predict heart rate condition based on thresholds
def predict_hr_condition(hr, activity="Resting"):
    # Default values
    condition = "Normal"
    confidence = 1.0
    action_alert = "No action required"
    
    # Check the heart rate thresholds
    for hr_threshold in thresholds_data["heart_rate"]:
        if hr >= hr_threshold["min"] and hr <= hr_threshold["max"]:
            if hr_threshold["activity"] == activity:
                condition = hr_threshold["condition"]
                action_alert = hr_threshold["action_alert"]
                break
            # If exact activity match not found, use the range match
            elif condition == "Normal":
                condition = hr_threshold["condition"]
                action_alert = hr_threshold["action_alert"]
    
    return condition, confidence, action_alert




def get_iot_data(db):
    """Fetch IoT data from MongoDB for PPG, ABP, SpO2, SBP, and DBP"""

    collection = db['vital_signs']  
    
    # Define the time range to fetch the most recent data (last 2 seconds)
    current_time = datetime.datetime.now()
    time_range = current_time - timedelta(seconds=2)  # Subtract 2 seconds from current time
    
    # Query the MongoDB collection for data within the last 2 seconds
    data = collection.find_one({"timestamp": {"$gte": time_range}})  # Adjust this query as needed
    
    # If data is found, extract values from the document
    if data:
        # Use .get() with a default value of 0 if the field is missing
        ppg = data.get('ppg', 0)  # Replace 'ppg' with the correct field name
        abp = data.get('abp', 0)  # Replace 'abp' with the correct field name
        spo2 = data.get('spo2', 0)  # Replace 'spo2' with the correct field name
        sbp = data.get('sbp', 0)  # Replace 'sbp' with the correct field name
        dbp = data.get('dbp', 0)  # Replace 'dbp' with the correct field name
        
        return ppg, abp, spo2, sbp, dbp
    else:
        print("No data found in MongoDB.")
        return 0, 0, 0, 0, 0  # Return 0s if no data found


# Function to generate response using Hugging Face QwQ-32B model
def get_llm_response(prompt,deepseek_model, chat_history=None):
    # Create a consolidated context from chat history
    formatted_messages = []
    
    # First add the system message with medical context
    system_message = f"""
            You are a medical assistant chatbot helping monitor a patient's vital signs.

            Patient's most recent data:
            - Heart Rate (PPG): {st.session_state.current_ppg:.2f} bpm
            - ABP: {st.session_state.current_abp:.2f} mmHg
            - SpO2: {st.session_state.current_spo2:.2f}%
            - SBP: {st.session_state.current_sbp:.2f} mmHg
            - DBP: {st.session_state.current_dbp:.2f} mmHg
            - Heart rate Status: {st.session_state.current_HR_status}
            - Blood pressure Status: {st.session_state.current_BP_status}
            - Confidence: {st.session_state.current_confidence:.2f}

            Behaviors:
            - If the user is asking a general medical question, provide a normal helpful response.
            - If the user is requesting a report (summaries, statistics, monitoring period, vitals evolution) the reports is send as an email, 
            reply ONLY with a JSON format:
            {{
                "normal_text": "string"(e.g:The report has been sent!),
                "name" :"string" (default: No),
                "email": "string" (default:No), 
                "period_range": "string" (default:No),
                "start_time": string,
                "end_time": string,
                "heart_rate": true or false (default:True),
                "blood_pressure": true or false (default:True),
                "spo2": true or false (default:True),
                "SBP": true or false (default:True),
                "DBP": true or false (default:True),
                "message_email": "string" ( this should be the body message, note that the destinater may not be the one requesting the report)
            }}
            - "normal_text" should always summarize the response normally even when sending the JSON.

            Important:
            - If you detect any emergency (e.g., Hypertensive Crisis, Tachycardia), politely advise immediate medical attention.
            - Never suggest any medication or specific treatments.
            - Never mix JSON and normal text outside of "normal_text" field.
            - Never mix JSON and normal text outside of "normal_text" field.
            """

    formatted_messages.append({
        "role": "system",
        "content": system_message
    })
    
    # Add chat history
    if chat_history:
        for chat in chat_history:
            formatted_messages.append({
                "role": chat["role"],
                "content": chat["content"]
            })
    
    # Add the current user message
    formatted_messages.append({
        "role": "user",
        "content": prompt
    })
    
    try:
        # Call DeepSeek API to get the response
        completion = deepseek_model.chat(
            messages=formatted_messages,
            temperature=0.7,  # You can tweak these parameters as needed
            max_tokens=512
        )

        # Extract the response text
        response_text = completion["choices"][0]["message"]["content"]
        return response_text
        
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "I'm having trouble connecting to my knowledge base. Please try again later."

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=[
        'Timestamp', 'PPG', 'ABP', 'SpO2', 'SBP', 'DBP', 'Age', 'Gender', 'Height', 'Weight', 'BMI', 
        'BP_Prediction', 'HR_Prediction', 'BP_Status', 'HR_Status'
    ])

# Load chat history from MongoDB when starting the app
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = load_chat_history()

# Initialize current values
if 'current_ppg' not in st.session_state:
    st.session_state.current_ppg = 0.0
if 'current_abp' not in st.session_state:
    st.session_state.current_abp = 0.0
if 'current_spo2' not in st.session_state:
    st.session_state.current_spo2 = 0.0
if 'current_sbp' not in st.session_state:
    st.session_state.current_sbp = 0.0
if 'current_dbp' not in st.session_state:
    st.session_state.current_dbp = 0.0
if 'current_HR_status' not in st.session_state:
    st.session_state.current_HR_status = "Normal"
if 'current_BP_status' not in st.session_state:
    st.session_state.current_BP_status = "Normal"
if 'current_confidence' not in st.session_state:
    st.session_state.current_confidence = 1.0

# Initialize chatbot visibility state
if 'show_chatbot' not in st.session_state:
    st.session_state.show_chatbot = False

# App title and description
st.title("Medical Monitoring System")

# Top section with current values display boxes
title = st.container()

with title:
    st.subheader("Real-time health monitoring with IoT integration")

top_row = st.container()
with top_row:
    # Reordered columns for vital signs display
    HR_status, hr_col, spo2_col, abp_col, BP_status, _ = st.columns([1.25, 1, 0.75, .8, 1.25, 1.5])
    
    # Create placeholder for status box
    with HR_status:
        HR_status_placeholder = st.empty()
        
        # Get status color based on current HR status
        HR_status_color = get_hr_status_color(st.session_state.current_HR_status)
            
        HR_status_placeholder.markdown(
            f"""
            <div style="background-color:{HR_status_color}30;padding:10px;border-radius:10px;text-align:center;border:2px solid {HR_status_color};">
                <h3 style="margin:0;color:#1e3a8a;">heart rate</h3>
                <h2 style="margin:5px 0;color:{HR_status_color};">{st.session_state.current_HR_status}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with BP_status:
        BP_status_placeholder = st.empty()
        
        # Get status color based on current BP status
        BP_status_color = get_bp_status_color(st.session_state.current_BP_status)

        BP_status_placeholder.markdown(
            f"""
            <div style="background-color:{BP_status_color}30;padding:10px;border-radius:10px;text-align:center;border:2px solid {BP_status_color};">
                <h3 style="margin:0;color:#1e3a8a;">blood pressure</h3>
                <h2 style="margin:5px 0;color:{BP_status_color};">{st.session_state.current_BP_status}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Create placeholder for HR/PPG box
    with hr_col:
        ppg_placeholder = st.empty()
        ppg_placeholder.markdown(
            """
            <div style="background-color:transparent;padding:10px;border-radius:10px;text-align:center;border:1px solid rgba(255,255,255,0.3);">
                <h3 style="margin:0;color:#1e3a8a;">HR</h3>
                <h2 style="margin:5px 0;color:#ff4b4b;">{:.2f} bpm</h2>
            </div>
            """.format(st.session_state.current_ppg),
            unsafe_allow_html=True
        )
    
    # Create placeholder for BP box (now showing SBP/DBP)
    with abp_col:
        bp_placeholder = st.empty()
        bp_placeholder.markdown(
            """
            <div style="background-color:transparent;padding:10px;border-radius:10px;text-align:center;border:1px solid rgba(255,255,255,0.3);">
                <h3 style="margin:0;color:#1e3a8a;">BP</h3>
                <h2 style="margin:5px 0;color:#0068c9;">{:.0f}</h2>
            </div>
            """.format(st.session_state.current_abp),
            unsafe_allow_html=True
        )
    
    # Create placeholder for SpO2 box
    with spo2_col:
        spo2_placeholder = st.empty()
        spo2_placeholder.markdown(
            """
            <div style="background-color:transparent;padding:10px;border-radius:10px;text-align:center;border:1px solid rgba(255,255,255,0.3);">
                <h3 style="margin:0;color:#1e3a8a;">SpO2</h3>
                <h2 style="margin:5px 0;color:#10b981;">{:.1f}%</h2>
            </div>
            """.format(st.session_state.current_spo2),
            unsafe_allow_html=True
        )

# Create dashboard layout with columns for main content and sidebar
col1, col2 = st.columns([3, 1])

# Right column with chatbot first, then patient info
with col2:
    chat_expander = st.expander("💬 Medical Assistant", expanded=st.session_state.show_chatbot)

    with chat_expander:
        st.subheader("Medical Assistant")

        chat_container = st.container()

        # Chat input form with Send and Reset on the same row
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("Ask the medical assistant...", key="chatbot_input_form")
            col21, col22,col23 = st.columns([1.25,3, 1.25])
            with col23:
                reset = st.form_submit_button("Reset")
            with col21:
                submit = st.form_submit_button("Send")
            
            

            if reset:
                st.session_state.chat_history = []
                st.rerun()

        if submit and user_input != '':
            # Save user message
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            save_chat_message("user", user_input)

            # Get assistant reply (pass deepseek_model as argument)
            response = get_llm_response(user_input, deepseek_model, st.session_state.chat_history)

            # Remove backticks if the response is wrapped in code block (i.e., triple backticks)
            if response.startswith("```json") and response.endswith("```"):
                response = response[7:-3].strip()  # Remove ```json at the start and ``` at the end

            # Try parsing response as JSON, if possible
            try:
                parsed_response = json.loads(response)
                if parsed_response.get('period_range') and parsed_response['period_range'] != "No":
                    period_range = parsed_response['period_range']
                    start_time_str, end_time_str = period_range.split(" to ")
                    
                    # Convert start and end times to datetime
                    start_time = datetime.strptime(parsed_response['start_time'], "%Y-%m-%d %H:%M:%S")  # Adjust format as per your input
                    end_time = datetime.strptime(parsed_response['end_time'], "%Y-%m-%d %H:%M:%S")  # Adjust format as per your input
                    
                    # Filter histories based on the time range
                    filtered_data = filter_history_by_period(start_time, end_time)
                else:
                    # If no period_range is given, filter the last 120 values
                    filtered_data = filter_history_by_period()



                fig_report = make_subplots(
                    rows=3, cols=1, 
                    shared_xaxes=True, 
                    subplot_titles=("Heart Rate History", "Blood Pressure History", "SpO2 History"),
                    vertical_spacing=0.1
                )

                # Add heart rate trace
                if filtered_data['ppg_history'] != []:
                    fig_report.add_trace(
                        go.Scatter(
                            x=filtered_data['time_history'], 
                            y=filtered_data['ppg_history'],
                            mode='lines+markers',
                            name='Heart Rate',
                            line=dict(color='red')
                        ),
                        row=1, col=1
                    )

                # Add blood pressure traces
                if filtered_data['sbp_history'] != []:
                    fig_report.add_trace(
                        go.Scatter(
                            x=filtered_data['time_history'], 
                            y=filtered_data['sbp_history'],
                            mode='lines+markers',
                            name='SBP',
                            line=dict(color='blue')
                        ),
                        row=2, col=1
                    )
                if filtered_data['dbp_history'] !=[]:
                    fig_report.add_trace(
                        go.Scatter(
                            x=filtered_data['time_history'], 
                            y=filtered_data['dbp_history'],
                            mode='lines+markers',
                            name='DBP',
                            line=dict(color='lightblue')
                        ),
                        row=2, col=1
                    )

                # Add SpO2 trace
                if filtered_data['spo2_history'] != []:
                    fig_report.add_trace(
                        go.Scatter(
                            x=filtered_data['time_history'], 
                            y=filtered_data['spo2_history'],
                            mode='lines+markers',
                            name='SpO2',
                            line=dict(color='green')
                        ),
                        row=3, col=1
                    )

                # Update layout with Y-axis labels and margins
                fig_report.update_layout(
                    height=600, 
                    margin=dict(l=100, r=0, t=30, b=0),  # Increase left margin for Y-axis labels
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    
                    # Add Y-axis labels with specific values
                    yaxis=dict(
                        tickvals=[100, 150, 200],  # Define Y-axis tick positions
                        ticktext=['100', '150', '200'],  # Customize Y-axis labels
                        title="Heart Rate",
                        scaleanchor="y",  # Anchor the y-axis for consistent scaling
                    ),
                    yaxis2=dict(
                        tickvals=[60, 100, 160],  # Define Y-axis tick positions
                        ticktext=['60', '100', '160'],  # Customize Y-axis labels for SBP/DBP
                        title="Blood Pressure (mmHg)",
                        scaleanchor="y2",  # Anchor the y-axis for consistent scaling
                    ),
                    yaxis3=dict(
                        tickvals=[90, 95, 100],  # Define Y-axis tick positions for SpO2
                        ticktext=['90', '95', '100'],  # Customize Y-axis labels
                        title="SpO2 (%)",
                        scaleanchor="y3",  # Anchor the y-axis for consistent scaling
                    ),
                )


                contacts_collection = db["contacts"]

                report_contacts = list(contacts_collection.find({"send": {"$in": ["Report", "Both"]}}))

                for contact in report_contacts:
                    try:
                        ai_email_agent(
                            contact=contact,
                            message=parsed_response['message_email'],
                            type_="Report",
                            plot_figs= [fig_report]
                        )
                    except Exception as e:
                        st.error(f"Failed to send report email to {contact['email']}: {e}")

                if isinstance(parsed_response, dict) and "normal_text" in parsed_response:
                    assistant_reply = parsed_response["normal_text"]
                else:
                    assistant_reply = response

            except json.JSONDecodeError:
                # If response isn't JSON, it's just normal text
                print('please god no')
                assistant_reply = response

            # Save assistant message
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
            save_chat_message("assistant", assistant_reply)

            # Set a flag to show immediately
            st.session_state.new_message = True

        # Display the chat
        with chat_container:
            for chat in st.session_state.chat_history:
                if chat["role"] == "user":
                    st.write(f"👤 **You**: {chat['content']}")
                else:
                    st.write(f"🤖 **Assistant**: {chat['content']}")

        # Optional: clean the flag
        if 'new_message' in st.session_state:
            del st.session_state['new_message']




    
    # PATIENT INFO SECTION SECOND
    st.markdown("---")
    st.subheader("Patient Information")
    
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
    weight = st.number_input("Weight (kg)", min_value=20, max_value=250, value=70)
    
    # Activity level selection
    activity = st.selectbox("Current Activity Level", [
        "Resting", 
        "Exercise/Moderate Activity", 
        "Exercise/Intense Activity"
    ])
    
    # Calculate BMI
    bmi = weight / ((height/100) ** 2)
    
    # Display BMI with category
    bmi_value = round(bmi, 2)
    
    # Determine BMI category
    if bmi < 18.5:
        bmi_category = "Underweight"
        bmi_color = "orange"
    elif bmi < 25:
        bmi_category = "Normal weight"
        bmi_color = "green"
    elif bmi < 30:
        bmi_category = "Overweight"
        bmi_color = "orange"
    else:
        bmi_category = "Obese"
        bmi_color = "red"
    
    # Display BMI with color and category
    st.markdown(
        f"""
        <div style="margin-bottom:10px;">
            <span style="font-weight:bold;">BMI:</span> 
            <span style="color:{bmi_color};font-weight:bold;">{bmi_value}</span> 
            <span style="color:{bmi_color};">({bmi_category})</span>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Save user info button
    if st.button("Update Patient Info"):
        # Save patient info to MongoDB
        patient_info = {
            "age": age,
            "gender": gender,
            "height": height,
            "weight": weight,
            "bmi": bmi_value,
            "bmi_category": bmi_category,
            "activity": activity,
            "timestamp": datetime.datetime.now()
        }
        save_vital_signs(patient_info)
        st.success("Patient information updated!")

# Main dashboard in left column
with col1:
    # Create tabs for different views
    tab1, tab2, tab3  = st.tabs(["Real-time Monitoring", "Historical Data","contact"])
    
    # Real-time monitoring tab
    with tab1:
        st.subheader("Vital Signs Monitor")
        
        start_monitoring = st.checkbox("Start Monitoring", value=False)
        # Create placeholders for real-time charts
        chart_placeholder = st.empty()
        
        if start_monitoring:
            # Create history for plotting
            if 'ppg_history' not in st.session_state:
                st.session_state.ppg_history = []
                st.session_state.abp_history = []
                st.session_state.spo2_history = []
                st.session_state.sbp_history = []
                st.session_state.dbp_history = []
                st.session_state.time_history = []
            
            # Number of points to show
            window_size = 100
            
            while start_monitoring:
                # Get new data point
                ppg, abp, spo2, sbp, dbp = get_iot_data(db)
                
                # Update current values in session state
                st.session_state.current_ppg = ppg
                st.session_state.current_abp = abp
                st.session_state.current_spo2 = spo2
                st.session_state.current_sbp = sbp
                st.session_state.current_dbp = dbp
                
                # Update the top boxes with current values
                ppg_placeholder.markdown(
                    """
                    <div style="background-color:transparent;padding:10px;border-radius:10px;text-align:center;border:1px solid rgba(255,255,255,0.3);">
                        <h3 style="margin:0;color:#1e3a8a;">HR</h3>
                        <h2 style="margin:5px 0;color:#ff4b4b;">{:.2f} bpm</h2>
                    </div>
                    """.format(ppg),
                    unsafe_allow_html=True
                )
                
                bp_placeholder.markdown(
                    """
                    <div style="background-color:transparent;padding:10px;border-radius:10px;text-align:center;border:1px solid rgba(255,255,255,0.3);">
                        <h3 style="margin:0;color:#1e3a8a;">BP</h3>
                        <h2 style="margin:5px 0;color:#0068c9;">{:.0f} mmHg</h2>
                    </div>
                    """.format(abp),
                    unsafe_allow_html=True
                )
                
                spo2_placeholder.markdown(
                    """
                    <div style="background-color:transparent;padding:10px;border-radius:10px;text-align:center;border:1px solid rgba(255,255,255,0.3);">
                        <h3 style="margin:0;color:#1e3a8a;">SpO2</h3>
                        <h2 style="margin:5px 0;color:#10b981;">{:.1f}%</h2>
                    </div>
                    """.format(spo2),
                    unsafe_allow_html=True
                )
                
                # Get current time
                now = datetime.datetime.now()
                
                # Append to history
                st.session_state.ppg_history.append(ppg)
                st.session_state.abp_history.append(abp)
                st.session_state.spo2_history.append(spo2)
                st.session_state.sbp_history.append(sbp)
                st.session_state.dbp_history.append(dbp)
                st.session_state.time_history.append(now)
                
                # Keep only the last window_size points
                if len(st.session_state.ppg_history) > window_size:
                    st.session_state.ppg_history = st.session_state.ppg_history[-window_size:]
                    st.session_state.abp_history = st.session_state.abp_history[-window_size:]
                    st.session_state.spo2_history = st.session_state.spo2_history[-window_size:]
                    st.session_state.sbp_history = st.session_state.sbp_history[-window_size:]
                    st.session_state.dbp_history = st.session_state.dbp_history[-window_size:]
                    st.session_state.time_history = st.session_state.time_history[-window_size:]
                
                # Create a plotly figure with 3 subplots
                fig = make_subplots(
                    rows=3, cols=1, 
                    shared_xaxes=True, 
                    subplot_titles=("Heart Rate", "Blood Pressure", "SpO2"),
                    vertical_spacing=0.08, 
                    row_heights=[0.33, 0.33, 0.33]
                )
                
                # Add trace for Heart Rate (PPG)
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(st.session_state.ppg_history))), 
                        y=st.session_state.ppg_history,
                        mode='lines',
                        name='Heart Rate',
                        line=dict(color='red')
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(st.session_state.abp_history))), 
                        y=st.session_state.sbp_history,
                        mode='lines',
                        name='ABP',
                        line=dict(color='blue')
                    ),
                    row=2, col=1
                )
                
                # Add trace for SpO2
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(st.session_state.spo2_history))), 
                        y=st.session_state.spo2_history,
                        mode='lines',
                        name='SpO2',
                        line=dict(color='green')
                    ),
                    row=3, col=1
                )
                
                # Update layout
                fig.update_layout(
                    height=450, 
                    margin=dict(l=0, r=0, t=30, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                # Remove x-axis labels except for the bottom plot
                fig.update_xaxes(showticklabels=False, row=1, col=1)
                fig.update_xaxes(showticklabels=False, row=2, col=1)
                
                # Display the chart
                chart_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Make predictions based on thresholds
                BP_status, bp_confidence, bp_action = predict_bp_condition(sbp, dbp)
                HR_status, hr_confidence, hr_action = predict_hr_condition(ppg, activity)
                
                # Update session state with prediction
                st.session_state.current_HR_status = HR_status
                st.session_state.current_BP_status = BP_status
                st.session_state.current_confidence = hr_confidence  # Just using one confidence value
                
                # Get updated colors based on new status
                HR_status_color = get_hr_status_color(HR_status)
                BP_status_color = get_bp_status_color(BP_status)
                
                # Update prediction box
                HR_status_placeholder.markdown(
                    f"""
                    <div style="background-color:{HR_status_color}30;padding:10px;border-radius:10px;text-align:center;border:2px solid {HR_status_color};">
                        <h3 style="margin:0;color:#1e3a8a;">Status</h3>
                        <h2 style="margin:5px 0;color:{HR_status_color};">{HR_status}</h2>
                        <p style="margin:0;color:{HR_status_color};"></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                BP_status_placeholder.markdown(
                    f"""
                    <div style="background-color:{BP_status_color}30;padding:10px;border-radius:10px;text-align:center;border:2px solid {BP_status_color};">
                        <h3 style="margin:0;color:#1e3a8a;">Status</h3>
                        <h2 style="margin:5px 0;color:{BP_status_color};">{BP_status}</h2>
                        <p style="margin:0;color:{BP_status_color};"></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Auto-open chatbot if severe condition detected
                if (BP_status == "Hypertensive Crisis" or HR_status == "Tachycardia (high)"):

                    if BP_status == "Hypertensive Crisis" and HR_status == "Tachycardia (high)":
                        if not st.session_state.show_chatbot:
                            st.session_state.show_chatbot = True

                            alert_prompt = f"""
                            The patient's vital signs monitoring system has detected a critical condition: {BP_status} and {HR_status}.
                            Current readings:
                            - Heart Rate (PPG): {ppg:.2f} bpm
                            - Blood Pressure: {abp:.0f} mmHg
                            - SpO2: {spo2:.1f}%
                            - Age: {age}
                            - Gender: {gender}
                            
                            Generate an urgent but calm medical alert message about this situation that provides immediate guidance. 
                            Do not provide specific medication advice but focus on immediate actions the patient should take.
                            """

                            alert_messages = [
                                {"role": "system", "content": "You are a medical emergency assistant. Provide urgent but calm guidance."},
                                {"role": "user", "content": alert_prompt}
                            ]

                            try:
                                completion = deepseek_model.chat(
                                messages=alert_messages,
                                temperature=0.7,  # You can tweak these parameters as needed
                                max_tokens=512
                            )
                                alert_response = completion.choices[0].message.content
                            except Exception as e:
                                alert_response = f"MEDICAL EMERGENCY DETECTED: {BP_status} and {HR_status} {bp_action}."

                            st.session_state.chat_history.append({"role": "assistant", "content": f"❗ ALERT: {alert_response}"})
                            save_chat_message("assistant", f"❗ ALERT: {alert_response}")

                            contacts_collection = db["contacts"]

                            alert_contacts = list(contacts_collection.find({"send": {"$in": ["Alerts", "Both"]}}))

                            for contact in alert_contacts:
                                try:
                                    ai_email_agent(
                                        contact=contact,
                                        message=f"❗ EMERGENCY ALERT ❗\n\n{alert_response}",
                                        type_="Alert",
                                        plot_figs=[fig]
                                    )
                                except Exception as e:
                                    st.error(f"Failed to send alert email to {contact['email']}: {e}")

                    else:
                        if not st.session_state.show_chatbot:
                            st.session_state.show_chatbot = True

                            alert_prompt = f"""
                            The patient's vital signs monitoring system has detected a critical condition: {HR_status if HR_status =="Tachycardia (high)" else BP_status}.
                            Current readings:
                            - Heart Rate (PPG): {ppg:.2f} bpm
                            - Blood Pressure: {abp:.0f} mmHg
                            - SpO2: {spo2:.1f}%
                            - Age: {age}
                            - Gender: {gender}
                            
                            Generate an urgent but calm medical alert message about this situation that provides immediate guidance. 
                            Do not provide specific medication advice but focus on immediate actions the patient should take.
                            """

                            alert_messages = [
                                {"role": "system", "content": "You are a medical emergency assistant. Provide urgent but calm guidance."},
                                {"role": "user", "content": alert_prompt}
                            ]

                            try:
                                completion = deepseek_model.chat(
                                messages=alert_messages,
                                temperature=0.7,  # You can tweak these parameters as needed
                                max_tokens=512
                            )
                                alert_response = completion.choices[0].message.content
                            except Exception as e:
                                alert_response = f"MEDICAL EMERGENCY DETECTED: {HR_status if HR_status == 'Tachycardia (high)' else BP_status}. {hr_action if HR_status == 'Tachycardia (high)' else bp_action}."

                            st.session_state.chat_history.append({"role": "assistant", "content": f"❗ ALERT: {alert_response}"})
                            save_chat_message("assistant", f"❗ ALERT: {alert_response}")

                            contacts_collection = db["contacts"]

                            alert_contacts = list(contacts_collection.find({"send": {"$in": ["Alerts", "Both"]}}))

                            for contact in alert_contacts:
                                try:
                                    ai_email_agent(
                                        contact=contact,
                                        message=f"❗ EMERGENCY ALERT ❗\n\n{alert_response}",
                                        type_="Alert",
                                        plot_figs=[fig]
                                    )
                                except Exception as e:
                                    st.error(f"Failed to send alert email to {contact['email']}: {e}")

                            # # Rerun to update UI
                            # st.rerun()
                    
                # Add to dataframe
                new_row = pd.DataFrame({
                    'Timestamp': [now],
                    'PPG': [ppg],
                    'ABP': [abp],
                    'SpO2': [spo2],
                    'SBP': [sbp],
                    'DBP': [dbp],
                    'Age': [age],
                    'Gender': [gender],
                    'Height': [height],
                    'Weight': [weight],
                    'BMI': [bmi],
                    'BP_status': [BP_status],
                    'HR_status': [BP_status],
                    # 'SpO2_Prediction': [spo2_prediction],
                    # 'Overall_Status': [overall_status],
                    # 'Confidence': [overall_confidence]
                })

                # Save to MongoDB
                vital_signs_data = {
                    'timestamp': now,
                    'ppg': ppg,
                    'abp': abp,
                    'spo2': spo2,
                    'sbp': sbp,
                    'dbp': dbp,
                    'age': age,
                    'gender': gender,
                    'height': height,
                    'weight': weight,
                    'bmi': bmi,
                    'BP_status': BP_status,
                    'HR_status': BP_status,
                    # 'spo2_prediction': spo2_prediction,
                    # 'overall_status': overall_status,
                    # 'confidence': overall_confidence,
                    # 'activity': activity
                }
                save_vital_signs(vital_signs_data)

                # Update DataFrame
                st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
                
                # Add some delay to simulate real-time data
                time.sleep(0.5)
    
    # Historical data tab
    with tab2:
        st.subheader("Historical Data")
        
        # Add filters for MongoDB data
        st.subheader("Data Filters")
        col_date1, col_date2 = st.columns(2)
        
        with col_date1:
            start_date = st.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=7))
        
        with col_date2:
            end_date = st.date_input("End Date", datetime.date.today())
        
        # Convert to datetime
        start_datetime = datetime.datetime.combine(start_date, datetime.time.min)
        end_datetime = datetime.datetime.combine(end_date, datetime.time.max)
        
        # Filter options
        filter_options = st.multiselect(
            "Filter by Condition", 
            [
                "All", "Normal", "Elevated", "Hypertension Stage 1", 
                "Hypertension Stage 2", "Hypertensive Crisis", 
                 "Elevated Resting HR", "Bradycardia (Low HR)",
            ],
            default=["All"]
        )
        
        # Query button
        if st.button("Query Database"):
            # Query MongoDB
            try:
                # Find data between dates
                base_query = {
                    "timestamp": {
                        "$gte": start_datetime,
                        "$lte": end_datetime
                    }
                }
                
                # Add condition filter if not "All"
                if "All" not in filter_options and filter_options:
                    condition_query = {
                        "$or": [
                            {"bp_prediction": {"$in": filter_options}},
                            {"hr_prediction": {"$in": filter_options}},
                            {"spo2_prediction": {"$in": filter_options}},
                            {"overall_status": {"$in": filter_options}}
                        ]
                    }
                    # Combine queries
                    query = {"$and": [base_query, condition_query]}
                else:
                    query = base_query
                
                # Query the data
                results = list(vital_signs_collection.find(query).sort("timestamp", 1))
                
                # Convert to DataFrame
                if results:
                    df_mongo = pd.DataFrame(results)
                    # Drop MongoDB _id column
                    if '_id' in df_mongo.columns:
                        df_mongo = df_mongo.drop('_id', axis=1)
                    
                    # Display results
                    st.dataframe(df_mongo)
                    
                    # Create summary statistics
                    st.subheader("Summary Statistics")
                    
                    # Create 2 columns for stats
                    col_stats1, col_stats2 = st.columns(2)
                    
                    with col_stats1:
                        # Calculate statistics
                        if 'ppg' in df_mongo.columns:
                            st.write(f"**Heart Rate (Average):** {df_mongo['ppg'].mean():.2f} bpm")
                        if 'sbp' in df_mongo.columns:
                            st.write(f"**SBP (Average):** {df_mongo['sbp'].mean():.2f} mmHg")
                        if 'dbp' in df_mongo.columns:
                            st.write(f"**DBP (Average):** {df_mongo['dbp'].mean():.2f} mmHg")
                        if 'spo2' in df_mongo.columns:
                            st.write(f"**SpO2 (Average):** {df_mongo['spo2'].mean():.2f}%")
                    
                    with col_stats2:
                        # Count occurrences of each status
                        if 'overall_status' in df_mongo.columns:
                            status_counts = df_mongo['overall_status'].value_counts()
                            st.write("**Status Distribution:**")
                            for status, count in status_counts.items():
                                st.write(f"- {status}: {count}")
                    
                    # Display trends over time
                    st.subheader("Trends")
                    
                    # Create a plotly figure for historical data
                    fig = make_subplots(
                        rows=3, cols=1, 
                        shared_xaxes=True, 
                        subplot_titles=("Heart Rate History", "Blood Pressure History", "SpO2 History"),
                        vertical_spacing=0.1
                    )
                    
                    # Add heart rate trace
                    if 'ppg' in df_mongo.columns and 'timestamp' in df_mongo.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=df_mongo['timestamp'], 
                                y=df_mongo['ppg'],
                                mode='lines+markers',
                                name='Heart Rate',
                                line=dict(color='red')
                            ),
                            row=1, col=1
                        )
                    
                    # Add blood pressure traces
                    if 'sbp' in df_mongo.columns and 'dbp' in df_mongo.columns and 'timestamp' in df_mongo.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=df_mongo['timestamp'], 
                                y=df_mongo['sbp'],
                                mode='lines+markers',
                                name='SBP',
                                line=dict(color='blue')
                            ),
                            row=2, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=df_mongo['timestamp'], 
                                y=df_mongo['dbp'],
                                mode='lines+markers',
                                name='DBP',
                                line=dict(color='lightblue')
                            ),
                            row=2, col=1
                        )
                    
                    # Add SpO2 trace
                    if 'spo2' in df_mongo.columns and 'timestamp' in df_mongo.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=df_mongo['timestamp'], 
                                y=df_mongo['spo2'],
                                mode='lines+markers',
                                name='SpO2',
                                line=dict(color='green')
                            ),
                            row=3, col=1
                        )
                    
                    # Update layout
                    fig.update_layout(
                        height=600, 
                        margin=dict(l=0, r=0, t=30, b=0),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    # Display the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download option
                    csv = df_mongo.to_csv(index=False)
                    st.download_button(
                        label="Download Query Results",
                        data=csv,
                        file_name=f"medical_data_{start_date}_to_{end_date}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No data found for the selected date range and filters.")
            except Exception as e:
                st.error(f"Error querying database: {e}")
        
        # Display current session data
        st.subheader("Current Session Data")
        if len(st.session_state.df) > 0:
            st.dataframe(st.session_state.df)
            
            if st.button("Download Session Data"):
                csv = st.session_state.df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="medical_session_data.csv",
                    mime="text/csv"
                )
        else:
            st.info("No data recorded in this session yet. Start monitoring to collect data.")
    with tab3:
        st.header("📇 Emergency Contact Information")

        contacts_collection = db["contacts"]

        if "contacts" not in st.session_state:
            # Load contacts from MongoDB if not in session state
            st.session_state.contacts = list(contacts_collection.find())

        if "edit_index" not in st.session_state:
            st.session_state.edit_index = None

        # Add new or edit form
        st.markdown("---")
        st.subheader("➕ Add/Edit Contact")

        # Load edit values if editing
        if st.session_state.edit_index is not None:
            contact = st.session_state.contacts[st.session_state.edit_index]
            default_name = contact["name"]
            default_email = contact["email"]
            default_who = contact["who"]
            default_send = contact["send"]
            print(contact)
        else:
            default_name = ""
            default_email = ""
            default_who = "Me"
            default_send = ["Alerts", "Report"]
        

        name = st.text_input("Contact Name", value=default_name)
        email = st.text_input("Contact Email", value=default_email)
        who = st.selectbox("Who is this?", ["Me", "Doctor", "Friend/Family"], index=["Me", "Doctor", "Friend/Family"].index(default_who))
        send = st.multiselect("Send What", ["Alerts", "Report"], default=default_send)

        if st.button("Save Contact"):
            contact_data = {"name": name, "email": email, "who": who, "send": send}
            
            if st.session_state.edit_index is not None:
                # Update contact in MongoDB
                contact_id = st.session_state.contacts[st.session_state.edit_index]["_id"]
                contacts_collection.update_one({"_id": contact_id}, {"$set": contact_data})
                st.session_state.edit_index = None
            else:
                # Insert new contact in MongoDB
                contacts_collection.insert_one(contact_data)
            
            # Reload contacts from MongoDB
            st.session_state.contacts = list(contacts_collection.find())
            st.rerun()

        st.subheader("📇 Saved Contacts")

        # Display headers
        header_cols = st.columns([2, 3, 2, 3, 1, 1])
        header_cols[0].markdown("**Name**")
        header_cols[1].markdown("**Email**")
        header_cols[2].markdown("**Who**")
        header_cols[3].markdown("**Send What**")
        header_cols[4].markdown("**✏️**")
        header_cols[5].markdown("**🗑️**")

        # Display saved contacts
        for i, contact in enumerate(st.session_state.contacts):
            cols = st.columns([2, 3, 2, 3, 1, 1])
            cols[0].write(contact["name"])
            cols[1].write(contact["email"])
            cols[2].write(contact["who"])
            cols[3].write(", ".join(contact["send"]))
            
            if cols[4].button("✏️", key=f"edit_{i}"):
                st.session_state.edit_index = i
            
            if cols[5].button("🗑️", key=f"delete_{i}"):
                # Delete contact from MongoDB
                contact_id = contact["_id"]
                contacts_collection.delete_one({"_id": contact_id})
                st.session_state.contacts.pop(i)
                st.rerun()

# Add footer with disclaimer
st.markdown("---")
st.caption("⚠️ Disclaimer: This application is for demonstration purposes only. Do not use for actual medical diagnosis or treatment. Always consult with qualified healthcare professionals for medical advice.")

# Custom CSS to ensure the chatbot expander is fixed on right side
st.markdown("""
<style>
    /* Custom styles for the chatbot */
    div[data-testid="stExpander"] {
        background-color: #f8f9fa;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    
    div[data-testid="stExpander"] > details {
        background-color: #f8f9fa;
    }
    
    div[data-testid="stExpander"] > details > summary {
        font-weight: bold;
        color: #1e3a8a;
    }
    
    /* Style the chat messages */
    div[data-testid="stContainer"] {
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)