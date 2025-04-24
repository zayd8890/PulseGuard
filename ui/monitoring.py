import streamlit as st 
import datetime
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from models.bp_model import load_bp_model, predict, CLASS_NAMES
from utils.data_simulation import get_iot_data, create_vital_signs_record, add_to_dataframe
from database.mongodb import save_vital_signs, save_chat_message
from services.llm_service import get_llm_response

def create_monitoring_tab(vital_signs_collection, chat_history_collection, hf_client):
    """Create the real-time monitoring tab"""
    st.subheader("Vital Signs Monitor")
    
    # Create placeholders for real-time charts
    chart_placeholder = st.empty()
    
    # Load BP model
    bp_model = load_bp_model()
    
    # Start monitoring button
    start_monitoring = st.checkbox("Start Monitoring", value=False)
    
    if start_monitoring:
        # Create history for plotting
        if 'ppg_history' not in st.session_state:
            st.session_state.ppg_history = []
            st.session_state.abp_history = []
            st.session_state.time_history = []
        
        # Number of points to show
        window_size = 100
        
        while start_monitoring:
            # Get new data point
            ppg, abp = get_iot_data()
            
            # Update current values in session state
            st.session_state.current_ppg = ppg
            st.session_state.current_abp = abp
            
            # Update top indicators
            update_indicators(ppg, abp)
            
            # Get current time
            now = datetime.datetime.now()
            
            # Append to history
            st.session_state.ppg_history.append(ppg)
            st.session_state.abp_history.append(abp)
            st.session_state.time_history.append(now)
            
            # Keep only the last window_size points
            if len(st.session_state.ppg_history) > window_size:
                st.session_state.ppg_history = st.session_state.ppg_history[-window_size:]
                st.session_state.abp_history = st.session_state.abp_history[-window_size:]
                st.session_state.time_history = st.session_state.time_history[-window_size:]
            
            # Create a plotly figure
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                subplot_titles=("PPG Signal", "ABP Signal"),
                                vertical_spacing=0.1, 
                                row_heights=[0.5, 0.5])
            
            # Add traces for PPG
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(st.session_state.ppg_history))), 
                    y=st.session_state.ppg_history,
                    mode='lines',
                    name='PPG',
                    line=dict(color='red')
                ),
                row=1, col=1
            )
            
            # Add traces for ABP
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(st.session_state.abp_history))), 
                    y=st.session_state.abp_history,
                    mode='lines',
                    name='ABP',
                    line=dict(color='blue')
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=400, 
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Remove x-axis labels except for the bottom plot
            fig.update_xaxes(showticklabels=False, row=1, col=1)
            
            # Display the chart
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Get patient information from session state
            age = st.session_state.get('age', 30)
            gender = st.session_state.get('gender', 'Male')
            height = st.session_state.get('height', 170)
            weight = st.session_state.get('weight', 70)
            bmi = weight / ((height/100) ** 2)
            
            # Create a data row for prediction
            # Gender encoding: Male=0, Female=1, Other=2
            gender_code = {"Male": 77, "Female": 70, "Other": 2}[gender]
            data_row = [ppg, abp, age, gender_code, height]
            
            # Make prediction
            prediction, confidence = predict(bp_model, data_row)
            prediction_class = CLASS_NAMES[prediction]
            
            # Update session state with prediction
            st.session_state.current_prediction = prediction_class
            st.session_state.current_confidence = confidence
            
            # Update prediction indicator
            update_prediction_indicator(prediction_class, confidence)
            
            # Auto-open chatbot if crisis detected (Hypertensive Crisis)
            handle_crisis_detection(prediction, confidence, ppg, abp, age, gender, chat_history_collection, hf_client)
            
            # Add to dataframe
            st.session_state.df = add_to_dataframe(
                st.session_state.df, now, ppg, abp, age, gender, 
                height, weight, bmi, prediction_class, confidence
            )

            # Save to MongoDB
            vital_signs_data = create_vital_signs_record(
                ppg, abp, age, gender, height, weight, bmi, prediction_class, confidence
            )
            save_vital_signs(vital_signs_collection, vital_signs_data)
            
            # Add some delay to simulate real-time data
            time.sleep(0.5)

def update_indicators(ppg, abp):
    """Update the top indicators with current values"""
    # This is just a placeholder function
    # In the actual app, we would update the indicators with new values
    pass

def update_prediction_indicator(prediction_class, confidence):
    """Update the prediction indicator with current values"""
    # This is just a placeholder function
    # In the actual app, we would update the prediction indicator with new values
    pass

def handle_crisis_detection(prediction, confidence, ppg, abp, age, gender, chat_history_collection, hf_client):
    """Handle crisis detection and generate alerts"""
    # Auto-open chatbot if crisis detected (Hypertensive Crisis is index 3)
    if prediction == 3 and confidence > 0.7:
        if not st.session_state.show_chatbot:
            st.session_state.show_chatbot = True
            # Generate alert message using Hugging Face model
            alert_prompt = f"""
            The patient's blood pressure monitoring system has detected a Hypertensive Crisis with confidence {confidence:.2f}.
            Current readings:
            - PPG: {ppg:.2f} bpm
            - ABP: {abp:.2f} mmHg
            - Age: {age}
            - Gender: {gender}
            
            Generate an urgent but calm medical alert message about this situation that provides immediate guidance. 
            Do not provide specific medication advice but focus on immediate actions the patient should take.
            """
            
            # Get alert response from LLM
            alert_response = get_llm_response(
                hf_client, 
                alert_prompt, 
                [], 
                ppg, 
                abp, 
                "Hypertensive Crisis", 
                confidence
            )
            
            # Add alert to chat history in session state
            st.session_state.chat_history.append({"role": "assistant", "content": f"❗ ALERT: {alert_response}"})
            
            # Save alert to MongoDB
            save_chat_message(chat_history_collection, "assistant", f"❗ ALERT: {alert_response}")
            
            # Rerun to update UI
            st.rerun()