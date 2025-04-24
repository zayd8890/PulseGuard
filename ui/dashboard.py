import streamlit as st
from ui.monitoring import create_monitoring_tab
from ui.historical import create_historical_tab
from ui.chatbot import create_chatbot
from services.llm_service import get_huggingface_client
import datetime

def create_dashboard(vital_signs_collection, chat_history_collection):
    """Create the main dashboard layout"""
    
    # Get Hugging Face client for chatbot
    hf_client = get_huggingface_client()
    
    # Top section with current values display boxes
    create_top_indicators()
    
    # Create dashboard layout with columns for main content and sidebar
    col1, col2 = st.columns([3, 1])

    # Right column with chatbot first, then patient info
    with col2:
        # CHATBOT SECTION FIRST
        create_chatbot(chat_history_collection, hf_client)
        
        # PATIENT INFO SECTION SECOND
        create_patient_info_section(vital_signs_collection)
        
    # Main dashboard in left column
    with col1:
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Real-time Monitoring", "Historical Data"])
        
        # Real-time monitoring tab
        with tab1:
            create_monitoring_tab(vital_signs_collection, chat_history_collection, hf_client)
            
        # Historical data tab
        with tab2:
            create_historical_tab(vital_signs_collection)

def create_top_indicators():
    """Create the top indicators row"""
    top_row = st.container()
    with top_row:
        # Reordered columns to put status on the left
        prediction_col, ppg_col, abp_col, title_col = st.columns([1, 1, 1, 2])
        
        # Create placeholder for status box (now first)
        with prediction_col:
            prediction_placeholder = st.empty()
            
            # Set status color based on prediction
            if st.session_state.current_prediction == "Normal BP":
                status_color = "green"
            elif st.session_state.current_prediction == "Elevated BP":
                status_color = "orange"
            elif st.session_state.current_prediction == "Hypertension Stage 1":
                status_color = "orange"
            elif st.session_state.current_prediction == "Hypertension Stage 2":
                status_color = "red"
            else:  # Hypertensive Crisis
                status_color = "red"
                
            prediction_placeholder.markdown(
                f"""
                <div style="background-color:{status_color}30;padding:10px;border-radius:10px;text-align:center;border:2px solid {status_color};">
                    <h3 style="margin:0;color:#1e3a8a;">Status</h3>
                    <h2 style="margin:5px 0;color:{status_color};">{st.session_state.current_prediction} ({st.session_state.current_confidence:.2f})</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Create placeholder for PPG box
        with ppg_col:
            ppg_placeholder = st.empty()
            ppg_placeholder.markdown(
                """
                <div style="background-color:transparent;padding:10px;border-radius:10px;text-align:center;border:1px solid rgba(255,255,255,0.3);">
                    <h3 style="margin:0;color:#1e3a8a;">PPG</h3>
                    <h2 style="margin:5px 0;color:#ff4b4b;">{:.2f} bpm</h2>
                </div>
                """.format(st.session_state.current_ppg),
                unsafe_allow_html=True
            )
        
        # Create placeholder for ABP box
        with abp_col:
            abp_placeholder = st.empty()
            abp_placeholder.markdown(
                """
                <div style="background-color:transparent;padding:10px;border-radius:10px;text-align:center;border:1px solid rgba(255,255,255,0.3);">
                    <h3 style="margin:0;color:#1e3a8a;">ABP</h3>
                    <h2 style="margin:5px 0;color:#0068c9;">{:.2f} mmHg</h2>
                </div>
                """.format(st.session_state.current_abp),
                unsafe_allow_html=True
            )
        
        with title_col:
            st.subheader("Real-time health monitoring with IoT integration")

def create_patient_info_section(vital_signs_collection):
    """Create the patient information section"""
    from database.mongodb import save_vital_signs
    
    st.markdown("---")
    st.subheader("Patient Information")
    
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
    weight = st.number_input("Weight (kg)", min_value=20, max_value=250, value=70)
    
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
            "timestamp": datetime.datetime.now()
        }
        save_vital_signs(vital_signs_collection, patient_info)
        st.success("Patient information updated!")