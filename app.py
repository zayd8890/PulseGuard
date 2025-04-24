import streamlit as st
from config.settings import set_page_config
from database.mongodb import get_mongodb_connection, initialize_collections
from ui.dashboard import create_dashboard

# Set page configuration
set_page_config()

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None

# Get MongoDB connection and collections
db = get_mongodb_connection()
vital_signs_collection, chat_history_collection = initialize_collections(db)

# Initialize current values
if 'current_ppg' not in st.session_state:
    st.session_state.current_ppg = 0.0
if 'current_abp' not in st.session_state:
    st.session_state.current_abp = 0.0
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = "Normal BP"
if 'current_confidence' not in st.session_state:
    st.session_state.current_confidence = 0.0
if 'show_chatbot' not in st.session_state:
    st.session_state.show_chatbot = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# App title and description
st.title("Medical Monitoring System")

# Create the main dashboard
create_dashboard(vital_signs_collection, chat_history_collection)

# Add footer with disclaimer
st.markdown("---")
st.caption("⚠️ Disclaimer: This application is for demonstration purposes only. Do not use for actual medical diagnosis or treatment. Always consult with qualified healthcare professionals for medical advice.")

# Add custom CSS
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