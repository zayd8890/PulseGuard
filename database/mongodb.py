import datetime
import streamlit as st
from pymongo import MongoClient

# MongoDB Connection
# Initialize MongoDB connection
def get_mongodb_connection():
    """Get MongoDB connection"""
    # Replace with your MongoDB connection string
    connection_string = "mongodb+srv://zayd88903:zayd202020@cluster0.mwmxpcy.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(connection_string)
    db = client["medical_monitoring"]
    return db

# Initialize MongoDB collections
def initialize_collections(db):
    """Initialize MongoDB collections and indexes"""
    # Create collections if they don't exist
    if "vital_signs" not in db.list_collection_names():
        db.create_collection("vital_signs")
    
    if "chat_history" not in db.list_collection_names():
        db.create_collection("chat_history")
    
    # Create indexes
    db.vital_signs.create_index("timestamp")
    db.chat_history.create_index("timestamp")
    
    return db.vital_signs, db.chat_history

# Load chat history from MongoDB
def load_chat_history(chat_history_collection):
    """Load chat history from MongoDB"""
    try:
        # Sort by timestamp to get history in chronological order
        chat_documents = chat_history_collection.find().sort("timestamp", 1)
        chat_history = []
        
        for doc in chat_documents:
            # Convert MongoDB document to chat format
            chat_history.append({
                "role": doc["role"],
                "content": doc["content"]
            })
        
        return chat_history
    except Exception as e:
        st.error(f"Error loading chat history: {e}")
        return []

# Save chat message to MongoDB
def save_chat_message(chat_history_collection, role, content):
    """Save chat message to MongoDB"""
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
def save_vital_signs(vital_signs_collection, data_dict):
    """Save vital signs data to MongoDB"""
    try:
        # Add timestamp if not already present
        if "timestamp" not in data_dict:
            data_dict["timestamp"] = datetime.datetime.now()
            
        # Insert the document
        vital_signs_collection.insert_one(data_dict)
    except Exception as e:
        st.error(f"Error saving vital signs: {e}")