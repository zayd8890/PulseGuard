import streamlit as st
from database.mongodb import save_chat_message, load_chat_history
from services.llm_service import get_llm_response

def create_chatbot(chat_history_collection, hf_client):
    """Create the chatbot component"""
    # Load chat history if not already done
    if len(st.session_state.chat_history) == 0:
        st.session_state.chat_history = load_chat_history(chat_history_collection)
    
    # Expandable chatbot section
    chat_expander = st.expander("💬 Medical Assistant", expanded=st.session_state.show_chatbot)
    
    with chat_expander:
        st.subheader("Medical Assistant")
        
        # Display chat history
        chat_container = st.container(height=400)
        with chat_container:
            for chat in st.session_state.chat_history:
                if chat["role"] == "user":
                    st.write(f"👤 **You**: {chat['content']}")
                else:
                    st.write(f"🤖 **Assistant**: {chat['content']}")
        
        # Chat input
        user_input = st.text_input("Ask the medical assistant...", key="chatbot_input")
        
        if user_input:
            process_chat_input(user_input, chat_history_collection, hf_client)

def process_chat_input(user_input, chat_history_collection, hf_client):
    """Process user input and generate response"""
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Save user message to MongoDB
    save_chat_message(chat_history_collection, "user", user_input)
    
    # Get response from HuggingFace model
    response = get_llm_response(
        hf_client,
        user_input, 
        st.session_state.chat_history,
        st.session_state.current_ppg,
        st.session_state.current_abp,
        st.session_state.current_prediction,
        st.session_state.current_confidence
    )
    
    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Save assistant response to MongoDB
    save_chat_message(chat_history_collection, "assistant", response)
    
    # Rerun to update the chat display
    st.rerun()