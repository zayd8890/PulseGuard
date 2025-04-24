from huggingface_hub import InferenceClient

# Configure Hugging Face Client
def get_huggingface_client():
    """Get HuggingFace inference client"""
    client = InferenceClient(
        provider="sambanova",
        api_key="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",
    )
    return client

# Function to generate response using Hugging Face QwQ-32B model
def get_llm_response(client, prompt, chat_history=None, current_ppg=0, current_abp=0, 
                     current_prediction="Normal BP", current_confidence=0):
    """Generate response using LLM"""
    # Create a consolidated context from chat history
    formatted_messages = []
    
    # First add the system message with medical context
    system_message = f"""
    You are a medical assistant chatbot helping monitor a patient's vital signs.
    The patient's most recent data shows:
    - PPG: {current_ppg:.2f} bpm
    - ABP: {current_abp:.2f} mmHg
    - Current Status: {current_prediction}
    - Confidence: {current_confidence:.2f}

    Provide a helpful, professional response. If you detect any medical emergency,
    advise the user to seek immediate medical attention. Don't provide specific 
    medication advice but offer general health guidance within your scope.
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
        # Call the QwQ-32B model
        completion = client.chat.completions.create(
            model="Qwen/QwQ-32B",
            messages=formatted_messages,
            max_tokens=512,
        )
        
        # Extract the response text
        response_text = completion.choices[0].message.content
        return response_text
        
    except Exception as e:
        return f"I'm having trouble connecting to my knowledge base. Please try again later. Error: {e}"