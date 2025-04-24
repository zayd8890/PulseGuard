from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="sambanova",
    api_key="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",
)

completion = client.chat.completions.create(
    model="Qwen/QwQ-32B",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    max_tokens=512,
)

print(completion.choices[0].message)