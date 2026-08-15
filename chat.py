"""
Chat with an NVIDIA NIM model from the terminal. No server, no CORS issues.

Setup:
1. pip install requests
2. Set your key:  export NVIDIA_API_KEY="nvapi-..."
   (or just paste it into API_KEY below)
3. Run: python chat.py
"""

import os
import requests

API_KEY = os.environ.get("NVIDIA_API_KEY", "PASTE_YOUR_KEY_HERE")
BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "meta/llama-3.1-8b-instruct"  # change to whatever model you're using

messages = []

print("Chat (type 'quit' to exit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ("quit", "exit"):
        break

    messages.append({"role": "user", "content": user_input})

    response = requests.post(
        BASE_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": messages,
            "max_tokens": 1000,
        },
    )

    data = response.json()

    if response.status_code != 200:
        print("Error:", data)
        continue

    reply = data["choices"][0]["message"]["content"]
    print(f"\nAI: {reply}\n")

    # Rate limit info, if the API sends it
    remaining = response.headers.get("x-ratelimit-remaining-requests")
    limit = response.headers.get("x-ratelimit-limit-requests")
    if remaining is not None:
        print(f"[requests left: {remaining}/{limit}]\n")

    messages.append({"role": "assistant", "content": reply})
