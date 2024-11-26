import requests
import json

url = "http://localhost:1337/v1/chat/completions"

headers = {
    "Content-Type": "application/json"
}

payload = {
    "messages": [
        {"role": "system", "content": "System prompt goes here"},
        {"role": "user", "content": "Give me a meal plan for today"}
    ],
    "model": "mistral-ins-7b-q4",
    "stream": False,
    "max_tokens": 2048,
    "stop": None,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "temperature": 0.7,
    "top_p": 0.95
}

response = requests.post(url, headers=headers, data=json.dumps(payload))

result = response.json()
content = result['choices'][0]['message']['content']
print(content)