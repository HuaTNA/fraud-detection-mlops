import requests

url = "http://localhost:5000/predict"

features = [0.1]* 28 + [100]

payload = {
    "features": features
}

response = requests.post(url, json=payload)

if response.ok:
    print("Prediction:", response.json())
else:
    print("Error:", response.status_code, response.text)