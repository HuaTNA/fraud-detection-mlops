import requests

url = "http://127.0.0.1:8000/predict"
features = [0.1] * 28 + [100]
response = requests.post(url, json={"features": features})

print(response.json())
