import requests

url = "http://127.0.0.1:8000"

# Check that the API is running
response = requests.get(f"{url}/health")
print("Health:", response.status_code, response.json())

# Client 1: low consumption and low seniority
client1 = {
    "age": 35,
    "salaire": 2500,
    "abonnement": "Standard",
    "anciennete": 1,
    "consommation": 10
}

response = requests.post(f"{url}/predict", json=client1) 

print("Client 1:", response.status_code, response.json())

# Client 2: higher consumption and seniority
client2 = {
    "age": 50,
    "salaire": 4000,
    "abonnement": "Premium",
    "anciennete": 10,
    "consommation": 80
}

response = requests.post(f"{url}/predict", json=client2)
print("Client 2:", response.status_code, response.json())

# Intentionally send an invalid age
invalid_client = client1.copy()
invalid_client["age"] = "abc"


response = requests.post(f"{url}/predict", json=invalid_client)
print("Invalid age:", response.status_code)