import requests
# import encoded threat and history here
threat = ""
resp = requests.post(
    "https://model-vq09kr2q.api.baseten.co/environments/production/predict",
    headers={"Authorization": "Api-Key 4ERqMtQ8.16dodkK3008VFCR6m4UfDsyQVBrK4DkLC"},
    json={"messages": [{"role": "user", f"content": "{threat}"}]}
)

print(resp.json())
