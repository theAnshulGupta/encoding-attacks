import requests

resp = requests.post(
    "https://model-vq09kr2q.api.baseten.co/environments/production/predict",
    headers={"Authorization": "Api-Key 4ERqMtQ8.16dodkK3008VFCR6m4UfDsyQVBrK4DkLC"},
    json={"messages": [{"role": "user", "content": "Hello"}]}
)

print(resp.json())
