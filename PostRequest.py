import requests

payload = {
    "github": "https://github.com/Ila-22/AI-Model",
    "email":  "ila.amini@outlook.com",
    "url":    "https://forecast11.herokuapp.com/forecast",
    "notes":  "Deployed with FastAPI on Heroku free tier"
}

resp = requests.post(
    "https://dps-challenge.netlify.app/.netlify/functions/api/challenge",
    json=payload,                # sets Content-Type: application/json
    timeout=10
)

print(resp.status_code)
print(resp.text)