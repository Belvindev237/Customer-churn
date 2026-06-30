import os
import requests

API_URL= os.getenv("API_URL","http://api:8000")

def predict_churn(input_data:dict)->dict:
  response=requests.post(f"{API_URL}/predict",json=input_data)

  response.raise_for_status()
  return response.json()

def explain_churn(input_data:dict)->dict:
  response=requests.post(f"{API_URL}/explain",json=input_data)
  response.raise_for_status()
  return response.json()