# Customer Churn Prediction

End-to-end machine learning project to predict customer churn using a trained classifier, a reproducible preprocessing pipeline, and a FastAPI deployment for real-time inference through a web form or JSON API.

## Overview

This project predicts whether a customer is likely to churn based on customer profile, contract, billing, and service usage data. The full pipeline includes data preprocessing, feature engineering, model training, serialization of artifacts, and deployment as a production-style API.

The application exposes a FastAPI backend that receives customer data, applies the saved preprocessing steps, runs the classifier, and returns a churn probability and binary prediction. A lightweight Streamlit interface can also be used to submit customer data and display the result in a user-friendly way. Streamlit is used as the frontend, while FastAPI handles the backend inference service. [web:99][web:97]

## Features

- Full preprocessing pipeline aligned with training-time transformations.
- Trained classification model for churn prediction.
- Real-time inference through FastAPI.
- Input validation with Pydantic schemas.
- Swagger/OpenAPI documentation generated automatically by FastAPI.
- Web form or JSON request support for predictions.
- Dockerized deployment with Docker Compose.
- Separate frontend and backend services for cleaner architecture. [web:97][web:98]

## Tech Stack

- Python
- FastAPI
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Uvicorn
- Docker
- Docker Compose

## Project Structure

```text
Churn_project/
├── api/
│   ├── main.py
│   ├── preprocessing.py
│   ├── config.py
│   ├── schemas.py
│   ├── requirements.txt
│   ├── Dockerfile.api
│   └── models/
│       ├── model.pkl
│       ├── scaler.pkl
│       ├── ordinal_encoder.pkl
│       └── feature_names.json
├── ui/
│   ├── api_ui.py
│   ├── requirements.txt
│   └── Dockerfile.ui
├── docker-compose.yml
└── README.md
```

## How It Works

1. The user fills in a web form or sends a JSON payload to the API.
2. FastAPI validates the input with a Pydantic schema.
3. The preprocessing pipeline applies the same transformations used during training.
4. The trained model loads the processed features and computes churn probability.
5. The API returns both the probability and the final class prediction.

## Preprocessing Pipeline

The preprocessing pipeline reproduces the training logic exactly and includes:

- Binary encoding for selected yes/no fields.
- Feature engineering such as `NbServices` and `TenureSegment`.
- Ordinal encoding of tenure segments.
- One-hot encoding of categorical variables.
- Standard scaling for numeric features.
- Strict alignment of columns with the trained model feature list.

This design ensures that inference-time inputs are transformed in the same way as training-time data, which is essential for stable model performance.

## API Endpoints

### `GET /`
Returns a simple message confirming that the API is running.

### `GET /health`
Returns information about the model status and loaded artifacts.

### `POST /predict`
Accepts customer data and returns:

- `churn_proba`
- `churn_prediction`
- `label`
- `seuil_utilise`

Example response:

```json
{
  "churn_proba": 0.7421,
  "churn_prediction": 1,
  "label": "Churner ⚠️",
  "seuil_utilise": 0.35
}
```

## Input Schema

The API expects structured customer data including:

- demographics,
- contract type,
- billing preferences,
- internet and phone services,
- usage-related fields,
- tenure and charges.

Validation is handled by FastAPI and Pydantic before the payload reaches the model.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/belvindev237/Churn_project.git
cd Churn_project
```

### 2. Run with Docker

```bash
docker compose up --build
```

### 3. Access the services

- FastAPI API: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- Streamlit UI: `http://localhost:8501`

## Local Development

### API

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload
```

### UI

```bash
cd ui
pip install -r requirements.txt
streamlit run api_ui.py
```

## Model and Artifacts

The deployed API uses saved artifacts from training:

- `model.pkl`: trained classifier
- `scaler.pkl`: fitted scaler
- `ordinal_encoder.pkl`: fitted ordinal encoder
- `feature_names.json`: expected feature columns

These artifacts are loaded at startup and reused for inference.

## Example Payload

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 5,
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.5,
  "TotalCharges": 352.5,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No"
}
```

## Possible Improvements

- Add automated tests for the preprocessing and API endpoints.
- Add logging and monitoring for prediction requests.
- Add model versioning with MLflow or a similar registry.
- Add explainability with SHAP for customer-level interpretation.
- Add CI/CD for automated build and deployment.

## Contributing

Contributions are welcome. You can improve the preprocessing, add new features, refine the interface, or expand the deployment setup.

## License

Add your preferred license here, such as MIT.