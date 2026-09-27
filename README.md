# 🏠 House Price Predictor

A Streamlit machine-learning application that predicts house prices across Indian cities and helps users find budget-matched properties and compare average prices by BHK.

## 🚀 Live Demo

Deploy this repository to your own Streamlit Community Cloud account and add the resulting URL here. The previous developer's deployment URL is intentionally not retained.

## ✨ Features

- **Multi-model evaluation** — compares Linear Regression, Random Forest, and Gradient Boosting using R² scores.
- **Automatic model selection** — uses the highest-scoring saved model for predictions.
- **Budget-based house search** — select a city and budget to see matching localities and predicted prices.
- **City comparison** — compare average prices for a selected BHK between two cities.
- **Model performance dashboard** — displays saved R² scores and a comparison chart.

## 🛠️ Tech Stack

- Python
- Streamlit 1.47.0
- pandas 2.3.1
- scikit-learn 1.7.0
- NumPy 2.3.1
- matplotlib 3.10.3
- joblib 1.5.1

## 📁 Project Structure

```text
house-price-predictor/
├── app.py
├── house_price_model.pkl
├── city_encoder.pkl
├── locality_encoder.pkl
├── features.pkl
├── r2_scores.pkl
├── house_price_train_dataset.csv
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

The saved model and encoders are included so the Streamlit app can run without retraining at startup.

## 🚀 Run Locally

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/house-price-predictor.git
cd house-price-predictor
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit, normally `http://localhost:8501`.

## ☁️ Streamlit Community Cloud

1. Push this project to your GitHub repository.
2. Open Streamlit Community Cloud and sign in with GitHub.
3. Create a new app from your repository.
4. Select the `main` branch.
5. Set the main file to `app.py`.
6. Deploy.

No database or API keys are required by the current application.

## 📊 Dataset

The included CSV contains 800 records with these fields:

| Column | Description |
|---|---|
| `City` | City name |
| `Locality` | Locality/area within the city |
| `Distance_km` | Distance from city centre in km |
| `Area_sqft` | Property area in square feet |
| `BHK` | Number of bedrooms |
| `Land_Price_per_sqft` | Land price per square foot |
| `Total_Price_INR` | Target house price in INR |

The application normalizes `Distance_km` to `Distance_from_CityCentre_km` internally because that is the feature name used by the saved model.

## 🧠 How It Works

1. `City` and `Locality` are transformed using the saved label encoders.
2. The saved regression models are evaluated using their stored R² scores.
3. The best saved model is selected automatically.
4. For a selected city and budget, the app predicts prices for the available properties and filters results around the selected budget.
5. The app returns up to 15 reproducible sample listings.

## ⚠️ Model Compatibility

The serialized models were created with **scikit-learn 1.7.0**. Keep the pinned dependency versions in `requirements.txt` unless you retrain and re-export the models with a newer scikit-learn version.

## 👤 Author

**Aman Gupta**

## 📄 License

MIT License. See `LICENSE`.
