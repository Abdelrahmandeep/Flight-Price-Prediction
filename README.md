# Flight Price Prediction Application ✈️

🚀 **[Click here to try the Live App](https://flight-price-prediction-ajb9whapsasgukyad5efks.streamlit.app/)**

![App Screenshot](screenshot.png)

## Project Overview

This project predicts flight ticket prices based on multiple features such as airline, source, destination, number of stops, duration, and date of journey. The prediction model is built using **Random Forest Regressor** with a complete preprocessing pipeline for feature extraction and transformation.

## Features

- **Airline** — the operating carrier
- **Source** (Departure City)
- **Destination** (Arrival City)
- **Total Stops** — number of layovers
- **Duration of Flight** (hours & minutes)
- **Days Left** until departure
- **Date of Journey** (automatically transformed into day, month, year)
- **Departure Time** (hour & minute)
- **Additional Info** — in-flight services information

## Model Performance

| Metric | Train | Test |
|--------|-------|------|
| MAE    | 0.13  | 0.14 |
| RMSE   | 0.19  | 0.20 |
| R²     | 0.87  | 0.85 |
| MAPE   | 1.40% | 1.54%|

> Metrics are computed on `log1p`-transformed prices.

## Project Structure

```
├── app.py                          # Streamlit web application
├── preprocessing.py                # Shared feature-engineering module
├── retrain_model.py                # Script to retrain the model
├── flight_price_prediction.ipynb   # Notebook: EDA, model building, tuning
├── best_model.pkl                  # Trained pipeline (preprocessing + RF)
├── Data_Train.xlsx                 # Training dataset
├── tests/
│   └── test_preprocessing.py      # Unit tests for preprocessing
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version for deployment
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/abdelrahman-beep/flight-price-prediction.git
   cd flight-price-prediction
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate   # macOS / Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Retrain the Model

If you modify the preprocessing or want to rebuild the model:

```bash
python retrain_model.py
```

This produces a new `best_model.pkl` with compressed storage.

### Run Tests

```bash
python -m pytest tests/ -v
```

## Tech Stack

- **Python 3.12**
- **Streamlit** — interactive web UI
- **scikit-learn** — ML pipeline & Random Forest
- **pandas / NumPy** — data processing
- **joblib** — model serialization

## License

This project is open source and available under the [MIT License](LICENSE).

---

🚀 *Built with Streamlit & scikit-learn*
