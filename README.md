# Flight Price Prediction Application ✈️

🚀 **[Click here to try the Live App](https://flight-price-prediction-34pkt7d67nh5jckymzmbak.streamlit.app/)**  
*(Alternative mirror: [Live App Mirror](https://flight-price-prediction-ajb9whapsasgukyad5efks.streamlit.app/))*

---

## 📌 Project Overview

This project provides an end-to-end Machine Learning solution to predict flight ticket prices. It features an end-to-end data processing pipeline, a tuned **Random Forest Regressor**, and a modern interactive **Streamlit Dashboard** equipped with real-time analytics, Lottie animations, and report export capabilities.

---

## 🚀 Key Features & New Additions

### 1. 🤖 Machine Learning Pipeline & Dual-Scale Evaluation
- **Feature Engineering & Preprocessing**:
  - `Date_of_Journey` extracted into Day, Month, and Year.
  - `Dep_Time` converted into departure hour and minute.
  - `Duration` normalized to total flight minutes.
  - `Total_Stops` mapped to numerical categories.
  - One-Hot Encoding for categorical features (`Airline`, `Source`, `Destination`, `Additional_Info`).
  - Standard scaling for numerical features.
- **Log Transformation & Original-Scale Evaluation [New 🆕]**:
  - The model is trained on log-transformed prices (`np.log1p`) to handle skewness.
  - **Comprehensive Evaluation**: In addition to standard log-scale metrics, the project now includes an **Original Price Scale Evaluation** using `np.expm1` to measure actual monetary errors and real percentage accuracy.

### 2. 🎨 Enhanced Arabic Interactive Web App (`app.py` / `add.py`) [New 🆕]
- **Modern UI**: Styled with Cairo font, card layouts, and subtle shadows.
- **Lottie Animations**: Fluid animated visuals for flight booking and analytics.
- **Three Core Tabs**:
  1. 🎫 **توقع فردي (Single Prediction)**: Interactive input for airlines, cities, dates, stops, and departure times with instant price prediction cards.
  2. 📊 **لوحة المعلومات والتحليلات (Analytics Dashboard)**: Real-time interactive charts powered by **Plotly** (Airlines average price, destination distributions, duration vs. price scatter plot).
  3. 📥 **تصدير السجل (Excel Export)**: One-click export of generated prediction sessions into an Excel spreadsheet.
  4. ℹ️ **معلومات النظام (System Info)**: Overview of model specifications and technical stack.

---

## 📊 Model Performance

### 1. Original Price Scale (Actual Currency / Real-World Evaluation) 🆕
> Evaluated by applying `np.expm1` back to the raw prices:

| Metric | Test Set | Train Set | Meaning in Practice |
|---|:---:|:---:|---|
| **MAE** | **1,293.92** | **1,194.20** | Average prediction error is only ~1,294 currency units |
| **RMSE** | **2,117.62** | **2,233.83** | Penalizes larger outlier errors |
| **R² Score** | **0.7693** | **0.7677** | Explains ~77% of real price variability |
| **MAPE** | **13.83%** | **12.61%** | Average relative percentage error is ~13.8% (**~86.2% accuracy**) |

*Note: The close alignment between Train and Test scores demonstrates that the model generalizes well and does not suffer from overfitting.*

### 2. Log-Transformed Scale (`log1p`)
> Metrics calculated directly on the log-transformed target:

| Metric | Test Set | Train Set |
|---|:---:|:---:|
| **MAE** | 0.14 | 0.13 |
| **RMSE** | 0.20 | 0.19 |
| **R² Score** | 0.8501 | 0.8685 |
| **MAPE** | 1.54% | 1.40% |

---

## 📁 Project Structure

```
├── app.py                          # Streamlit web application (Main entrypoint)
├── add.py                          # Interactive dashboard alias
├── preprocessing.py                # Shared feature engineering & preprocessing
├── retrain_model.py                # Script to retrain and compress the model
├── flight_price_prediction.ipynb   # Complete notebook: EDA, tuning, & dual-scale evaluation
├── best_model.pkl                  # Trained Scikit-Learn pipeline
├── Data_Train.xlsx                 # Flight dataset
├── requirements.txt                # Project dependencies
├── runtime.txt                     # Python runtime specification
├── .gitignore                      # Git ignore rules
└── README.md                       # Comprehensive project documentation
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Abdelrahmandeep/Flight-Price-Prediction.git
   cd Flight-Price-Prediction
   ```

2. **Create and activate a virtual environment:**
   ```powershell
   # Windows
   python -m venv .venv
   .\.venv\Scripts\activate

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### Run the Interactive Streamlit App
```bash
streamlit run app.py
```
*The app will automatically launch in your browser at `http://localhost:8501`.*

### Explore the Jupyter Notebook
Open and run [flight_price_prediction.ipynb](flight_price_prediction.ipynb) to inspect EDA, cross-validation, and the newly added original-scale evaluation cell.

### Retrain the Model
```bash
python retrain_model.py
```

---

## 🧰 Tech Stack
- **Python 3.12+**
- **Machine Learning**: `scikit-learn`, `RandomForestRegressor`, `joblib`
- **Data Manipulation**: `pandas`, `numpy`, `openpyxl`
- **Visualization & UI**: `streamlit`, `plotly`, `streamlit-lottie`, `matplotlib`

---

## 📄 License
This project is open source and available under the [MIT License](LICENSE).
