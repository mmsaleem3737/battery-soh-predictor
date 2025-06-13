# ProSoH: Professional Battery State of Health (SoH) Predictor 🔋

This repository contains a complete, end-to-end system for predicting the State of Health (SoH) of Lithium-ion batteries. The project features a highly accurate XGBoost machine learning model (R² > 0.997) wrapped in a user-friendly Streamlit web application.

The application allows non-technical users to get instant SoH predictions by simply uploading a raw data log from a battery cycler, making advanced data science accessible to a wider audience.




---

## ✨ Key Features

- **State-of-the-Art Model:** Utilizes a fine-tuned XGBoost Regressor to achieve exceptionally high prediction accuracy.
- **User-Friendly Interface:** A multi-tab Streamlit application that is intuitive and requires no technical knowledge to operate.
- **Automatic Feature Engineering:** The app automatically processes raw time-series data to calculate the 9 complex features required by the model. The user only needs to provide a data file and a cycle number.
- **Informative & Transparent:** Includes detailed sections on model performance, the training dataset, and clear instructions for use.
- **Theme-Aware UI:** The interface is designed to work flawlessly in both light and dark modes.

---

## 📈 Model Details

The predictive power of this application comes from a robust machine learning model and a high-quality dataset.

- **Algorithm:** XGBoost Regressor
- **Performance Metrics:**
  - **R-squared (R²):** `0.9979` (Indicates the model explains 99.79% of the variance in the data)
  - **Mean Absolute Error (MAE):** `0.00187 Ah` (The predictions are, on average, off by only 0.00187 Ah)
- **Training Data:**
  - The model was trained on the renowned **CALCE Battery Research Group dataset**.
  - **Battery Type:** CX2-34 Lithium Cobalt Oxide (LCO) prismatic cells.
  - **Dataset Size:** The cleaned dataset contains nearly 400,000 measurements across more than 1,700 charge/discharge cycles.

---

## 🚀 How to Run the Application

To run the predictor on your local machine, please follow these steps.

### 1. Prerequisites

- Python 3.8 - 3.11
- A way to clone this repository (e.g., Git)

### 2. Setup

**Step A: Clone the Repository**
```bash
git clone <your-repository-url>
cd <repository-folder>
```

**Step B: Create a Virtual Environment (Recommended)**
It's highly recommended to use a virtual environment to avoid conflicts with other projects.
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**Step C: Install Required Libraries**
This repository includes a `requirements.txt` file that lists all necessary libraries. Install them with pip:
```bash
pip install -r requirements.txt
```

### 3. Launch the App

Once the setup is complete, you can launch the Streamlit application with a single command:
```bash
streamlit run app.py
```
This command will start the web server and open the application in your default web browser.

---

## 📁 File Structure

- `app.py`: The main Python script for the Streamlit web application.
- `xgboost_battery_soh_model.pkl`: The pre-trained, optimized XGBoost model file.
- `scaler.pkl`: The pre-fitted StandardScaler object for preprocessing data.
- `requirements.txt`: A list of all Python libraries required to run the project.
- `generate_synthetic_data.py`: A utility script to generate a sample CSV file for testing.
- `README.md`: This file.

---

## 📜 Disclaimer

This application and its predictions are provided for academic, research, and informational purposes only. It is not intended to be a substitute for professional engineering advice or certified testing. The developers assume no liability for the accuracy or reliability of the predictions. By using this application, you acknowledge and agree to these terms.
