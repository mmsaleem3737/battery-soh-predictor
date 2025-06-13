import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Core Application Setup ---
st.set_page_config(
    page_title="ProSoH | Professional Battery SoH Predictor",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Asset Loading ---
@st.cache_resource
def load_assets():
    """Loads and caches the pre-trained model and scaler for performance."""
    try:
        model = joblib.load('xgboost_battery_soh_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_assets()

# --- 3. Feature Engineering Logic ---
def engineer_features(df, cycle_index):
    """Calculates the 9 required model features from raw data for a given cycle."""
    cycle_df = df[df['Cycle_Index'] == cycle_index]
    
    if cycle_df.empty:
        return None, f"Error: No data found for Cycle Index {cycle_index} in the uploaded file."

    if len(cycle_df) < 2:
        return None, f"Error: Not enough data points ({len(cycle_df)}) for Cycle Index {cycle_index} to calculate reliable statistics. At least 2 points are needed."

    # Define the statistics to calculate.
    agg_funcs = {
        'Voltage (V)': ['mean', 'std', 'min', 'max'],
        'Current (A)': ['mean', 'std', 'min', 'max'],
    }
    
    # This produces a DataFrame with stats as rows and measures as columns
    summary_vertical = cycle_df.agg(agg_funcs)

    # 1. Unstack the DataFrame to turn it into a Series with a MultiIndex.
    # 2. Convert this Series back to a single-row DataFrame.
    # 3. Transpose it to get features as columns.
    cycle_summary = summary_vertical.unstack().to_frame().T
    
    # Flatten the MultiIndex columns from ('Voltage (V)', 'mean') to 'Voltage (V)_mean'
    cycle_summary.columns = ['_'.join(col).strip() for col in cycle_summary.columns.values]
    
    # Fill any NaN values that might occur (e.g., std dev of a single point) with 0
    cycle_summary.fillna(0, inplace=True)

    # Add the Cycle_Index itself as a feature
    cycle_summary['Cycle_Index'] = cycle_index
    
    return cycle_summary, None

# --- 4. Model & Data Constants ---
FEATURE_ORDER = [
    'Cycle_Index',
    'Voltage (V)_mean', 'Voltage (V)_std', 'Voltage (V)_min', 'Voltage (V)_max',
    'Current (A)_mean', 'Current (A)_std', 'Current (A)_min', 'Current (A)_max'
]
INITIAL_CAPACITY = 0.62 

# --- 5. UI Styling ---
st.markdown("""
<style>
    /* General Styling */
    /* Let Streamlit manage the app background for theme consistency */

    /* Main Content */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 5px;
        width: 100%;
        height: 3em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 6. Application Header ---
st.title("ProSoH: Professional Battery Health Predictor")
st.markdown("A sophisticated tool for accurately predicting the State of Health (SoH) of Lithium-ion batteries.")

# Check if model assets are loaded correctly
if not model or not scaler:
    st.error("🔴 Critical Error: Model assets ('xgboost_battery_soh_model.pkl', 'scaler.pkl') not found. The application cannot proceed.")
    st.stop()

# --- 7. Tabbed Interface ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Predictor", "📈 About the Model", "📖 How to Use", "📜 Terms & Conditions"])

# == TAB 1: PREDICTOR ==
with tab1:
    st.header("Predict Battery Health from Data")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("1. Upload Data File")
        uploaded_file = st.file_uploader(
            "Upload the raw time-series CSV file for your battery.",
            type="csv",
            help="The file must contain 'Cycle_Index', 'Voltage (V)', and 'Current (A)' columns."
        )
        
        if uploaded_file:
            st.subheader("2. Select Cycle")
            try:
                raw_df = pd.read_csv(uploaded_file)
                available_cycles = sorted(raw_df['Cycle_Index'].unique())
                target_cycle = st.selectbox(
                    "Choose the cycle to analyze:",
                    options=available_cycles
                )
                
                st.subheader("3. Get Prediction")
                if st.button("⚡ Run Prediction", type="primary"):
                    st.session_state.prediction_made = True
                    st.session_state.raw_df = raw_df
                    st.session_state.target_cycle = target_cycle
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.session_state.prediction_made = False

    with col2:
        if 'prediction_made' in st.session_state and st.session_state.prediction_made:
            with st.spinner('Analyzing cycle data and running model...'):
                feature_df, error = engineer_features(st.session_state.raw_df, st.session_state.target_cycle)
                
                if error:
                    st.error(f"An error occurred: {error}")
                else:
                    feature_df = feature_df[FEATURE_ORDER]
                    scaled_features = scaler.transform(feature_df)
                    predicted_capacity = model.predict(scaled_features)[0]
                    soh_percentage = (predicted_capacity / INITIAL_CAPACITY) * 100
                    soh_percentage = max(0, min(100, soh_percentage))

                    st.metric(label="Predicted Capacity", value=f"{predicted_capacity:.3f} Ah")
                    st.metric(label="State of Health (SoH)", value=f"{soh_percentage:.1f}%")
                    
                    with st.expander("View Calculated Features (for technical users)"):
                        st.write("The model's prediction was based on these 9 features, which were automatically calculated from your file:")
                        st.dataframe(feature_df)
        else:
            st.info("Upload a file and select a cycle to see the prediction here.")

# == TAB 2: ABOUT THE MODEL ==
with tab2:
    st.header("Model Performance & Training Data")
    st.markdown("""
    This prediction is powered by an XGBoost Regressor model, a powerful gradient boosting algorithm known for its high accuracy. 
    The model was rigorously trained, tuned, and evaluated to ensure its reliability.
    """)
    
    st.subheader("Performance Metrics")
    metric1, metric2 = st.columns(2)
    metric1.metric(label="R-squared (R²)", value="0.9979", delta="State-of-the-Art Accuracy", delta_color="off")
    metric2.metric(label="Mean Absolute Error (MAE)", value="0.00187 Ah", delta="Highly Precise", delta_color="off")
    
    st.subheader("Training Dataset")
    st.markdown("""
    The model was trained on the renowned **CALCE Battery Research Group dataset**.
    - **Battery Type:** CX2-34 Lithium Cobalt Oxide (LCO) prismatic cells.
    - **Conditions:** Cycled at 25°C with a 0.5C charge/discharge rate.
    - **Data Size:** The cleaned dataset contains nearly 400,000 individual measurements across more than 1,700 cycles.
    
    This extensive dataset allows the model to learn the subtle nuances of battery degradation over a long lifespan.
    """)
    
    st.subheader("Most Important Predictive Feature")
    st.markdown("""
    Through advanced model interpretation (SHAP analysis), we identified that the **Minimum Current (`Current (A)_min`)** during a cycle is the single most powerful predictor of a battery's health. This physically corresponds to the battery's ability to handle a full discharge, which weakens as it ages.
    """)

# == TAB 3: HOW TO USE ==
with tab3:
    st.header("Step-by-Step Guide")
    st.markdown("""
    To use this tool, you need a specific type of data file. Here's how to get your prediction:

    **1. Obtain Your Data Log:**
       - You must have a system that can record your battery's `Voltage (V)` and `Current (A)` over at least one full charge-and-discharge cycle.
       - This is typically done with a battery cycler, a data logger, or a custom sensor setup.

    **2. Format Your Data as a CSV File:**
       - Save the data log as a Comma-Separated Values (`.csv`) file.
       - **Crucially**, your file must contain the following columns with these exact names:
         - `Cycle_Index`: The number of the cycle (e.g., 1, 2, 50, 150).
         - `Voltage (V)`: The battery's voltage at a point in time.
         - `Current (A)`: The battery's current at a point in time (positive for charge, negative for discharge).

    **3. Use the Predictor:**
       - Go to the **📊 Predictor** tab.
       - Click the "Browse files" button and upload your CSV file.
       - Once uploaded, select the specific `Cycle_Index` you wish to analyze from the dropdown menu.
       - Click the **⚡ Run Prediction** button. Your results will appear instantly.
    """)

# == TAB 4: TERMS & CONDITIONS ==
with tab4:
    st.header("Disclaimer and Terms of Use")
    st.warning("Please read the following terms carefully before using this application.")
    st.markdown("""
    - **For Informational Purposes Only:** This application and its predictions are provided for academic, research, and informational purposes only. It is not intended to be a substitute for professional engineering advice or certified testing.
    
    - **No Guarantees:** The predictions are based on a machine learning model trained on a specific type of battery under specific lab conditions. While the model is highly accurate for data similar to its training set, its performance on different battery chemistries, formats, or operating conditions is not guaranteed. The developers assume no liability for the accuracy or reliability of the predictions.
    
    - **Not for Critical Applications:** Do not use this application for any safety-critical, mission-critical, or commercial purposes where an inaccurate prediction could lead to financial loss, equipment damage, or harm to individuals.
    
    - **Data Privacy:** The application processes the uploaded file in memory to generate a prediction. No uploaded data is saved or stored on any server.
    
    By using this application, you acknowledge that you have read, understood, and agree to these terms and conditions.
    """)
