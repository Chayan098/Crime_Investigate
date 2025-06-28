import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from fairlearn.metrics import demographic_parity_difference
import shap
import matplotlib.pyplot as plt

st.title("🚓 AI-Powered Predictive Policing Tool")
st.write("""
This tool predicts potential crime hotspots, checks for bias, and explains predictions.
""")

# Upload CSV
uploaded_file = st.file_uploader("/data.csv", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(data.head())

    # Preprocess
    X = data[['location', 'time_of_day', 'day_of_week', 'crime_type', 'arrests_last_year', 'population_density']].copy()
    X['location'] = X['location'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4})
    y = data['reported_crime']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    st.write("### Prediction Results")
    st.write(f"Sample Predictions: {preds[:10]}")

    # Bias check
    sensitive_feature = X_test['location']
    dpd = demographic_parity_difference(y_test, preds, sensitive_features=sensitive_feature)
    st.write(f"### Demographic Parity Difference (Location Bias): {dpd:.4f}")

    if abs(dpd) > 0.1:
        st.warning("⚠️ Possible bias detected! Review or retrain your model.")
    else:
        st.success("✅ No significant bias detected.")

    # Explainability
    st.write("### Explainable AI (SHAP Summary Plot)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # SHAP summary plot
    plt.title("Feature Importance")
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(plt.gcf())
    plt.clf()

    st.info("Model trained & explained successfully.")
else:
    st.info("Please upload a CSV file to begin.")
