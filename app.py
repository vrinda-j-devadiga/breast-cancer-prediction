import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load Model
model = joblib.load("breast_cancer_model.pkl")

st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🩺")
st.title("🔬 Breast Cancer Prediction & Severity Analysis")
st.write("Enter the details below to predict breast cancer and view risk severity.")

# Input Fields
radius_mean = st.number_input("Radius Mean", value=14.5)
texture_mean = st.number_input("Texture Mean", value=20.3)
perimeter_mean = st.number_input("Perimeter Mean", value=96.5)
area_mean = st.number_input("Area Mean", value=655.0)
smoothness_mean = st.number_input("Smoothness Mean", value=0.10)
compactness_mean = st.number_input("Compactness Mean", value=0.13)
concavity_mean = st.number_input("Concavity Mean", value=0.09)
symmetry_mean = st.number_input("Symmetry Mean", value=0.18)

if st.button("Predict"):
    input_data = [[radius_mean, texture_mean, perimeter_mean, area_mean,
                   smoothness_mean, compactness_mean, concavity_mean, symmetry_mean]]
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display Prediction
    result = "🟥 Malignant (Cancer Detected)" if prediction == 1 else "🟩 Benign (No Cancer)"
    st.subheader(f"**Prediction Result:** {result}")

    # Severity Score
    severity_score = (probability * 0.6) + ((radius_mean + perimeter_mean + area_mean) / 2000 * 0.4)
    severity_score = round(severity_score * 10, 2)
    st.write(f"📍 **Severity Score:** `{severity_score} / 10`")

    # Severity Message
    if severity_score < 3:
        st.success("✅ **Low Risk:** No immediate concern, but regular screening is advised.")
    elif 3 <= severity_score < 7:
        st.warning("🟧 **Moderate Risk:** Consider consulting a physician for further tests.")
    else:
        st.error("🟥 **High Risk:** Immediate medical consultation is strongly recommended.")

    # Recommendations
    st.write("### 🩺 Suggested Next Steps:")
    if prediction == 0:
        st.write("- Maintain a healthy lifestyle and continue regular self-examinations.")
        st.write("- Schedule routine mammograms as per medical guidelines.")
    else:
        st.write("- Consult an oncologist or breast specialist for further evaluation.")
        st.write("- Consider diagnostic tests such as Mammogram, Ultrasound, or Biopsy.")
        st.write("- Early detection significantly improves treatment outcomes.")

    # Charts
    fig1 = go.Figure(data=[go.Pie(labels=["Benign", "Malignant"], values=[1 - probability, probability])])
    fig1.update_layout(title="Cancer Probability Distribution")
    st.plotly_chart(fig1)

    fig2 = go.Figure(data=[go.Bar(x=["Severity Score"], y=[severity_score])])
    fig2.update_layout(title="Severity Score (0 - 10)", yaxis=dict(range=[0, 10]))
    st.plotly_chart(fig2)

    fig3 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=severity_score,
        gauge={"axis": {"range": [0, 10]}},
        title={"text": "Severity Level Indicator"}
    ))
    st.plotly_chart(fig3)

# Disclaimer
st.write("---")
st.caption("⚠️ **Disclaimer:** This tool is for educational and research purposes only and should not be used as a substitute for professional medical diagnosis or treatment.")



