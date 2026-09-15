import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Graduate Admission Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Graduate Admission Prediction Dashboard")
st.write("Interactive ML Dashboard for Predicting University Admission Chances (DSCI 2334)")

# Sidebar input options
st.sidebar.header("User Input Parameters")

def user_input_features():
    gre = st.sidebar.slider("GRE Score", 290, 340, 320)
    toefl = st.sidebar.slider("TOEFL Score", 92, 120, 108)
    univ_rating = st.sidebar.slider("University Rating", 1, 5, 3)
    sop = st.sidebar.slider("SOP Strength", 1.0, 5.0, 3.5, 0.5)
    lor = st.sidebar.slider("LOR Strength", 1.0, 5.0, 3.5, 0.5)
    cgpa = st.sidebar.slider("CGPA", 6.8, 9.92, 8.5, 0.01)
    research = st.sidebar.radio("Research Experience", (1, 0), format_func=lambda x: "Yes" if x == 1 else "No")
    
    data = {
        'GRE Score': gre,
        'TOEFL Score': toefl,
        'University Rating': univ_rating,
        'SOP': sop,
        'LOR': lor,
        'CGPA': cgpa,
        'Research': research
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display input parameters
st.subheader("Input Student Parameters")
st.dataframe(input_df)

# Target calculation formula
calculated_chance = (
    (input_df['CGPA'][0] / 10.0) * 0.45 +
    (input_df['GRE Score'][0] / 340.0) * 0.25 +
    (input_df['TOEFL Score'][0] / 120.0) * 0.15 +
    (input_df['University Rating'][0] / 5.0) * 0.05 +
    (input_df['SOP'][0] / 5.0) * 0.04 +
    (input_df['LOR'][0] / 5.0) * 0.04 +
    (input_df['Research'][0]) * 0.02
)

prediction = np.clip(calculated_chance, 0.0, 1.0)

# Display prediction results
st.subheader("Admission Prediction Result")
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Estimated Chance of Admission", value=f"{prediction * 100:.2f}%")

with col2:
    if prediction >= 0.75:
        st.success("High Chance of Admission! 🎉")
    elif prediction >= 0.50:
        st.warning("Moderate Chance of Admission. ⚖️")
    else:
        st.error("Low Chance of Admission. ⚠️")

st.markdown("---")
st.subheader("SageMaker Canvas Batch Output Data")

# Load Canvas prediction output
try:
    canvas_df = pd.read_csv("Reem_MLDeployment_Canvas_Output.csv")
    st.dataframe(canvas_df.head(10))
except Exception:
    st.info("Uploaded SageMaker Canvas predictions CSV file will display here.")