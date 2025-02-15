import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

def load_model():
    """Loads the saved model and its dependencies from a pickle file."""
    with open('model.pkl', 'rb') as file:
        model, scaler, le = pickle.load(file)
    return model, scaler, le

def processing_input_data(data, scaler, le):
    """Processes the input data to prepare it for the model."""
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])[0]
    data = pd.DataFrame([data])
    data_transformed = scaler.transform(data)
    return data

def predict_data(data):
    """Predicts the output based on the input data."""
    model, scaler, le = load_model()
    processed_data = processing_input_data(data, scaler, le)
    prediction = model.predict(processed_data)
    return prediction

def main():
    st.set_page_config(page_title="Student Performance Prediction", layout="wide")

    st.title("Student Performance Prediction")
    st.markdown("""
        <style>
            .main-title {
                font-size: 24px;
                color: #4CAF50;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar Layout
    st.sidebar.header("Input Parameters")
    hours_studied = st.sidebar.slider("Hours Studied", 1, 10, 5)
    pre_score = st.sidebar.slider("Previous Scores", 40, 100, 40)
    exe_activity = st.sidebar.selectbox("Extracurricular Activities", ["Yes", "No"])
    sleep_hours = st.sidebar.slider("Sleep Hours", 4, 10, 7)
    no_que_solved = st.sidebar.slider("Number of Question Papers Solved", 4, 10, 7)

    user_data = {
        "Hours Studied": hours_studied,
        "Previous Scores": pre_score,
        "Extracurricular Activities": exe_activity,
        "Sleep Hours": sleep_hours,
        "Sample Question Papers Practiced": no_que_solved
    }

    # Prediction Button
    if st.sidebar.button("Predict Performance"):
        prediction = predict_data(user_data)
        
        # Display Prediction Results
        st.markdown(f"### Prediction Result")
        st.markdown(f"""
            **Predicted Performance Index**: 
            **{prediction[0]:.2f}**
        """, unsafe_allow_html=True)
        
        # Additional info (can be customized further)
        st.markdown("### Explanation of Prediction")
        st.write("The prediction is based on the input factors such as study hours, past performance, extracurricular activities, sleep hours, and the number of practice questions solved. The result indicates your potential performance based on these factors.")
        
    # Add footer or additional elements (optional)
    st.markdown("""
        ---
        *Note: The prediction is based on the data model and may not fully capture all real-world factors. Please use it as a guideline.*
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
