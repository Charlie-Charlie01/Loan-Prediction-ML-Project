import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="💰",
    layout="centered"
)

# Load the saved model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Title and description
st.title("Loan Approval Prediction")
st.markdown("""
This app predicts whether your loan application will be approved based on various factors.
Fill in the details below to get a prediction.
""")

# Create columns for better layout
col1, col2 = st.columns(2)

# Form inputs
with col1:
    st.subheader("Personal Information")
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    
with col2:
    st.subheader("Loan & Financial Information")
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=100)
    loan_amount_term = st.selectbox("Loan Term (months)", [36, 60, 84, 120, 180, 240, 300, 360, 480])
    credit_history = st.selectbox("Credit History (1: Good, 0: Bad)", [1, 0])
    property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

# Function to preprocess input data
def preprocess_input():
    # Create a dictionary with user inputs
    data = {
        'Gender': 1 if gender == "Male" else 0,
        'Married': 1 if married == "Yes" else 0,
        'Dependents': 3 if dependents == "3+" else int(dependents),
        'Education': 1 if education == "Graduate" else 0,
        'Self_Employed': 1 if self_employed == "Yes" else 0,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area_Rural': 1 if property_area == "Rural" else 0,
        'Property_Area_Semiurban': 1 if property_area == "Semiurban" else 0,
        'Property_Area_Urban': 1 if property_area == "Urban" else 0,
    }
    
    # Create dataframe from the dictionary
    df = pd.DataFrame(data, index=[0])
    
    # Add log of loan amount (if your model uses this feature)
    df['LoanAmount_log'] = np.log(df['LoanAmount'] + 1)
    
    # Create dummies for Dependents
    df['Dependents_1'] = 1 if dependents == "1" else 0
    df['Dependents_2'] = 1 if dependents == "2" else 0
    df['Dependents_3'] = 1 if dependents == "3+" else 0
    
    # Add Gender dummy (if your model uses this)
    df['Gender_1'] = 1 if gender == "Male" else 0
    
    return df

# Prediction function
def predict_approval(df):
    # Make sure the DataFrame columns match the model's expected features
    prediction = model.predict(df)
    probability = model.predict_proba(df)
    return prediction[0], probability[0][1]  # Return prediction and probability of approval

# Button to make prediction
if st.button("Predict Loan Approval"):
    # Preprocess the input
    input_df = preprocess_input()
    
    # Make prediction
    prediction, probability = predict_approval(input_df)
    
    # Show the result with a divider
    st.markdown("---")
    st.subheader("Prediction Result")
    
    if prediction == 1:
        st.success(f"✅ Loan Approved! (Probability: {probability:.2%})")
        st.balloons()
    else:
        st.error(f"❌ Loan Not Approved (Probability: {probability:.2%})")
    
    # Show explanation
    st.subheader("Key Factors")
    
    # Display some key factors that influence loan approval
    factors = []
    if credit_history == 1:
        factors.append("✓ Good credit history")
    else:
        factors.append("✗ Poor credit history")
    
    if input_df['LoanAmount'][0] / input_df['ApplicantIncome'][0] < 0.3:
        factors.append("✓ Good loan amount to income ratio")
    else:
        factors.append("✗ High loan amount relative to income")
    
    if property_area == "Semiurban":
        factors.append("✓ Property in favorable area")
    
    # Display the factors
    for factor in factors:
        st.write(factor)
    
    # Add a disclaimer
    st.markdown("---")
    st.caption("Note: This prediction is based on historical data and should be used as a guideline only.")

# Add information at the bottom
st.markdown("---")
st.markdown("### How it works")
st.write("""
This application uses a machine learning model trained on historical loan data to predict loan approval.
The model considers factors such as your income, credit history, loan amount, and personal details
to determine the likelihood of loan approval.
""")

# Sidebar with additional information
with st.sidebar:
    st.header("About")
    st.info("""
    This loan prediction app helps you understand your chances of loan approval.
    
    The prediction is based on a machine learning model trained on historical loan data.
    
    For questions or feedback, please reach out to the support team.
    """)
    
    st.header("Instructions")
    st.markdown("""
    1. Fill in all required fields
    2. Click 'Predict Loan Approval'
    3. Review the prediction result
    4. Understand the key factors affecting your prediction
    """)