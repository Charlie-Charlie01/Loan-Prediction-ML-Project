import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Loan Eligibility Predictor",
    page_icon="💰",
    layout="wide"
)

# Application title and description
st.markdown("# 💰 Loan Eligibility Prediction App")
st.markdown("""
This application predicts whether your loan application will be approved based on your inputs.
Fill in the details below and click the 'Predict' button to see the result.
""")

# Check for required packages
missing_packages = []
try:
    import sklearn
except ImportError:
    missing_packages.append("scikit-learn")

# Show warning if packages are missing
if missing_packages:
    st.error(f"""
    ### Missing Required Packages
    
    The following packages need to be installed:
    **{', '.join(missing_packages)}**
    
    Please run this command in your terminal:
    ```
    pip install {' '.join(missing_packages)}
    ```
    
    After installation, restart the Streamlit app with:
    ```
    streamlit run app.py
    ```
    """)
    st.stop()  # Stop execution if packages are missing

# Function to load the pre-trained model
@st.cache_resource
def load_model():
    try:
        # First check if scikit-learn is installed
        try:
            import sklearn
        except ImportError:
            st.error("""
            Error: scikit-learn is not installed. 
            
            Please install it by running this command in your terminal:
            ```
            pip install scikit-learn
            ```
            
            Then restart the Streamlit app.
            """)
            return None
            
        # In a real application, you would have your model saved as a pickle file
        # For demonstration, we'll create a dummy model if file doesn't exist
        if os.path.exists('loan_model.pkl'):
            with open('loan_model.pkl', 'rb') as file:
                model = pickle.load(file)
        else:
            # Dummy model for demonstration purposes only
            # In a real application, you'd handle this differently
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=42)
            # Train with dummy data
            X_dummy = np.random.rand(100, 11)
            y_dummy = np.random.choice([0, 1], size=100)
            model.fit(X_dummy, y_dummy)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_model()

# Create columns for better layout
col1, col2 = st.columns(2)

# User Input Section - Left Column
with col1:
    st.subheader("Personal Information")
    
    gender = st.selectbox("Gender", ["Male", "Female"])
    
    marital_status = st.selectbox("Marital Status", ["Married", "Not Married"])
    
    dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
    
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    
    credit_history = st.selectbox("Credit History (Debt obligations met on time?)", ["Yes", "No"])
    
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# User Input Section - Right Column
with col2:
    st.subheader("Loan Information")
    
    applicant_income = st.number_input("Applicant Monthly Income ($)", 
                                     min_value=0, 
                                     max_value=100000, 
                                     value=5000,
                                     step=100)
    
    coapplicant_income = st.number_input("Coapplicant Monthly Income ($)", 
                                       min_value=0, 
                                       max_value=100000, 
                                       value=0,
                                       step=100)
    
    loan_amount = st.slider("Loan Amount ($)", 
                          min_value=0, 
                          max_value=100000, 
                          value=10000,
                          step=1000)
    
    loan_amount_term = st.slider("Loan Term (months)", 
                               min_value=12, 
                               max_value=480, 
                               value=360,
                               step=12)
    
    # Calculate the loan amount to income ratio as an additional feature
    total_income = applicant_income + coapplicant_income
    if total_income > 0:
        loan_amount_income_ratio = loan_amount / total_income
    else:
        loan_amount_income_ratio = 0

    st.markdown(f"**Loan Amount to Income Ratio:** {loan_amount_income_ratio:.2f}")

# Function to preprocess inputs for model prediction
def preprocess_inputs(data):
    # Create a DataFrame with the input data
    df = pd.DataFrame(data, index=[0])
    
    # Encode categorical features
    # Gender
    df['Gender'] = 1 if data['Gender'] == 'Male' else 0
    
    # Marital Status
    df['Married'] = 1 if data['Married'] == 'Married' else 0
    
    # Dependents - One-hot encode
    if data['Dependents'] == '0':
        df['Dependents_0'] = 1
        df['Dependents_1'] = 0
        df['Dependents_2'] = 0
        df['Dependents_3+'] = 0
    elif data['Dependents'] == '1':
        df['Dependents_0'] = 0
        df['Dependents_1'] = 1
        df['Dependents_2'] = 0
        df['Dependents_3+'] = 0
    elif data['Dependents'] == '2':
        df['Dependents_0'] = 0
        df['Dependents_1'] = 0
        df['Dependents_2'] = 1
        df['Dependents_3+'] = 0
    else:  # '3+'
        df['Dependents_0'] = 0
        df['Dependents_1'] = 0
        df['Dependents_2'] = 0
        df['Dependents_3+'] = 1
    
    # Education
    df['Education'] = 1 if data['Education'] == 'Graduate' else 0
    
    # Self Employed
    df['Self_Employed'] = 1 if data['Self_Employed'] == 'Yes' else 0
    
    # Credit History
    df['Credit_History'] = 1 if data['Credit_History'] == 'Yes' else 0
    
    # Property Area - One-hot encode
    if data['Property_Area'] == 'Urban':
        df['Property_Area_Urban'] = 1
        df['Property_Area_Semiurban'] = 0
        df['Property_Area_Rural'] = 0
    elif data['Property_Area'] == 'Semiurban':
        df['Property_Area_Urban'] = 0
        df['Property_Area_Semiurban'] = 1
        df['Property_Area_Rural'] = 0
    else:  # 'Rural'
        df['Property_Area_Urban'] = 0
        df['Property_Area_Semiurban'] = 0
        df['Property_Area_Rural'] = 1
    
    # Standard scaling for numerical features (in a real app, you would use the same scaler used during training)
    df['ApplicantIncome'] = data['ApplicantIncome'] / 10000  # Simple scaling
    df['CoapplicantIncome'] = data['CoapplicantIncome'] / 10000  # Simple scaling
    df['LoanAmount'] = data['LoanAmount'] / 10000  # Simple scaling
    df['Loan_Amount_Term'] = data['Loan_Amount_Term'] / 100  # Simple scaling
    df['Loan_Amount_Income_Ratio'] = data['Loan_Amount_Income_Ratio']
    
    # Select and order features to match the training data
    # In a real application, this would be determined by your model's training features
    # For demonstration, we'll use a subset of features
    features = ['Gender', 'Married', 'Education', 'Self_Employed', 'ApplicantIncome',
                'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
                'Property_Area_Urban', 'Property_Area_Semiurban', 'Loan_Amount_Income_Ratio']
    
    return df[features]

# Prediction button
st.markdown("---")
predict_button = st.button("Predict Loan Eligibility", type="primary")

# Make prediction when the button is clicked
if predict_button:
    # Check if all required inputs are provided
    if not model:
        st.error("Model could not be loaded. Please check the model file.")
    else:
        try:
            # Prepare input data
            input_data = {
                'Gender': gender,
                'Married': marital_status,
                'Dependents': dependents,
                'Education': education,
                'Self_Employed': self_employed,
                'ApplicantIncome': applicant_income,
                'CoapplicantIncome': coapplicant_income,
                'LoanAmount': loan_amount,
                'Loan_Amount_Term': loan_amount_term,
                'Credit_History': credit_history,
                'Property_Area': property_area,
                'Loan_Amount_Income_Ratio': loan_amount_income_ratio
            }
            
            # Preprocess the input data
            preprocessed_data = preprocess_inputs(input_data)
            
            # Make prediction
            prediction = model.predict(preprocessed_data)
            probability = model.predict_proba(preprocessed_data)
            
            # Display prediction result
            st.markdown("---")
            st.subheader("Prediction Result")
            
            if prediction[0] == 1:
                st.markdown(
                    """
                    <div style="background-color:#c8e6c9; padding:20px; border-radius:10px; text-align:center;">
                        <h2 style="color:#2e7d32; margin:0;">✅ Loan Approved</h2>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                st.markdown(f"**Approval Confidence:** {probability[0][1]*100:.2f}%")
            else:
                st.markdown(
                    """
                    <div style="background-color:#ffcdd2; padding:20px; border-radius:10px; text-align:center;">
                        <h2 style="color:#c62828; margin:0;">❌ Loan Rejected</h2>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                st.markdown(f"**Rejection Confidence:** {probability[0][0]*100:.2f}%")
            
            # Display key factors (simplified - in a real app you might use SHAP values or feature importance)
            st.subheader("Key Factors")
            
            factors = []
            if credit_history == "No":
                factors.append("Poor credit history reduces approval chances.")
            if loan_amount_income_ratio > 1:
                factors.append("High loan amount to income ratio reduces approval chances.")
            if applicant_income < 3000:
                factors.append("Low applicant income reduces approval chances.")
                
            if not factors:
                factors = ["Your application looks good overall."]
                
            for factor in factors:
                st.markdown(f"• {factor}")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.markdown("Please check your inputs and try again.")

# Add app instructions at the bottom
with st.expander("How to use this app"):
    st.markdown("""
    1. **Fill in your personal information** in the left column.
    2. **Fill in your loan information** in the right column.
    3. **Click the 'Predict' button** to see if your loan would be approved.
    4. **Review the prediction result** and key factors affecting the decision.
    
    **Note**: This is a prediction model and does not guarantee actual loan approval from any financial institution.
    """)

# Add information about the model
with st.expander("About the Model"):
    st.markdown("""
    This application uses a Logistic Regression model trained on historical loan application data. The model considers various factors like:
    
    * Income details
    * Loan amount and duration
    * Credit history
    * Personal information
    * Property details
    
    Logistic Regression is well-suited for binary classification problems like loan approval, providing both a decision and probability score for each prediction.
    """)

# Footer
st.markdown("---")
st.markdown("© 2025 Loan Eligibility Predictor | For demonstration purposes only")
