import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder

# Load the trained Random Forest model
model = load('svc_model.joblib')

# Create a Streamlit app
st.title("Customer Churn Prediction")

# Input fields for feature values on the main screen
st.header("Enter Customer Information")
options = ['No', 'Yes']  # Mapping to 0 (No) and 1 (Yes)
optionswithinternet=['Yes','No','No Internet']
genderoption = ['Male', 'Female']


col1, col2 = st.columns(2)

with col1:
    senior_citizen = st.radio('Is the customer a senior citizen?', options)
    partner = st.radio('Is the Partner also a customer?', options)
    dependent = st.radio('Is there any Dependent a customer?', options)
    phoneservice = st.radio('Do the customer have other Phone Service?', options)
    multipleline = st.selectbox("Multiple Lines", ('Yes', 'No', 'No Phone Service'))
    online_security = st.radio('Online Security opted?', optionswithinternet)
    online_backup = st.radio('Online backup option is there?', optionswithinternet)
    device_protection = st.radio('Device protection included?', optionswithinternet)
    tech_support = st.radio('Tech Support included?', optionswithinternet)
    gender = st.radio('Gender', genderoption)
with col2:
    tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100, value=1)
    internet_service = st.selectbox("Internet Service", ('DSL', 'Fiber optic', 'No'))
    contract = st.selectbox("Contract", ('Month-to-month', 'One year', 'Two year'))
    monthly_charges = st.number_input("Monthly Charges", min_value=0, max_value=200, value=50)
    total_charges = st.number_input("Total Charges", min_value=0, max_value=10000, value=0)
    streaming_tv = st.radio('Streaming TV?', optionswithinternet)
    streaming_movies = st.radio('Streaming Movie?', optionswithinternet)
    paperless = st.radio('Is paperless Billing?', options)
    paymentmethod= st.selectbox("Payment Method", ('Bank Transfer', 'Credit Card', 'Electronic cheque','Mailed Cheque'))
# Map input values to numeric using the label mapping
label_mapping = {
    'DSL': 0,
    'Fiber optic': 1,
    'No': 1,
    'Month-to-month': 0,
    'One year': 1,
    'Two year': 1,
    'Bank Transfer': 0,
    'Credit Card' : 1,
    'Electronic cheque': 1,
    'Mailed cheque' :1
}
map_multipleEntries ={
     'No Phone Service' : 1,
     'No Internet Service' : 1,
     'Yes' : 1,
     'No': 0,
     'Male' :1,
     'Female' :0
}



internet_services = label_mapping[internet_service]
contracts = label_mapping[contract]
multiplelines = map_multipleEntries[multipleline]
onlineSecurities =map_multipleEntries[online_security]
backup= map_multipleEntries[online_backup]
protection =map_multipleEntries[online_backup]
techSupport =map_multipleEntries[tech_support]
streamingtv = map_multipleEntries[streaming_tv]
streamingmovies = map_multipleEntries[streaming_movies]
payment = label_mapping[paymentmethod]
senior = map_multipleEntries[senior_citizen]
gender_male = map_multipleEntries[gender]
partneroption = map_multipleEntries[partner]
dependentoption = map_multipleEntries[dependent]
phone = map_multipleEntries[phoneservice]
paperlessoption = map_multipleEntries[paperless]
# Make a prediction using the model
prediction = model.predict([[senior,tenure,monthly_charges,total_charges,gender_male,partneroption,
                             dependentoption,phone, multiplelines,multiplelines,internet_services,internet_services
                             ,onlineSecurities,onlineSecurities,backup,backup,protection,protection,
                             techSupport,techSupport,streamingtv,streamingtv,streamingmovies,
                             streamingmovies,contracts,contracts,paperlessoption,payment,payment,payment]])

# Display the prediction result on the main screen
st.header("Prediction Result")
if prediction[0] == 0:
    st.success("This customer is likely to stay.")
else:
    st.error("This customer is likely to churn.")

# Add any additional Streamlit components or UI elements as needed.
