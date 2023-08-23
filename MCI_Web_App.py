import pandas as pd
import xgboost as xgb
import streamlit as st

# Read the CSV file into a DataFrame
df = pd.read_csv('Myocardial infarction complications.csv')

# Define the target variable
target_column = 'LET_IS'

# Define the selected features
selected_features = ['AGE', 'SEX', 'RAZRIV', 'K_SH_POST', 'ritm_ecg_p_04', 'SVT_POST', 'endocr_02']

# Split the data into features (X) and target (y)
X = df[selected_features]
y = df[target_column]

# Initialize the XGBoost classifier with the best parameters
model = xgb.XGBClassifier(learning_rate=0.2, max_depth=3, n_estimators=300, random_state=42)

# Fit the model on the training data
model.fit(X, y)

# Streamlit Input
st.set_page_config(page_title="Myocardial Infarction Predictor", page_icon="heart1.png")
st.sidebar.header('Predict Lethal Outcome with XGBoost Classifier')

# Styling
st.markdown(
    """
    <style>
    .info-box {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    .info-title {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .info-description {
        font-size: 14px;
        color: #666;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Class labels mapping
class_labels = {
    0: 'unknown (alive)',
    1: 'cardiogenic shock',
    2: 'pulmonary edema',
    3: 'myocardial rupture',
    4: 'progress of congestive heart failure',
    5: 'thromboembolism',
    6: 'asystole',
    7: 'ventricular fibrillation'
}

# Additional descriptions for each class
class_descriptions = {
    0: "The predicted outcome suggests that the individual is alive and in an unknown condition.",
    1: "The predicted outcome suggests the possibility of cardiogenic shock, a serious condition that requires immediate medical attention.",
    2: "The predicted outcome indicates the potential for pulmonary edema, which involves fluid accumulation in the lungs.",
    3: "The predicted outcome suggests the risk of myocardial rupture, a serious complication of heart attacks.",
    4: "The predicted outcome indicates the potential for progression of congestive heart failure.",
    5: "The predicted outcome suggests the possibility of thromboembolism, which involves blood clot formation.",
    6: "The predicted outcome indicates the possibility of asystole, a type of irregular heartbeat.",
    7: "The predicted outcome suggests the risk of ventricular fibrillation, a serious arrhythmia."
}



# Create input fields for the selected features in a single column
input_values = {}
for feature in selected_features:
    if feature == 'SEX':
        value = st.sidebar.selectbox('Gender', ['female', 'male'], key=feature)
        value = 0 if value == 'female' else 1
    elif feature in ['zab_leg_03', 'SVT_POST', 'ritm_ecg_p_04', 'n_p_ecg_p_07', 'n_p_ecg_p_10', 'RAZRIV',
                    'endocr_02', 'K_SH_POST', 'FIBR_JELUD', 'n_r_ecg_p_04']:
        if feature == 'RAZRIV':
            label = "Myocardial Rupture"
        elif feature == 'K_SH_POST':
            label = "Cardiogenic Shock After MI"
        elif feature == 'ritm_ecg_p_04':
            label = "ECG Rhythm Abnormality"
        elif feature == 'SVT_POST':
            label = "Supraventricular Tachycardia Post MI"
        elif feature == 'endocr_02':
            label = "Obesity in the Anamnesis (Endocrine Disorder)"
        else:
            label = feature
        value = st.sidebar.selectbox(f'{label}', ['no', 'yes'], key=feature)
        value = 0 if value == 'no' else 1
    elif feature == 'AGE':
        value = st.sidebar.number_input(f'Enter Age', key=feature, step=1, value=30)
        value = int(value)
    input_values[feature] = value

# Convert input values to a DataFrame
input_data = pd.DataFrame([input_values], columns=selected_features)

# Predict the outcome using the trained model
prediction_class = model.predict(input_data)[0]
predicted_class_label = class_labels[prediction_class]


# Display the predicted class and description in separate sections
st.title("Myocardial Infarction Outcome Predictor")
st.subheader("Predicted Lethal Outcome:")

with st.container():
    st.info(f"Class: {prediction_class} - {predicted_class_label}")
    st.write(class_descriptions[prediction_class])

# Display the input values
st.subheader("Input Values:")
input_values_str = ', '.join([f"{feature}: {value}" for feature, value in input_values.items()])
st.write(input_values_str)

# Get the count of people with the same input age
input_age = input_values['AGE']
people_with_input_age = df[df['AGE'] == input_age]
count_same_age = people_with_input_age.shape[0]

st.subheader("Count of People with the Same Age:")
st.write(f"There are {count_same_age} people with the same age as the input value.")

# Get the count of people with the same input age and the same predicted outcome
people_with_same_outcome = people_with_input_age[people_with_input_age[target_column] == prediction_class]
count_same_outcome = people_with_same_outcome.shape[0]

st.subheader("Count of People with the Same Age and Predicted Outcome:")
st.write(f"There are {count_same_outcome} people with the same age and predicted outcome.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Disclaimer: This is a prediction tool and not a substitute for medical advice.")