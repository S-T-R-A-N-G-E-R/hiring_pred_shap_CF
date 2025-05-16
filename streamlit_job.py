import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import dice_ml
from dice_ml.utils import helpers
import time

# Custom CSS for styling
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stForm {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stNumberInput label, .stSelectbox label {
        font-weight: bold;
        color: #333;
    }
    .stSuccess, .stError, .stWarning {
        border-radius: 4px;
        padding: 10px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stMarkdown p {
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and reference data
try:
    rf_classifier = joblib.load('rf_classifier.joblib')
    reference_data = joblib.load('reference_data.joblib')
    X_ref = reference_data['X']
    y_ref = reference_data['y']
except FileNotFoundError:
    st.error("Model or reference data file not found. Ensure 'rf_classifier.joblib' and 'reference_data.joblib' are in the same directory.")
    st.stop()

# Define features
model_features = ['Age', 'Gender', 'EducationLevel', 'ExperienceYears', 'PreviousCompanies',
                  'DistanceFromCompany', 'InterviewScore', 'SkillScore', 'PersonalityScore', 'RecruitmentStrategy']
changeable_features = ['EducationLevel', 'ExperienceYears', 'InterviewScore', 'SkillScore', 'PersonalityScore', 'RecruitmentStrategy']
categorical_features = ['Gender', 'EducationLevel', 'PreviousCompanies', 'RecruitmentStrategy']

# Convert categorical features to strings in reference data
X_ref = X_ref.copy()
for feature in categorical_features:
    X_ref[feature] = X_ref[feature].astype(str)

# Input validation and processing
def process_job_seeker_input(input_dict, expected_features):
    missing = [f for f in expected_features if f not in input_dict]
    extra = [f for f in input_dict if f not in expected_features]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    if extra:
        st.warning(f"Ignoring extra features: {extra}")
    
    input_df = pd.DataFrame([input_dict], columns=expected_features)
    if input_df.isna().any().any():
        raise ValueError("Input contains missing values.")
    
    # Convert categorical features to strings
    for feature in categorical_features:
        input_df[feature] = input_df[feature].astype(str)
    
    return input_df

# Generate suggestions
def generate_suggestions(rf_classifier, input_df, changeable_features, model_features, X_ref, y_ref):
    prediction = rf_classifier.predict(input_df)[0]
    prob = rf_classifier.predict_proba(input_df)[0]
    result = {
        'prediction': 'Hired' if prediction == 1 else 'Not Hired',
        'probability': prob[1],
        'explanation': None,
        'suggestions': []
    }

    if prediction == 1:
        return result

    explainer = shap.TreeExplainer(rf_classifier)
    shap_values = explainer.shap_values(input_df)
    
    if shap_values.ndim == 3 and shap_values.shape[2] == 2:
        shap_values_array = shap_values[:, :, 1]
    elif isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values_array = shap_values[1]
    else:
        raise ValueError(f"Unexpected shap_values format: {shap_values.shape}")

    shap_contributions = pd.DataFrame({
        'feature': model_features,
        'shap_value': shap_values_array[0]
    })
    most_negative = shap_contributions[shap_contributions['shap_value'] < 0].sort_values(by='shap_value').iloc[0]
    result['explanation'] = {
        'feature': most_negative['feature'],
        'shap_value': most_negative['shap_value']
    }

    X_dice = X_ref[model_features]
    d = dice_ml.Data(
        dataframe=pd.concat([X_dice, y_ref], axis=1),
        continuous_features=['Age', 'ExperienceYears', 'DistanceFromCompany', 'InterviewScore', 'SkillScore', 'PersonalityScore'],
        categorical_features=['Gender', 'EducationLevel', 'PreviousCompanies', 'RecruitmentStrategy'],
        outcome_name=y_ref.name,
        categorical_levels={
            'EducationLevel': ['1', '2', '3'],
            'Gender': ['0', '1'],
            'PreviousCompanies': ['1', '2', '3', '4', '5'],
            'RecruitmentStrategy': ['1', '2', '3']
        }
    )
    m = dice_ml.Model(model=rf_classifier, backend='sklearn')
    exp = dice_ml.Dice(d, m, method='random')

    counterfactuals = exp.generate_counterfactuals(
        query_instances=input_df,
        total_CFs=3,
        desired_class="opposite",
        features_to_vary=changeable_features,
        permitted_range={
            'InterviewScore': [input_df['InterviewScore'].iloc[0], 100],
            'SkillScore': [input_df['SkillScore'].iloc[0], 100],
            'PersonalityScore': [input_df['PersonalityScore'].iloc[0], 100],
            'ExperienceYears': [input_df['ExperienceYears'].iloc[0], 15]
        },
        random_seed=42
    )

    cf_df = counterfactuals.cf_examples_list[0].final_cfs_df
    if cf_df is not None:
        for i, cf in cf_df.iterrows():
            cf_features = pd.DataFrame([cf[X_dice.columns]], columns=X_dice.columns)
            cf_pred = rf_classifier.predict(cf_features)[0]
            cf_prob = rf_classifier.predict_proba(cf_features)[0][1]
            changes = []
            for feature in changeable_features:
                original_val = input_df[feature].iloc[0]
                cf_val = cf[feature]
                if feature in categorical_features:
                    if str(original_val) != str(cf_val):
                        if feature == 'RecruitmentStrategy':
                            mapping = {1: 'Aggressive', 2: 'Moderate', 3: 'Conservative'}
                            advice = f"Switch to {mapping.get(int(cf_val), cf_val)} recruitment strategy"
                        elif feature == 'EducationLevel':
                            mapping = {1: 'Bachelor’s', 2: 'Master’s', 3: 'PhD'}
                            advice = f"Upgrade to {mapping.get(int(cf_val), cf_val)} degree (e.g., pursue a {'Master’s' if int(cf_val) == 2 else 'PhD'})"
                        changes.append({'feature': feature, 'advice': advice, 'new_value': cf_val})
                else:
                    if not np.isclose(float(original_val), float(cf_val), atol=1e-5):
                        if feature == 'ExperienceYears':
                            advice = f"Gain {float(cf_val) - float(original_val):.1f} more years of experience"
                        elif feature == 'InterviewScore':
                            advice = f"Improve Interview Score to {float(cf_val)} (e.g., practice interview techniques)"
                        elif feature == 'SkillScore':
                            advice = f"Raise Skill Score to {float(cf_val)} (e.g., take relevant courses or certifications)"
                        elif feature == 'PersonalityScore':
                            advice = f"Enhance Personality Score to {float(cf_val)} (e.g., improve communication or teamwork skills)"
                        changes.append({'feature': feature, 'advice': advice, 'new_value': cf_val})
            
            if changes and cf_pred == 1:
                result['suggestions'].append({
                    'probability': cf_prob,
                    'changes': changes
                })

    return result

# Streamlit App
st.title("Candidate Hiring Predictor")
st.markdown("""
Welcome! Enter candidate details to predict whether they'll be hired. If not hired, we'll suggest ways to improve their chances based on the most influential factors.
""")


# Input form
with st.form("job_seeker_form"):
    st.subheader("Enter Candidate Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input(
            "Age (20–50 years)",
            min_value=20, max_value=50, value=30, step=1,
            help="Your age in years."
        )
        gender = st.selectbox(
            "Gender",
            options=[(0, "Male"), (1, "Female")],
            format_func=lambda x: x[1],
            help="Select your gender."
        )[0]
        education_level = st.selectbox(
            "Education Level",
            options=[
                (1, "Bachelor’s"),
                (2, "Master’s"),
                (3, "PhD")
            ],
            format_func=lambda x: x[1],
            help="Your highest education level (Bachelor’s includes all undergraduate degrees)."
        )[0]
        experience_years = st.number_input(
            "Years of Experience (0–15)",
            min_value=0, max_value=15, value=5, step=1,
            help="Number of years of professional experience."
        )
        previous_companies = st.number_input(
            "Previous Companies (1–5)",
            min_value=1, max_value=5, value=2, step=1,
            help="Number of companies you’ve worked for."
        )
    
    with col2:
        distance_from_company = st.number_input(
            "Distance from Company (1–50 km)",
            min_value=1.0, max_value=50.0, value=20.0, step=0.1,
            help="Distance from your residence to the company in kilometers."
        )
        interview_score = st.number_input(
            "Interview Score (0–100)",
            min_value=0, max_value=100, value=70, step=1,
            help="Your score from the interview process."
        )
        skill_score = st.number_input(
            "Skill Score (0–100)",
            min_value=0, max_value=100, value=75, step=1,
            help="Your technical skills assessment score."
        )
        personality_score = st.number_input(
            "Personality Score (0–100)",
            min_value=0, max_value=100, value=60, step=1,
            help="Your personality traits evaluation score."
        )
        recruitment_strategy = st.selectbox(
            "Recruitment Strategy",
            options=[
                (1, "Aggressive"),
                (2, "Moderate"),
                (3, "Conservative")
            ],
            format_func=lambda x: x[1],
            help="The hiring team’s recruitment strategy (e.g., Aggressive may prioritize top talent)."
        )[0]

    col_submit, col_reset = st.columns([1, 1])
    with col_submit:
        submitted = st.form_submit_button("Predict Outcome")
    with col_reset:
        reset = st.form_submit_button("Reset Form")

if reset:
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

if submitted:
    input_dict = {
        'Age': age,
        'Gender': gender,
        'EducationLevel': education_level,
        'ExperienceYears': experience_years,
        'PreviousCompanies': previous_companies,
        'DistanceFromCompany': distance_from_company,
        'InterviewScore': interview_score,
        'SkillScore': skill_score,
        'PersonalityScore': personality_score,
        'RecruitmentStrategy': recruitment_strategy
    }

    try:
        input_df = process_job_seeker_input(input_dict, model_features)
        
        with st.spinner("Analyzing your profile..."):
            time.sleep(1)
            result = generate_suggestions(rf_classifier, input_df, changeable_features, model_features, X_ref, y_ref)

        st.subheader("Prediction Result")
        st.write(f"**Outcome**: {result['prediction']}")
        st.write(f"**Probability of Being Hired**: {result['probability']:.3f}")

        if result['prediction'] == 'Hired':
            st.success("Congratulations! You are predicted to be hired.")
        else:
            st.error("You are predicted to be not hired. See suggestions below to improve your chances.")
            st.subheader("Reason for Rejection")
            st.write(f"The feature most responsible is **{result['explanation']['feature']}** "
                     f"(reducing your hiring probability by {abs(result['explanation']['shap_value']):.3f}).")

            if result['suggestions']:
                st.subheader("Suggestions to Get Hired")
                for i, suggestion in enumerate(result['suggestions'], 1):
                    st.write(f"**Suggestion {i} (Predicted Hiring Probability: {suggestion['probability']:.3f})**:")
                    for change in suggestion['changes']:
                        st.write(f"- {change['advice']}")
            else:
                st.warning("No suggestions generated. Try improving multiple skills or applying under a different recruitment strategy.")

    except Exception as e:
        st.error(f"Error: {e}")
