import os
import openai
import pandas as pd
import asyncio
import streamlit as st

# Load patient data
csv_file_path = '/home/cdsw/2_datasets/patient_data.csv'
patient_df = pd.read_csv(csv_file_path)
patient_options = [f"{row['patient_id']}, {row['name']}" for _, row in patient_df.iterrows()]
sample_questions = [
    "What are some medications to treat this condition?",
    "What is a preliminary diagnosis for this patient?",
    "What are suggestions to improve the patient's health?"
]

async def get_openai_response(messages):
    openai.api_key = os.getenv("OPENAI_KEY")
    for _ in range(3):
        try:
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-4",
                messages=messages,
                timeout=30
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            await asyncio.sleep(2)
    return "Error: Failed to fetch response."

async def get_patient_info_and_suggestions(patient_info, question, predefined_question):
    if not patient_info or ', ' not in patient_info:
        return "Error: Invalid selection", "No suggestions available"

    patient_id, patient_name = patient_info.split(', ', 1)
    matching_patient = patient_df[patient_df['patient_id'] == patient_id]
    if matching_patient.empty:
        return "Error: Patient not found", "No suggestions available"

    profile = matching_patient.iloc[0].to_dict()
    profile_str = "\n".join([f"{key}: {value}" for key, value in profile.items() if key != 'patient_id'])
    actual_question = predefined_question if not question else question
    suggestions = await get_openai_response([
        {"role": "system", "content": f"Patient profile: {profile}"},
        {"role": "user", "content": actual_question}
    ])
    return profile_str, suggestions

def main():
    st.title("Patient Data Insights")
    
    st.sidebar.header("Patient Selector")
    selected_patient = st.sidebar.selectbox("Select Patient", patient_options, index=0)
    st.sidebar.header("Questions")
    question = st.sidebar.text_input("Enter your question")
    predefined_question = st.sidebar.radio("Predefined Questions", sample_questions)
    
    if st.sidebar.button("Get Suggestions"):
        with st.spinner("Fetching suggestions..."):
            patient_info, suggestions = asyncio.run(
                get_patient_info_and_suggestions(selected_patient, question, predefined_question)
            )
        st.subheader("Patient Profile")
        st.text(patient_info)
        st.subheader("Suggestions")
        st.text(suggestions)

if __name__ == "__main__":
    main()
