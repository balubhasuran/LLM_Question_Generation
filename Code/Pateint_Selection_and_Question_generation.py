import pandas as pd


lab_tests=pd.read_csv('OutPateint_Labs.csv',low_memory=False)
med=pd.read_csv('OutPateint_Med.csv',low_memory=False)
diag=pd.read_csv('OutPateint_diags.csv',low_memory=False)
demographics=pd.read_csv('OutPateint_demo.csv',low_memory=False)

#Version 3 for evaluation
import pandas as pd
from datetime import datetime

def generate_clinical_prompt_for_patient(patient_id, demographics, lab_tests, med, diag):
    # Ensure dates are in datetime format
    demographics['DOB'] = pd.to_datetime(demographics['BIRTH_DATE'])
    lab_tests['RESULT_DATE'] = pd.to_datetime(lab_tests['RESULT_DATE'])
    med['RX_START_DATE'] = pd.to_datetime(med['RX_START_DATE'])
    diag['ADMIT_DATE'] = pd.to_datetime(diag['ADMIT_DATE'])

    # Calculate the latest lab test date for the patient
    latest_lab_date = lab_tests[lab_tests['patient_id'] == patient_id]['RESULT_DATE'].max()

    # Calculate the age as of the latest lab test date
    dob = demographics.loc[demographics['patient_id'] == patient_id, 'DOB'].iloc[0]
    def calculate_age(dob, test_date):
        if pd.isnull(test_date) or pd.isnull(dob):
            return None
        return test_date.year - dob.year - ((test_date.month, test_date.day) < (dob.month, dob.day))
    
    age_as_of_lab = calculate_age(dob, latest_lab_date)

    # Filter data for the given patient ID
    patient_demo = demographics[demographics['patient_id'] == patient_id]
    patient_labs = lab_tests[lab_tests['patient_id'] == patient_id]
    patient_med = med[med['patient_id'] == patient_id]
    patient_diag = diag[diag['patient_id'] == patient_id]

    # Filter labs of interest
    labs_of_interest = [
        'Alanine aminotransferase', 'Albumin', 'Albumin/Globulin', 'Alkaline phosphatase',
        'Aspartate aminotransferase', 'Bilirubin', 'Bilirubin.glucuronidated+Bilirubin.albumin bound',
        'Bilirubin.non-glucuronidated', 'C reactive protein', 'Calcium', 'Chloride', 'Cholesterol',
        'Cholesterol.in HDL', 'Cholesterol.in LDL', 'Cholesterol.non HDL',
        'Cholesterol.total/Cholesterol.in HDL', 'Creatine kinase', 'Creatinine',
        'Estimated average glucose', 'Ferritin', 'Globulin', 'Glomerular filtration rate/1.73 sq M.predicted',
        'Glomerular filtration rate/1.73 sq M.predicted.black',
        'Glomerular filtration rate/1.73 sq M.predicted.non black', 'Glucose', 'Hematocrit',
        'Hemoglobin', 'Hemoglobin A1c/Hemoglobin.total', 'Phosphate', 'Platelets', 'Potassium',
        'Protein', 'Sodium', 'Triglyceride', 'Urea nitrogen', 'Urea nitrogen/Creatinine'
    ]
    patient_labs = patient_labs[patient_labs['COMPONENT'].isin(labs_of_interest)]
    
    # Process labs
    latest_labs = patient_labs.groupby('COMPONENT', group_keys=False).apply(lambda x: x.nlargest(4, 'RESULT_DATE'))
    latest_labs_sorted = latest_labs.sort_values(by='RESULT_DATE', ascending=False)
    latest_labs_sorted['RESULT_DATE'] = latest_labs_sorted['RESULT_DATE'].dt.strftime('%Y-%m-%d')

    # Map 'ABN_IND' values to their corresponding meanings
    abnormal_ind_map = {
        'AB': 'Abnormal',
        'AH': 'Abnormally high',
        'AL': 'Abnormally low',
        'CH': 'Critically high',
        'CL': 'Critically low',
        'CR': 'Critical',
        'IN': 'Inconclusive',
        'NL': 'Normal',
        'NI': 'No information',
        'UN': 'Unknown',
        'OT': 'Other'
    }
    latest_labs_sorted['ABN_IND'] = latest_labs_sorted['ABN_IND'].map(abnormal_ind_map)

    # Create a filtered lab list
    def format_lab_row(row):
        components = [str(row['RESULT_DATE']), str(row['LAB_LOINC']), row['COMPONENT'], str(row['RESULT_NUM'])]
        if row['RESULT_UNIT'] != 'NI':
            components.append(str(row['RESULT_UNIT']))  # Ensure string conversion
        if pd.notna(row['NORM_RANGE_LOW']) and pd.notna(row['NORM_RANGE_HIGH']):
            components.append(f"{str(row['NORM_RANGE_LOW'])} - {str(row['NORM_RANGE_HIGH'])}")  # Ensure string conversion
        if row['ABN_IND'] != 'No information':
            components.append(str(row['ABN_IND']))  # Ensure string conversion
        return ', '.join(components)

    
    lab_list = latest_labs_sorted.apply(format_lab_row, axis=1).tolist()
    
    # Process medications and remove duplicates, sorted by descending dates
    latest_med = patient_med.groupby('RXNORM_CUI', group_keys=False).apply(lambda x: x.nlargest(4, 'RX_START_DATE'))
    latest_med_sorted = latest_med.sort_values(by='RX_START_DATE', ascending=False).drop_duplicates(
        subset=['RXNORM_CUI']
    )
    latest_med_sorted['RX_START_DATE'] = latest_med_sorted['RX_START_DATE'].dt.strftime('%Y-%m-%d')
    med_list = latest_med_sorted[['RX_START_DATE', 'RXNORM_CUI', 'Drug']].apply(
        lambda x: ', '.join(x.astype(str)), axis=1).tolist()
    
    # Process diagnoses and remove duplicates, sorted by descending dates
    latest_diag = patient_diag.groupby('DX', group_keys=False).apply(lambda x: x.nlargest(4, 'ADMIT_DATE'))
    latest_diag_sorted = latest_diag.sort_values(by='ADMIT_DATE', ascending=False).drop_duplicates(
        subset=['DX']
    )
    latest_diag_sorted['ADMIT_DATE'] = latest_diag_sorted['ADMIT_DATE'].dt.strftime('%Y-%m-%d')
    diag_list = latest_diag_sorted[['ADMIT_DATE', 'DX', 'ICD10 String']].apply(
        lambda x: ', '.join(x.astype(str)), axis=1).tolist()
    
    # Extract demographic information
    gen = patient_demo['SEX'].iloc[0]
    eth = patient_demo['HISPANIC'].iloc[0]
    gender = 'Female' if gen == 'F' else 'Male'
    race_code = str(patient_demo['RACE'].iloc[0])
    race_mapping = {
        '1': 'American Indian or Alaska Native', '2': 'Asian', '3': 'African American',
        '4': 'Native Hawaiian or Other Pacific Islander', '5': 'White', '6': 'Multiple race'
    }
    race = race_mapping.get(race_code, 'OT')
    if race == 'OT' and eth == 'Y':
        race = 'Hispanic'
    elif race == 'OT':
        race = 'Non-Hispanic'

    # Base info
    base_info = f"I am a {age_as_of_lab} year old {race.lower()} {gender.lower()}."
    
    # Generate the prompt
    prompt = f"""
    Generate a list of 20 precise clinical questions for a patient to ask their clinician regarding abnormal lab test results. Follow these guidelines:
    Guidelines:
    Order of Questions: Rank questions by clinical urgency, starting with life-threatening concerns such as kidney function, glucose levels, and cardiovascular risks.
    Highlight Abnormal Values: Explicitly mention which lab test values are abnormal and explain their significance in the question.
    Medication and Disease Context: Align each question with the patient's medical history (e.g., diabetes, hypertension) and current medications (e.g., insulin detemir, diltiazem, levothyroxine).
    Actionable Outcomes: Focus on specific steps the patient can take, including treatment adjustments, follow-ups, and lifestyle changes.
    Patient-Friendly Language: Use 6th-grade reading level and patient-friendly terms.
    Consistency: Maintain the sequence of lab tests as provided in the clinical data.
    Time-Efficient: Limit to 20 questions to fit into a 15-minute discussion with the clinician.

    Patient's Profile and Clinical Data:
    {base_info}

    The following are my latest lab test results:
    Date, LOINC, Lab Test name, Test Value, Unit, Normal Range Low-High, Abnormal Indication
    {chr(10).join(['    ' + lab for lab in lab_list])}

    The following medications were recently used:
    Date, Drug_CUI, Medication
    {chr(10).join(['    ' + med for med in med_list])}

    The following diagnoses were listed into my chart:
    Date, ICD Code, Diagnosis
    {chr(10).join(['    ' + diag for diag in diag_list])}

    Output:
    Generate only the questions in plain text without extra context or numbering. Each question should:
    - Explicitly state abnormal lab values.
    - Mention the patient’s diseases (e.g., Type 2 diabetes, kidney disease, peripheral vascular disease, hyperlipidemia).
    - Include relevant medications (e.g., insulin detemir, levothyroxine). 
    - Focus on proactive actions the patient can take, such as adjusting medications or following up with specific tests.
    - The goal is to support informed discussions during the clinical visit by focusing on urgent and actionable health issues.
    """
    return prompt


# Loop through each patient ID and generate prompts, saving them as .txt files
i = 0
for patient_id in demographics['patient_id'].unique():
    # Generate the prompt for the current patient
    prompt = generate_clinical_prompt_for_patient(patient_id, demographics, lab_tests, med, diag)
    i += 1
    # Define the file name, including the patient ID
    filename = f"D:\\MIMICIV_20_Patients\\Round 3\\patient_prompt_{i}.txt"
    
    # Save the prompt to a .txt file
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(prompt)
    
    print(f"Prompt for patient {patient_id} saved as {filename}")

