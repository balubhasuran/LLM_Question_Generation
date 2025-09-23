#!/usr/bin/env python
# coding: utf-8

# In[7]:


#Round 3 Feb 11 2025
import pandas as pd
from datetime import datetime
import os
import glob


# In[8]:


lab_tests=pd.read_csv('OutPateint_Labs.csv',low_memory=False)
med=pd.read_csv('OutPateint_Med.csv',low_memory=False)
diag=pd.read_csv('OutPateint_diags.csv',low_memory=False)
demographics=pd.read_csv('OutPateint_demo.csv',low_memory=False)


# In[9]:


# filters and prints only abnormal values for HbA1c, eGFR, and creatinine from the patient lab profile


def generate_clinical_prompt_for_patient(patient_id, demographics, lab_tests, med, diag):
    # Define reference ranges for selected labs.
    # For Creatinine, a default is provided but will be overridden by gender below.
    reference_ranges = {
        "Hemoglobin A1c/Hemoglobin.total": (4.0, 5.6),  # Example: ratio normal range
        "Glomerular filtration rate/1.73 sq M.predicted.black": (60, 120),
        "Glomerular filtration rate/1.73 sq M.predicted.non black": (60, 120),
        "Creatinine": (0.6, 1.3),  # Default value; will be replaced based on gender.
        "Urea nitrogen/Creatinine": (10, 20),  # Example values
        "Creatine kinase": (30, 200)  # Example reference range; adjust as needed.
    }

    # Ensure date columns are in datetime format.
    demographics['DOB'] = pd.to_datetime(demographics['BIRTH_DATE'])
    lab_tests['RESULT_DATE'] = pd.to_datetime(lab_tests['RESULT_DATE'])
    med['RX_START_DATE'] = pd.to_datetime(med['RX_START_DATE'])
    diag['ADMIT_DATE'] = pd.to_datetime(diag['ADMIT_DATE'])

    # Filter lab tests for the given patient.
    patient_labs = lab_tests[lab_tests['patient_id'] == patient_id]
    if patient_labs.empty:
        return None

    # Calculate the latest lab date and the patient's age as of that date.
    latest_lab_date = patient_labs['RESULT_DATE'].max()
    dob = demographics.loc[demographics['patient_id'] == patient_id, 'DOB'].iloc[0]
    def calculate_age(dob, test_date):
        if pd.isnull(test_date) or pd.isnull(dob):
            return None
        return test_date.year - dob.year - ((test_date.month, test_date.day) < (dob.month, dob.day))
    age_as_of_lab = calculate_age(dob, latest_lab_date)

    # Filter other patient data.
    patient_demo = demographics[demographics['patient_id'] == patient_id]
    patient_med = med[med['patient_id'] == patient_id]
    patient_diag = diag[diag['patient_id'] == patient_id]

    # --- Extract Demographic Information Needed for Lab Reference ---
    # Get gender early so that the helper function can use it.
    gen = patient_demo['SEX'].iloc[0]
    # Assuming 'F' indicates female; otherwise, male.
    gender = 'Female' if gen == 'F' else 'Male'

    # --- Check for Abnormal Labs ---
    # Restrict to lab tests that have a defined reference range.
    labs_with_range = patient_labs[patient_labs['COMPONENT'].isin(reference_ranges.keys())]
    # Identify abnormal labs where the RESULT_NUM is either below the lower bound or above the upper bound.
    abnormal_labs = labs_with_range[
        (labs_with_range['RESULT_NUM'] < labs_with_range['COMPONENT'].map(lambda x: reference_ranges[x][0])) |
        (labs_with_range['RESULT_NUM'] > labs_with_range['COMPONENT'].map(lambda x: reference_ranges[x][1]))
    ]
    # If there are no abnormal labs, skip this patient.
    if abnormal_labs.empty:
        return None

    # --- Process Lab Tests ---
    # Select labs from the most recent date; if fewer than 50 labs exist, take the 50 most recent.
    most_recent_labs = patient_labs[patient_labs['RESULT_DATE'] == latest_lab_date]
    if len(most_recent_labs) < 50:
        most_recent_labs = patient_labs.nlargest(50, 'RESULT_DATE')
    latest_labs_sorted = most_recent_labs.sort_values(by='RESULT_DATE', ascending=False)
    latest_labs_sorted['RESULT_DATE'] = latest_labs_sorted['RESULT_DATE'].dt.strftime('%Y-%m-%d')

    # Helper function to format each lab result line.
    def format_lab_result(row):
        date_str = str(row['RESULT_DATE'])
        component_str = str(row['COMPONENT'])
        unit_str = str(row['RESULT_UNIT'])
        value_str = str(row['RESULT_NUM'])
        normal_range = ""
        abnormal_indication = ""
        # For labs with a defined reference range...
        if component_str in reference_ranges:
            # For Creatinine, override based on gender.
            if component_str == "Creatinine":
                if gender == "Male":
                    low, high = 0.7, 1.3
                else:  # Female
                    low, high = 0.5, 1.1
            else:
                low, high = reference_ranges[component_str]
            normal_range = f"{low} - {high}"
            try:
                numeric_value = float(value_str)
                if numeric_value < low:
                    abnormal_indication = "Abnormally low"
                elif numeric_value > high:
                    abnormal_indication = "Abnormally high"
            except Exception:
                pass
        # Construct the output line.
        parts = [date_str, component_str, unit_str, value_str]
        if normal_range:
            parts.append(normal_range)
        if abnormal_indication:
            parts.append(abnormal_indication)
        return ", ".join(parts)

    lab_list = latest_labs_sorted.apply(format_lab_result, axis=1).tolist()

    # --- Process Medications ---
    # Keep the most recent 10 medications (ensuring no duplicate RXNORM_CUI).
    latest_med_sorted = patient_med.sort_values(by='RX_START_DATE', ascending=False)
    latest_med_sorted = latest_med_sorted.drop_duplicates(subset=['RXNORM_CUI']).head(30)
    latest_med_sorted['RX_START_DATE'] = latest_med_sorted['RX_START_DATE'].dt.strftime('%Y-%m-%d')
    med_list = latest_med_sorted[['RX_START_DATE', 'RXNORM_CUI', 'Drug']].apply(
        lambda x: ", ".join(x.astype(str)), axis=1).tolist()

    # --- Process Diagnoses ---
    # Keep the most recent 10 diagnoses (ensuring no duplicate ICD codes).
    latest_diag_sorted = patient_diag.sort_values(by='ADMIT_DATE', ascending=False)
    latest_diag_sorted = latest_diag_sorted.drop_duplicates(subset=['DX']).head(30)
    latest_diag_sorted['ADMIT_DATE'] = latest_diag_sorted['ADMIT_DATE'].dt.strftime('%Y-%m-%d')
    diag_list = latest_diag_sorted[['ADMIT_DATE', 'DX', 'ICD10 String']].apply(
        lambda x: ", ".join(x.astype(str)), axis=1).tolist()

    # --- Extract Additional Demographic Information ---
    # (Now including race, which is used only for the profile text.)
    eth = patient_demo['HISPANIC'].iloc[0]
    race_code = str(patient_demo['RACE'].iloc[0])
    race_mapping = {
        '1': 'American Indian or Alaska Native',
        '2': 'Asian',
        '3': 'African American',
        '4': 'Native Hawaiian or Other Pacific Islander',
        '5': 'White',
        '6': 'Multiple race'
    }
    race = race_mapping.get(race_code, 'OT')
    if race == 'OT' and eth == 'Y':
        race = 'Hispanic'
    elif race == 'OT':
        race = 'Non-Hispanic'

    # Base patient information.
    base_info = f"I am a {age_as_of_lab} year old {race.lower()} {gender.lower()}."

    # --- Generate the Clinical Prompt ---
    prompt = f"""
Generate a list of 20 precise clinical questions for a patient to ask their clinician regarding abnormal lab test results. Follow these guidelines:

Patient's Profile and Clinical Data:
{base_info}

The following are my latest lab test results:
Date, Lab Test Name, Unit, Test Value, Normal Range, Abnormal Indication (if applicable)
{chr(10).join(lab_list)}

The following medications were recently used:
Date, Drug_CUI, Medication
{chr(10).join(med_list)}

The following diagnoses were listed into my chart:
Date, ICD Code, Diagnosis
{chr(10).join(diag_list)}

Output:
Generate only the questions in plain text without extra context or numbering. Each question should:
- Explicitly state abnormal lab values.
- Mention the patient’s diseases (e.g., Type 2 diabetes, kidney disease, peripheral vascular disease, hyperlipidemia).
- Include relevant medications (e.g., insulin detemir, levothyroxine).
- Focus on proactive actions the patient can take, such as adjusting medications or following up with specific tests.
- The goal is to support informed discussions during the clinical visit by focusing on urgent and actionable health issues.
"""
    return prompt

# --- Loop through Each Patient ---
i = 0
for patient_id in demographics['patient_id'].unique():
    prompt = generate_clinical_prompt_for_patient(patient_id, demographics, lab_tests, med, diag)
    if prompt:
        i += 1
        filename = f"D:\\MIMICIV_20_Patients\\Round 3\\patient_question_prompt_{i}.txt"
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(prompt)
        print(f"Prompt for patient {patient_id} saved as {filename}")


# In[10]:


def filter_sort_and_save_profiles(source_directory, target_directory):
    # Pattern to match the clinical profile files in the source directory.
    file_pattern = os.path.join(source_directory, "patient_question_prompt_*.txt")
    profile_files = glob.glob(file_pattern)
    
    profiles = []
    
    # Loop over each file, read its content, and analyze abnormal lab lines.
    for file in profile_files:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Split content into lines.
        lines = content.splitlines()
        # Identify lines that mention an abnormal lab value.
        abnormal_lab_lines = [line for line in lines if "Abnormally high" in line or "Abnormally low" in line]
        
        # Extract lab test names from abnormal lines.
        abnormal_lab_names = set()
        total_abnormal_count = 0
        for line in abnormal_lab_lines:
            # Expecting a format like:
            # Date, Lab Test Name, Unit, Test Value, Normal Range, Abnormal Indication
            fields = line.split(",")
            if len(fields) >= 2:
                lab_name = fields[1].strip()  # Lab test name is the second field.
                abnormal_lab_names.add(lab_name)
                total_abnormal_count += 1
        
        distinct_abnormal_count = len(abnormal_lab_names)
        
        # Only include profiles that have abnormalities in at least two different labs.
        if distinct_abnormal_count >= 2:
            profiles.append((file, distinct_abnormal_count, total_abnormal_count, content))
    
    # Sort profiles:
    # - First by the number of distinct abnormal labs (descending)
    # - Then by the total abnormal count (descending)
    profiles_sorted = sorted(profiles, key=lambda x: (x[1], x[2]), reverse=True)
    
    # Create target directory if it doesn't exist.
    os.makedirs(target_directory, exist_ok=True)
    
    # Save the sorted profiles into the target directory.
    for i, (orig_file, distinct_count, total_count, content) in enumerate(profiles_sorted, start=1):
        new_filename = os.path.join(target_directory, f"sorted_profile_{i}.txt")
        with open(new_filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Saved sorted profile {i} (Distinct abnormal labs: {distinct_count}, Total abnormal count: {total_count}) to {new_filename}")

    # Optionally, print out the sorted profiles.
    for file, distinct_count, total_count, content in profiles_sorted:
        print(f"File: {file} - Distinct abnormal labs: {distinct_count}, Total abnormal count: {total_count}")
        print(content)
        print("=" * 100)  # Separator between profiles

# Specify the directory where the clinical profiles are saved.
source_directory = r"D:\MIMICIV_20_Patients\Round 3"
# Specify the target directory where the sorted profiles should be saved.
target_directory = r"D:\MIMICIV_20_Patients\Round 3\Sorted_Profiles"

filter_sort_and_save_profiles(source_directory, target_directory)


# In[6]:


import os
import glob
from datetime import datetime

def process_lab_section(lab_lines):
    """
    Process the lab test results so that:
      1. Only records from the overall latest date in the lab section are retained.
      2. For tests that appear more than once on that date, only the first occurrence is kept.
      
    If a header line (e.g., "Date, Lab Test Name, ...") is present, it is preserved.
    """
    # Separate header (if present) from data lines.
    header_line = None
    data_lines = lab_lines[:]
    if data_lines and data_lines[0].strip().lower().startswith("date"):
        header_line = data_lines.pop(0).rstrip("\n")
    
    parsed = []  # List of tuples: (date_obj, lab_name, line)
    max_date = None
    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(',')]
        # Expect at least 4 fields: Date, Lab Test Name, Unit, and Test Value.
        if len(parts) < 4:
            continue
        date_str = parts[0]
        lab_name = parts[1]
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            continue
        parsed.append((dt, lab_name, line))
        if max_date is None or dt > max_date:
            max_date = dt

    # Filter to keep only records with the maximum date.
    latest_records = [ (dt, lab_name, line) for (dt, lab_name, line) in parsed if dt == max_date ]
    
    # Now group by lab test name: keep only the first occurrence if duplicates exist.
    latest_by_lab = {}
    for dt, lab_name, line in latest_records:
        if lab_name not in latest_by_lab:
            latest_by_lab[lab_name] = line

    # Prepare the output list.
    processed_lines = list(latest_by_lab.values())
    
    # If a header was present, insert it at the top.
    if header_line:
        processed_lines.insert(0, header_line)
        
    return processed_lines

def get_lab_section(file_path):
    """
    Extracts the lab section lines from a file. The lab section is defined as all lines
    between the line containing "The following are my latest lab test results:" and the line
    containing "The following medications were recently used:".
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    lab_start_index = None
    lab_end_index = None
    for i, line in enumerate(lines):
        if "The following are my latest lab test results:" in line:
            lab_start_index = i
        if "The following medications were recently used:" in line:
            lab_end_index = i
            if lab_start_index is not None:
                break

    if lab_start_index is None or lab_end_index is None:
        return []
    
    # Return the lab section lines (everything after the lab header until the medications section)
    return lines[lab_start_index + 1:lab_end_index]

def count_abnormal_in_file(file_path):
    """
    Processes the lab section (filtering to only the latest date and unique lab tests)
    and counts the number of lab records flagged as either "Abnormally high" or "Abnormally low".
    """
    lab_section = get_lab_section(file_path)
    if not lab_section:
        return 0
    processed_lab = process_lab_section(lab_section)
    count = 0
    for line in processed_lab:
        # Check (case-insensitive) for abnormal flags.
        if "abnormally high" in line.lower() or "abnormally low" in line.lower():
            count += 1
    return count

def get_top_20_abnormal_files(input_folder):
    """
    Scans all .txt files in the input folder, counts the number of abnormal lab results
    (both "Abnormally high" and "Abnormally low") in each file (after filtering to only the latest date),
    and returns the top 20 files with the highest abnormal counts.
    """
    abnormal_counts = {}
    file_pattern = os.path.join(input_folder, '*.txt')
    for file_path in glob.glob(file_pattern):
        count = count_abnormal_in_file(file_path)
        abnormal_counts[file_path] = count

    # Sort files by abnormal count in descending order.
    sorted_files = sorted(abnormal_counts.items(), key=lambda x: x[1], reverse=True)
    top_20 = sorted_files[:20]
    return top_20

def process_and_write_file(file_path, output_folder):
    """
    Processes a single file:
      - Only the lab test section is processed so that only labs from the latest date (and only one per test)
        are retained.
      - All other parts of the file remain unchanged.
    The modified file is then written to the output folder.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    lab_start_index = None
    lab_end_index = None
    for i, line in enumerate(lines):
        if "The following are my latest lab test results:" in line:
            lab_start_index = i
        if "The following medications were recently used:" in line:
            lab_end_index = i
            if lab_start_index is not None:
                break

    if lab_start_index is None or lab_end_index is None:
        print(f"Lab section markers not found in {file_path}. File will be written unmodified.")
        output_lines = lines
    else:
        # Split the file into three parts.
        before_lab = lines[:lab_start_index + 1]
        lab_section = lines[lab_start_index + 1:lab_end_index]
        after_lab = lines[lab_end_index:]
        
        # Process the lab section.
        processed_lab = process_lab_section(lab_section)
        # Ensure each processed lab line ends with a newline.
        processed_lab = [line if line.endswith("\n") else line + "\n" for line in processed_lab]
        
        # Reassemble the file content.
        output_lines = before_lab + processed_lab + after_lab

    # Ensure the output folder exists.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    base_name = os.path.basename(file_path)
    output_file_path = os.path.join(output_folder, f"processed_{base_name}")
    with open(output_file_path, 'w') as out_f:
        out_f.writelines(output_lines)
    
    print(f"Processed file written to: {output_file_path}")

def process_folder(input_folder, output_folder):
    """
    Processes all .txt files in the input folder:
      - For each file, only the lab test section is modified (keeping only the labs from the latest date).
      - The complete file is then written to the output folder.
    """
    file_pattern = os.path.join(input_folder, '*.txt')
    for file_path in glob.glob(file_pattern):
        process_and_write_file(file_path, output_folder)

# -------------------------
# Main execution starts here
# -------------------------
# Example usage:
if __name__ == "__main__":
    input_folder = "D:\MIMICIV_20_Patients\Round 3\Sorted_Profiles"   # Replace with the path to your folder containing text files
    output_folder = "D:\MIMICIV_20_Patients\Round 3"    # Output folder to store processed files
    

    # 1. Identify and print the top 20 files having the highest combined count of abnormal labs.
    top_20_files = get_top_20_abnormal_files(input_folder)
    print("Top 20 files with highest number of combined 'Abnormally high' and 'Abnormally low' labs:")
    for file_path, abnormal_count in top_20_files:
        print(f"File: {file_path}  |  Abnormal Count: {abnormal_count}")

    # 2. Process each file (modify only the lab section) and write the full file to the output folder.
    process_folder(input_folder, output_folder)


# In[12]:


#all in +/- 6 months
import os
import glob
from datetime import datetime, timedelta

# Define the allowed time window (in days) for medications and diagnoses relative to the lab's latest date.
TIME_DELTA_DAYS = 180

def process_lab_section(lab_lines):
    """
    Process the lab test results so that:
      1. Only records from the overall latest date in the lab section are retained.
      2. For tests that appear more than once on that date, only the first occurrence is kept.
      
    If a header line (e.g., "Date, Lab Test Name, ...") is present, it is preserved.
    
    Returns:
        processed_lines: List of processed lab lines.
        max_date: The latest lab test date (datetime object) found.
    """
    header_line = None
    data_lines = lab_lines[:]
    if data_lines and data_lines[0].strip().lower().startswith("date"):
        header_line = data_lines.pop(0).rstrip("\n")
    
    parsed = []  # List of tuples: (date_obj, lab_name, line)
    max_date = None
    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(',')]
        # Expect at least 4 fields: Date, Lab Test Name, Unit, and Test Value.
        if len(parts) < 4:
            continue
        date_str = parts[0]
        lab_name = parts[1]
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            continue
        parsed.append((dt, lab_name, line))
        if max_date is None or dt > max_date:
            max_date = dt

    # Filter to keep only records with the maximum date.
    latest_records = [(dt, lab_name, line) for (dt, lab_name, line) in parsed if dt == max_date]
    
    # Group by lab test name: keep only the first occurrence if duplicates exist.
    latest_by_lab = {}
    for dt, lab_name, line in latest_records:
        if lab_name not in latest_by_lab:
            latest_by_lab[lab_name] = line

    processed_lines = list(latest_by_lab.values())
    if header_line:
        processed_lines.insert(0, header_line)
        
    return processed_lines, max_date

def process_medication_section(med_lines, reference_date, delta_days=TIME_DELTA_DAYS):
    """
    Process the medication section lines, filtering only rows with dates within ± delta_days of reference_date.
    If a header line (e.g., "Date, Drug_CUI, Medication") is present, it is preserved.
    """
    header_line = None
    data_lines = med_lines[:]
    if data_lines and data_lines[0].strip().lower().startswith("date"):
        header_line = data_lines.pop(0).rstrip("\n")
    
    filtered = []
    if reference_date is None:
        # If no reference date, return all lines unfiltered.
        filtered = data_lines
    else:
        for line in data_lines:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            # Expect at least 3 fields: Date, Drug_CUI, Medication.
            if len(parts) < 3:
                continue
            date_str = parts[0]
            try:
                dt = datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                continue
            if abs((dt - reference_date).days) <= delta_days:
                filtered.append(line)
    if header_line:
        filtered.insert(0, header_line)
    return filtered

def process_diagnosis_section(diag_lines, reference_date, delta_days=TIME_DELTA_DAYS):
    """
    Process the diagnosis section lines, filtering only rows with dates within ± delta_days of reference_date.
    If a header line (e.g., "Date, ICD Code, Diagnosis") is present, it is preserved.
    """
    header_line = None
    data_lines = diag_lines[:]
    if data_lines and data_lines[0].strip().lower().startswith("date"):
        header_line = data_lines.pop(0).rstrip("\n")
    
    filtered = []
    if reference_date is None:
        filtered = data_lines
    else:
        for line in data_lines:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            # Expect at least 3 fields: Date, ICD Code, Diagnosis.
            if len(parts) < 3:
                continue
            date_str = parts[0]
            try:
                dt = datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                continue
            if abs((dt - reference_date).days) <= delta_days:
                filtered.append(line)
    if header_line:
        filtered.insert(0, header_line)
    return filtered

def get_lab_section(file_path):
    """
    Extracts the lab section lines from a file. The lab section is defined as all lines
    between the line containing "The following are my latest lab test results:" and the line
    containing "The following medications were recently used:".
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    lab_start_index = None
    lab_end_index = None
    for i, line in enumerate(lines):
        if "The following are my latest lab test results:" in line:
            lab_start_index = i
        if "The following medications were recently used:" in line:
            lab_end_index = i
            if lab_start_index is not None:
                break

    if lab_start_index is None or lab_end_index is None:
        return []
    
    return lines[lab_start_index + 1:lab_end_index]

def count_abnormal_in_file(file_path):
    """
    Processes the lab section (filtering to only the latest date and unique lab tests)
    and counts the number of lab records flagged as either "Abnormally high" or "Abnormally low".
    """
    lab_section = get_lab_section(file_path)
    if not lab_section:
        return 0
    processed_lab, _ = process_lab_section(lab_section)
    count = 0
    for line in processed_lab:
        if "abnormally high" in line.lower() or "abnormally low" in line.lower():
            count += 1
    return count

def get_top_40_abnormal_files(input_folder):
    """
    Scans all .txt files in the input folder, counts the number of abnormal lab results
    (both "Abnormally high" and "Abnormally low") in each file (after filtering to only the latest date),
    and returns the top 20 files with the highest abnormal counts.
    """
    abnormal_counts = {}
    file_pattern = os.path.join(input_folder, '*.txt')
    for file_path in glob.glob(file_pattern):
        count = count_abnormal_in_file(file_path)
        abnormal_counts[file_path] = count

    sorted_files = sorted(abnormal_counts.items(), key=lambda x: x[1], reverse=True)
    top_40 = sorted_files[:40]
    return top_40

def process_and_write_file(file_path, output_folder):
    """
    Processes a single file:
      - The lab test section is processed so that only labs from the latest date (and only one per test)
        are retained.
      - The medications and diagnoses sections are filtered to only include entries within ±TIME_DELTA_DAYS
        of the latest lab test date.
      - All other parts of the file remain unchanged.
    The modified file is then written to the output folder.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Identify section markers.
    lab_idx = med_idx = diag_idx = None
    for i, line in enumerate(lines):
        if "The following are my latest lab test results:" in line:
            lab_idx = i
        elif "The following medications were recently used:" in line:
            med_idx = i
        elif "The following diagnoses were listed into my chart:" in line:
            diag_idx = i

    if lab_idx is None or med_idx is None or diag_idx is None:
        print(f"Section markers not found in {file_path}. File will be written unmodified.")
        output_lines = lines
    else:
        # Split the file into parts.
        before_lab = lines[:lab_idx]
        lab_header = lines[lab_idx]
        lab_section = lines[lab_idx+1:med_idx]
        med_header = lines[med_idx]
        med_section = lines[med_idx+1:diag_idx]
        diag_header = lines[diag_idx]
        diag_section = lines[diag_idx+1:]
        
        # Process each section.
        processed_lab, lab_max_date = process_lab_section(lab_section)
        processed_lab = [line if line.endswith("\n") else line + "\n" for line in processed_lab]
        
        processed_med = process_medication_section(med_section, lab_max_date, delta_days=TIME_DELTA_DAYS)
        processed_med = [line if line.endswith("\n") else line + "\n" for line in processed_med]
        
        processed_diag = process_diagnosis_section(diag_section, lab_max_date, delta_days=TIME_DELTA_DAYS)
        processed_diag = [line if line.endswith("\n") else line + "\n" for line in processed_diag]
        
        # Reassemble the file.
        output_lines = []
        output_lines.extend(before_lab)
        output_lines.append(lab_header)
        output_lines.extend(processed_lab)
        output_lines.append(med_header)
        output_lines.extend(processed_med)
        output_lines.append(diag_header)
        output_lines.extend(processed_diag)
    
    # Ensure the output folder exists.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    base_name = os.path.basename(file_path)
    output_file_path = os.path.join(output_folder, f"processed_{base_name}")
    with open(output_file_path, 'w') as out_f:
        out_f.writelines(output_lines)
    
    print(f"Processed file written to: {output_file_path}")

def process_folder(input_folder, output_folder):
    """
    Processes all .txt files in the input folder:
      - For each file, the lab test section is modified (keeping only the labs from the latest date).
      - The medications and diagnoses sections are filtered based on the latest lab test date.
      - The complete file is then written to the output folder.
    """
    file_pattern = os.path.join(input_folder, '*.txt')
    for file_path in glob.glob(file_pattern):
        process_and_write_file(file_path, output_folder)

# -------------------------
# Main execution starts here
# -------------------------
if __name__ == "__main__":
    input_folder = r"D:\MIMICIV_20_Patients\Round 3\Sorted_Profiles"   # Replace with your input folder path.
    output_folder = r"D:\MIMICIV_20_Patients\Round 3\Processed"                 # Replace with your desired output folder.

    # 1. Identify and print the top 20 files with the highest abnormal lab counts.
    top_40_files = get_top_40_abnormal_files(input_folder)
    print("Top 20 files with highest number of combined 'Abnormally high' and 'Abnormally low' labs:")
    for file_path, abnormal_count in top_40_files:
        print(f"File: {file_path}  |  Abnormal Count: {abnormal_count}")

    # 2. Process each file and write the full modified file to the output folder.
    process_folder(input_folder, output_folder)


# In[ ]:




