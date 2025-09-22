#!/usr/bin/env python
# coding: utf-8

# In[2]:


import openai
import os
import pandas as pd

# Set the API key manually
api_key = ""  # Replace with your actual OpenAI API key

# Initialize OpenAI client
client = openai.OpenAI(api_key=api_key)

if not client.api_key:
    raise ValueError("Please set your OpenAI API key.")

# Define the folder containing clinical profiles
clinical_profiles_folder = "D:\\MIMICIV_20_Patients\\Round 3\\Clinical Profiles\\"  # Update with the actual path

def read_clinical_profiles(folder_path):
    """
    Reads clinical profiles from text files in a given folder.
    """
    clinical_profiles = {}
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, "r", encoding="utf-8") as file:
                    clinical_profiles[filename] = file.read().strip()
    return clinical_profiles

def get_model_response(prompt):
    """
    Calls the GPT-4 API with the given prompt and returns the response.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a medical professional."},
                      {"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API call error: {e}")
        return None

def generate_questions_with_gpt4(clinical_profile):
    """
    Calls GPT-4 API to generate 20 patient-friendly clinical questions 
    based on the given clinical profile and abnormal lab test results.
    """
    prompt = f"""**Your task is to generate a list of 20 patient-friendly clinical questions for a patient to ask their clinician regarding abnormal lab test results.**  

### **Steps to Follow:**  
1. **Review the patient’s clinical profile** to understand their general health background, including medical history and current medications.  
2. **Analyze the lab test data** to identify abnormal values, clearly noting which values fall outside the reference ranges and understanding their clinical significance.  
3. **Generate a list of 20 precise clinical questions** that address these abnormal lab results.  
4. **Sort the questions by clinical urgency**, starting with potentially life-threatening concerns (e.g., kidney function, high glucose, cardiovascular risks) and progressing to less critical issues.  
5. **For each question:**  
   - Explicitly state the **abnormal lab test values** and why they are significant.  
   - Reference the patient’s **medical conditions** (e.g., Type 2 diabetes, kidney disease, peripheral vascular disease, hyperlipidemia) and **current medications** (e.g., insulin detemir, diltiazem, levothyroxine).  
   - Focus on **actionable outcomes**, such as adjusting medications, scheduling follow-up tests, or implementing lifestyle changes.  
6. **Maintain the sequence of lab tests** as provided in the clinical data and limit the output to exactly **20 questions**, ensuring the discussion fits within a **15-minute clinical visit**.  
7. **Use simple, 6th-grade reading level language** to ensure the questions are patient-friendly.  

### **Output Format:**  
The output should be structured as follows:  

`Lab test name \t Question`  

### **Example Output:**  
```
Glucose	My glucose levels were high on recent tests (229 mg/dL and 228 mg/dL). Given my Type 2 diabetes, should I adjust my insulin detemir dose or add other medications to control my blood sugar?  
estimated glomerular filtration rate (eGFR)	My estimated glomerular filtration rate (eGFR) is low (46 mL/min/1.73m² for Black and 40 mL/min/1.73m² for non-Black patients). Does this suggest kidney disease, and should we adjust any medications like diltiazem that might impact kidney function?  
```

### **Examples of Highly Rated Questions:**  
- **Actionable:** *My triglyceride level is 163 mg/dL, which seems elevated. Should I make specific dietary changes or increase my exercise to lower this, considering my Type 2 diabetes and hyperlipidemia?*  
- **Specific:** *My hemoglobin A1c is elevated at 7.6%, indicating poor blood sugar control. Can we discuss lifestyle or dietary changes to improve this, or consider different diabetes treatments?*  
- **Medication-aware:** *The creatinine level is slightly high at 1.26 mg/dL. Given my kidney function decline, would you recommend any lifestyle or dietary changes, or additional testing for kidney health?*  

### **Examples of Lower Rated Questions to Avoid:**  
- **Too vague:** *Is my chloride level of 105 mmol/L normal, or could it indicate dehydration or other electrolyte imbalances that need attention?*  
- **Not actionable:** *My ALT and AST levels are both low (ALT at 11 IU/L and AST at 16 IU/L). Does this indicate any issues with my liver function, especially with my hyperlipidemia?*  
- **Lacks context:** *My sodium and potassium levels are within range, but is there any specific diet or hydration advice to ensure these remain balanced, especially considering my hypertension?*  

### **Final Notes:**  
- **Ensure all 20 questions strictly follow the requested format.**  
- **Focus on lab test abnormalities and their implications.**  
- **Avoid general symptoms-based questions (e.g., cough, cardiomyopathy) unless linked directly to lab abnormalities.**  


"""

    return get_model_response(prompt)

# Read clinical profiles
clinical_profiles = read_clinical_profiles(clinical_profiles_folder)

# Process each clinical profile and generate questions
all_questions = []
for filename, profile_content in clinical_profiles.items():
    questions_text = generate_questions_with_gpt4(profile_content)
    if questions_text:
        questions = [line.split("\t") for line in questions_text.split("\n") if "\t" in line]
        for question in questions:
            all_questions.append([filename] + question)

# Create a DataFrame and save it as a CSV file
df = pd.DataFrame(all_questions, columns=["Patient File", "Lab test name", "Question"])
csv_file_path = "D:\\MIMICIV_20_Patients\\Round 3\\Clinical Profiles\\generated_lab_questions_gpt4o.csv"  # Update the path
df.to_csv(csv_file_path, index=False)

print(f"Results saved to {csv_file_path}")


# In[ ]:




