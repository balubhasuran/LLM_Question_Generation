# Generating Question Prompt Lists From Electronic Health Record Data Using Large Language Models: Iterative Evaluation Study
Evaluating the feasibility of using large language models (LLMs) to generate patient-friendly, clinically relevant questions grounded in electronic health record laboratory data. 

Abstract
Background: Patients frequently access laboratory results through patient portals, but many struggle to interpret these values and formulate relevant questions for their clinicians. Question prompt lists (QPLs) can enhance communication but are rarely tailored to individual clinical contexts. Objective: This study evaluated the feasibility of using large language models (LLMs) to generate patient-friendly, clinically relevant questions grounded in electronic health record (EHR) laboratory data. Methods: We extracted deidentified clinical profiles, including laboratory results, diagnoses, and medications, from patients with chronic conditions (eg, diabetes and chronic kidney disease). Using 9 deidentified clinical profiles from the OneFlorida Data Trust, we generated 486 questions across all rounds: 126 with GPT-4o in round 1, 120 with GPT-4o in round 2, 180 with GPT-4o in round 3, and 60 with LLaMA 3.2 in round 3. Prompt refinements were informed by clinician ratings consisting of 2 binary questions (ie, clear phrasing and clinical validity) and 3 Likert-scale questions (ie, significance for the patient’s health; clinical appropriateness, that is, the likelihood of being asked in primary care setting; and willingness to answer). Refinements were incorporated after each round. Patient participants then evaluated selected questions for understandability, perceived usefulness, and intention to use. Readability was assessed with standard indices. Results: Iterative clinician feedback improved question clarity and reduced clinically irrelevant suggestions. Across rounds, GPT-4o consistently produced more coherent and patient-friendly questions, while LLaMA 3.2 demonstrated competitive performance on Likert-scale metrics. It exhibited greater variability in clinical appropriateness as noted by clinicians. In round 3, the binary metric “clear phrasing” reached a ceiling effect for both models, while clinical validity ratings showed greater variability, particularly from one clinician. Likert-scale evaluations tended to favor LLaMA 3.2 across all 3 clinicians for clinical appropriateness (3.37‐4.82 vs 3.02‐4.67), significance for the patient’s health (3.38‐4.28 vs 2.97‐3.83), and willingness to answer (3.17‐4.70 vs 2.82‐4.47), with multiple comparisons reaching statistical significance after Bonferroni correction. Patient evaluation (N=134) of GPT-4o–generated questions showed that 25 of 30 questions had moderate to high understandability (average Likert score ≥3.5), and 19 of 30 questions had moderate to high usefulness (average Likert score ≥3.5). Conclusions: This study supports the feasibility of using LLMs with structured EHR-derived laboratory data to generate contextualized QPLs, but model outputs varied in clinical appropriateness and readability. Clinician-in-the-loop review remains necessary before patient-facing use.

This project provides a complete workflow for processing electronic health record (EHR) data to automatically generate relevant, patient-friendly questions for clinical appointments. The pipeline uses Python for data processing and leverages a Large Language Model (LLM), specifically GPT-4o, to create clinically relevant questions based on a patient's lab results, diagnoses, and medications.

## 📝 Project Overview

The primary goal is to empower patients by providing them with a list of precise, urgent, and actionable questions to discuss with their healthcare providers. This is achieved by:
1.  **Ingesting and processing** outpatient EHR data (demographics, labs, medications, diagnoses).
2.  **Identifying patients** with abnormal lab results that require clinical attention.
3.  **Generating structured prompts** that summarize a patient's clinical profile.
4.  **Refining these profiles** to include only the most recent and relevant information.
5.  **Using the OpenAI GPT-4o API** to generate a list of 20 patient-friendly questions from the refined profiles.
6.  **Analyzing and visualizing** the quality of the generated questions based on clinician evaluations.

---

##  Workflow

The project is structured as a multi-step data processing pipeline:

**1. Patient Profile Generation:**
* Raw patient data from CSV files (`OutPateint_demo.csv`, `OutPateint_Labs.csv`, etc.) is loaded.
* The scripts (`Pateint_Selection_and_Question_generation.py`, `Round 1 and 2.py`) process this data to create text-based clinical profiles for each patient.
* Patients are filtered to include only those with clinically significant abnormal lab results.

**2. Profile Refinement & Sorting:**
* The generated profiles undergo several cleaning steps to improve relevance.
* The lab section is filtered to show only results from the most recent date.
* Medication and diagnosis histories are filtered to a specific time window (e.g., ±180 days) around the latest lab date.
* Profiles are sorted based on the number of distinct abnormal lab tests to prioritize more complex cases.

**3. LLM-Based Question Generation:**
* The final, refined text profiles are fed as prompts to the GPT-4o model via the OpenAI API (`Round 3.py`).
* The model generates a list of 20 questions for each patient, tailored to their specific abnormal labs, existing conditions, and medications.

**4. Statistical Analysis and Visualization:**
* The generated questions are evaluated by clinicians (data stored in `Plot.xlsx`, `HeatMap_Readability.xlsx`).
* The `Stastistical analysis and figures.py` script computes readability scores, performs statistical tests (Mann-Whitney U, Cohen's Kappa), and generates figures (heatmaps, boxplots, radar charts) to visualize the evaluation results.

---

## 📂 File Descriptions

* `Pateint_Selection_and_Question_generation.py`: An initial script to generate patient clinical profiles from raw CSV files. It establishes the core logic for data filtering and prompt formatting.
* `Round 1 and 2.py`: Contains functions for advanced processing and refinement of the patient profiles. This script filters patients based on the count of abnormal labs, sorts them, and cleans the lab, medication, and diagnosis sections to ensure temporal relevance.
* `Round 3.py`: The final question-generation script. It reads the processed patient profiles, sends them to the OpenAI GPT-4o API, and saves the generated questions to a CSV file.
* `Stastistical analysis and figures.py`: A comprehensive script for analyzing clinician feedback and readability metrics. It reads evaluation data from Excel files and produces all the figures and statistical outputs for the project.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* An OpenAI API key

### Installation
1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Setup and Execution
1.  **Place your data files** in the appropriate directories. The scripts expect the input CSVs (e.g., `OutPateint_Labs.csv`) and evaluation Excel files (`Plot.xlsx`, `HeatMap_Readability.xlsx`) to be in the project's root or a specified data directory.
2.  **Generate and refine patient profiles:** Run the functions in `Round 1 and 2.py` to create the sorted and processed `.txt` prompt files.
3.  **Configure the API key:** Open `Round 3.py` and replace `""` with your actual OpenAI API key:
    ```python
    api_key = "YOUR_OPENAI_API_KEY"
    ```
4.  **Generate questions:** Run the `Round 3.py` script to call the API and generate the questions.
    ```bash
    python "Round 3.py"
    ```
5.  **Analyze results:** Run the `Stastistical analysis and figures.py` script to produce visualizations from the evaluation data.
    ```bash
    python "Stastistical analysis and figures.py"
    ```

Cite

He Z, Bhasuran B, Lustria M, Hanna K, Killian M, Shavor C, Dailey M, Manikandan S, Luo X Generating Question Prompt Lists From Electronic Health Record Data Using Large Language Models: Iterative Evaluation Study J Med Internet Res 2026;28:e87280
URL: https://www.jmir.org/2026/1/e87280
DOI: 10.2196/87280

