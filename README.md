# Generating Question Prompt Lists From Electronic Health Record Data Using Large Language Models

## Iterative Evaluation Study

This repository contains the data-processing, question-generation, and evaluation workflow developed for the study:

> **Generating Question Prompt Lists From Electronic Health Record Data Using Large Language Models: Iterative Evaluation Study**

The project evaluates the feasibility of using large language models (LLMs) to generate patient-friendly and clinically relevant questions grounded in electronic health record (EHR) data, including laboratory results, diagnoses, and medications.

---

## Table of Contents

* [Study Overview](#study-overview)
* [Abstract](#abstract)
* [Project Objectives](#project-objectives)
* [Workflow](#workflow)
* [Repository Files](#repository-files)
* [Getting Started](#getting-started)
* [Data Requirements](#data-requirements)
* [API Key Configuration](#api-key-configuration)
* [Running the Pipeline](#running-the-pipeline)
* [Evaluation and Statistical Analysis](#evaluation-and-statistical-analysis)
* [Privacy and Security](#privacy-and-security)
* [Citation](#citation)
* [Disclaimer](#disclaimer)

---

## Study Overview

Patients increasingly access laboratory test results through patient portals. However, many patients experience difficulty interpreting these results and identifying appropriate questions to ask their clinicians.

Question prompt lists (QPLs) can support patient–clinician communication by helping patients prepare relevant questions before clinical appointments. Traditional QPLs, however, are generally not personalized to an individual patient’s laboratory results, diagnoses, or medications.

This project provides an end-to-end workflow for:

1. Processing structured outpatient EHR data.
2. Identifying patients with clinically significant abnormal laboratory results.
3. Creating longitudinal clinical profiles.
4. Refining profiles to retain the most recent and clinically relevant information.
5. Generating patient-friendly QPLs using LLMs.
6. Evaluating the generated questions through clinician and patient assessments.
7. Analyzing clinical quality, readability, and interrater agreement.

The study evaluated GPT-4o across three iterative prompting rounds and compared GPT-4o with LLaMA 3.2 during the final round.

---

## Abstract

### Background

Patients frequently access laboratory results through patient portals, but many struggle to interpret these values and formulate relevant questions for their clinicians. Question prompt lists can enhance communication but are rarely tailored to individual clinical contexts.

### Objective

This study evaluated the feasibility of using LLMs to generate patient-friendly, clinically relevant questions grounded in EHR laboratory data.

### Methods

We extracted deidentified clinical profiles containing laboratory results, diagnoses, and medications from patients with chronic conditions, including diabetes and chronic kidney disease.

Using nine deidentified clinical profiles from the OneFlorida Data Trust, we generated 486 questions across three evaluation rounds:

| Evaluation round | Model     | Questions generated |
| ---------------- | --------- | ------------------: |
| Round 1          | GPT-4o    |                 126 |
| Round 2          | GPT-4o    |                 120 |
| Round 3          | GPT-4o    |                 180 |
| Round 3          | LLaMA 3.2 |                  60 |
| **Total**        | —         |             **486** |

Prompt refinements were informed by clinician ratings consisting of:

* Two binary measures:

  * Clear phrasing
  * Clinical validity
* Three Likert-scale measures:

  * Clinical appropriateness
  * Significance for the patient’s health
  * Clinician willingness to answer

Prompt refinements were incorporated after each evaluation round. Patient participants subsequently evaluated selected questions for understandability, perceived usefulness, and intention to use. Readability was assessed using standard readability indices.

### Results

Iterative clinician feedback improved question clarity and reduced clinically irrelevant suggestions. Across rounds, GPT-4o consistently generated coherent and patient-friendly questions, whereas LLaMA 3.2 demonstrated competitive performance on Likert-scale measures but greater variability in clinical appropriateness.

In round 3, the binary measure of clear phrasing reached a ceiling effect for both models. Clinical validity ratings showed greater variability, particularly for one clinician.

Likert-scale evaluations tended to favor LLaMA 3.2 across all three clinicians for:

* Clinical appropriateness: 3.37–4.82 versus 3.02–4.67
* Significance for the patient’s health: 3.38–4.28 versus 2.97–3.83
* Willingness to answer: 3.17–4.70 versus 2.82–4.47

Multiple comparisons remained statistically significant after Bonferroni correction.

Among 134 patient participants who evaluated GPT-4o–generated questions:

* 25 of 30 questions demonstrated moderate to high understandability, defined as an average Likert score of at least 3.5.
* 19 of 30 questions demonstrated moderate to high perceived usefulness, defined as an average Likert score of at least 3.5.

### Conclusions

This study supports the feasibility of using LLMs with structured, EHR-derived laboratory data to generate contextualized QPLs. However, model outputs varied in clinical appropriateness and readability. Clinician-in-the-loop review remains necessary before these questions can be used in patient-facing applications.

---

## Project Objectives

The primary objective is to help patients prepare precise, relevant, and actionable questions for conversations with their health care professionals.

The workflow performs the following tasks:

1. **EHR data ingestion**
   Loads outpatient demographic, laboratory, medication, and diagnosis data.

2. **Patient identification**
   Identifies patients with abnormal laboratory results that may require clinical attention.

3. **Clinical profile generation**
   Creates structured, text-based summaries of patients’ laboratory results, diagnoses, and medications.

4. **Temporal profile refinement**
   Retains the most recent and clinically relevant information within predefined time windows.

5. **LLM-based question generation**
   Uses GPT-4o to generate 20 patient-friendly questions tailored to each clinical profile.

6. **Clinician and patient evaluation**
   Assesses the clarity, clinical validity, appropriateness, usefulness, and readability of generated questions.

7. **Statistical analysis and visualization**
   Generates descriptive statistics, inferential tests, heatmaps, boxplots, and radar charts.

---

## Workflow

```text
Raw EHR data
     │
     ▼
Patient selection
     │
     ▼
Identification of abnormal laboratory results
     │
     ▼
Initial clinical profile generation
     │
     ▼
Temporal filtering and profile refinement
     │
     ▼
Patient prioritization based on abnormal laboratory tests
     │
     ▼
LLM prompt construction
     │
     ▼
Question prompt list generation
     │
     ▼
Clinician and patient evaluation
     │
     ▼
Statistical analysis and visualization
```

### 1. Patient Profile Generation

Raw patient data are loaded from CSV files containing information such as:

* Demographics
* Laboratory results
* Diagnoses
* Medications

The scripts process these data and create a text-based clinical profile for each patient.

Patients are filtered to retain those with clinically significant abnormal laboratory results.

Primary scripts:

* `Pateint_Selection_and_Question_generation.py`
* `Round 1 and 2.py`

### 2. Profile Refinement and Sorting

The generated profiles undergo several refinement steps to improve clinical relevance and reduce unnecessary information.

These steps include:

* Retaining laboratory results from the most recent relevant date.
* Filtering medication and diagnosis histories using a specified time window around the latest laboratory date.
* Removing duplicate or irrelevant information.
* Counting distinct abnormal laboratory tests.
* Sorting patients according to the number of abnormal laboratory tests to prioritize more complex clinical profiles.

For example, medication and diagnosis records may be restricted to a window of ±180 days around the latest laboratory date.

### 3. LLM-Based Question Generation

The final clinical profiles are incorporated into structured prompts and submitted to GPT-4o through the OpenAI API.

The model generates 20 questions for each patient. The questions are designed to be:

* Patient-friendly
* Clinically relevant
* Specific to the patient’s abnormal laboratory results
* Contextualized using diagnoses and medications
* Appropriate for discussion during a clinical appointment

Primary script:

* `Round 3.py`

### 4. Statistical Analysis and Visualization

Generated questions are evaluated by clinicians and patient participants.

Evaluation data are stored in files such as:

* `Plot.xlsx`
* `HeatMap_Readability.xlsx`

The analysis script performs tasks including:

* Descriptive statistical analysis
* Readability assessment
* Mann–Whitney U tests
* Interrater reliability analysis
* Multiple-comparison correction
* Heatmap generation
* Boxplot generation
* Radar chart generation

Primary script:

* `Stastistical analysis and figures.py`

---

## Repository Files

| File                                           | Description                                                                                                                                                                                                          |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Pateint_Selection_and_Question_generation.py` | Generates initial patient clinical profiles from raw CSV files. The script contains the core logic for patient selection, laboratory-result filtering, and profile formatting.                                       |
| `Round 1 and 2.py`                             | Refines clinical profiles generated during the first two prompting rounds. It filters patients by abnormal laboratory-test count, applies temporal constraints, cleans profile sections, and sorts patient profiles. |
| `Round 3.py`                                   | Reads the final refined clinical profiles, submits them to GPT-4o through the OpenAI API, and saves the generated question prompt lists.                                                                             |
| `Stastistical analysis and figures.py`         | Analyzes clinician and patient evaluations, calculates readability and agreement measures, performs statistical tests, and generates study figures.                                                                  |
| `Plot.xlsx`                                    | Contains evaluation data used to compare rounds, models, or clinician ratings.                                                                                                                                       |
| `HeatMap_Readability.xlsx`                     | Contains data used for heatmap and readability analyses.                                                                                                                                                             |
| `requirements.txt`                             | Lists the Python packages required to run the project.                                                                                                                                                               |

> **Note:** Some filenames contain spelling inconsistencies because they reflect the original pipeline. Rename them only after updating all corresponding file references in the source code.

---

## Getting Started

### Prerequisites

Before running the project, ensure that the following are available:

* Python 3.8 or later
* `pip`
* An OpenAI API key
* Input EHR data in the expected CSV format
* Evaluation data in the expected Excel format

### Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### Create a Virtual Environment

Creating an isolated Python environment is recommended.

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS or Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Data Requirements

The scripts expect structured outpatient EHR files containing information such as:

```text
OutPateint_demo.csv
OutPateint_Labs.csv
OutPateint_Medications.csv
OutPateint_Diagnoses.csv
```

Evaluation files may include:

```text
Plot.xlsx
HeatMap_Readability.xlsx
```

Depending on the local configuration, these files should be placed either:

* In the repository root directory, or
* In the input directory specified within each Python script.

### Recommended Directory Structure

```text
project-root/
├── data/
│   ├── raw/
│   │   ├── OutPateint_demo.csv
│   │   ├── OutPateint_Labs.csv
│   │   ├── OutPateint_Medications.csv
│   │   └── OutPateint_Diagnoses.csv
│   ├── evaluation/
│   │   ├── Plot.xlsx
│   │   └── HeatMap_Readability.xlsx
│   └── processed/
│       └── patient_profiles/
├── outputs/
│   ├── generated_questions/
│   ├── figures/
│   └── statistical_results/
├── Pateint_Selection_and_Question_generation.py
├── Round 1 and 2.py
├── Round 3.py
├── Stastistical analysis and figures.py
├── requirements.txt
└── README.md
```

Update the file paths in the scripts according to the local directory structure.

---

## API Key Configuration

Do not place an OpenAI API key directly in a script or commit it to GitHub.

### Option 1: Environment Variable

#### Windows Command Prompt

```bash
set OPENAI_API_KEY=your_api_key_here
```

#### Windows PowerShell

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

#### macOS or Linux

```bash
export OPENAI_API_KEY="your_api_key_here"
```

The key can then be loaded in Python:

```python
import os

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not configured.")
```

### Option 2: `.env` File

Create a file named `.env` in the repository root:

```text
OPENAI_API_KEY=your_api_key_here
```

Load it in Python:

```python
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY was not found in the environment.")
```

Add `.env` to `.gitignore`:

```gitignore
.env
venv/
__pycache__/
*.pyc
outputs/
data/raw/
```

---

## Running the Pipeline

### Step 1: Generate Initial Patient Profiles

Run the initial patient-selection and profile-generation script:

```bash
python "Pateint_Selection_and_Question_generation.py"
```

This step:

* Loads the source EHR files.
* Identifies eligible patients.
* Detects abnormal laboratory results.
* Creates initial text-based clinical profiles.

### Step 2: Refine and Sort Patient Profiles

Run the round 1 and round 2 processing script:

```bash
python "Round 1 and 2.py"
```

This step:

* Filters laboratory results.
* Applies temporal windows to diagnoses and medications.
* Removes irrelevant profile information.
* Counts distinct abnormal laboratory tests.
* Sorts and saves the processed patient profiles.

### Step 3: Generate Question Prompt Lists

After configuring the OpenAI API key, run:

```bash
python "Round 3.py"
```

This step submits the refined profiles to GPT-4o and saves the generated questions to the configured output location.

### Step 4: Analyze Evaluation Results

Run the statistical-analysis and figure-generation script:

```bash
python "Stastistical analysis and figures.py"
```

This step produces:

* Summary statistics
* Statistical-test results
* Readability results
* Heatmaps
* Boxplots
* Radar charts
* Other study visualizations

---

## Evaluation and Statistical Analysis

The generated questions were evaluated using clinician- and patient-centered measures.

### Clinician Evaluation

Clinicians assessed each question using two binary criteria:

1. Clear phrasing
2. Clinical validity

They also used Likert scales to assess:

1. Clinical appropriateness
2. Significance for the patient’s health
3. Willingness to answer

### Patient Evaluation

Patient participants evaluated selected questions based on:

* Understandability
* Perceived usefulness
* Intention to use the question during a clinical encounter

### Readability Analysis

Standard readability indices were used to assess whether generated questions were understandable to patients.

### Statistical Methods

Depending on the comparison, the analysis included:

* Descriptive statistics
* Mann–Whitney U tests
* Interrater reliability measures
* Cohen’s kappa
* Multiple-comparison corrections
* Heatmap-based visualization
* Boxplots and radar charts

Refer to the published article for the complete evaluation protocol and statistical methodology.

---

## Privacy and Security

This research used deidentified clinical profiles. No protected health information should be uploaded to an external LLM service without appropriate institutional approval, data-use authorization, and security safeguards.

Before running this pipeline:

* Remove direct patient identifiers.
* Follow applicable institutional review board requirements.
* Follow the relevant data-use agreement.
* Confirm whether the selected LLM service is approved for the intended data.
* Do not commit patient-level data, API keys, or confidential outputs to GitHub.
* Review all generated questions before patient-facing use.

---

## Citation

When using this repository, methodology, or associated materials, cite the following article:

> He Z, Bhasuran B, Lustria M, Hanna K, Killian M, Shavor C, Dailey M, Manikandan S, Luo X. Generating Question Prompt Lists From Electronic Health Record Data Using Large Language Models: Iterative Evaluation Study. *J Med Internet Res.* 2026;28:e87280. doi:10.2196/87280.

### BibTeX

```bibtex
@article{he2026generating,
  title   = {Generating Question Prompt Lists From Electronic Health Record Data Using Large Language Models: Iterative Evaluation Study},
  author  = {He, Zhe and Bhasuran, Balu and Lustria, Mia and Hanna, Karim and Killian, Michael and Shavor, Casey and Dailey, Michelle and Manikandan, Senthilkumar and Luo, Xiang},
  journal = {Journal of Medical Internet Research},
  year    = {2026},
  volume  = {28},
  pages   = {e87280},
  doi     = {10.2196/87280},
  url     = {https://www.jmir.org/2026/1/e87280}
}
```

### Article Links

* **Full text:** https://www.jmir.org/2026/1/e87280
* **DOI:** https://doi.org/10.2196/87280

---

## Disclaimer

This repository is intended for research and evaluation purposes only.

The generated questions are not medical advice and should not be used as a replacement for professional clinical judgment. LLM-generated outputs may contain incomplete, inappropriate, or inaccurate information. Clinician review remains necessary before any generated content is presented to patients or incorporated into clinical workflows.
