# Generating Question Prompt Lists From Electronic Health Record Data Using Large Language Models

## Iterative Evaluation Study

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Article](https://img.shields.io/badge/JMIR-e87280-red.svg)](https://www.jmir.org/2026/1/e87280)
[![DOI](https://img.shields.io/badge/DOI-10.2196%2F87280-blue.svg)](https://doi.org/10.2196/87280)

This repository contains the code, prompt templates, and illustrative examples developed for the study:

> **Generating Question Prompt Lists From Electronic Health Record Data Using Large Language Models: Iterative Evaluation Study**

The project evaluates the feasibility of using large language models (LLMs) to generate patient-friendly and clinically relevant question prompt lists (QPLs) grounded in electronic health record (EHR) data, including laboratory results, diagnoses, and medications.

---

## Table of Contents

- [Study Overview](#study-overview)
- [Abstract](#abstract)
- [Repository Structure](#repository-structure)
- [Workflow](#workflow)
- [File Descriptions](#file-descriptions)
- [Getting Started](#getting-started)
- [Required Local Data](#required-local-data)
- [Configuration](#configuration)
- [Running the Code](#running-the-code)
- [Evaluation and Statistical Analysis](#evaluation-and-statistical-analysis)
- [Important Implementation Notes](#important-implementation-notes)
- [Privacy and Security](#privacy-and-security)
- [Citation](#citation)
- [Disclaimer](#disclaimer)

---

## Study Overview

Patients increasingly access laboratory test results through patient portals, but many experience difficulty interpreting the results and identifying appropriate questions to discuss with clinicians.

Question prompt lists can support patient–clinician communication by helping patients prepare relevant questions before clinical appointments. However, conventional QPLs are generally not personalized to an individual patient's laboratory results, diagnoses, or medications.

This project demonstrates a workflow for:

1. Processing structured outpatient EHR data.
2. Identifying patients with abnormal laboratory results.
3. Creating text-based clinical profiles.
4. Refining profiles to retain recent and clinically relevant information.
5. Generating patient-friendly questions using an LLM.
6. Iteratively refining prompts using clinician feedback.
7. Evaluating question quality, readability, and interrater agreement.

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

| Evaluation round | Model | Questions generated |
|---|---|---:|
| Round 1 | GPT-4o | 126 |
| Round 2 | GPT-4o | 120 |
| Round 3 | GPT-4o | 180 |
| Round 3 | LLaMA 3.2 | 60 |
| **Total** | — | **486** |

Prompt refinements were informed by clinician ratings consisting of:

- Two binary measures:
  - Clear phrasing
  - Clinical validity
- Three Likert-scale measures:
  - Clinical appropriateness
  - Significance for the patient's health
  - Clinician willingness to answer

Patient participants subsequently evaluated selected questions for understandability, perceived usefulness, and intention to use. Readability was assessed using standard readability indices.

### Results

Iterative clinician feedback improved question clarity and reduced clinically irrelevant suggestions. Across rounds, GPT-4o consistently generated coherent and patient-friendly questions, whereas LLaMA 3.2 demonstrated competitive performance on Likert-scale measures but greater variability in clinical appropriateness.

In round 3, the clear-phrasing measure reached a ceiling effect for both models. Clinical validity ratings showed greater variability, particularly for one clinician.

Likert-scale evaluations tended to favor LLaMA 3.2 across all three clinicians for:

- Clinical appropriateness: 3.37–4.82 versus 3.02–4.67
- Significance for the patient's health: 3.38–4.28 versus 2.97–3.83
- Willingness to answer: 3.17–4.70 versus 2.82–4.47

Multiple comparisons remained statistically significant after Bonferroni correction.

Among 134 patient participants who evaluated GPT-4o–generated questions:

- 25 of 30 questions demonstrated moderate to high understandability, defined as an average Likert score of at least 3.5.
- 19 of 30 questions demonstrated moderate to high perceived usefulness, defined as an average Likert score of at least 3.5.

### Conclusions

This study supports the feasibility of using LLMs with structured, EHR-derived laboratory data to generate contextualized QPLs. However, model outputs varied in clinical appropriateness and readability. Clinician-in-the-loop review remains necessary before patient-facing use.

---

## Repository Structure

```text
LLM_Question_Generation/
├── Code/
│   ├── Pateint_Selection_and_Question_generation.py
│   ├── Round 1 and 2.py
│   ├── Round 3.py
│   └── Stastistical analysis and figures.py
├── Data/
│   ├── Clinical Profile.txt
│   └── Sample Questions.txt
├── prompts/
│   ├── Round 1.txt
│   ├── Round 2.txt
│   └── Round 3.txt
├── README.md
└── requirements.txt
```

### Top-Level Directories

| Directory | Contents |
|---|---|
| [`Code/`](Code) | Python scripts for profile generation, profile refinement, LLM-based question generation, statistical analysis, and figure generation. |
| [`Data/`](Data) | An illustrative clinical profile and sample generated questions. This folder does not contain the source EHR dataset. |
| [`prompts/`](prompts) | Prompt templates used in rounds 1, 2, and 3 of the iterative evaluation. |

---

## Workflow

```text
Local structured EHR files
          │
          ▼
Patient selection and profile generation
          │
          ▼
Identification of abnormal laboratory results
          │
          ▼
Profile filtering, sorting, and temporal refinement
          │
          ▼
Round-specific prompt construction
          │
          ▼
GPT-4o question generation
          │
          ▼
Clinician and patient evaluation
          │
          ▼
Statistical analysis and figure generation
```

### 1. Patient Selection and Profile Generation

The pipeline loads local CSV files containing:

- Demographic information
- Laboratory results
- Medication records
- Diagnosis records

It creates a text-based clinical profile for each eligible patient and includes laboratory results, recent medications, recent diagnoses, and basic demographic context.

### 2. Profile Refinement

The profile-processing code performs tasks such as:

- Identifying abnormal laboratory values.
- Retaining laboratory results from the most recent relevant date.
- Removing duplicate laboratory tests.
- Ranking profiles by the number of abnormal laboratory results.
- Filtering medication and diagnosis records to a defined interval around the latest laboratory date.
- Selecting profiles with multiple distinct abnormal laboratory tests.

### 3. Prompt Refinement

The [`prompts/`](prompts) directory documents the iterative prompt-development process:

- **Round 1:** Broad question generation based on the full clinical profile.
- **Round 2:** Greater emphasis on abnormal values, urgency, actionability, clinical context, and patient-friendly language.
- **Round 3:** A more structured prompt with explicit output formatting, examples of highly rated questions, and examples of lower-rated questions to avoid.

### 4. Question Generation

The final-round script:

1. Reads clinical profiles from text files.
2. Sends each profile to GPT-4o.
3. Requests exactly 20 patient-friendly questions.
4. Organizes the output by patient file and laboratory test.
5. Saves the generated questions to a CSV file.

### 5. Evaluation and Analysis

Clinician and patient evaluation data are analyzed to assess:

- Clear phrasing
- Clinical validity
- Clinical appropriateness
- Significance for the patient's health
- Clinician willingness to answer
- Patient understandability
- Patient-perceived usefulness
- Intention to use
- Readability
- Interrater agreement
- Differences between models and prompting rounds

---

## File Descriptions

### Code

#### [`Code/Pateint_Selection_and_Question_generation.py`](Code/Pateint_Selection_and_Question_generation.py)

Creates initial patient-specific prompts from local EHR-derived CSV files.

The script:

- Loads laboratory, medication, diagnosis, and demographic data.
- Calculates patient age using the most recent laboratory date.
- Selects laboratory tests of interest.
- Formats abnormality indicators and reference ranges.
- Retains recent medications and diagnoses.
- Creates a structured patient profile and question-generation prompt.
- Writes one text file per patient.

> The filename contains the original spelling of `Pateint`. Keep this spelling when running the existing file unless the file and all references are renamed.

#### [`Code/Round 1 and 2.py`](Code/Round%201%20and%202.py)

Contains multiple stages of profile generation and refinement used during the iterative prompting workflow.

The script includes logic to:

- Detect selected abnormal laboratory results.
- Filter profiles to patients with at least two distinct abnormal laboratory tests.
- Sort profiles according to abnormal laboratory-test counts.
- Retain laboratory results from the latest date.
- Remove duplicate tests from the latest date.
- Filter medications and diagnoses to a ±180-day window around the latest laboratory date.
- Save processed profiles to configured output folders.

#### [`Code/Round 3.py`](Code/Round%203.py)

Implements the final GPT-4o question-generation workflow.

The script:

- Reads `.txt` clinical profiles from a configured folder.
- Uses the OpenAI Python client.
- Calls the `gpt-4o` model with temperature set to `0`.
- Requests 20 questions at an approximately sixth-grade reading level.
- Expects tab-separated output in the format `Lab test name<TAB>Question`.
- Saves results with the columns:
  - `Patient File`
  - `Lab test name`
  - `Question`

#### [`Code/Stastistical analysis and figures.py`](Code/Stastistical%20analysis%20and%20figures.py)

Contains the study's statistical-analysis and visualization code.

The script includes analyses and visualizations involving:

- Descriptive statistics
- Mann–Whitney U tests
- Cohen's kappa
- Intraclass correlation coefficients
- Clinician-level comparisons
- Round-level comparisons
- GPT-4o versus LLaMA comparisons
- Readability measures
- Heatmaps
- Bar charts
- Boxplots
- Radar charts

The file was exported from an interactive analysis workflow and contains multiple analysis sections. Run only the sections relevant to the available input files and intended analysis.

### Data Examples

#### [`Data/Clinical Profile.txt`](Data/Clinical%20Profile.txt)

Provides an illustrative clinical-profile format containing:

- Patient demographic context
- Laboratory results
- Medication history
- Diagnosis history

Dates are masked in the example.

#### [`Data/Sample Questions.txt`](Data/Sample%20Questions.txt)

Provides examples of patient-facing questions generated from laboratory values, diagnoses, and medication context.

### Prompt Templates

#### [`prompts/Round 1.txt`](prompts/Round%201.txt)

The initial prompt asks the model to generate questions for each laboratory test and prioritize them according to medical urgency.

#### [`prompts/Round 2.txt`](prompts/Round%202.txt)

The revised prompt adds explicit instructions regarding:

- Abnormal laboratory values
- Clinical urgency
- Medication and disease context
- Actionable outcomes
- Patient-friendly language
- Question limits appropriate for a clinical visit

#### [`prompts/Round 3.txt`](prompts/Round%203.txt)

The final prompt adds:

- A stepwise reasoning framework
- A strict 20-question limit
- A defined output format
- Highly rated examples
- Lower-rated examples to avoid
- Greater emphasis on specificity, actionability, and laboratory abnormalities

---

## Getting Started

### Prerequisites

- Python 3.8 or later
- `pip`
- An OpenAI API key for question generation
- Local EHR-derived CSV files in the expected format
- Local evaluation spreadsheets for statistical analysis

### Clone the Repository

```bash
git clone https://github.com/balubhasuran/LLM_Question_Generation.git
cd LLM_Question_Generation
```

### Create a Virtual Environment

#### Windows Command Prompt

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The repository specifies packages for data processing, statistical analysis, visualization, readability assessment, machine learning utilities, and OpenAI API access.

---

## Required Local Data

The source patient-level EHR data are not included in this public repository.

The patient-profile scripts expect the following filenames unless the paths in the scripts are changed:

```text
OutPateint_Labs.csv
OutPateint_Med.csv
OutPateint_diags.csv
OutPateint_demo.csv
```

The analysis script refers to local evaluation files including:

```text
Plot.xlsx
HeatMap_Readability.xlsx
Radar_Table.xlsx
```

Not every analysis section uses every spreadsheet. The exact file requirements depend on the section being executed.

### Expected CSV Roles

| File | Expected content |
|---|---|
| `OutPateint_Labs.csv` | Laboratory results, dates, units, reference ranges, LOINC codes, and abnormality indicators. |
| `OutPateint_Med.csv` | Medication records, RxNorm identifiers, medication names, and start dates. |
| `OutPateint_diags.csv` | Diagnosis codes, diagnosis descriptions, and dates. |
| `OutPateint_demo.csv` | Patient identifiers and demographic variables such as birth date, sex, race, and ethnicity. |

The scripts assume specific column names. Review the code before execution and map local columns to the expected schema when necessary.

---

## Configuration

The current scripts contain local Windows paths from the original research environment. These paths must be changed before the code can be run on another computer.

### Input CSV Locations

Both patient-profile scripts load CSV files using relative paths, for example:

```python
lab_tests = pd.read_csv("OutPateint_Labs.csv", low_memory=False)
med = pd.read_csv("OutPateint_Med.csv", low_memory=False)
diag = pd.read_csv("OutPateint_diags.csv", low_memory=False)
demographics = pd.read_csv("OutPateint_demo.csv", low_memory=False)
```

You can either:

1. Place these CSV files in the directory from which the script is executed, or
2. Replace the filenames with absolute or project-relative paths.

### Output Folders

Update variables such as:

```python
source_directory = r"D:\MIMICIV_20_Patients\Round 3"
target_directory = r"D:\MIMICIV_20_Patients\Round 3\Sorted_Profiles"
input_folder = r"D:\MIMICIV_20_Patients\Round 3\Sorted_Profiles"
output_folder = r"D:\MIMICIV_20_Patients\Round 3\Processed"
```

A portable alternative is to use `pathlib`:

```python
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
input_folder = project_root / "local_data" / "clinical_profiles"
output_folder = project_root / "local_outputs" / "processed_profiles"

output_folder.mkdir(parents=True, exist_ok=True)
```

Keep patient-level data and generated patient-level outputs outside the public repository unless sharing is explicitly authorized.

### OpenAI API Key

The current `Round 3.py` file contains an empty API-key variable:

```python
api_key = ""
```

Do not hard-code or commit an API key. Replace the API-key configuration with an environment variable:

```python
import os
import openai

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY is not configured. Set it before running the script."
    )

client = openai.OpenAI(api_key=api_key)
```

Set the environment variable before execution.

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

---

## Running the Code

Because the scripts contain original local paths and research-specific data assumptions, review and configure each script before execution.

### Step 1: Generate Initial Patient Profiles

From the repository root:

```bash
python "Code/Pateint_Selection_and_Question_generation.py"
```

This script reads the four EHR-derived CSV files and writes patient-specific prompt files to the configured output directory.

### Step 2: Refine and Sort Profiles

```bash
python "Code/Round 1 and 2.py"
```

This script contains multiple processing stages. Confirm all configured source and target directories before running it.

Depending on the selected stage, it can:

- Generate abnormal-laboratory profiles.
- Select profiles with multiple abnormal tests.
- Rank profiles by abnormal-test count.
- Retain the most recent laboratory date.
- Apply the ±180-day medication and diagnosis window.
- Write sorted and processed text profiles.

### Step 3: Generate Questions With GPT-4o

First update:

- `clinical_profiles_folder`
- `csv_file_path`
- OpenAI API-key handling

Then run:

```bash
python "Code/Round 3.py"
```

The script saves a CSV containing the patient filename, laboratory-test name, and generated question.

### Step 4: Run Statistical Analyses and Generate Figures

Place the required Excel files in the working directory or update their paths in the script.

Then run:

```bash
python "Code/Stastistical analysis and figures.py"
```

The script contains several analysis blocks and may generate files such as:

```text
FIGURE_1.png
FIGURE_2.png
FIGURE_3.png
FIGURE_4.png
LLM_Heatmaps/
LLM_Combined_Likert_Segmented/
```

Output depends on the sections executed and the available input spreadsheets.

---

## Evaluation and Statistical Analysis

### Clinician Evaluation

Clinicians assessed generated questions using two binary criteria:

1. Clear phrasing
2. Clinical validity

They also used Likert scales to assess:

1. Clinical appropriateness
2. Significance for the patient's health
3. Willingness to answer

### Patient Evaluation

Patient participants evaluated selected questions based on:

- Understandability
- Perceived usefulness
- Intention to use the question during a clinical encounter

### Statistical Methods

The repository includes code for methods such as:

- Descriptive statistics
- Mann–Whitney U tests
- Cohen's kappa
- Intraclass correlation coefficients
- Multiple-comparison correction
- Readability analysis
- Heatmap visualization
- Boxplots
- Radar charts

Refer to the published article for the finalized study protocol, outcome definitions, statistical procedures, and interpretation.

---

## Important Implementation Notes

1. **The public repository does not include the source EHR data.**  
   The `Data/` directory contains only an illustrative clinical profile and sample questions.

2. **Local paths must be updated.**  
   Several scripts contain absolute Windows paths from the original research environment.

3. **The code assumes specific input schemas.**  
   Column names and formats must match those referenced in the scripts.

4. **The analysis script contains multiple notebook-style sections.**  
   Some sections require different Excel files or previously created variables. Review the relevant section before executing it.

5. **The existing filenames contain spelling inconsistencies.**  
   Examples include `Pateint` and `Stastistical`. These names are retained in the repository and in this README so that commands match the current files.

6. **Generated questions require clinical review.**  
   The pipeline is intended for research and evaluation, not autonomous patient-facing deployment.

7. **API usage may incur costs.**  
   Review the selected model, token limits, and OpenAI account settings before processing multiple profiles.

---

## Privacy and Security

This research used deidentified clinical profiles. Do not send protected health information or identifiable patient data to an external LLM service without appropriate authorization and safeguards.

Before using the pipeline:

- Remove direct and indirect patient identifiers.
- Follow institutional review board requirements.
- Follow all applicable data-use agreements.
- Confirm that the selected LLM service is approved for the intended data.
- Store API credentials in environment variables or an approved secret manager.
- Do not commit patient-level data, credentials, or confidential outputs to GitHub.
- Review all generated questions before any patient-facing use.

A recommended `.gitignore` configuration is:

```gitignore
.env
.venv/
venv/
__pycache__/
*.pyc

# Local patient-level data and outputs
local_data/
local_outputs/
*.csv
*.xlsx

# Generated figures and analysis directories
FIGURE_*.png
LLM_Heatmaps/
LLM_Combined_Likert_Segmented/
```

Review these exclusions before use so that intentionally shareable non-sensitive files are not omitted.

---

## Citation

When using this repository, methodology, or associated materials, cite:

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

### Publication

- **Full text:** https://www.jmir.org/2026/1/e87280
- **DOI:** https://doi.org/10.2196/87280

---

## Disclaimer

This repository is intended for research and evaluation purposes only.

The generated questions are not medical advice and should not replace professional clinical judgment. LLM-generated outputs may contain inaccurate, incomplete, irrelevant, or inappropriate information. Clinician review remains necessary before generated content is presented to patients or incorporated into clinical workflows.
