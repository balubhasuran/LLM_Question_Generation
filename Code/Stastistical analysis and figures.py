#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load both sheets from the Excel file
file_path = "Plot.xlsx"
sheet_names = pd.ExcelFile(file_path).sheet_names
df_rounds_gpt4 = pd.read_excel(file_path, sheet_name=sheet_names[0])
df_gpt4_vs_llama = pd.read_excel(file_path, sheet_name=sheet_names[1])


# In[2]:


df_rounds_gpt4.head()


# In[3]:


df_gpt4_vs_llama.head()


# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns

# Combine both sheets into a single DataFrame
df_combined = pd.concat([df_rounds_gpt4, df_gpt4_vs_llama], ignore_index=True)

# Set visual style
sns.set(style="whitegrid")

# Separate binary and likert types
df_binary = df_combined[df_combined["Type"] == "binary"]
df_likert = df_combined[df_combined["Type"] == "likert"]

# Create bar plot for binary scores across Rounds and LLMs
plt.figure(figsize=(12, 6))
sns.barplot(data=df_binary, x="Question", y="Score", hue="Round", ci="sd", estimator="mean")
plt.title("Binary Questions: Mean Scores by Round")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create bar plot for likert scores across Rounds and LLMs
plt.figure(figsize=(12, 6))
sns.barplot(data=df_likert, x="Question", y="Score", hue="LLM", ci="sd", estimator="mean")
plt.title("Likert Questions: Mean Scores by LLM (Round 3)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[5]:


from scipy.stats import mannwhitneyu
from sklearn.metrics import cohen_kappa_score
import itertools

# Prepare ICC matrix for each type
clinicians = df_combined['Clinican'].unique()
questions = df_combined['Question'].unique()

# Pivot data for ICC (each row: a question, each column: clinician scores)
icc_data_likert = df_likert.pivot_table(index=['Round', 'LLM', 'Question'], columns='Clinican', values='Score')

# Compute ICC(2,1) using pingouin
import pingouin as pg

icc_results = pg.intraclass_corr(data=icc_data_likert.reset_index().melt(id_vars=['Round', 'LLM', 'Question']),
                                 targets='Question',
                                 raters='Clinican',
                                 ratings='value')

# Compute pairwise Cohen's kappa for binary questions (round 2 only)
df_binary_round2 = df_binary[df_binary['Round'] == 2].pivot(index='Question', columns='Clinican', values='Score')
cohen_kappas = {}
for (a, b) in itertools.combinations(df_binary_round2.columns, 2):
    kappa = cohen_kappa_score(df_binary_round2[a].round(), df_binary_round2[b].round())
    cohen_kappas[f"{a} vs {b}"] = kappa

# Mann-Whitney U tests for Likert scores between GPT-4 and LLaMA
mannwhitney_results = {}
for question in df_likert['Question'].unique():
    scores_gpt4 = df_likert[(df_likert['LLM'] == 'GPT-4') & (df_likert['Question'] == question)]['Score']
    scores_llama = df_likert[(df_likert['LLM'] == 'LLaMA') & (df_likert['Question'] == question)]['Score']
    if not scores_gpt4.empty and not scores_llama.empty:
        stat, p = mannwhitneyu(scores_gpt4, scores_llama)
        mannwhitney_results[question] = {'U': stat, 'p-value': p}

# Display ICC, Cohen's Kappa, and Mann-Whitney U test results
icc_results_important = icc_results[['Type', 'ICC', 'F', 'pval']]



# In[6]:


icc_results_important


# In[10]:


# --- Renaming remains the same ---
clinician_map = {'Karim': 'Clinician 1', 'Cindy': 'Clinician 2', 'Mandy': 'Clinician 3'}
df_binary = df_binary.replace({'Clinican': clinician_map})
df_likert = df_likert.replace({'Clinican': clinician_map})

# --- Normalize binary scores to 0–1 ---
# If 'Score' is currently a count up to N (e.g., 20), this divides by the per-group max N
df_binary['Score_norm'] = (
    df_binary['Score'] /
    df_binary.groupby(['Round', 'Clinican', 'Question'])['Score'].transform('max')
)

# --- Prep data for bar plots (use normalized score for binary) ---
df_binary_bar = (
    df_binary.groupby(['Round', 'Clinican', 'Question'])['Score_norm']
    .mean().reset_index(name='Score')  # keep the column name 'Score' for plotting
)
df_likert_bar = (
    df_likert.groupby(['Round', 'Clinican', 'Question'])['Score']
    .mean().reset_index()
)

# --- Colors ---
clinician_palette = {
    'Clinician 1': '#B4D6C8',
    'Clinician 2': '#A5A29F',
    'Clinician 3': '#F6A35F'
}

# --- Binary bar plot (0–1) ---
plt.figure(figsize=(12, 6))
sns.barplot(
    data=df_binary_bar,
    x='Question', y='Score', hue='Clinican',
    palette=clinician_palette, width=0.3, errorbar='se'
)
plt.title("Binary Scores: Average per Question by Clinician (Rounds 2 & 3)")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(r"FIGURE_1.png", dpi=300)
plt.show()

# --- Likert bar plot (unchanged, still averages) ---
plt.figure(figsize=(12, 6))
sns.barplot(
    data=df_likert_bar,
    x='Question', y='Score', hue='Clinican',
    palette=clinician_palette, width=0.5, errorbar='se'
)
plt.title("Likert Scores: Average per Question by Clinician (Rounds 2 & 3)")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(r"FIGURE_2.png", dpi=300)
plt.show()


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ------------------------
# Global font size settings
# ------------------------
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

# Define question types explicitly
binary_questions = ["Clearly Phrased", "Clinical Sense"]
likert_questions = [
    "Likely to answer",
    "Likely to be asked in primary care",
    "Significant for Patient's Health"
]
all_clinicians = ['Clinician 1', 'Clinician 2', 'Clinician 3']

# Update clinician names
clinician_map = {'Karim': 'Clinician 1', 'Cindy': 'Clinician 2', 'Mandy': 'Clinician 3'}
df_binary['Clinican'] = df_binary['Clinican'].replace(clinician_map)
df_likert['Clinican'] = df_likert['Clinican'].replace(clinician_map)

# Helper function for pivot table
def complete_pivot(df, round_num, questions):
    pivot = df[(df["Round"] == round_num) & (df["Question"].isin(questions))] \
        .pivot_table(index='Question', columns='Clinican', values='Score', aggfunc='mean')
    return pivot.reindex(index=questions, columns=all_clinicians)

# Create separate pivot tables
binary_r2 = complete_pivot(df_binary, 2, binary_questions) / 20.0
binary_r3 = complete_pivot(df_binary, 3, binary_questions) / 20.0
likert_r2 = complete_pivot(df_likert, 2, likert_questions)
likert_r3 = complete_pivot(df_likert, 3, likert_questions)

# Draw heatmaps
fig, axs = plt.subplots(2, 2, figsize=(16, 10))

heatmaps = [
    (binary_r2, axs[0, 0], 'Round 2: Binary Scores', 'Blues'),
    (likert_r2, axs[0, 1], 'Round 2: Likert Scores', 'Oranges'),
    (binary_r3, axs[1, 0], 'Round 3: Binary Scores', 'Blues'),
    (likert_r3, axs[1, 1], 'Round 3: Likert Scores', 'Oranges'),
]

for data, ax, title, cmap in heatmaps:
    norm = plt.Normalize(vmin=np.nanmin(data.values), vmax=np.nanmax(data.values))
    sns.heatmap(data, cmap=cmap, cbar=True, ax=ax, annot=False,
                fmt=".2f", linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontsize=18, pad=12)
    
    # Annotate cells
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            val = data.iloc[y, x]
            if not np.isnan(val):
                color = "white" if norm(val) > 0.5 else "black"
                ax.text(x + 0.5, y + 0.5, f"{val:.2f}",
                        ha='center', va='center', color=color,
                        fontsize=13, weight="bold")

    ax.set_ylabel("")
    ax.set_xlabel("")

# ---- Add panel labels (a, b, c, d) ----
panel_labels = ['(a)', '(c)', '(b)', '(d)']
for ax, label in zip(axs.flatten(), panel_labels):
    ax.text(-0.1, 1.05, label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='bottom', ha='right')

plt.tight_layout()
plt.savefig("FIGURE_3.png", dpi=300)
plt.show()


# In[17]:


# ============================
# Clinician Heatmaps with p-values
# ============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------
# Configuration (edit here)
# ------------------------
SAVE_PATH = r"FIGURE_3.png"
FIGSIZE = (20, 12)
DPI = 300

# Choose colormaps: pick any of {"coolwarm", "RdYlGn_r", "YlGnBu"}
CMAP_BINARY = "YlGnBu"    # good for high=better, sequential
CMAP_LIKERT = "coolwarm"  # red→yellow→green (reversed), good for rating scales

ANNOT_FMT = ".2f"   # numeric annotation format
STAR_P_THRESH = 0.05
STAR_SYMBOL = "*"   # change to "†" or "**" if you like
STAR_OFFSET = (0.28, -0.28)  # (dx, dy) in cell units from center toward top-right corner

# ------------------------
# Global figure & font settings
# ------------------------
plt.rcParams.update({
    "font.size": 16,        # base font size
    "axes.titlesize": 20,   # title size
    "axes.labelsize": 16,   # axis label size
    "xtick.labelsize": 14,  # x-axis tick labels
    "ytick.labelsize": 14   # y-axis tick labels
})
# ------------------------
# Build data from Tables 2 & 3
# ------------------------

# Table 2 (Binary) — Means and P-values
binary_means = [
    # Question, Clinician, Round, Mean
    ("Clearly Phrased", "Clinician 1", 2, 1.00),
    ("Clearly Phrased", "Clinician 1", 3, 1.00),
    ("Clearly Phrased", "Clinician 2", 2, 1.00),
    ("Clearly Phrased", "Clinician 2", 3, 1.00),
    ("Clearly Phrased", "Clinician 3", 2, 0.99),
    ("Clearly Phrased", "Clinician 3", 3, 1.00),

    ("Clinical Sense",  "Clinician 1", 2, 0.97),
    ("Clinical Sense",  "Clinician 1", 3, 0.97),
    ("Clinical Sense",  "Clinician 2", 2, 0.98),
    ("Clinical Sense",  "Clinician 2", 3, 1.00),
    ("Clinical Sense",  "Clinician 3", 2, 0.71),
    ("Clinical Sense",  "Clinician 3", 3, 0.74),
]
df_binary = pd.DataFrame(binary_means, columns=["Question", "Clinician", "Round", "Score"])

# p-values comparing Round 2 vs Round 3 (per Question × Clinician)
binary_pvals = {
    ("Clearly Phrased", "Clinician 1"): 1.000,
    ("Clearly Phrased", "Clinician 2"): 1.000,
    ("Clearly Phrased", "Clinician 3"): 0.321,
    ("Clinical Sense",  "Clinician 1"): 0.704,
    ("Clinical Sense",  "Clinician 2"): 0.158,
    ("Clinical Sense",  "Clinician 3"): 0.565,
}

# Table 3 (Likert) — Means and p-values
likert_means = [
    ("Likely to be asked in primary care", "Clinician 1", 2, 3.85),
    ("Likely to be asked in primary care", "Clinician 1", 3, 3.92),
    ("Likely to be asked in primary care", "Clinician 2", 2, 4.21),
    ("Likely to be asked in primary care", "Clinician 2", 3, 4.22),
    ("Likely to be asked in primary care", "Clinician 3", 2, 3.80),
    ("Likely to be asked in primary care", "Clinician 3", 3, 3.79),

    ("Significant for Patient's Health", "Clinician 1", 2, 3.64),
    ("Significant for Patient's Health", "Clinician 1", 3, 3.71),
    ("Significant for Patient's Health", "Clinician 2", 2, 4.00),
    ("Significant for Patient's Health", "Clinician 2", 3, 4.06),
    ("Significant for Patient's Health", "Clinician 3", 2, 3.29),
    ("Significant for Patient's Health", "Clinician 3", 3, 3.33),

    ("Likely to answer", "Clinician 1", 2, 4.29),
    ("Likely to answer", "Clinician 1", 3, 4.45),
    ("Likely to answer", "Clinician 2", 2, 4.32),
    ("Likely to answer", "Clinician 2", 3, 4.32),
    ("Likely to answer", "Clinician 3", 2, 3.37),
    ("Likely to answer", "Clinician 3", 3, 3.29),
]
df_likert = pd.DataFrame(likert_means, columns=["Question", "Clinician", "Round", "Score"])

likert_pvals = {
    ("Likely to be asked in primary care", "Clinician 1"): 0.351,
    ("Likely to be asked in primary care", "Clinician 2"): 0.882,
    ("Likely to be asked in primary care", "Clinician 3"): 0.984,

    ("Significant for Patient's Health", "Clinician 1"): 0.477,
    ("Significant for Patient's Health", "Clinician 2"): 0.238,
    ("Significant for Patient's Health", "Clinician 3"): 0.925,

    ("Likely to answer", "Clinician 1"): 0.057,
    ("Likely to answer", "Clinician 2"): 0.987,
    ("Likely to answer", "Clinician 3"): 0.781,
}

# ------------------------
# Utilities
# ------------------------
binary_questions = ["Clearly Phrased", "Clinical Sense"]
likert_questions = [
    "Likely to answer",
    "Likely to be asked in primary care",
    "Significant for Patient's Health",
]
all_clinicians = ["Clinician 1", "Clinician 2", "Clinician 3"]

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16
})

def make_pivot(df, round_num, questions, clinicians):
    p = (df[(df["Round"] == round_num) & (df["Question"].isin(questions))]
         .pivot_table(index="Question", columns="Clinician", values="Score", aggfunc="mean"))
    return p.reindex(index=questions, columns=clinicians)

def annotate_cells(ax, data, norm, use_stars=False, pvals=None, fmt=".2f"):
    """
    Annotate each cell with value and optional significance star (top-right).
    If use_stars=True, looks up p-values in pvals[(question, clinician)] and adds STAR_SYMBOL if < STAR_P_THRESH.
    """
    nrows, ncols = data.shape
    for y in range(nrows):
        for x in range(ncols):
            val = data.iloc[y, x]
            if np.isnan(val):
                continue

            # main numeric annotation
            color = "white" if norm(val) > 0.5 else "black"
            ax.text(x + 0.5, y + 0.5, f"{val:{fmt}}",
                    ha="center", va="center", color=color, fontsize=13, weight="bold")

            # significance star (if requested and significant)
            if use_stars and pvals is not None:
                q = data.index[y]
                c = data.columns[x]
                p = pvals.get((q, c), None)
                if p is not None and p < STAR_P_THRESH:
                    dx, dy = STAR_OFFSET
                    ax.text(x + 0.5 + dx, y + 0.5 + dy, STAR_SYMBOL,
                            ha="center", va="center", color="black", fontsize=18, weight="bold")

def draw_heatmap(ax, df_mat, title, cmap):
    vmin = float(np.nanmin(df_mat.values))
    vmax = float(np.nanmax(df_mat.values))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sns.heatmap(
        df_mat,
        cmap=cmap,
        norm=norm,
        cbar=True,
        ax=ax,
        annot=False,
        fmt=ANNOT_FMT,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.85}
    )
    ax.set_title(title, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    return norm

# ------------------------
# Build matrices
# ------------------------
binary_r2 = make_pivot(df_binary, 2, binary_questions, all_clinicians)
binary_r3 = make_pivot(df_binary, 3, binary_questions, all_clinicians)
likert_r2 = make_pivot(df_likert, 2, likert_questions, all_clinicians)
likert_r3 = make_pivot(df_likert, 3, likert_questions, all_clinicians)

# ------------------------
# Plot
# ------------------------
# Set figure size when creating subplots
fig, axs = plt.subplots(2, 2, figsize=(22, 14))  # wider & taller

# Round 2: Binary
norm_b2 = draw_heatmap(axs[0, 0], binary_r2, "Round 2: Binary (Mean)", CMAP_BINARY)
annotate_cells(axs[0, 0], binary_r2, norm_b2, use_stars=False, pvals=None, fmt=ANNOT_FMT)

# Round 2: Likert
norm_l2 = draw_heatmap(axs[0, 1], likert_r2, "Round 2: Likert (Mean)", CMAP_LIKERT)
annotate_cells(axs[0, 1], likert_r2, norm_l2, use_stars=False, pvals=None, fmt=ANNOT_FMT)

# Round 3: Binary (add stars where R3 vs R2 is significant)
norm_b3 = draw_heatmap(axs[1, 0], binary_r3, "Round 3: Binary (Mean)", CMAP_BINARY)
annotate_cells(axs[1, 0], binary_r3, norm_b3, use_stars=True, pvals=binary_pvals, fmt=ANNOT_FMT)

# Round 3: Likert (add stars where R3 vs R2 is significant)
norm_l3 = draw_heatmap(axs[1, 1], likert_r3, "Round 3: Likert (Mean)", CMAP_LIKERT)
annotate_cells(axs[1, 1], likert_r3, norm_l3, use_stars=True, pvals=likert_pvals, fmt=ANNOT_FMT)

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=DPI, bbox_inches="tight")
plt.show()



# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# Load Excel
file_path = "Plot.xlsx"
df_rounds_gpt4 = pd.read_excel(file_path, sheet_name=0)
df_gpt4_vs_llama = pd.read_excel(file_path, sheet_name=1)

# Use only GPT4 vs LLaMA sheet
df_combined = df_gpt4_vs_llama

# Rename clinicians
clinician_map = {'Karim': 'Clinician 1', 'Cindy': 'Clinician 2', 'Mandy': 'Clinician 3'}
df_combined['Clinican'] = df_combined['Clinican'].replace(clinician_map)

# Split by type and round
df_binary = df_combined[df_combined['Type'] == 'binary']
df_likert = df_combined[df_combined['Type'] == 'likert']
df_round3 = df_combined[df_combined['Round'] == 3]
binary_r3 = df_round3[df_round3['Type'] == 'binary']
likert_r3 = df_round3[df_round3['Type'] == 'likert']

# Function to draw boxplot with Mann–Whitney U test annotation
def plot_boxplot_with_stats(data, x, y, title, ax, palette, ylim=None):
    sns.boxplot(data=data, x=x, y=y, palette=palette, ax=ax)
    llms = data[x].dropna().unique()
    if len(llms) == 2:
        scores_1 = data[data[x] == llms[0]][y]
        scores_2 = data[data[x] == llms[1]][y]
        stat, p = mannwhitneyu(scores_1, scores_2)
        ax.set_title(f"{title}\nMann–Whitney U p = {p:.4f}", fontsize=16, pad=12)
    else:
        ax.set_title(f"{title}\n(Not enough groups for test)", fontsize=16, pad=12)
    if ylim:
        ax.set_ylim(ylim)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_xlabel("")
    ax.set_ylabel(y, fontsize=14)

# Define the custom LLM palette
llm_palette = {
    'GPT-4o': '#BAC7CB',
    'Llama 3.2': '#C31E3F'
}

# Plot side-by-side boxplots for Round 3 binary and likert comparisons
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))  # larger figure

# Ensure grid lines are in the background
for ax in axes:
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

# Plot Binary scores with y-axis limit
plot_boxplot_with_stats(binary_r3, 'LLM', 'Score',
                        "Binary Score Distribution by LLM (Round 3)", axes[0],
                        palette=llm_palette, ylim=(19, 20))

# Plot Likert scores
plot_boxplot_with_stats(likert_r3, 'LLM', 'Score',
                        "Likert Score Distribution by LLM (Round 3)", axes[1],
                        palette=llm_palette)

# Title and layout
plt.suptitle("Comparison of GPT-4 vs LLaMA Scores with Statistical Tests (Round 3)",
             fontsize=18, weight="bold", y=1.02)

plt.tight_layout()
plt.savefig("FIGURE_4.png", dpi=300)
plt.show()



# In[20]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# Load Excel
file_path = "Plot.xlsx"
df_rounds_gpt4 = pd.read_excel(file_path, sheet_name=0)
df_gpt4_vs_llama = pd.read_excel(file_path, sheet_name=1)

# Use only GPT4 vs LLaMA sheet
df_combined = df_gpt4_vs_llama

# Rename clinicians
clinician_map = {'Karim': 'Clinician 1', 'Cindy': 'Clinician 2', 'Mandy': 'Clinician 3'}
df_combined['Clinican'] = df_combined['Clinican'].replace(clinician_map)

# Split by type and round
df_round3 = df_combined[df_combined['Round'] == 3]
binary_r3 = df_round3[df_round3['Type'] == 'binary'].copy()
likert_r3 = df_round3[df_round3['Type'] == 'likert'].copy()

# --- Scale binary scores to 0–1 range ---
binary_r3['Score'] = binary_r3['Score'] / 20.0

# Function to draw boxplot with Mann–Whitney U test annotation
def plot_boxplot_with_stats(data, x, y, title, ax, palette, ylim=None):
    sns.boxplot(data=data, x=x, y=y, palette=palette, ax=ax)
    llms = data[x].dropna().unique()
    if len(llms) == 2:
        scores_1 = data[data[x] == llms[0]][y]
        scores_2 = data[data[x] == llms[1]][y]
        stat, p = mannwhitneyu(scores_1, scores_2)
        ax.set_title(f"{title}\nMann–Whitney U p = {p:.4f}", fontsize=16, pad=12)
    else:
        ax.set_title(f"{title}\n(Not enough groups for test)", fontsize=16, pad=12)
    if ylim:
        ax.set_ylim(ylim)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_xlabel("")
    ax.set_ylabel(y, fontsize=14)

# Custom palette
llm_palette = {
    'GPT-4o': '#BAC7CB',
    'Llama 3.2': '#C31E3F'
}

# --- Create side-by-side plots ---
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))

# Ensure grid lines are in the background
for ax in axes:
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

# Left: Binary scores (scaled 0–1) with tighter range 0.6–1.0
plot_boxplot_with_stats(binary_r3, 'LLM', 'Score',
                        "Binary Score Distribution by LLM (Round 3)", axes[0],
                        palette=llm_palette, ylim=(0.95, 1.0))

# Right: Likert scores
plot_boxplot_with_stats(likert_r3, 'LLM', 'Score',
                        "Likert Score Distribution by LLM (Round 3)", axes[1],
                        palette=llm_palette)

# Add subplot labels (a), (b)
panel_labels = ['(a)', '(b)']
for ax, label in zip(axes, panel_labels):
    ax.text(-0.05, 1.08, label, transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='bottom', ha='right')

plt.tight_layout()
plt.savefig("FIGURE_4.png", dpi=300)
plt.show()


# In[5]:


#HeatMap for binary and likert reponses
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Excel file
file_path = 'HeatMap_Readability.xlsx'
xls = pd.ExcelFile(file_path)
df = xls.parse('Sheet1')

#Assign Q1–Q60 per unique question for each LLM
df['Question'] = None

for llm in df['LLM'].unique():
    unique_ids = df[df['LLM'] == llm]['Question ID'].unique()
    q_labels = {qid: f"Q{i+1}" for i, qid in enumerate(unique_ids)}
    df.loc[df['LLM'] == llm, 'Question'] = df[df['LLM'] == llm]['Question ID'].map(q_labels)

# Define Likert-scale columns and shorter labels
likert_cols = [
    'Likely to be asked in primary care visits (1 unlikely  - 5 very likely)',
    "Significant for patient's health? (1 very insignificant - 5  very significant)",
    'Would you be happy to answer this question? (1 would not answer - 5 would very much like to answer)'
]
likert_labels = [
    'Asked in Primary Care',
    'Significant for Health',
    'Willing to Answer'
]

# Group by LLM and plot
llm_groups = df.groupby("LLM")
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 30), sharey=True)
cmap = sns.color_palette("YlGnBu", as_cmap=True)

for ax, (llm_name, group) in zip(axes, llm_groups):
    heatmap_data = group[["Question"] + likert_cols].set_index("Question")
    heatmap_data.columns = likert_labels
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap=cmap,
        vmin=1,
        vmax=5,
        linewidths=0.5,
        linecolor='white',
        cbar=True,
        annot=False
    )
    ax.set_title(f"{llm_name} – Likert Ratings", fontsize=18)
    ax.set_xlabel("Evaluation Criterion", fontsize=14)
    ax.set_ylabel("Question", fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.show()


# In[6]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Load Excel file
file_path = 'HeatMap_Readability.xlsx'
xls = pd.ExcelFile(file_path)
df = xls.parse('Sheet1')

#Assign Q1–Q60 per unique question for each LLM
df['Question'] = None

for llm in df['LLM'].unique():
    unique_ids = df[df['LLM'] == llm]['Question ID'].unique()
    q_labels = {qid: f"Q{i+1}" for i, qid in enumerate(unique_ids)}
    df.loc[df['LLM'] == llm, 'Question'] = df[df['LLM'] == llm]['Question ID'].map(q_labels)

# Define binary columns and convert Y/N to 1/0
binary_cols = [
    'Is the question clearly phrased and easily understandable? (Y/N)',
    'Does the question make clinical sense (Y/N)'
]
binary_labels = ['Clearly Phrased', 'Clinically Sensible']
df_binary = df.copy()
df_binary[binary_cols] = df_binary[binary_cols].applymap(
    lambda x: 1 if str(x).strip().lower() in ['y', 'yes'] else 0
)

# Group by LLM and plot
llm_groups = df_binary.groupby("LLM")
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 30), sharey=True)
binary_cmap = ListedColormap(["#00A9F4", "#BF0F05"])  # Blue = 0, Red = 1

for ax, (llm_name, group) in zip(axes, llm_groups):
    heatmap_data = group[["Question"] + binary_cols].set_index("Question")
    heatmap_data.columns = binary_labels
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap=binary_cmap,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor='white',
        cbar=False,
        annot=False
    )
    ax.set_title(f"{llm_name} – Binary Ratings", fontsize=18)
    ax.set_xlabel("Evaluation Criterion", fontsize=14)
    ax.set_ylabel("Question", fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

# Add binary color legend
red_patch = mpatches.Patch(color="#BF0F05", label="1 (Yes)")
blue_patch = mpatches.Patch(color="#00A9F4", label="0 (No)")
fig.legend(handles=[red_patch, blue_patch], loc='lower center', ncol=2, fontsize=14)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()


# In[11]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import os

# Load Excel file
file_path = r'HeatMap_Readability.xlsx'
df = pd.read_excel(file_path)

# Output directory
output_dir = "LLM_Heatmaps"
os.makedirs(output_dir, exist_ok=True)

# Map clinician names to numeric labels
df['Clinician'] = df['Clinician'].map({
    'Cindy': 'Clinician 1',
    'Karim': 'Clinician 2',
    'Mandy': 'Clinician 3'
})

# Assign Q1–Q60 per unique question for each LLM
df['Question'] = None
for llm in df['LLM'].unique():
    unique_ids = df[df['LLM'] == llm]['Question ID'].unique()
    q_labels = {qid: f"Q{i+1}" for i, qid in enumerate(unique_ids)}
    df.loc[df['LLM'] == llm, 'Question'] = df[df['LLM'] == llm]['Question ID'].map(q_labels)

# === Likert-Scale Ratings ===
likert_cols = {
    'Likely to be asked in primary care visits (1 unlikely  - 5 very likely)': 'Asked in Primary Care',
    "Significant for patient's health? (1 very insignificant - 5  very significant)": 'Significant for Health',
    'Would you be happy to answer this question? (1 would not answer - 5 would very much like to answer)': 'Willing to Answer'
}

for llm in df['LLM'].unique():
    for original_col, title in likert_cols.items():
        pivot_df = df[df["LLM"] == llm].pivot(index="Question", columns="Clinician", values=original_col)
        pivot_df = pivot_df.reindex([f"Q{i+1}" for i in range(60)])  # Sort Q1 to Q60
        plt.figure(figsize=(10, 14))
        sns.heatmap(
            pivot_df,
            cmap="YlGnBu",
            vmin=1,
            vmax=5,
            linewidths=0.5,
            linecolor='white',
            cbar=True,
            annot=False  # Remove numbers
        )
        plt.title(f"{llm} – {title}", fontsize=16)
        plt.xlabel("Clinician")
        plt.ylabel("Question")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{llm.replace(' ', '_')}_{title.replace(' ', '_')}.png"))
        plt.close()

# === Binary Ratings ===
binary_cols = {
    'Is the question clearly phrased and easily understandable? (Y/N)': 'Clearly Phrased',
    'Does the question make clinical sense (Y/N)': 'Clinically Sensible'
}

# Convert Y/N to 1/0
for col in binary_cols.keys():
    df[col] = df[col].apply(lambda x: 1 if str(x).strip().lower() in ['y', 'yes'] else 0)

binary_cmap = ListedColormap(["#00A9F4", "#BF0F05"])  # Blue = 0, Red = 1

for llm in df['LLM'].unique():
    for original_col, title in binary_cols.items():
        pivot_df = df[df["LLM"] == llm].pivot(index="Question", columns="Clinician", values=original_col)
        pivot_df = pivot_df.reindex([f"Q{i+1}" for i in range(60)])  # Sort Q1 to Q60
        plt.figure(figsize=(10, 14))
        sns.heatmap(
            pivot_df,
            cmap=binary_cmap,
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor='white',
            cbar=False,
            annot=False  # Remove numbers
        )
        plt.title(f"{llm} – {title}", fontsize=16)
        plt.xlabel("Clinician")
        plt.ylabel("Question")
        red_patch = mpatches.Patch(color="#BF0F05", label="1 (Yes)")
        blue_patch = mpatches.Patch(color="#00A9F4", label="0 (No)")
        plt.legend(handles=[red_patch, blue_patch], loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{llm.replace(' ', '_')}_{title.replace(' ', '_')}.png"))
        plt.close()




# In[17]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# Load and prepare
file_path = r'HeatMap_Readability.xlsx'
output_dir = "LLM_Combined_Likert_Segmented"
os.makedirs(output_dir, exist_ok=True)
df = pd.read_excel(file_path)

# Clinician mapping
df['Clinician'] = df['Clinician'].map({
    'Cindy': 'Clinician 1',
    'Karim': 'Clinician 2',
    'Mandy': 'Clinician 3'
})

# Assign Q1–Q60 per LLM
df['Question'] = None
for llm in df['LLM'].unique():
    unique_ids = df[df['LLM'] == llm]['Question ID'].unique()
    q_labels = {qid: f"Q{i+1}" for i, qid in enumerate(unique_ids)}
    df.loc[df['LLM'] == llm, 'Question'] = df[df['LLM'] == llm]['Question ID'].map(q_labels)

# Likert criteria
likert_map = {
    'Likely to be asked in primary care visits (1 unlikely  - 5 very likely)': 'Asked in Primary Care',
    "Significant for patient's health? (1 very insignificant - 5  very significant)": 'Significant for Health',
    'Would you be happy to answer this question? (1 would not answer - 5 would very much like to answer)': 'Willing to Answer'
}

# Plot one combined heatmap per LLM
for llm in df['LLM'].unique():
    subset = df[df["LLM"] == llm]

    # Combine Likert columns: multi-index columns for each criterion x clinician
    blocks = []
    for original_col, title in likert_map.items():
        p = subset.pivot(index="Question", columns="Clinician", values=original_col)
        p.columns = pd.MultiIndex.from_product([[title], p.columns])
        blocks.append(p)

    combined = pd.concat(blocks, axis=1)
    combined = combined.reindex([f"Q{i+1}" for i in range(60)])

    # Color bar for patient group
    patient_map = subset.drop_duplicates("Question").set_index("Question")["Patient"]
    patient_cat = patient_map.loc[combined.index].astype("category")
    palette = sns.color_palette("Set2", n_colors=patient_cat.cat.categories.size)
    row_colors = [palette[i] for i in patient_cat.cat.codes]

    # Plot
    fig, ax = plt.subplots(figsize=(18, 14))
    sns.heatmap(
        combined,
        cmap="YlGnBu",
        vmin=1,
        vmax=5,
        linewidths=0.5,
        linecolor='white',
        cbar=True,
        annot=False,
        ax=ax
    )

    # Add patient color strip on left
    for y, color in enumerate(row_colors):
        ax.add_patch(plt.Rectangle((-0.5, y), 0.25, 1, color=color, lw=0))

    # Add horizontal lines to show groups (20-question segmentation)
    for i in [20, 40]:
        ax.axhline(i, color='black', linewidth=1)

    # Legend for patient groups
    handles = [mpatches.Patch(color=palette[i], label=label) for i, label in enumerate(patient_cat.cat.categories)]
    plt.legend(
        handles=handles,
        title='Patient Group',
        bbox_to_anchor=(0.5, -0.15),
        loc='lower center',
        fontsize=10,
        ncol=2
    )

    plt.title(f"{llm} – Combined Likert Ratings (Grouped by Patient)", fontsize=18)
    plt.xlabel("Evaluation Criterion × Clinician")
    plt.ylabel("Question")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{llm.replace(' ', '_')}_Combined_Likert_Segmented.png"))
    plt.close()


# In[2]:


# Install package first (run in terminal if needed)
# pip install py-readability-metrics

import pandas as pd
from readability import Readability

# Load your dataset
df = pd.read_excel("HeatMap_Readability.xlsx")

# Filter for Q1–Q120
df_q120 = df.iloc[0:120].copy()
df_q120



# In[3]:


import textstat

def compute_scores_textstat(text):
    return pd.Series({
        'Flesch_Kincaid': textstat.flesch_kincaid_grade(text),
        'Flesch_Reading_Ease': textstat.flesch_reading_ease(text),
        'Gunning_Fog': textstat.gunning_fog(text),
        'Dale_Chall': textstat.dale_chall_readability_score(text),
        'SMOG_Index': textstat.smog_index(text),
        'ARI': textstat.automated_readability_index(text)
    })
readability_scores = df_q120['Questions'].apply(compute_scores_textstat)
# Combine with original info
df_results = pd.concat([df_q120[['Question ID', 'LLM']], readability_scores], axis=1)
df_results


# In[11]:


readability_cols=['Flesch_Kincaid', 'Flesch_Reading_Ease',
       'Gunning_Fog', 'Dale_Chall', 'SMOG_Index', 'ARI']
agg_by_llm = df_results.groupby('LLM')[readability_cols].agg(['mean', 'std', 'min', 'max'])
agg_by_llm


# In[5]:


df_results.to_csv('Readability_Score.csv')


# Metric	GPT-4 (Mean)	LLaMA (Mean)	Interpretation
# Flesch-Kincaid	6.79	9.34	GPT-4 is easier to read (lower grade level)
# Flesch Reading Ease	66.05	56.58	GPT-4 questions are easier to understand
# Gunning Fog	9.01	11.53	LLaMA questions are more complex
# Dale-Chall	10.99	11.55	Slightly harder vocabulary in LLaMA
# SMOG Index	1.85	2.34	LLaMA may contain more polysyllabic words
# ARI	7.06	10.64	GPT-4 is simpler overall

# In[14]:


import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define your readability columns
readability_cols = ['Flesch_Kincaid', 'Flesch_Reading_Ease', 'Gunning_Fog', 
                    'Dale_Chall', 'ARI', 'SMOG_Index']

# Read your data
readability_df = pd.read_csv('Readability_Score.csv')

# Create a folder to save plots
output_folder = r"Score_Plots"
os.makedirs(output_folder, exist_ok=True)
readability_df
# Melt the DataFrame for plotting
melted_df = readability_df.melt(id_vars=["LLM"], value_vars=readability_cols, var_name="Metric", value_name="Score")

# Boxplots for each readability metric
for metric in readability_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=readability_df, x='LLM', y=metric)
    plt.title(f'Boxplot of {metric} by LLM')
    plt.tight_layout()
    plot_path = os.path.join(output_folder, f"{metric}_boxplot.png")
    plt.savefig(plot_path, dpi=300)
    plt.show()





# In[35]:


# Bar plots for mean readability scores
mean_scores = readability_df.groupby("LLM")[readability_cols].mean().T
mean_scores.plot(kind='bar', figsize=(10, 6))
plt.title("Mean Readability Scores by LLM")
plt.ylabel("Score")
plt.tight_layout()
plt.show()


# In[36]:


# Statistical Tests
ttest_results = []
mannwhitney_results = []

for metric in readability_cols:
    gpt_scores = readability_df[readability_df["LLM"] == "GPT-4"][metric]
    llama_scores = readability_df[readability_df["LLM"] == "Llama"][metric]
    
    t_stat, t_p = ttest_ind(gpt_scores, llama_scores, equal_var=False)
    u_stat, u_p = mannwhitneyu(gpt_scores, llama_scores, alternative='two-sided')
    
    ttest_results.append((metric, t_stat, t_p))
    mannwhitney_results.append((metric, u_stat, u_p))

# Correlation matrix
correlation_matrix = readability_df[readability_cols].corr()


# In[37]:


correlation_matrix


# In[38]:


ttest_results


# In[39]:


mannwhitney_results


# In[15]:


# Compute number of words and character length for each question
readability_df["Num_Words"] = readability_df["Question ID"].map(
    df_q120.set_index("Question ID")["Questions"].str.split().str.len()
)

readability_df["Num_Chars"] = readability_df["Question ID"].map(
    df_q120.set_index("Question ID")["Questions"].str.len()
)

readability_df


# In[16]:


readability_df.to_csv('Readability_Score_all.csv')


# In[42]:


# Aggregate number of words and characters by LLM
length_metrics = ['Num_Words', 'Num_Chars']
agg_length_stats = readability_df.groupby("LLM")[length_metrics].agg(['mean', 'std', 'min', 'max'])
agg_length_stats



# In[17]:


import os
import matplotlib.pyplot as plt
import seaborn as sns

# Create a folder to save plots
output_folder = r"Score_Plots"
os.makedirs(output_folder, exist_ok=True)

# Define readability metrics
metrics = ['Flesch_Kincaid', 'Flesch_Reading_Ease', 'Gunning_Fog', 
           'Dale_Chall', 'SMOG_Index', 'ARI']

# Re-plot scatter plots with cross markers and save them
for metric in metrics:
    # Words vs metric
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=readability_df,
        x='Num_Words',
        y=metric,
        hue='LLM',
        marker='x'
    )
    plt.title(f'{metric} vs. Number of Words')
    plt.tight_layout()
    file_path_words = os.path.join(output_folder, f"{metric}_vs_NumWords.png")
    plt.savefig(file_path_words, dpi=300)
    plt.close()

    # Chars vs metric
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=readability_df,
        x='Num_Chars',
        y=metric,
        hue='LLM',
        marker='x'
    )
    plt.title(f'{metric} vs. Number of Characters')
    plt.tight_layout()
    file_path_chars = os.path.join(output_folder, f"{metric}_vs_NumChars.png")
    plt.savefig(file_path_chars, dpi=300)
    plt.close()




# In[3]:


import matplotlib.pyplot as plt
import numpy as np

# Categories
categories = ['Clarity', 'Completeness', 'Conciseness', 'Utility']
N = len(categories)

# Dummy values for each category (0 to 5 scale)
healthcare_provider = [4.5, 3.0, 4.0, 4.0]
gpt4_complex = [3.5, 2.5, 3.0, 3.5]

# Close the plot by repeating the first value
values1 = healthcare_provider + [healthcare_provider[0]]
values2 = gpt4_complex + [gpt4_complex[0]]

# Create angle list
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # complete the loop

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Draw labels
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="gray", size=8)
plt.ylim(0, 5)

# Plot each dataset
ax.plot(angles, values1, color='mediumseagreen', linewidth=2, label='Healthcare Provider')
ax.fill(angles, values1, color='mediumseagreen', alpha=0.3)

ax.plot(angles, values2, color='skyblue', linewidth=2, label='GPT4 (Complex)')
ax.fill(angles, values2, color='skyblue', alpha=0.3)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# Show the plot
plt.title("Radar Chart Comparison", size=15, y=1.08)
plt.show()


# In[43]:


# ─────────────────────────────────────────────────────────────────────────────
# Radar plots of clinicians’ Likert ratings — 1 figure per LLM
# ─────────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# ── 1. Load Excel ────────────────────────────────────────────────────────────
file_path = Path(r"HeatMap_Readability.xlsx")
df = pd.read_excel(file_path)

# ── 2. Clean headers & rename columns ────────────────────────────────────────
df.columns = (df.columns
                .str.strip()
                .str.replace(r"\s+", " ", regex=True))

df = df.rename(columns={
    "Likely to be asked in primary care visits (1 unlikely - 5 very likely)":
        "Primary care relevance",
    "Significant for patient's health? (1 very insignificant - 5 very significant)":
        "Significance",
    "Would you be happy to answer this question? (1 would not answer - 5 would very much like to answer)":
        "Clinician willingness"
})

likert_metrics = ["Primary care relevance",
                  "Significance",
                  "Clinician willingness"]

# ── 3. Remap clinicians ──────────────────────────────────────────────────────
clinician_map = {'Karim': 'Clinician 1',
                 'Cindy': 'Clinician 2',
                 'Mandy': 'Clinician 3'}
df["Clinician"] = df["Clinician"].map(clinician_map)

# ── 4. Radar‑plot helper that RETURNS fig so caller can save -----------------
def plot_radar_likert(data, title, clinician_order,
                      r_max=5, label_pad=0.4):

    angles = np.linspace(0, 2*np.pi, len(likert_metrics),
                         endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    ax.set_ylim(0, r_max)

    colours = ["mediumseagreen", "skyblue", "coral"]
    for colour, clinician in zip(colours, clinician_order):
        if clinician not in data.index:
            continue
        vals = data.loc[clinician].tolist() + data.loc[clinician].tolist()[:1]
        ax.plot(angles, vals, linewidth=2, color=colour, label=clinician)
        ax.fill(angles, vals, alpha=0.10, color=colour)

    ax.set_yticks(range(1, r_max+1))
    ax.set_yticklabels([str(i) for i in range(1, r_max+1)], fontsize=8)
    ax.set_rlabel_position(0)

    # horizontal axis labels
    ax.set_xticks([])
    for ang, lab in zip(angles[:-1], likert_metrics):
        extra = 0.25 if lab == "Clinician willingness" else 0
        r_lbl = r_max + label_pad + extra

        ang_deg = np.degrees(ang)
        if -30 <= ang_deg <= 30 or ang_deg >= 330:
            ha, va = "left", "center"
        elif 60 <= ang_deg <= 150:
            ha, va = "right", "center"
        else:
            ha, va = "center", "top"

        ax.text(ang, r_lbl, lab, ha=ha, va=va, fontsize=10)

    ax.set_title(title, fontsize=14, y=1.10)
    ax.legend(loc='upper right', bbox_to_anchor=(1.32, 1.10))
    plt.tight_layout()
    return fig          # <‑‑ return the figure so we can save it

# ── 5. Output directory for PNGs ─────────────────────────────────────────────
out_dir = Path("llm_radar_plots")
out_dir.mkdir(parents=True, exist_ok=True)

# ── 6. Draw & save one radar chart per LLM ───────────────────────────────────
clinician_order = ['Clinician 1', 'Clinician 2', 'Clinician 3']

for llm in df["LLM"].unique():
    llm_means = (df[df["LLM"] == llm]
                 .groupby("Clinician")[likert_metrics]
                 .mean())

    fig = plot_radar_likert(llm_means,
                            title=f"Clinician Ratings (Likert) – {llm}",
                            clinician_order=clinician_order)

    # safe filename e.g. "Clinician_Ratings_Likert_GPT-4.png"
    safe_llm = llm.replace(" ", "_").replace("/", "_")
    png_path = out_dir / f"Clinician_Ratings_Likert_{safe_llm}.png"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()            # comment this out if you don’t need the popup
    plt.close(fig)
    print(f"Saved {png_path}")


# In[31]:


import pandas as pd
from pathlib import Path

# ── CONFIGURATION ────────────────────────────────────────────────────────────
xlsx_path   = Path(r"Radar_Table.xlsx")
labs_per_pt = 3
random_pick = False           # True → pick random labs, False → first N alphabetically

# ── LOAD ─────────────────────────────────────────────────────────────────────
df = pd.read_excel(xlsx_path)

required_cols = {"Patient", "LLM", "Lab test", "Questions"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in {xlsx_path.name}: {missing}")

df["Lab test"]  = df["Lab test"].astype(str).str.strip()
df["Questions"] = df["Questions"].astype(str).str.strip()

# evaluation columns to preserve
extra_cols = [
    "Clinician",
    "Is the question clearly phrased and easily understandable? (Y/N)",
    "Does the question make clinical sense (Y/N)",
    "Likely to be asked in primary care visits (1 unlikely  - 5 very likely)",
    "Significant for patient's health? (1 very insignificant - 5  very significant)",
    "Would you be happy to answer this question? (1 would not answer - 5 would very much like to answer)"
]

# ── VERIFY *exactly* TWO LLMs ────────────────────────────────────────────────
llms = df["LLM"].unique().tolist()
if len(llms) != 2:
    raise ValueError(f"Expected exactly two LLMs, found {len(llms)}: {llms}")
llm1, llm2 = llms

# ── BUILD OUTPUT ────────────────────────────────────────────────────────────
rows_out = []

for patient_id, grp in df.groupby("Patient"):

    # 1. labs common to both LLMs
    labs_llm1 = set(grp.loc[grp["LLM"] == llm1, "Lab test"])
    labs_llm2 = set(grp.loc[grp["LLM"] == llm2, "Lab test"])
    shared_labs = sorted(labs_llm1 & labs_llm2)

    if len(shared_labs) < labs_per_pt:
        print(f"Patient {patient_id}: only {len(shared_labs)} shared labs – skipping.")
        continue

    # 2. choose the labs we will keep
    chosen_labs = (
        pd.Series(shared_labs).sample(labs_per_pt, random_state=0).tolist()
        if random_pick else shared_labs[:labs_per_pt]
    )

    subset = grp[grp["Lab test"].isin(chosen_labs)].copy()

    # 3. keep ONE row per (Clinician, LLM, Lab) so each clinician’s rating survives
    key_cols = ["Clinician", "LLM", "Lab test"]
    if "Rank" in subset.columns:
        subset = (subset.sort_values("Rank")
                        .groupby(key_cols, as_index=False)
                        .head(1))      # best‑ranked for that clinician
    else:
        subset = (subset.groupby(key_cols, as_index=False)
                        .head(1))      # first encountered for that clinician

    # 4. collect desired columns
    keep_cols = ["Patient", "Lab test", "LLM", "Questions"] + extra_cols
    rows_out.extend(subset[keep_cols].to_dict(orient="records"))

# ── RESULT DF ───────────────────────────────────────────────────────────────
result = (pd.DataFrame(rows_out)
            .sort_values(["Patient", "Lab test", "LLM", "Clinician"])
            .reset_index(drop=True))

print(result.head(15))
# result.to_excel("Selected_3_Lab_Qs_with_all_clinicians.xlsx", index=False)


# In[32]:


result


# In[33]:


result.to_csv('Radar_Table_final.csv')


# In[42]:


# ─────────────────────────────────────────────────────────────────────────────
# Radar plots of clinicians’ Likert ratings
#   • two LLMs × three patients  →  6 PNG files in ./radar_plots
# ─────────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# ── 1. Load the CSV / Excel with the selected‑question rows ──────────────────
data_path = Path(r"C:\D\e Health Lab projects\Question_Answering\Question template generation\Results\Evaluation_results_of_AI_question_generation_module_Round3\Radar_Table_final.csv")
df = pd.read_csv(data_path)                     # or pd.read_excel(...)

# ── 2. Clean & rename columns ────────────────────────────────────────────────
df.columns = (df.columns
                .str.strip()
                .str.replace(r"\s+", " ", regex=True))

df = df.rename(columns={
    "Likely to be asked in primary care visits (1 unlikely - 5 very likely)":
        "Primary care relevance",
    "Significant for patient's health? (1 very insignificant - 5 very significant)":
        "Significance",
    "Would you be happy to answer this question? (1 would not answer - 5 would very much like to answer)":
        "Clinician willingness"
})

likert_metrics = ["Primary care relevance", "Significance", "Clinician willingness"]

# ── 3. Remap clinician names (optional) ──────────────────────────────────────
clinician_map = {'Karim': 'Clinician 1',
                 'Cindy': 'Clinician 2',
                 'Mandy': 'Clinician 3'}
df["Clinician"] = df["Clinician"].map(clinician_map).fillna(df["Clinician"])

# ── 4. Helper that RETURNS the figure ---------------------------------------
def plot_radar_likert(data, title, clinician_order, r_max=5, label_pad=0.4):
    angles = np.linspace(0, 2*np.pi, len(likert_metrics), endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_ylim(0, r_max)

    colours = ["mediumseagreen", "skyblue", "coral"]
    for colour, clinician in zip(colours, clinician_order):
        if clinician not in data.index:
            continue
        vals = data.loc[clinician].tolist() + data.loc[clinician].tolist()[:1]
        ax.plot(angles, vals, linewidth=2, color=colour, label=clinician)
        ax.fill(angles, vals, alpha=0.10, color=colour)

    ax.set_yticks(range(1, r_max+1))
    ax.set_yticklabels([str(i) for i in range(1, r_max+1)], fontsize=8)
    ax.set_rlabel_position(0)

    # horizontal axis labels
    ax.set_xticks([])
    for ang, lab in zip(angles[:-1], likert_metrics):
        extra = 0.25 if lab == "Clinician willingness" else 0
        r_lbl = r_max + label_pad + extra

        ang_deg = np.degrees(ang)
        if -30 <= ang_deg <= 30 or ang_deg >= 330:
            ha, va = "left", "center"
        elif 60 <= ang_deg <= 150:
            ha, va = "right", "center"
        else:
            ha, va = "center", "top"

        ax.text(ang, r_lbl, lab, ha=ha, va=va, fontsize=9)

    ax.set_title(title, fontsize=13, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1.08), fontsize=8)
    plt.tight_layout()
    return fig  # ← return so caller can save &/or show

# ── 5. Create output directory ───────────────────────────────────────────────
out_dir = Path("radar_plots")
out_dir.mkdir(parents=True, exist_ok=True)

# ── 6. Generate & save six radar charts ──────────────────────────────────────
clinician_order = ['Clinician 1', 'Clinician 2', 'Clinician 3']

for pid in df["Patient"].unique():
    patient_df = df[df["Patient"] == pid]

    for llm in patient_df["LLM"].unique():           # e.g. GPT‑4, LLaMA
        subset = patient_df[patient_df["LLM"] == llm]
        means  = subset.groupby("Clinician")[likert_metrics].mean()

        fig = plot_radar_likert(means,
                                 title=f"Patient {pid} – {llm}",
                                 clinician_order=clinician_order)

        # safe filename like: Patient_69_Male_GPT-4.png
        safe_title = (f"Patient_{pid}_{llm}"
                      .replace(" ", "_")
                      .replace("/", "_"))
        png_path = out_dir / f"{safe_title}.png"

        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.show()        # comment out if you don’t want the interactive window
        plt.close(fig)    # free memory
        print(f"Saved {png_path}")


# In[2]:


#Round2 vs round 3
import pandas as pd
import textstat

# Load the Excel file
df = pd.read_excel("Readability_Score_Round 2_3.xlsx")

# Define a function to compute readability metrics
def compute_scores_textstat(text):
    return pd.Series({
        'Flesch_Kincaid': textstat.flesch_kincaid_grade(text),
        'Flesch_Reading_Ease': textstat.flesch_reading_ease(text),
        'Gunning_Fog': textstat.gunning_fog(text),
        'Dale_Chall': textstat.dale_chall_readability_score(text),
        'SMOG_Index': textstat.smog_index(text),
        'ARI': textstat.automated_readability_index(text),
        'Num_Words': len(text.split()),
        'Num_Chars': len(text)
    })

# Apply to the Question column
readability_scores = df['Question'].apply(compute_scores_textstat)

# Combine with metadata
df_results = pd.concat([df[['Quesation ID', 'Round', 'Patient']], readability_scores], axis=1)
df_results


# In[8]:


# 5) Select only the numeric columns we want to summarize
numeric_cols = [
    'Flesch_Kincaid',
    'Flesch_Reading_Ease',
    'Gunning_Fog',
    'Dale_Chall',
    'SMOG_Index',
    'ARI',
    'Num_Words',
    'Num_Chars'
]

# 6) Group by Round and compute mean & std
round_stats = (
    df_results
      .groupby('Round')[numeric_cols]
      .agg(['mean', 'std','min', 'max'])
)

# 7) (Optional) Clean up the MultiIndex columns
round_stats.columns = [f"{metric}_{stat}" for metric, stat in round_stats.columns]
round_stats = round_stats.reset_index()
round_stats





