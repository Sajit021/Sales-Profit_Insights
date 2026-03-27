# Global Superstore Analysis Project

This folder contains sales analysis code and a Streamlit dashboard built on the `Global_Superstore2.csv` dataset.

## Project Files

- `salesDashboard.py`: Interactive Streamlit dashboard.
- `sales.py`: Python script for analysis and visualizations.
- `sales.ipynb`: Notebook version of the analysis.
- `Global_Superstore2.csv`: Dataset required by the code.

## Prerequisites

- Python 3.9 or newer
- `pip` available in terminal

## 1) Open this folder

Open a terminal in this project folder (where the CSV and Python files exist).

## 2) Create and activate a virtual environment (recommended)

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once in PowerShell, then try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 3) Install required packages

```powershell
pip install numpy pandas plotly scipy streamlit
```

Optional (only if you want to run the notebook):

```powershell
pip install notebook jupyter
```

## 4) Run the dashboard (main app)

```powershell
streamlit run salesDashboard.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## 5) Run the analysis script

```powershell
python sales.py
```

This runs the analysis and opens/prints chart outputs depending on your environment.

## 6) Run the notebook (optional)

```powershell
jupyter notebook
```

Then open `sales.ipynb` in the browser/Jupyter UI.

## Troubleshooting

- **File not found (`Global_Superstore2.csv`)**:
  Make sure the CSV is in the same folder as `salesDashboard.py` and `sales.py`.
- **`streamlit` command not recognized**:
  Ensure your virtual environment is activated, or run:
  `python -m streamlit run salesDashboard.py`
- **Dependency errors**:
  Reinstall packages in the same environment used to run the commands.
