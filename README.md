# PhishLens

PhishLens is a **Python machine-learning project** that classifies website records as **phishing**, **suspicious**, or **legitimate** using decision trees. Built with **scikit-learn**, **Pandas**, **SciPy**, and **Matplotlib**, it processes the bundled [UCI Website Phishing dataset](https://archive.ics.uci.edu/dataset/379/website+phishing), compares baseline and tuned models, and identifies the most influential website attributes.

## How to run

Install **Python 3** (tested with Python 3.13), then download or clone this project. Open a terminal inside the `phish-lens` folder and keep `PhishingData.arff` in the same folder as `script_Linh_Nguyen.py`.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install scikit-learn pandas scipy matplotlib
.\.venv\Scripts\python.exe script_Linh_Nguyen.py
```

### macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install scikit-learn pandas scipy matplotlib
.venv/bin/python script_Linh_Nguyen.py
```

### Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install scikit-learn pandas scipy matplotlib
.venv/bin/python script_Linh_Nguyen.py
```

The script prints the dataset statistics, baseline and tuned model accuracies, selected hyperparameters, and most important feature.

