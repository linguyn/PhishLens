# PhishLens

**Explainable machine learning for phishing website classification.**

PhishLens is a **Python** machine-learning project that classifies website records as **phishing**, **suspicious**, or **legitimate** using an **entropy-based decision tree**. It combines data preprocessing, stratified sampling, cross-validated hyperparameter tuning, and feature-importance analysis to evaluate both predictive performance and the rules behind each model.

The project operates on a labeled dataset of precomputed website attributes. It is an offline classification experiment; it does not scan live URLs.

## Highlights

- Process **1,353 website records** with **9 input features** from an ARFF dataset.
- Preserve class proportions with a **stratified 70/30 train-test split**.
- Compare a baseline decision tree with a model selected through **GridSearchCV**.
- Evaluate **49 hyperparameter combinations** using **5-fold cross-validation**.
- Identify influential attributes through **impurity-based feature importance**.
- Visualize the selected model's decision rules with **Matplotlib**.

## Technology stack

| Technology | Role |
| --- | --- |
| **Python** | Data-processing and machine-learning workflow |
| **Pandas** | Numeric conversion, feature/target separation, and dataset statistics |
| **SciPy** | Loading the ARFF dataset with `scipy.io.arff` |
| **scikit-learn** | Decision trees, stratified splitting, grid search, accuracy, and tree plotting |
| **Matplotlib** | Rendering the decision-tree visualization |

## Dataset

The project uses the [Website Phishing dataset from the UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/379/website+phishing), included locally as [`PhishingData.arff`](PhishingData.arff).

| Property | Value |
| --- | --- |
| Records | 1,353 |
| Input features | 9 |
| Target column | `Result` |
| Missing values in the bundled file | 0 |
| Training records | 947 |
| Test records | 406 |

Target labels follow the UCI dataset documentation:

| Label | Class | Records |
| --- | --- | --- |
| `-1` | Phishing | 702 |
| `0` | Suspicious | 103 |
| `1` | Legitimate | 548 |

The input columns are `SFH`, `popUpWidnow`, `SSLfinal_State`, `Request_URL`, `URL_of_Anchor`, `web_traffic`, `URL_Length`, `age_of_domain`, and `having_IP_Address`. Column names, including the spelling of `popUpWidnow`, are preserved from the bundled dataset.

## Methodology

1. **Load and preprocess:** Read the ARFF file with SciPy, create a Pandas DataFrame, and convert encoded values to numeric data.
2. **Separate features and labels:** Use the first nine columns as inputs and the final `Result` column as the target. Print record counts, feature counts, and class distribution.
3. **Create a holdout set:** Split the data into 70% training and 30% testing with `stratify=y` and `random_state=42`.
4. **Train a baseline:** Fit `DecisionTreeClassifier(criterion="entropy", random_state=42)` on the training set.
5. **Tune on training data:** Run five-fold `GridSearchCV` with accuracy scoring over the following parameter grid:

   ```python
   parameters = {
       "max_depth": [None, 3, 4, 5, 6, 7, 8],
       "min_samples_split": [2, 3, 4, 5, 6, 8, 10],
   }
   ```

   The search evaluates 49 combinations across five folds, then refits the selected estimator on the full training set. The held-out test set is not used to select hyperparameters.

6. **Evaluate and interpret:** Compare baseline and selected-model test accuracy, rank feature importance, print the highest-ranked attribute, and construct a labeled decision-tree figure.

## Verified results

The following results were reproduced from the current script and bundled dataset:

| Model | Test accuracy |
| --- | --- |
| Baseline decision tree | **87.19%** |
| Cross-validation-selected decision tree | **86.70%** |

| Selected-model detail | Value |
| --- | --- |
| `max_depth` | `None` |
| `min_samples_split` | `5` |
| Mean cross-validation accuracy | 87.44% |
| Highest-ranked feature | `SFH` |
| `SFH` feature importance | 0.3945 |

The baseline performed about **0.49 percentage points** better on the held-out test set. Grid search selects the model with the best mean cross-validation score on the training data; this does not guarantee higher accuracy on a separate test set.

These results were verified with **Python 3.13.15**, **scikit-learn 1.9.1**, **Pandas 3.0.6**, **SciPy 1.18.1**, and **Matplotlib 3.11.2**. Dependency versions can affect reproducibility.

The included [project report](report_Linh_Nguyen.pdf) records an earlier tuned-model accuracy of **85.71%**. The table above reflects the current executable script. Class meanings in this README follow the UCI documentation.

## Getting started

Install Python and open a terminal in the `phish-lens` directory. Keep `PhishingData.arff` alongside `script_Linh_Nguyen.py`; the script resolves the dataset path relative to its own location.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install scikit-learn pandas scipy matplotlib
python script_Linh_Nguyen.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install scikit-learn pandas scipy matplotlib
python script_Linh_Nguyen.py
```

The script prints dataset statistics, both test accuracies, the selected hyperparameters, and the most important feature. No API keys, database, or external service is required to run the experiment after installing its dependencies.

### Save the decision-tree visualization

The script creates a Matplotlib figure but does not explicitly display or save it. With the virtual environment active, run this command from the project directory to export the figure without changing the source:

```bash
python -c "import matplotlib; matplotlib.use('Agg'); import runpy; import matplotlib.pyplot as plt; runpy.run_path('script_Linh_Nguyen.py'); plt.savefig('decision_tree.png', dpi=150, bbox_inches='tight')"
```

This reruns the experiment and writes `decision_tree.png` to the current directory.

## Project structure

```text
phish-lens/
|-- README.md                 # Project documentation
|-- script_Linh_Nguyen.py      # Preprocessing, training, tuning, and evaluation
|-- PhishingData.arff         # Labeled website feature dataset
`-- report_Linh_Nguyen.pdf    # Original experiment report
```

## Scope and limitations

- Predictions require the nine encoded attributes used during training. Extracting these attributes from a new URL is outside the current implementation.
- Evaluation currently uses accuracy on one fixed holdout split. Per-class precision, recall, F1 scores, and a confusion matrix would provide a more detailed view of performance.
- Feature importance measures a feature's contribution to impurity reduction within this tree; it does not establish causation.
- The trained estimator remains in memory during execution. The project does not currently export a model or provide an inference API.

## Dataset attribution

Abdelhamid, N. (2014). *Website Phishing* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5B301](https://doi.org/10.24432/C5B301).

The dataset is distributed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**, as stated on the [UCI dataset page](https://archive.ics.uci.edu/dataset/379/website+phishing). This attribution applies to the dataset and does not assign a license to the project code.
