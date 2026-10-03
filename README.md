Breast Cancer Diagnosis System (KNN)
A desktop application built with Python + Tkinter that trains a K-Nearest Neighbors classifier on breast-tumor cell measurements and classifies a new tumor as Malignant (M) or Benign (B).
> ⚠️ **Disclaimer:** This is an educational / research project. It is **not a medical device** and must not be used to make diagnosis or treatment decisions. Always consult a qualified healthcare professional.
---
Features
Automatically loads and trains from a default CSV file at start-up
Load any other compatible CSV through a file dialog
One-click model training with test-set accuracy report (80/20 stratified split)
8 numeric inputs with validation (empty / non-numeric values are rejected)
Colour-coded result (red = malignant, green = benign) with a confidence percentage
Status bar showing the current data and model state
Requirements
Python 3.8+ (with Tkinter)
`pandas`, `numpy`, `scikit-learn`
```bash
pip install pandas numpy scikit-learn
```
> On Debian/Ubuntu, Tkinter may need to be installed separately: `sudo apt install python3-tk`
Quick Start
Put `qq.py` and your dataset in the same folder.
Name the dataset `knn_breast_cancer-selected-columns.csv` (or load another file from inside the app).
Run the app from that folder:
```bash
python qq.py
```
Data Format
The CSV must contain `id`, `diagnosis` (`M` / `B`) and exactly these eight feature columns, in this order, with no missing values and no other columns:
#	Column	Description
1	`radius_mean`	Mean distance from centre to perimeter points
2	`texture_mean`	Standard deviation of gray-scale values
3	`perimeter_mean`	Mean size of the core tumor perimeter
4	`area_mean`	Mean tumor area
5	`smoothness_mean`	Mean local variation in radius lengths
6	`compactness_mean`	Perimeter² / area − 1.0
7	`concavity_mean`	Severity of concave portions of the contour
8	`concave points_mean`	Number of concave portions of the contour
The column names follow the Wisconsin Diagnostic Breast Cancer dataset naming, reduced to the eight "mean" measurements.
Usage
Button	What it does
📂 Load New Data	Choose a CSV file. You must then click Train Model.
🔄 Train Model	Splits, scales, trains and shows the test accuracy.
🗑️ Clear Fields	Empties all inputs and resets the result.
🔍 Diagnose Tumor	Validates the inputs and shows the prediction.
Typical workflow: start the app → (optionally load + train another dataset) → enter the 8 values (use a dot as the decimal separator) → click Diagnose Tumor.
Reading the confidence
The model uses `k = 5`, so confidence can only be 60 %, 80 % or 100 %. It is the share of the 5 nearest neighbours voting for the predicted class — not a calibrated probability.
How It Works
Drop `id` and `diagnosis` → features `X`, target `y`
Encode labels with `LabelEncoder` (`B` → 0, `M` → 1)
`train_test_split` — 80 % / 20 %, `random_state=42`, stratified
`StandardScaler` fitted on the training set only
`KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)` (Euclidean)
Accuracy computed on the 20 % test set
For a new sample: scale → `predict` / `predict_proba` → decode label
Code Structure
```
qq.py
└── BreastCancerApp
    ├── __init__(root)          # window setup, loads default data
    ├── create_widgets()        # builds the whole UI
    ├── load_default_data()     # reads the default CSV, auto-trains
    ├── load_data()             # file dialog -> loads CSV (no retrain)
    ├── train_model()           # preprocessing + KNN training + evaluation
    ├── predict()               # input validation + prediction + display
    ├── clear_inputs()          # reset fields and result
    └── update_status(message)  # status bar text
```
Troubleshooting
Problem	Fix
"Data file not found" at start-up	Run from the folder containing the CSV, or use Load New Data
"Please train the model first"	Click Train Model
`['id'] not found in axis`	Add `id` and `diagnosis` columns to the CSV
`Input contains NaN`	Remove or fill rows with empty cells
`X has N features, but StandardScaler is expecting M`	Remove extra columns (e.g. `Unnamed: 32`)
`ModuleNotFoundError`	`pip install pandas numpy scikit-learn`
Known Limitations
Training uses all columns except `id`/`diagnosis`, while prediction uses the 8 form fields in fixed order — extra or reordered CSV columns break or corrupt results.
Load New Data does not retrain; the previous model stays active until Train Model is clicked.
The default CSV is searched in the current working directory, not the script folder.
Accuracy is from a single 80/20 split; the final model is not refit on all data.
The label `'M'` is hard-coded.
No range/plausibility checks on the entered values.
Ideas for Improvement
Select training columns explicitly: `df[[key for key, _ in self.features]]`
Auto-train after loading a new file
Cross-validation, confusion matrix, precision/recall (especially recall for the malignant class)
Tune `k` with `GridSearchCV`
Save / load the model with `joblib`
Separate ML logic from the GUI and add unit tests
Package as a standalone executable with PyInstaller
License
Add your preferred license here (e.g. MIT).
