# Threat Detection using Machine Learning

## Overview
This project explores how machine learning can be used to identify suspicious behavior in web traffic data. Instead of relying on manual monitoring, the system is designed to automatically detect anomalies and highlight potential threats.

The project follows a complete pipeline starting from raw data processing to model evaluation, making it a practical implementation of data-driven cybersecurity analysis.

---

## Objective
The main objectives of this project are:
- Detect unusual patterns in web traffic
- Identify potential cybersecurity threats
- Build a structured and scalable machine learning pipeline

---

## Dataset
- Source: AWS CloudWatch Web Traffic Dataset (Kaggle)
- The dataset contains:
  - Network traffic information (bytes in/out)
  - IP addresses and country codes
  - Protocol and rule data
  - Timestamps and detection indicators

---

## Methodology

### Data Preprocessing
- Converted timestamp columns into datetime format  
- Checked and handled missing values  
- Encoded categorical variables using Label Encoding  

---

### Exploratory Data Analysis
- Analyzed statistical properties of the dataset  
- Used box plots to detect outliers  
- Generated a correlation heatmap to understand feature relationships  

---

### Feature Engineering
- Created new features such as:
  - Session duration  
  - Hour of activity  
- Standardized numerical features for better model performance  

---

### Anomaly Detection
- Applied Isolation Forest to identify unusual traffic patterns  
- Classified data points as normal or anomalous  

---

### Model Building
The following machine learning models were implemented:
- Random Forest  
- Logistic Regression  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
- Decision Tree  

---

### Model Evaluation
Models were evaluated using:
- Accuracy Score  
- Classification Report  
- Confusion Matrix  
- ROC-AUC Score  
- Cross-validation  

---

## Key Highlights
- End-to-end machine learning pipeline (data → model → evaluation)  
- Combination of anomaly detection and classification techniques  
- Comparison of multiple ML models  
- Visualization of patterns and anomalies  

---

## Tech Stack

**Programming Language**
- Python  

**Libraries**
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  

**Tools**
- Jupyter Notebook  

---

## How to Run

1. Clone the repository:
git clone https://github.com/Ashita-gup/Cybersecurity-threat-detection.git

2. Install required libraries:
pip install -r requirements.txt

3. Run the project:
- Open `Project.ipynb` in Jupyter Notebook  
OR  
- Run `PythonFile.py`  

---

## Future Scope
- Improve detection using deep learning models  
- Build a real-time threat detection system  
- Integrate with cloud-based monitoring tools  

---

## Author
Ashita Gupta  
B.Tech in Computer Science with Specialization in Cyber Security at
Indian Institute of Information Technology Kottayam
