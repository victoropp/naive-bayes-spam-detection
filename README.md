# Naive Bayes Spam Detection in MATLAB



This project implements a **real-world spam email classifier** using the Spambase dataset. It applies **Naive Bayes classification with hyperparameter tuning**, **feature selection**, **bootstrap confidence intervals**, and includes **model comparison with Logistic Regression**.

---

##  Key Features
- Feature pruning (low variance & high correlation)
- Hyperparameter tuning (kernel + width)
- Stratified cross-validation
- Evaluation metrics: Accuracy, Precision, Recall, F1, AUC
- Bootstrap Confidence Interval
- ROC and Precision-Recall curves
- Model comparison with Logistic Regression
- MATLAB deployable model (`SpamClassifierModel.mat`)

---

##  Dataset

- **Name**: [Spambase Dataset](https://archive.ics.uci.edu/ml/datasets/spambase) from UCI
- **Size**: 4,601 emails (57 features + 1 target)
- **Target Classes**: Spam (1), Non-Spam (0)

---

##  Folder Structure

```plaintext
naive-bayes-spam-detection/
├── NaiveBayes_SpamDetection.m       # Main MATLAB script
├── spambase.csv                     # Dataset file (not included in public repo)
├── outputs/                         # Output visualizations
│   ├── class_distribution.png
│   ├── hyperparameter_eval.png
│   ├── optimization_surface.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── precision_recall_curve.png
├── SpamClassifierModel.mat          # Saved deployable model
└── README.md                        # Project documentation

```

---

##  Results Summary

| Metric                           | Value            |
|----------------------------------|------------------|
| Cross-Validated Accuracy (Tuning) | 85.07%          |
| 5-Fold CV Accuracy               | 84.85%           |
| Test Set Accuracy                | 85.94%           |
| **Precision**                    | 0.83752          |
| **Recall**                       | 0.79742          |
| **F1-Score**                     | 0.81698          |
| **AUC**                          | 0.91582          |
| **Bootstrap CI (95%)**           | [83.99%, 87.61%] |
| Logistic Regression Accuracy     | 87.68%           |


---

##  Visualizations

### Class Distribution
![Class Distribution](outputs/class_distribution.png)

### Hyperparameter Optimization
![Hyperparameter Evaluation](outputs/hyperparameter_eval.png)

### Optimization Surface
![Optimization Surface](outputs/optimization_surface.png)

### Confusion Matrix
![Confusion Matrix](outputs/confusion_matrix.png)

### ROC Curve
![ROC Curve](outputs/roc_curve.png)

### Precision-Recall Curve
![Precision-Recall](outputs/precision_recall_curve.png)

---

##  Getting Started

1. Download the dataset from [UCI Spambase](https://archive.ics.uci.edu/ml/datasets/spambase)
2. Open `NaiveBayes_SpamDetection.m` in MATLAB
3. Run the script — metrics and figures will be generated automatically

---

##  Model Deployment
A trained model is saved using MATLAB's `saveLearnerForCoder` for production deployment:
```matlab
saveLearnerForCoder(finalModel, 'SpamClassifierModel')

```
---
##  Author

**Victor Collins Oppon**  
*MSc Data Science | Chartered Accountant*  
[LinkedIn Profile](https://www.linkedin.com/in/victor-collins-oppon-fcca-mba-bsc-01541019/)  
Building a bridge between finance and intelligent automation.


##  License

This project is licensed under the [MIT License](LICENSE).  
You are free to use, modify, and distribute this software for both personal and commercial use, provided proper attribution is given.


