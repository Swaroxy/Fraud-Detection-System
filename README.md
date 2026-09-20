# 💳 Fraudulent Transaction Detection with Neural Networks

An end-to-end **credit card fraud detection** project using **TensorFlow/Keras** and classical machine learning. The project focuses on handling extreme class imbalance and building a neural-network classifier capable of identifying potentially fraudulent transactions.

## 🎯 Project Objective

Build a machine-learning system that classifies credit card transactions as:

* `0` — Legitimate
* `1` — Fraudulent

Because fraud represents only a very small portion of transactions, the project focuses on **Precision, Recall, F1-score, PR-AUC, ROC-AUC, and threshold tuning** rather than relying on accuracy alone.

## 🧠 Approach

The project follows a complete ML workflow:

1. Load and inspect the credit-card transaction dataset
2. Analyze missing values, duplicates, class imbalance, transaction amount, time, and PCA features
3. Remove duplicate records
4. Create stratified **70/15/15 train-validation-test splits**
5. Scale `Time` and `Amount` using training data only
6. Train a **balanced Logistic Regression baseline**
7. Build a regularized **Keras Multi-Layer Perceptron (MLP)**
8. Handle class imbalance using class weights
9. Compare **Adam vs SGD + momentum**
10. Train the final model using **EarlyStopping** and **ReduceLROnPlateau**
11. Evaluate model performance on validation and test data
12. Tune the fraud classification threshold using validation F1-score
13. Evaluate the selected threshold on the untouched test set
14. Save the trained model, scaler, metrics, and visualizations
15. Demonstrate live-style transaction screening

## 🏗️ Neural Network

The main model is a fully connected Keras MLP:

```text
Input Features
      ↓
Dense(64, ReLU + L2)
      ↓
Dropout(30%)
      ↓
Dense(32, ReLU + L2)
      ↓
Dropout(30%)
      ↓
Dense(16, ReLU)
      ↓
Dense(1, Sigmoid)
```

The model uses **binary cross-entropy**, class weighting, dropout, and L2 regularization to address the highly imbalanced fraud-detection problem.

## 📊 Evaluation

The project evaluates:

* Precision
* Recall
* F1-score
* PR-AUC / Average Precision
* ROC-AUC
* Confusion Matrix
* Precision-Recall Curve
* ROC Curve

The classification threshold is also tuned on the validation set rather than blindly relying on `0.5`.

## 🛠️ Technologies

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow / Keras
* Joblib
* Jupyter Notebook

## 📁 Project Outputs

The notebook generates:

```text
outputs/
├── figures/
├── models/
└── reports/
```

Saved artifacts include the preprocessing scaler, Logistic Regression baseline, trained neural-network model, evaluation figures, metrics, and threshold-comparison reports.

## ▶️ Running the Project

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow joblib jupyter
```

Place the dataset at:

```text
data/creditcard.csv
```

Then open and run:

```text
Fraud_Detection_Project.ipynb
```

## 🔑 Key Concepts Demonstrated

* Binary classification
* Severe class imbalance
* Data preprocessing
* Stratified train/validation/test splitting
* Feature scaling
* Logistic Regression
* Neural networks / MLPs
* ReLU and Sigmoid activations
* Binary cross-entropy
* Class weighting
* Dropout regularization
* L2 regularization
* Adam optimization
* SGD with momentum
* Early stopping
* Learning-rate scheduling
* Precision-Recall analysis
* Threshold optimization
* Model persistence
* Transaction-level inference

## 📌 Project Structure

```text
Fraud-Detection/
│
├── data/
│   └── creditcard.csv
│
├── outputs/
│   ├── figures/
│   ├── models/
│   └── reports/
│
├── Fraud_Detection_Project.ipynb
└── README.md
```

👨‍💻 Author

Swagat Bhattarai

AI / ML Engineer | Python | Deep Learning | RAG | Generative AI

GitHub: [Swaroxy](https://github.com/Swaroxy)
