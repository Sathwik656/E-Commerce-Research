# Datasets
Dataset1: https://archive.ics.uci.edu/dataset/502/online%2Bretail%2Bii
Dataset1: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce select=olist_orders_dataset.csv

# Models To Be Created
Random Forest Classifier → Churn Prediction
Random Forest Regressor → Customer Spending Prediction

# Tasks
1. Inspect and clean Online Retail II
2. Construct customer features + churn target
3. Train/evaluate/save Random Forest Classifier
customer_churn_dataset.csv
          ↓
     3.1 Prepare X/y
          ↓
     3.2 Train/Test Split
          ↓
     3.3 Train Random Forest
          ↓
     3.4 Generate Predictions
          ↓
     3.5 Evaluate Model
          ↓
     3.6 Feature Importance
          ↓
     3.7 Save + Reload Model
          ↓
     churn_model.pkl
4. Inspect and Clean Olist Spending Dataset
5. Construct Spending Features + Target
6. Train/evaluate/save Random Forest Regressor
                 TASK 6
                    │
                    ↓
        6.0 Validate Target
                    │
                    ↓
          6.1 Prepare X and y
                    │
                    ↓
          6.2 Train/Test Split
                    │
                    ↓
        6.3 Train Random Forest
              Regressor
                    │
                    ↓
        6.4 Generate Predictions
                    │
                    ↓
          6.5 Evaluate Model
          ├── MAE
          ├── RMSE
          └── R²
                    │
                    ↓
          6.6 Feature Importance
                    │
                    ↓
          6.7 Save Model +Reload & Verify
                    │
                    ↓
                     
7. Build the Customer Risk Value calculation
8. Build the simple Flask application
9. Test the complete pipeline and generate the research-paper results/figures

# Customer Churn Prediction Model
A Random Forest Classifier was trained using customer purchasing behaviour features. The model was evaluated on the held-out test set using accuracy, precision, recall, F1-score and ROC-AUC. A confusion matrix was also generated to examine correct and incorrect classifications.

# Freature Importance
Feature importance analysis was performed using the Random Forest model to identify the relative contribution of customer behavioural variables to churn prediction

# How model 2 Works
X → 9 historical behaviour features
y → NextOrderSpending