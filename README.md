# customer-360-segmentation

## Project Overview
A full Customer 360 Intelligence System that combines customer segmentation, explainable AI and intent detection to help financial institutions better understand and serve their customers.

## Business Goals
- Know your customer better through segmentation.
- Reduce customer churn by identifying At Risk customers.
- Increase revenue by targeting Champions.
- Understand customer intent for faster service routing
- Support fraud detection through unusual pattern flagging.

# Business Impact
Goal 
Retain At Risk Customers | Save 15-20% revenue.
Target Champions | 30% higher campaign ROI
Intent detection | 40% faster service routing.
Personalization | 25% increase in satisfaction.

## Dataset
- Source: Bank Transaction Dataset
- Size : 2512 transactions, 16 features
- Features: TransactionAmount,  AccountBalance, TransactionType, Channel, CustomerAge, CustomerOccupation, LoginAttempts, TransactionDuration

## Technical Approach
## Model 1 - Customer Segmentation
- RFM Analysis - Recency, Frequency, Monetary
- KMeans Clustering - 4 optimal clusters
- SHAP Explainability - Why each customer is in their segment

## Model 2 - Intent Detection
- Algorithm: Random Forest Classifier
- Accuracy: 100%
- 6 Intent Classes:
- Cash Withdrawal
- Bill Payment
- Online Purchase
- Deposit
- Online Transfer
- Branch Transfer
- 
## Segmentation Results
Champion | 169 | High spend, frequent, recent
Potential Loyalist | 153 | Growing, needs nurturing |
At Risk | 138 | We're good, now inactive |
Loyal Customer | 35 | Consistent, smaller group

## How to Run
## Clone the repo
git clone https://github.com/Santandave961/customer-360-segmentation

## Install dependencies
pip install -r requirements. txt

## Run the streamlit dashboard
streamlit run app.py

# Technologies Used
Python - Core Language.
Pandas and Numpy - Data manipulation
Scikit learn - ML models
SHAP - Explainable AI
KMeans - Customer clustering
Streamlit - Dashboard
Matplotlib and Seaborn - Visualizations
