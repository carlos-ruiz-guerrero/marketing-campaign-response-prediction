# Bank Marketing Campaign Response Prediction

## Project Overview

This project analyzes the Bank Marketing dataset from the UCI Machine Learning Repository to identify the factors that influence customer subscription to term deposit products.

The project combines exploratory data analysis, customer segmentation, statistical validation, feature engineering, and predictive modeling techniques to transform raw campaign data into actionable business insights.

### Business Objective

The objective of this analysis is to determine which customer characteristics and campaign factors are associated with successful term deposit subscriptions, enabling more targeted and effective marketing strategies.

---

## Dataset

**Source:** UCI Machine Learning Repository \- Bank Marketing Dataset

**Target Variable:**

- subscribed (Yes / No)

The dataset contains customer demographic attributes, financial information, campaign details, and historical marketing interactions.

### Main Variables

**Customer Information**

- Age
- Job
- Marital Status
- Education

**Financial Information**

- Account Balance
- Housing Loan
- Personal Loan
- Credit Default

**Campaign Information**

- Contact Method
- Month of Contact
- Campaign Duration
- Number of Contacts

**Historical Information**

- Previous Contacts
- Days Since Last Contact
- Previous Campaign Outcome

---

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit\-Learn

---

## Project Structure

bank\-marketing\-analysis/

│

├── src/

│   ├── data\_preparation.py

│   ├── eda.py

│   ├── customer\_segmentation.py

│   ├── statistical\_analysis.py

│   ├── feature\_engineering.py

│   └── modeling.py

│

├── data/

│   ├── raw/

│   └── processed/

│

├── reports/

│   ├── figures/

│   └── findings/

│

├── README.md

├── requirements.txt

└── config.py

---

## Current Progress

✅ Data Acquisition and Validation

✅ Exploratory Data Analysis (EDA)

✅ Customer Segmentation

✅ Statistical Significance Testing

✅ Effect Size Analysis (Cramér's V)

✅ Feature Engineering

🔄 Predictive Modeling (In Progress)

⬜ SQLite Database Integration

⬜ Streamlit Dashboard

---

# Executive Summary

## Overall Campaign Performance

The overall subscription rate was approximately 11%, indicating that successful subscriptions represented a relatively small proportion of all campaign contacts.

This highlights the importance of customer targeting and campaign optimization in improving marketing performance.

---

## Key Findings

### 1\. Older Customers Represent the Most Valuable Segment

Customers aged 60 and above achieved a conversion rate of 42.3%, substantially outperforming all other age groups.

### 2\. Higher Account Balances Are Associated with Greater Conversion Rates

Customers in the highest balance segment achieved a conversion rate of 16.2%, compared with 7.2% for customers in the lowest segment.

### 3\. Campaign Fatigue Reduces Marketing Effectiveness

Customers contacted only once converted at a rate of 14.6%, while customers contacted six or more times converted at just 5.8%.

This suggests diminishing returns from repeated campaign contacts.

### 4\. Previous Customer Engagement Improves Outcomes

Customers who had been contacted previously converted at a rate of 23.1%, compared with 9.2% for customers with no prior contact history.

### 5\. High\-Performing Customer Profiles Exist

Segmentation analysis identified several high\-converting profiles concentrated within the 60\+ age group, particularly among self\-employed, entrepreneur, management, and retired customers.

---

# Statistical Validation

Chi\-Square tests confirmed statistically significant relationships between all analyzed categorical variables and subscription behavior.

However, effect size analysis using Cramér's V demonstrated that not all statistically significant variables have meaningful business impact.

| Variable | Cramér's V | Effect Size |
| --- | --- | --- |
| Previous Campaign Outcome (poutcome) | 0.470 | Strong |
| Age Group | 0.173 | Weak |
| Job | 0.136 | Weak |
| Education | 0.073 | Negligible |
| Marital Status | 0.066 | Negligible |
| Contact Method | 0.012 | Negligible |

### Key Statistical Insight

Previous campaign outcome emerged as the strongest predictor of subscription behavior.

Age group and occupation contribute modestly to customer conversion, while education, marital status, and contact channel exhibit limited practical impact despite statistical significance.

This analysis highlights an important analytical principle:

Statistical significance does not necessarily imply business significance.

---

# Customer Segmentation

Customer segmentation was performed to identify high\-value customer groups and uncover patterns in subscription behavior.

Key segmentation dimensions included:

- Age Groups
- Balance Segments
- Campaign Intensity
- Previous Contact History

The analysis revealed that:

- Customers aged 60\+ convert significantly more often.
- Higher\-balance customers show higher conversion rates.
- Previously contacted customers exhibit substantially higher engagement.
- Excessive campaign contacts are associated with lower conversion rates.

---

# Feature Engineering

To improve model performance and business interpretability, several engineered features were created:

- age\_group
- balance\_segment
- campaign\_group
- contacted\_before
- recently\_contacted
- high\_balance\_customer
- campaign\_fatigue
- subscribed\_target

These features transform raw operational variables into meaningful business indicators suitable for predictive modeling.

---

# Preliminary Recommendations

1. Prioritize marketing efforts toward customers aged 60\+.
2. Focus campaigns on higher\-balance customer segments.
3. Reduce excessive contact attempts to avoid campaign fatigue.
4. Prioritize retention and re\-engagement campaigns.
5. Leverage historical campaign outcomes when selecting future targets.

---

# Next Steps

- Build predictive models (Logistic Regression and Random Forest)
- Evaluate model performance using ROC\-AUC, Precision, Recall, and F1 Score
- Analyze feature importance
- Create a SQLite analytical database
- Develop an interactive Streamlit dashboard

---

# Citation

Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5K306](https://doi.org/10.24432/C5K306)


