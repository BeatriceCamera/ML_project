# ML_project
Machine Learning project using fairness-aware modeling and SHAP interpretability on a simulated HR dataset to detect potential bias in managerial role prediction.


# Fairness-Aware Machine Learning on DEI Dataset
- **Language:** Python  
- **Libraries:**
  - **Data Analysis & Visualization:** pandas, numpy, seaborn, matplotlib, missingno 
  - **Machine Learning:** scikit-learn (model selection, preprocessing,     classification, evaluation), xgboost  
  - **Feature Engineering & Sampling:** imbalanced-learn (SMOTE, RandomUnderSampler), mlxtend (Sequential Feature Selector)  
  - **Explainability:** SHAP  
  - **Statistical Modeling:** scipy.stats (for randomized search distributions)

## Overview
This project analyzes fairness and bias in machine learning using a simulated Diversity, Equity, and Inclusion (DEI) dataset containing sensitive attributes (gender, ethnicity, disability, sexual orientation).  
The goal is to predict managerial roles and evaluate whether sensitive features influence the model’s decisions, a proxy for potential systemic bias.


## Methodology
The dataset was preprocessed with imputation, encoding, and standardization, addressing class imbalance through **SMOTE**.  
A pipeline combining sampling, dimensionality reduction, and classification was optimized via **RandomizedSearchCV** and **Cross-Validation**.

**Best configuration:**
- **Sampler:** SMOTE (strategy=1.0)  
- **Dimensionality Reduction:** SequentialFeatureSelector (DecisionTree)  
- **Classifier:** XGBoost (`learning_rate=0.146`, `max_depth=9`, `n_estimators=100`)  

## Evaluation
Model performance was assessed with **F1-score**, **Precision-Recall**, **Learning Curve**, and **Confusion Matrix**.

- Moderate predictive strength.  
- **SHAP analysis** showed that *division* and *age* were most influential, while sensitive attributes contributed little to prediction.  
- Indicates limited bias but potential structural imbalance in the dataset.


## Final Consideration
The model did not identify strong or well-defined patterns within the data, suggesting that there are no evident or systematic discriminations in the company structure.  Its moderate performance is therefore meaningful: it indicates that sensitive attributes are not significantly correlated with being a manager.

### View the Full Notebook
The complete notebook can be viewed interactively on **nbviewer**:

[Open in nbviewer](https://nbviewer.org/github/BeatriceCamera/ML_project/blob/main/DEI_Fairness_Analysis.ipynb)



**Author:** Beatrice Camera  
B.Sc. Artificial Intelligence @ University of Pavia, University of Milan, University of Milano-Bicocca  
