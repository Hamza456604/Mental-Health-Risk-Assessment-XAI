# Mental-Health-Risk-Assessment-XAI
An Interpretable Framework for Mental Health Risk Assessment using Explainable AI


Author: Hamza Ahmed
Institution: Ulster University
Department: Department of Computer Science and Technology

---

## PROJECT OVERVIEW

This project presents an interpretable machine learning framework for **mental health risk assessment** using Explainable Artificial Intelligence (XAI).

The framework uses the **MindManta dataset**, containing more than 8,300 records and over 150 binary symptomatic features. The original multi-class disease information is transformed into a binary classification problem consisting of:

* Healthy (0)
* Illness (1)

The project evaluates multiple machine learning approaches and develops a **two-stage Stacked Ensemble Architecture**. The Level-0 layer consists of heterogeneous machine learning models, while a Logistic Regression model is used as the Level-1 meta-classifier.

The framework also integrates **SHAP (SHapley Additive exPlanations)** to provide global feature importance and instance-level explanations for individual predictions.

The main objective is to combine strong predictive performance with interpretable, human-understandable reasoning suitable for a Decision Support System (DSS).

---

## PROJECT FILES

### Models / Machine Learning Notebook

Contains the machine learning implementation, including:

* Data loading and inspection
* Exploratory Data Analysis (EDA)
* Data cleansing
* Feature encoding
* Target variable binarisation
* Feature standardisation
* Stratified train-test splitting
* Machine learning model training
* Stacked ensemble implementation
* Hyperparameter optimisation
* Model evaluation
* Confusion matrices
* ROC curve analysis
* Cross-validation
* Comparative performance analysis

### XAI Analysis Notebook

Contains the Explainable AI implementation using SHAP, including:

* Global feature importance
* SHAP bar plots
* SHAP summary/beeswarm plots
* Feature contribution analysis
* Instance-level prediction explanations
* Individual diagnostic rationale / “Reason Cards”

### Dataset

**MindManta Dataset**

The dataset contains more than 8,300 records with 150+ binary symptomatic indicators. The features represent different cognitive, physiological, emotional, social, trauma and stress-related symptoms.

### Supporting Material

Contains the complete extended project documentation, including:

* Context and problem statement
* Research aim and objectives
* Research questions
* Literature review
* Methodology
* Model architecture
* Evaluation protocol
* Results
* XAI analysis
* Critical appraisal
* Key findings
* Ethical considerations
* References


## MACHINE LEARNING MODELS

The framework evaluates heterogeneous machine learning algorithms at the base-learning level:

* **Naive Bayes** – probabilistic baseline
* **Decision Tree** – hierarchical rule-based learning
* **Random Forest** – bagging-based ensemble learning
* **XGBoost** – gradient boosting for complex non-linear relationships

The outputs of the Level-0 models are passed to a **Logistic Regression meta-classifier**, which learns from the predictions of the individual models and produces the final classification.

---

## PROPOSED ARCHITECTURE

The proposed framework consists of two main stages:

### Stage 1 – Level-0 Base Learners

The first stage contains heterogeneous machine learning models:

1. Naive Bayes
2. Decision Tree
3. Random Forest
4. XGBoost

### Stage 2 – Level-1 Meta-Classifier

The probability outputs of the base learners are provided to a **Logistic Regression meta-learner**. The meta-classifier combines the different model predictions to generate the final Healthy/Illness classification.

---

## DATA PREPROCESSING

The preprocessing pipeline consists of the following stages:

1. **Data Cleansing**
   Records containing missing values are removed using `dropna()`.

2. **Categorical Encoding**
   Binary symptom indicators are converted into numerical 0/1 representations.

3. **Target Binarisation**
   The original disease variable is converted into:

   * Healthy = 0
   * Illness = 1

4. **Feature Standardisation**
   `StandardScaler` is used to standardise the feature space.

5. **Stratified Data Splitting**
   The dataset is divided using an **80:20 train-test split**, while maintaining the class distribution between the training and testing subsets.

---

## HYPERPARAMETER OPTIMISATION

Hyperparameter optimisation is performed using **GridSearchCV with 5-fold stratified cross-validation**.

The reported configurations include:

| Model               | Key Parameters                            |
| ------------------- | ----------------------------------------- |
| Decision Tree       | `max_depth=15`, `criterion='gini'`        |
| Random Forest       | `n_estimators=200`, `max_features='sqrt'` |
| XGBoost             | `learning_rate=0.1`, `n_estimators=200`   |
| Logistic Regression | `C=1.0`, `solver='lbfgs'`                 |

The optimisation process was used to improve generalisation and control model complexity.

---

## EXPLAINABLE AI

The framework integrates **SHAP (SHapley Additive exPlanations)** to address the black-box nature of machine learning models.

SHAP is used for:

* Global feature importance analysis
* Understanding the contribution of individual symptoms
* Analysing the direction of feature influence
* Generating instance-level explanations
* Producing individual “Reason Cards”

The SHAP analysis identified features such as **difficulty_concentrating, insomnia, and sleep_disturbances** among the most influential predictors in the reported experiment.

The framework moves beyond global feature importance by using SHAP explanations to show why a particular individual received a specific prediction.

---

## EVALUATION

The models are evaluated using several standard classification metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

Additional evaluation includes:

* Confusion matrices
* ROC curves
* 10-fold stratified cross-validation
* Comparative model analysis

Recall is given particular importance because the project focuses on identifying individuals at potential mental health risk.

---

## KEY RESULTS

The reported experimental results are:

| Model                |   Accuracy |  Precision |     Recall |   F1-Score |  ROC-AUC |
| -------------------- | ---------: | ---------: | ---------: | ---------: | -------: |
| Naive Bayes          |     74.77% |     0.9902 |     0.7489 |     0.8528 |     0.80 |
| Decision Tree        |     97.35% |     0.9758 |     0.9975 |     0.9865 |     0.76 |
| Random Forest        |     98.91% |     0.9890 |     1.0000 |     0.9944 |     0.98 |
| XGBoost              |     98.61% |     0.9860 |     1.0000 |     0.9929 |     0.95 |
| **Stacked Ensemble** | **99.32%** | **0.9890** | **1.0000** | **0.9945** | **0.96** |

The **Stacked Ensemble** achieved the highest reported accuracy of **99.32%**, with a Recall of **1.0000** and F1-Score of **0.9945**.

---

## IMPLEMENTATION ENVIRONMENT

The project was implemented using Python and Jupyter Notebook.

The documented software environment includes:

* Python 3.9.12
* scikit-learn 1.0.2
* XGBoost 1.6.1
* SHAP 0.41.0
* Jupyter Notebook 6.4.12
* Pandas
* NumPy
* Matplotlib
* Seaborn

The reported hardware environment was an Apple M2 Max system with 32 GB RAM and a 12-core CPU.

---

## REPRODUCIBILITY

The implementation uses a global `random_state=42` for the models and data splitting to support reproducible experimental results. The preprocessing, model training and interpretability stages are maintained as separate components to reduce the possibility of data leakage.

---

## HOW TO RUN

1. Install Python and Jupyter Notebook.

2. Install the required Python libraries:

```bash
pip install -r requirements.txt
```

3. Place the dataset in the same project directory as the machine learning notebook.

4. Open Jupyter Notebook:

```bash
jupyter notebook
```

5. Open the machine learning notebook.

6. Run the notebook cells sequentially or select **Cell → Run All**.

7. The notebook will perform data preprocessing, model training, hyperparameter optimisation and model evaluation.

8. Open the XAI notebook to reproduce the SHAP analysis.

9. Run the XAI notebook cells sequentially to generate global and instance-level explanations.

---

## PROJECT CONTRIBUTIONS

The project makes the following key contributions:

1. Develops an interpretable machine learning framework for mental health risk assessment.
2. Compares multiple heterogeneous machine learning models.
3. Implements a two-stage stacked ensemble architecture.
4. Uses SHAP to improve model transparency and interpretability.
5. Provides instance-level diagnostic rationales through SHAP-based explanations.
6. Achieves a reported Stacked Ensemble accuracy of **99.32%** and Recall of **1.0000**.

---

## ETHICAL NOTE

The framework is designed as a **Decision Support System (DSS)** and should not be considered a replacement for qualified mental health professionals.

The supporting documentation states that the dataset used was anonymised and that no personally identifiable information was processed. The intended approach is human-in-the-loop, where the final decision remains with the relevant healthcare professional.

---

## PROJECT STRUCTURE

A typical project directory can be organised as:

```text
Mental-Health-XAI/
│
├── Dataset/
│   └── MindManta dataset
│
├── Models/
│   └── Machine Learning Notebook
│
├── XAI/
│   └── XAI Analysis Notebook
│
├── Research Paper/
│   └── IEEE Research Paper
│
├── Supporting Material/
│   └── Supporting Material.pdf
│
├── requirements.txt
└── README.txt
```

---

## AUTHOR

**Hamza Ahmed**
Department of Computer Science and Technology
Ulster University
Supervisor: **Dr. Nalinda Somasiri**
