# Assignment-6-INFO--I-211
# Breast Cancer Classification using Scikit Learn

## Project Purpose

The purpose of this project is to perform machine learning classification
using the breast cancer dataset available in Scikit Learn. The goal of
this project is to build and evaluate multiple machine learning models
that can classify tumors as malignant or benign.

Machine learning has played an important role in improving early
detection of breast cancer, which has contributed to a significant
reduction in mortality rates.

---

## Dataset

The dataset used in this project is the Breast Cancer dataset from
Scikit Learn. It contains 569 samples and 30 numerical features
extracted from digitized images of breast mass tissue.

Target values:

0 = Malignant  
1 = Benign

---

## Models Implemented

Three machine learning models were implemented:

1. Logistic Regression
2. Random Forest
3. Support Vector Machine (SVM)

Each model was trained using an 80/20 train-test split.

---

## Evaluation Metrics

The models were evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

These metrics help evaluate model performance beyond simple accuracy,
which is important in medical prediction tasks.

---

## Best Performing Model

After evaluating all models, the Random Forest classifier produced
the best overall performance. It achieved the highest ROC-AUC score
and maintained strong precision and recall values. This indicates
that Random Forest was highly effective in correctly identifying
both malignant and benign tumors.

Logistic Regression also performed well but assumes a linear
relationship between features and the target variable. Support
Vector Machine performed competitively but required more tuning.

Overall, Random Forest provided the most balanced performance.

---

## Class Design

### CancerModels
This class initializes the three classification models and returns
them as a dictionary for easy iteration.

### ModelTuner
This class performs hyperparameter tuning using GridSearchCV and
cross-validation to improve model performance.

### ModelEvaluator
This class evaluates trained models using several performance metrics.

---

## Limitations

- Dataset size is relatively small
- Only three models were tested
- Limited hyperparameter tuning

Future improvements could include additional models,
feature selection, and deep learning approaches.


## Model Performance Analysis

Three classification models—Logistic Regression, Random Forest, and Support Vector Machine—were trained and evaluated using the breast cancer dataset. After analyzing multiple evaluation metrics, Random Forest emerged as the best-performing model overall. It achieved the highest accuracy, meaning it correctly classified the majority of tumors in the dataset. In addition, the model produced strong precision and recall scores, indicating that it was effective at both correctly identifying malignant tumors and minimizing false predictions. The F1 score, which balances precision and recall, was also high for Random Forest, demonstrating consistent performance across both classes. Furthermore, the model achieved the strongest ROC-AUC score, showing that it was highly capable of distinguishing between malignant and benign tumors.

One important reason Random Forest performs well on this dataset is that it is an ensemble learning method that combines many decision trees. Each tree captures different patterns in the data, and when combined, they create a more robust and accurate prediction system. This allows the model to detect complex nonlinear relationships between the features and the target variable that simpler models may miss.

Although Logistic Regression and Support Vector Machine also produced strong results, they each have certain limitations. Logistic Regression assumes a linear relationship between the input features and the target variable, which can restrict its ability to model more complex interactions in the data. The Support Vector Machine model performed competitively but required more tuning and computational effort to achieve similar performance levels. Overall, Random Forest provided the best balance between predictive accuracy, robustness, and interpretability, making it the most reliable model for this classification task.