# Simple Random Forest Model — Working Logic

## 1. What This Project Does

This project predicts whether a student will **Pass (`1`)** or **Fail (`0`)** using:

- `study_hours` — number of hours studied
- `attendance` — attendance percentage

The model uses `RandomForestClassifier` from `scikit-learn`.

## 2. Dataset

The notebook creates a small dataset with 20 rows:

| study_hours | attendance | passed |
|-------------|------------|--------|
| Input feature | Input feature | Target/output |

- **Features (`X`)**: `study_hours` and `attendance`
- **Target (`y`)**: `passed`

```python
X = df[["study_hours", "attendance"]]
y = df["passed"]
```

## 3. Train–Test Split

The data is divided into:

- **80% training data** — used to teach the model
- **20% testing data** — used to check how well the model performs

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

`stratify=y` keeps a similar pass/fail ratio in both sets.

## 4. How a Random Forest Works

A Random Forest combines many decision trees:

1. It creates multiple random subsets of the training data.
2. It builds one decision tree for each subset.
3. Each tree makes its own prediction.
4. For classification, the final prediction is decided by a **majority vote**.

For example:

- 60 trees predict **Pass**
- 40 trees predict **Fail**

The forest returns **Pass**, because Pass received the most votes.

## 5. Model Configuration

```python
model = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=3,
    max_features="sqrt",
    random_state=42
)
```

- `n_estimators=100`: builds 100 decision trees.
- `criterion="gini"`: uses the Gini measure to decide the best splits.
- `max_depth=3`: limits each tree to three levels.
- `max_features="sqrt"`: considers a random square-root subset of features at each split.
- `random_state=42`: makes the random results reproducible.

## 6. Training the Model

```python
model.fit(X_train, y_train)
```

During training, the forest learns rules such as:

- Students with low study hours are more likely to fail.
- Students with high attendance are more likely to pass.
- The effect of one feature may depend on the value of the other feature.

## 7. Making Predictions

```python
y_pred = model.predict(X_test)
```

The model compares `X_test` with the rules learned during training and creates predicted pass/fail values.

## 8. Accuracy

```python
accuracy = accuracy_score(y_test, y_pred)
```

Accuracy measures the percentage of correct predictions:

$$
\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}
$$

## 9. Confusion Matrix

The confusion matrix compares actual values with predicted values:

|  | Predicted Fail | Predicted Pass |
|--|----------------|----------------|
| Actual Fail | True Negative | False Positive |
| Actual Pass | False Negative | True Positive |

It helps identify whether the model is confusing one class with another.

## 10. Classification Report

The report displays:

- **Precision** — how many predicted passes or failures were correct
- **Recall** — how many actual passes or failures were found
- **F1-score** — balance of precision and recall
- **Support** — number of samples for each class

## 11. Decision Tree Visualization

```python
plot_tree(model.estimators_[0])
```

This shows the first of the 100 decision trees.

A node uses a feature and threshold to split the data. For example:

```text
attendance <= 70?
├── Yes → Likely Fail
└── No  → Likely Pass
```

One tree is not the whole model. The Random Forest combines all 100 trees.

## 12. Feature Importance

```python
model.feature_importances_
```

Feature importance shows how much each input contributes to the model's decisions:

- Higher importance — more useful for prediction
- Lower importance — less useful for prediction
- The notebook plots both features using a bar chart

## 13. Predicting a New Student

```python
new_student = pd.DataFrame({
    "study_hours": [6],
    "attendance": [85]
})

prediction = model.predict(new_student)
```

The new student is passed through all 100 trees:

```text
Study hours = 6
Attendance = 85%
        ↓
100 decision trees
        ↓
Pass / Fail majority vote
        ↓
Final prediction
```

## 14. Complete Prediction Flow

```text
Raw dataset
    ↓
Separate features and target
    ↓
Split into training and testing data
    ↓
Create Random Forest with 100 trees
    ↓
Train trees using training data
    ↓
Predict test data
    ↓
Compare predictions with actual values
    ↓
Measure accuracy and classification metrics
    ↓
Visualize trees and feature importance
```

## 15. Why Use Random Forest?

- Combines many trees instead of relying on one tree
- Usually reduces overfitting
- Handles feature interactions
- Works for both classification and regression
- Provides feature-importance scores
- More reliable than a single decision tree in many problems
