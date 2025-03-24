# Salary Prediction system

This project focuses on predicting employee salary categories (Low, Medium, or High) using machine learning techniques. By analyzing various factors such as years of experience, education level, job title, and gender, the model provides accurate salary predictions, which can assist HR departments and management in making informed decisions.

🔎 Problem Statement
The goal was to build a machine learning model that classifies employees into one of the three salary categories based on relevant features. Ensuring a high accuracy in classification allows organizations to predict and analyze salary structures effectively.

📥 Data Preprocessing
Data Collection: The dataset contained employee information with features like:

Years of Experience

Education Level

Job Title

Gender

Salary (Target Variable)

Data Cleaning: Checked for missing values and handled them using appropriate imputation techniques.

Feature Engineering: Created a new categorical target variable (Salary_Class) by binning the salary into categories:

Low for salaries ≤ 50,000

Medium for salaries between 50,000 and 100,000

High for salaries ≥ 100,000

Encoding: Converted categorical data using Label Encoding.

Feature Scaling: Applied StandardScaler to normalize the feature values for consistent model training.

🧪 Model Building
Two classification algorithms were implemented to predict the salary category:

Random Forest Classifier:

Used an ensemble of decision trees for better accuracy and reduced overfitting.

Hyperparameter tuning was applied using GridSearchCV to optimize n_estimators, max_depth, and min_samples_split.

Logistic Regression:

Applied as a baseline model for comparison.

Logistic Regression performed well on simpler data, but struggled with non-linear relationships.

📊 Model Evaluation
The models were evaluated using the following metrics:

Accuracy: Overall correctness of the model.

Precision: Ability to identify correct positive predictions.

Recall: Ability to detect actual positives.

F1-Score: Harmonic mean of precision and recall for a balanced evaluation.

Confusion Matrix: Visual representation of true positives, false positives, true negatives, and false negatives.

🔎 Random Forest Classifier achieved an accuracy of 87% on the test data, outperforming Logistic Regression. It also demonstrated robust performance in handling non-linear relationships and noisy data.

📦 Model Saving and Deployment
The final trained Random Forest model was saved using Joblib for future use.

It can be loaded and used for real-time predictions in salary prediction applications.

🔔 Tools and Technologies Used
Programming: Python

Libraries: Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib

Machine Learning Models: Random Forest Classifier, Logistic Regression

Data Processing: Label Encoding, StandardScaler, Train-Test Split

Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

Model Deployment: Joblib for model serialization

📌 Conclusion
The Salary Classification Model successfully classified employee salaries with 87% accuracy. The model’s robust performance and effective feature analysis make it suitable for real-world HR applications, assisting companies in salary analysis, budgeting, and workforce management. Further improvements can be made by incorporating additional employee attributes or using more advanced ensemble models.
