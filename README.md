# disease-prediction-using-machine-learning
A machine learning-based disease prediction system designed to assist healthcare providers by offering accurate and early diagnoses from patient symptoms, leveraging data-driven models for improved clinical decision-making.

🔍 Overview
We trained multiple machine learning models on a symptom-based dataset and combined their outputs to build a robust prediction system. The final model can take symptoms as input and return the most likely disease.

⚙️ Features
Input symptoms, get predicted disease.

Model ensemble: SVM, Naive Bayes, and Random Forest.

Cross-validation using Stratified K-Fold for fair evaluation.

Confusion matrix visualization for performance insight.

Combined prediction using mode of individual model outputs.

🧠 ML Models Used
Support Vector Machine (SVM)

Gaussian Naive Bayes

Random Forest Classifier

🛠️ Libraries & Tools
pandas, numpy, scipy – Data handling and computation

matplotlib, seaborn – Data visualization

scikit-learn – Model building and evaluation

🧪 Model Workflow
Preprocess data and encode symptoms numerically.

Perform Stratified K-Fold Cross-Validation on models.

Train each model on resampled training data.

Evaluate performance using confusion matrices.

Combine model predictions using majority voting (mode).

Use the final model to predict disease based on user symptoms.

🚀 Usage
python
Copy code
predict_disease(['fever', 'headache', 'nausea'])
Returns the predicted disease based on trained ensemble model.









Ask ChatGPT



Tools


