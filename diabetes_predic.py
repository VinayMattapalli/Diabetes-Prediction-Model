# Importing Dependencies
import numpy as np
import pandas as pd
import joblib
import gradio as gr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
DATA_PATH = r"C:/Users/gkeer/OneDrive/Desktop/ML Projects/Diabetes Prediction/diabetes.csv"
MODEL_PATH = r"C:/Users/gkeer/OneDrive/Desktop/ML Projects/Diabetes Prediction/diabetes_model.joblib"

# Load dataset
diabetes_dset = pd.read_csv(DATA_PATH)

# Splitting Features & Labels
X = diabetes_dset.drop(columns="Outcome", axis=1)
Y = diabetes_dset["Outcome"]

# Splitting Data into Training & Testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Standardizing the Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit & Transform Training Data
X_test = scaler.transform(X_test)  # Transform Test Data

# Model Selection & Training
svm_model = svm.SVC(kernel='linear', C=1, probability=True)  # Enable probability estimates
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=2)

svm_model.fit(X_train, Y_train)
rf_model.fit(X_train, Y_train)

# Evaluate Models (This was missing before!)
svm_accuracy = accuracy_score(Y_test, svm_model.predict(X_test))
rf_accuracy = accuracy_score(Y_test, rf_model.predict(X_test))

print(f"🔹 SVM Model Accuracy: {svm_accuracy:.4f}")
print(f"🔹 Random Forest Accuracy: {rf_accuracy:.4f}")

# Save Best Model (NOW it's correctly using the defined accuracy scores!)
best_model = rf_model if rf_accuracy > svm_accuracy else svm_model
joblib.dump(best_model, MODEL_PATH)

# Prediction Function
def predict_diabetes(input_data):
    try:
        # Convert input data into numpy array
        input_data_as_array = np.array([float(value.strip()) for value in input_data.split(',')]).reshape(1, -1)

        # Standardize Input Data
        std_data = scaler.transform(input_data_as_array)

        # Load Trained Model
        trained_model = joblib.load(MODEL_PATH)

        # Make Prediction
        prediction = trained_model.predict(std_data)[0]
        probability = trained_model.predict_proba(std_data)[0][1] * 100  # Get probability percentage

        # Return Result
        if prediction == 0:
            return f"✅ The person is **NOT Diabetic** ({probability:.2f}% confidence)."
        else:
            return f"⚠️ The person **HAS Diabetes** ({probability:.2f}% confidence)."

    except Exception as e:
        return f"⚠️ Error: {str(e)}. Please enter valid numerical values."

# Gradio Web Interface
iface = gr.Interface(
    fn=predict_diabetes,
    inputs=gr.Textbox(lines=2, placeholder="Enter 8 values separated by commas..."),
    outputs="text",
    title="Diabetes Prediction System",
    description="Enter 8 health parameters to predict Diabetes risk with high accuracy.",
)

iface.launch(share=True)
