# 🩺 Diabetes Prediction System

**A machine learning-based model to predict diabetes risk with high accuracy.**  
This system takes **8 health parameters** as input and predicts whether a person has diabetes or not, along with a probability percentage.

![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange)
![ML Model](https://img.shields.io/badge/SVM%20%26%20RandomForest-Sklearn-green)

---

## **🔍 Project Overview**
- Uses **Support Vector Machine (SVM) and Random Forest** for prediction.
- Automatically selects the **best model based on accuracy**.
- Accepts **8 health parameters** (e.g., Glucose, BMI, Age, etc.).
- Provides a **probability score** with each prediction.

---

## **📁 Dataset**
The model is trained on the **PIMA Indian Diabetes Dataset**, containing:
- **768 samples**, **8 health features**, and an **Outcome label** (Diabetes: `1`, No Diabetes: `0`).

---

## **🛠️ Installation**
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/VinayMattapalli/Diabetes-Prediction-Model.git
cd Diabetes-Prediction-Model
2️⃣ Install Dependencies
sh
Copy
Edit
pip install -r requirements.txt
3️⃣ Run the Application
sh
Copy
Edit
python diab_pred.py
Then open http://127.0.0.1:7860 in your browser.

📊 Model Performance
Metric	SVM	Random Forest
Accuracy	79.5%	85.2%
Precision (Diabetes)	78.9%	83.7%
Recall (Diabetes)	80.2%	87.1%
📌 Random Forest performed better, so it is used by default! 🚀

🖥️ Usage
🎯 Using the Model via Gradio UI
Run:
sh
Copy
Edit
python diab_pred.py
Open http://127.0.0.1:7860 in your browser.
Enter 8 comma-separated values (e.g., 5,166,72,19,175,25.8,0.587,51).
Click "Submit" to get the result:
✅ "Not Diabetic" (90.2% confidence)
⚠️ "Diabetic" (85.3% confidence)
🔗 Technologies Used
Python 3.8+
Scikit-Learn
Pandas & NumPy
Joblib (for model persistence)
Gradio (for web UI)
📝 License
This project is licensed under the MIT License. Feel free to modify and use it.

📬 Contact
👨‍💻 Developed by: Vinay Mattapalli
🔗 GitHub: https://github.com/VinayMattapalli

🙌 Contributions & feedback are welcome! If you find issues or want to improve the model, feel free to create a pull request.

🚀 Star ⭐ the Repository if You Like It!
If this project helps you, consider giving it a ⭐ on GitHub!

Happy Coding! 🎯🔥

