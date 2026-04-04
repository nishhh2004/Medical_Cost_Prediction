# 🏥 Medical Cost Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange?style=flat-square&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?style=flat-square&logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green?style=flat-square&logo=pandas)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)

---

## 📌 Problem Statement

Medical insurance costs vary significantly based on personal and lifestyle factors. Predicting these costs accurately can help insurance companies price policies fairly and help individuals understand what influences their charges. This project builds a **Machine Learning web application** using **Linear Regression** that predicts annual medical insurance charges based on a patient's personal attributes such as age, BMI, smoking status, and region. The model is deployed as an interactive **Streamlit web app**.

## 📁 Folder Structure
```
Medical_Cost_Prediction/
│
├── app.py                         
├── medical_cost_model.pkl       
├── scaler.pkl                     
├── insurance.csv                  
├── Medical_Cost_Prediction.ipynb  
├── LICENSE
├── requirements.txt                     
└── README.md                      
```

## 📊 Dataset

- **Source:** Kaggle — Medical Cost Personal Dataset
- **Records:** 1,338 rows
- **Features:** 6 (age, sex, bmi, children, smoker, region)
- **Target:** charges (annual medical insurance cost in USD)

| Feature  | Type        | Description                      |
|----------|-------------|----------------------------------|
| age      | Numerical   | Age of the primary beneficiary   |
| sex      | Categorical | Gender (male / female)           |
| bmi      | Numerical   | Body Mass Index                  |
| children | Numerical   | Number of dependents             |
| smoker   | Categorical | Smoking status (yes / no)        |
| region   | Categorical | US residential region            |
| charges  | Numerical   | Annual medical insurance charges |

---

## ⚙️ ML Pipeline
Raw Data → EDA → Preprocessing → Model Training → Evaluation → Deployment

---

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/Medical_Cost_Prediction.git
cd Medical_Cost_Prediction
```

**2. Install dependencies**
```bash
pip install streamlit scikit-learn pandas numpy
```

**3. Run the app**
```bash
streamlit run app.py
```

**4. Open in browser**
http://localhost:8501

---

## 💡 Key Insights

- **Smoking** is the most impactful variable — smokers pay ≈3–4x more than non-smokers
- **BMI** has a positive correlation with charges — higher BMI leads to higher costs
- **Age** shows a steady positive relationship with insurance charges
- The model explains **78% of the variation** in insurance charges (R² = 0.78)

---

## 🛠️ Tech Stack

| Tool         | Purpose                    |
|--------------|----------------------------|
| Python       | Core programming language  |
| Pandas       | Data manipulation          |
| Matplotlib   | Data visualization         |
| Seaborn      | Statistical plots          |
| Scikit-Learn | ML model and preprocessing |
| Streamlit    | Web app deployment         |
| Pickle       | Model serialization        |

---

## 👤 Author

**Nishanth MS**
[GitHub](https://github.com/nishhh2004) · [LinkedIn](https://linkedin.com/in/nishanth-ms)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
