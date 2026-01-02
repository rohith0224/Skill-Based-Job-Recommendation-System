# 🧠 Skill-Based Job Recommendation System

This project implements an **AI-driven job recommendation system** that analyzes resumes, predicts suitable job roles, and identifies missing skills required for each role. The system combines **text-based machine learning** with **language-model-powered skill extraction and explanation**, and is deployed as an interactive **Streamlit web application**.

---

## 📌 Project Overview

The application accepts resumes in PDF or plain text format, extracts relevant skills, and recommends job roles based on learned relationships between skills and job postings.  
In addition to role prediction, the system performs **skill gap analysis** and provides **human-readable explanations** for missing skills to guide users on what to learn next.

---

## 🧩 System Components

- **Machine Learning**
  - TF-IDF feature representation for skills
  - Linear multi-class classifier trained using stochastic gradient descent
  - Ranking-based output (Top-K job role recommendations)

- **Language Model Integration**
  - Extracts structured skills from unstructured resume text
  - Generates concise explanations for missing skills

- **Application Layer**
  - Built using Streamlit
  - Supports resume upload (PDF) or manual text input
  - Displays job match scores, skill match, and missing skills

---


# 🚀 How to Run the Project

This guide walks you through setting up the environment, training the model, and running the Streamlit application.

---

## 1️⃣ Create and Activate a Virtual Environment (Recommended)

### macOS / Linux
```bash
python -m venv venv
source venv/bin/activate
```

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

---

## 2️⃣ Install Required Dependencies

Upgrade `pip` and install all required packages:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3️⃣ Configure API Key

Before running the application, you must add your API key.

1. Open the file **`python.py`**
2. Create a variable and paste your API key:
```python
API_KEY = "your_api_key_here"
```

⚠️ **Important:** Do not share or commit your API key publicly.

---

## 4️⃣ Train the Model

Navigate to the model directory:
```bash
cd model
```

Run the training script:
```bash
python recommender.py
```

After successful execution, you should see:
```text
Model and vectorizer saved successfully as recommender.pkl
```

---

## 5️⃣ Run the Streamlit Application

```bash
streamlit run chain.py
```

The Streamlit application will automatically open in your browser.

---

## ✅ Notes
- Ensure Python **3.8+** is installed.
- Always activate the virtual environment before running the project.
- Keep your API keys secure.

---

