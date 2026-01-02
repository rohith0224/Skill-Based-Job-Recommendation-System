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

