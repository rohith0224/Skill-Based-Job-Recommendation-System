import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.linear_model import SGDClassifier
import numpy as np
df=pd.read_csv('/Users/rohithsunkara/Desktop/Datamining project/Naukri Jobs Data.csv')
df = df.dropna(subset=['required_skills', 'job_post'])
df["required_skills"] = df["required_skills"].apply(
    lambda x: [
        s.strip().lower().replace("_", " ").replace("-", " ")
        for s in str(x).replace("\\n", "\n").split("\n")
        if s.strip()
    ]
)
import re

def normalize_job_title(title: str) -> str:
    """
    Cleans up job titles:
    - Removes seniority (Senior, Jr, Lead, etc.)
    - Removes numeric levels (I, II, III, 1, 2, 3)
    - Normalizes spacing
    - Keeps main title text (like 'Software Engineer', 'Data Scientist')
    """
    title = str(title).lower().strip()

    # Remove common seniority prefixes/suffixes
    title = re.sub(r'\b(senior|sr\.?|lead|principal|junior|jr\.?|entry[-\s]*level|intern|apprentice)\b', '', title)

    # Remove level indicators (I, II, III, 1, 2, 3, iv, v)
    title = re.sub(r'\b(i{1,3}|iv|v|vi{0,2}|[1-9])\b', '', title)

    # Remove extra hyphens, commas, parentheses
    title = re.sub(r'[-,()/]', ' ', title)

    # Collapse multiple spaces into one
    title = re.sub(r'\s+', ' ', title).strip()

    # Capitalize words for neatness
    title = title.title()

    return title



X = df["required_skills"].apply(lambda x: " , ".join(x) if isinstance(x, list) else x)
y = df['job_post'].apply(normalize_job_title)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
X_test = X_test.apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=8000,
    ngram_range=(1, 3), 
    token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z+\-_. ]+[a-zA-Z]\b'
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
def simplify_job_title(title):
    title = title.lower().strip()
    if any(w in title for w in ["data scientist", "research scientist", "quant"]):
        return "data_scientist"
    if any(w in title for w in ["machine learning", "ml engineer", "ai engineer", "deep learning"]):
        return "ml_engineer"
    if any(w in title for w in ["data engineer", "etl", "pipeline", "big data"]):
        return "data_engineer"
    if "analyst" in title or "business intelligence" in title:
        return "data_analyst"
    if any(w in title for w in ["software engineer", "sde", "backend", "full stack", "full-stack"]):
        return "software_engineer"
    if any(w in title for w in ["frontend", "front-end", "ui developer", "react developer"]):
        return "frontend_engineer"
    if any(w in title for w in ["devops", "site reliability", "sre", "cloud engineer"]):
        return "devops_engineer"
    if any(w in title for w in ["mobile developer", "android", "ios", "flutter", "react native"]):
        return "mobile_engineer"
    if any(w in title for w in ["product manager", "pm", "program manager"]):
        return "product_manager"
    if any(w in title for w in ["project manager", "scrum master"]):
        return "project_manager"
    if any(w in title for w in ["security engineer", "cybersecurity", "infosec"]):
        return "security_engineer"
    if any(w in title for w in ["aws", "azure", "gcp", "cloud architect"]):
        return "cloud_architect"
    if any(w in title for w in ["qa", "test engineer", "automation engineer"]):
        return "qa_engineer"
    if re.search(r'\b(research\s*scientist|researcher|r&d)\b', title):
        return 'research_scientist'
    if re.search(r'\b(marketing|seo|content|digital\s*marketing|social\s*media)\b', title):
        return 'marketing_specialist'
    if re.search(r'\b(human\s*resources|hr|recruiter|talent\s*acquisition)\b', title):
        return 'hr_specialist'
    if re.search(r'\b(accountant|finance|financial|auditor|tax|payroll)\b', title):
        return 'finance_specialist'
    if re.search(r'\b(support|helpdesk|customer\s*service|technical\s*support)\b', title):
        return 'support_engineer'
    if re.search(r'\b(intern|trainee|apprentice)\b', title):
        return 'intern'
    return "other"
y_train = y_train.apply(simplify_job_title)
y_test = y_test.apply(simplify_job_title)
model = SGDClassifier(
    loss="log_loss",
    max_iter=1000,
    tol=1e-3,
    n_jobs=-1,
    learning_rate="optimal",
    alpha=1e-4,
    random_state=42   
)
model.fit(X_train_tfidf, y_train)
with open('recommender.pkl', 'wb') as f:
    pickle.dump((model, vectorizer), f)

print("✅ Model and vectorizer saved successfully as recommender.pkl")