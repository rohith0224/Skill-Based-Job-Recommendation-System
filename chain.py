import streamlit as st
from langchain_openai import ChatOpenAI
from python import OPENAI_API_KEY
import pdfplumber
import re
import numpy as np
import pickle

llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini",temperature=0)

st.set_page_config(page_title="🧠 Skill-Based Job Recommendation System", layout="centered")
st.title("🧠 Skill-Based Job Recommendation System")

with open("recommender.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

for key in ["resume_text", "skills_text", "results", "missing_skills_all"]:
    if key not in st.session_state:
        st.session_state[key] = None

st.markdown("### 📄 Paste your resume text **or** upload a PDF file below:")

resume_text_input = st.text_area(
    "📝 Paste your resume text here", 
    height=250, 
    placeholder="Paste your resume content..."
)

st.markdown("---")
st.markdown("### 📎 Or upload your resume (PDF)")

uploaded_file = st.file_uploader("Upload your resume", type=["pdf"])
extracted_text = ""

if uploaded_file is not None:
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
    st.success("✅ Text extracted successfully from PDF!")
    if not resume_text_input.strip():
        resume_text_input = extracted_text

if st.button("🚀 Extract & Recommend Jobs"):
    if resume_text_input.strip():
        with st.spinner("Analyzing your resume and extracting skills..."):
            prompt = (
                "Extract all the technical, analytical, and soft skills from the following text. "
                "Return them as a clean Python list with no explanations.\n\n"
                f"{resume_text_input}"
            )
            response = llm.invoke(prompt)
            st.session_state.skills_text = response.content.strip().strip("[]")
            st.session_state.resume_text = resume_text_input

        st.subheader("🎯 Extracted Skills")
        st.write(st.session_state.skills_text)

        def suggest_jobs(skills_text, top_n=5):
            skills_text = skills_text.lower().replace(",", " ")
            X_input = vectorizer.transform([skills_text])
            probs = model.predict_proba(X_input)[0]
            sorted_idx = np.argsort(probs)[::-1][:top_n]
            return [(model.classes_[i], round(probs[i] * 100, 2)) for i in sorted_idx]

        def get_top_skills_per_job(model, vectorizer, top_n=20):
            feature_names = np.array(vectorizer.get_feature_names_out())
            job_skill_map = {}
            for i, job in enumerate(model.classes_):
                coefs = model.coef_[i]
                top_indices = np.argsort(coefs)[::-1][:top_n]
                job_skill_map[job] = list(feature_names[top_indices])
            return job_skill_map

        job_top_skills = get_top_skills_per_job(model, vectorizer, top_n=20)

        def clean_skill_text(text):
            text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
            text = text.lower().replace("_", " ").replace("-", " ").replace("\n", " ")
            return " ".join(text.split())

        missing_skills_all = {}
        skills_text = st.session_state.skills_text
        skills_text_clean = clean_skill_text(skills_text)
        user_skills = set([s.strip() for s in skills_text_clean.replace(",", " , ").split(" , ") if s.strip()])
        results = suggest_jobs(skills_text, top_n=5)

        st.session_state.results = results
        st.session_state.missing_skills_all = {}

        st.markdown("## 💼 Top Job Matches:")
        for job, score in results:
            st.markdown(f"### 🔹 {job} ({score}%)")

            ideal_skills = set([" ".join(skill.split()) for skill in job_top_skills.get(job, [])])
            learned, missing = set(), set()

            for skill in ideal_skills:
                if any(skill in u or u in skill for u in user_skills):
                    learned.add(skill)
                else:
                    missing.add(skill)

            st.session_state.missing_skills_all[job] = list(missing)

            if ideal_skills:
                match_pct = round((len(learned) / len(ideal_skills)) * 100, 1)
                st.write(f"**✅ Skill Match:** {match_pct}% ({len(learned)}/{len(ideal_skills)} skills)")

                if learned:
                    st.write(f"**🧠 You already know:** {', '.join(sorted(learned))}")
                if missing:
                    st.write(f"**🚀 Learn next:** {', '.join(sorted(missing))}")
                if not missing:
                    st.write("🎯 You already have all key skills for this role!")
            else:
                st.write("_No top skill data found for this role._")

            st.markdown("---")


if st.session_state.results:
    st.markdown("## 💼 Your Top Job Matches (Persistent View):")
    for job, score in st.session_state.results:
        st.markdown(f"**🔹 {job} ({score}%)**")


if st.session_state.missing_skills_all:
    st.subheader("💬 Want a brief about missing skills?")
    st.write("Select which job role you want me to explain:")

    job_options = list(st.session_state.missing_skills_all.keys())
    selected_job = st.selectbox("🔍 Choose a job role", options=[""] + job_options)

    if st.button("📖 Brief Me on Missing Skills"):
        if selected_job and selected_job in st.session_state.missing_skills_all:
            missing_skills = st.session_state.missing_skills_all[selected_job]
            if missing_skills:
                st.markdown(f"### 🧠 Explanation for missing skills in **{selected_job}**:")
                prompt = (
                    f"Explain in one line each why these skills are important for a {selected_job.replace('_', ' ')} role: "
                    + ", ".join(missing_skills)
                )
                with st.spinner(f"Generating explanation for {selected_job}..."):
                    explanation = llm.invoke(prompt)
                st.write(explanation.content)
            else:
                st.success(f"✅ You have all the key skills for {selected_job}!")
        else:
            st.warning("⚠️ Please select a valid job role.")
else:
    st.info("👆 Paste or upload your resume, then click **Extract & Recommend Jobs**.")
