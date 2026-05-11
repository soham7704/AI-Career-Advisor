from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader
import streamlit as st
import os
from dotenv import load_dotenv
import json, re, ast
import tempfile

load_dotenv()
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", api_key=groq_api_key)

template = """
You are an expert AI Career Advisor. Evaluate how well the candidate's resume aligns with the job description.

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

Return ONLY valid JSON with these exact keys:
- match_score (int 0-100)
- missing_skills (list of strings)
- recommendations (list of 2-3 short strings)
- feedback (string, 1-2 sentences)

No markdown, no backticks, no explanation. Raw JSON only.
"""

prompt = PromptTemplate(input_variables=["resume_text", "jd_text"], template=template)
chain = prompt | llm

def extract_resume_text(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    loader = PyMuPDFLoader(tmp_path)
    pages = loader.load()
    return "\n".join([page.page_content for page in pages])

def analyze_resume_vs_jd(resume_path, jd_text):
    resume_text = extract_resume_text(resume_path)
    response = chain.invoke({"resume_text": resume_text, "jd_text": jd_text})
    response_text = response.content  # pipe returns AIMessage, not a dict

    try:
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            return json.loads(json_match.group(0))
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(response_text.strip())
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {"error": "Could not parse response.", "raw_output": response_text}
