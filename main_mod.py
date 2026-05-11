# main_mod.py

# Importing the necessary libraries
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyMuPDFLoader
import os
from dotenv import load_dotenv
import json, re, ast
import tempfile
import streamlit as st

# Load environment variables
load_dotenv()
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

# Initialize Groq LLM
llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", api_key=groq_api_key)

# Prompt template for resume and job description analysis
template = """
You are an expert AI Career Advisor. Your task is to evaluate how well a candidate's resume aligns with a specific job description (JD).

Start by carefully reading and extracting key information:

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

Now perform the following in order:
1. **Extract** the candidate's name, technical skills, experience (roles + duration), and education from the resume.
2. **Extract** the most relevant skills, qualifications, and responsibilities from the JD. Focus on what the company is **specifically looking for**.
3. **Do a skill gap analysis** by comparing resume and JD. List **only the missing or weak skills** relative to the JD (not skills unrelated to the job).
4. **Suggest 2-3 concise, practical resume improvements** to make it more aligned with this JD (e.g., adding projects, skills, keywords).
5. **Give personalized career advice** (short, actionable, and tailored to the candidate’s profile and the job).

🎯 Only consider skills that are **directly relevant** to the JD role (e.g., if JD is for AI/ML engineer, ignore missing frontend skills).

Return ONLY valid JSON with these keys:
- match_score (int from 0 to 100)
- missing_skills (list of relevant but absent skills)
- recommendations (2–3 short, practical resume improvement tips)
- feedback (1–2 sentence personalized career advice)

Do NOT include markdown, triple backticks, or explanations — just raw JSON output.
"""

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["resume_text", "jd_text"],
    template=template
)

# Create LLM chain with the prompt
chain = prompt | llm
response = chain.invoke({"resume_text": resume_text, "jd_text": jd_text})
response_text = response.content  # AIMessage object → .content

# Extract text content from PDF resume
def extract_resume_text(uploaded_file): # Renamed from pdf_path → uploaded_file
    # Change: Save uploaded file to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Pass temp file path to the loader
    loader = PyMuPDFLoader(tmp_path)
    pages = loader.load()
    return "\n".join([page.page_content for page in pages])

# Analyze resume against job description using the LLM chain
def analyze_resume_vs_jd(resume_path, jd_text):
    resume_text = extract_resume_text(resume_path)
    response = chain.invoke({"resume_text": resume_text, "jd_text": jd_text})

    # Ensure response is a string
    if isinstance(response, dict) and "text" in response:
        response_text = response["text"]
    else:
        response_text = str(response)

    # Try extracting JSON from LLM response
    try:
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            cleaned_json = json_match.group(0)
            return json.loads(cleaned_json)
    except json.JSONDecodeError:
        pass

    # Fallback to Python dict parsing if JSON fails
    try:
        parsed = ast.literal_eval(response_text.strip())
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Return raw output on failure
    return {
        "error": "Could not parse response as valid JSON or Python dict.",
        "raw_output": response_text
    }
