import ast
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

st.set_page_config(
    page_title="Personal Skill Gap Analyzer", layout="wide"
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"], .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }
    
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: #0f172a !important;
    }

    label {
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-radius: 6px;
        border: 1px solid #cbd5e1 !important;
    }
    
    div[data-baseweb="select"]:focus-within > div,
    div[data-baseweb="select"] > div:hover {
        border-color: #4A6378 !important;
        box-shadow: 0 0 0 1px #4A6378 !important;
    }

    div[data-baseweb="select"] * {
        color: #0f172a !important;
    }

    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
    }

    .stTextArea textarea:focus {
        border-color: #4A6378 !important;
        box-shadow: 0 0 0 1px #4A6378 !important;
    }

    .stButton > button, .stDownloadButton > button {
        background-color: #1F2A33 !important;
        color: #ffffff !important;
        border: 1px solid #1F2A33 !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        outline: none !important;
    }

    .stButton > button *, .stDownloadButton > button * {
        color: #ffffff !important;
    }
    
    .stButton > button:hover, .stDownloadButton > button:hover,
    .stButton > button:focus, .stDownloadButton > button:focus,
    .stButton > button:active, .stDownloadButton > button:active {
        background-color: #4A6378 !important;
        color: #ffffff !important;
        border-color: #4A6378 !important;
        box-shadow: 0 0 0 2px #4A6378 !important;
    }

    .stAlert {
        background-color: #fef2f2 !important;
        color: #991b1b !important;
        border: 1px solid #fecaca !important;
        border-radius: 6px !important;
    }

    .info-card {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-left: 5px solid #4194cb !important;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    .info-card p, .info-card strong {
        color: #0f172a !important;
    }

    .matched-skills {
        color: #2563eb !important;
        font-weight: 600;
    }

    .missing-skills {
        color: #dc2626 !important;
        font-weight: 600;
    }

    .recommendations {
        color: #16a34a !important;
        font-weight: 600;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("Personalized Skill Gap Analyzer")


def parse_skills(skill_entry):
  if pd.isna(skill_entry):
    return []
  if isinstance(skill_entry, str) and skill_entry.startswith("["):
    try:
      return [s.strip().lower() for s in ast.literal_eval(skill_entry)]
    except (ValueError, SyntaxError):
      pass
  return [s.strip().lower() for s in str(skill_entry).split(",") if s.strip()]


@st.cache_data
def load_and_prep_data():
  jobs_df = pd.read_csv("all_job_post.csv")
  jobs_df["parsed_skills"] = jobs_df["job_skill_set"].apply(parse_skills)
  jobs_df["skill_text"] = jobs_df["parsed_skills"].apply(lambda x: " ".join(x))

  jobs_df["cat_clean"] = (
      jobs_df["category"].astype(str).str.upper().str.replace("-", " ").str.strip()
  )

  all_unique_skills = set(
      s for sublist in jobs_df["parsed_skills"] for s in sublist
  )

  tfidf = TfidfVectorizer(max_features=300, stop_words="english")
  X_jobs = tfidf.fit_transform(jobs_df["skill_text"])

  kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
  jobs_df["cluster"] = kmeans.fit_predict(X_jobs)

  return jobs_df, tfidf, kmeans, all_unique_skills


jobs_df, tfidf, kmeans, master_skills_set = load_and_prep_data()

st.subheader("Enter Your Profile Details")

available_departments = sorted(jobs_df["cat_clean"].unique())

col_input1, col_input2 = st.columns([1, 2])

with col_input1:
  department = st.selectbox(
      "Select Your Department / Field:", available_departments
  )

with col_input2:
  user_skills_input = st.text_area(
      "Enter Your Skills (comma separated):",
      placeholder="e.g. communication, lead generation, excel, python, negotiation",
  )

if st.button("Analyze Skill Gap"):
  if not user_skills_input.strip():
    st.warning("Please enter your skills before running the analysis.")
  else:
    raw_skills = [
        s.replace("and ", "").strip().lower()
        for s in user_skills_input.split(",")
        if s.strip()
    ]

    valid_user_skills = [
        s
        for s in raw_skills
        if s in master_skills_set
        or any(token in master_skills_set for token in s.split())
    ]

    if not valid_user_skills:
      st.error(
          "No recognized skills were found in your input. Please enter valid"
          " professional skills."
      )
    else:
      user_skills_list = valid_user_skills
      user_skills_set = set(user_skills_list)
      user_skill_text = " ".join(user_skills_list)

      user_vec = tfidf.transform([user_skill_text])
      similarities = cosine_similarity(user_vec, kmeans.cluster_centers_)[0]
      matched_cluster = int(np.argmax(similarities))

      matching_jobs = jobs_df[jobs_df["cat_clean"] == department]

      if matching_jobs.empty:
        matching_jobs = jobs_df

      all_dept_skills = [
          s for sublist in matching_jobs["parsed_skills"] for s in sublist
      ]
      skill_freq = Counter(all_dept_skills)

      matched_skills = [
          s for s in user_skills_list if s in set(all_dept_skills)
      ]
      missing_top_skills = [
          skill
          for skill, count in skill_freq.most_common()
          if skill not in user_skills_set
      ][:5]

      recommended_courses = [
          f"Course: {s.title()}" for s in missing_top_skills[:3]
      ]

      st.write("---")
      st.subheader("Analysis Results")

      col1, col2 = st.columns(2)

      with col1:
        st.markdown(
            f"""
          <div class="info-card">
              <p><strong>Selected Department:</strong> {department}</p>
              <p><strong>Matched Job Cluster:</strong> Cluster {matched_cluster}</p>
              <p><strong>Total Skills Entered:</strong> {len(user_skills_list)}</p>
              <p><strong>Matched Skills:</strong> <span class="matched-skills">{', '.join(matched_skills) if matched_skills else 'None'}</span></p>
          </div>
          """,
            unsafe_allow_html=True,
        )

      with col2:
        st.markdown(
            f"""
          <div class="info-card">
              <p><strong>Top Missing Skills:</strong> <span class="missing-skills">{', '.join(missing_top_skills) if missing_top_skills else 'None'}</span></p>
              <p><strong>Recommended Action Items:</strong> <span class="recommendations">{', '.join(recommended_courses) if recommended_courses else 'None'}</span></p>
          </div>
          """,
            unsafe_allow_html=True,
        )

      st.subheader("Detailed Gap Report")
      report_df = pd.DataFrame([{
          "Department": department,
          "Matched Cluster": matched_cluster,
          "Entered Skills": ", ".join(user_skills_list),
          "Matched Skills": ", ".join(matched_skills),
          "Top Missing Skills": ", ".join(missing_top_skills),
          "Recommendations": ", ".join(recommended_courses),
      }])

      st.dataframe(report_df, use_container_width=True)

      csv_data = report_df.to_csv(index=False).encode("utf-8")
      st.download_button(
          label="Download Gap Report CSV",
          data=csv_data,
          file_name="skill_gap_analysis.csv",
          mime="text/csv",
      )
