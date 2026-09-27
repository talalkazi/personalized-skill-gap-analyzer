import ast
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def parse_skills(skill_entry):
  if pd.isna(skill_entry):
    return []
  if isinstance(skill_entry, str) and skill_entry.startswith("["):
    try:
      return [s.strip().lower() for s in ast.literal_eval(skill_entry)]
    except (ValueError, SyntaxError):
      pass
  return [s.strip().lower() for s in str(skill_entry).split(",") if s.strip()]


def main():
  interns_df = pd.read_csv("intern_skills.csv")
  jobs_df = pd.read_csv("all_job_post.csv")
  software_df = pd.read_csv("software_skills.csv")

  jobs_df["parsed_skills"] = jobs_df["job_skill_set"].apply(parse_skills)
  interns_df["parsed_skills"] = interns_df["Skills"].apply(parse_skills)

  jobs_df["skill_text"] = jobs_df["parsed_skills"].apply(lambda x: " ".join(x))
  interns_df["skill_text"] = interns_df["parsed_skills"].apply(
      lambda x: " ".join(x)
  )

  tfidf = TfidfVectorizer(max_features=300, stop_words="english")
  X_jobs = tfidf.fit_transform(jobs_df["skill_text"])

  n_clusters = 5
  kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
  jobs_df["cluster"] = kmeans.fit_predict(X_jobs)

  jobs_df["cat_clean"] = (
      jobs_df["category"].astype(str).str.upper().str.replace("-", " ")
  )

  results = []
  for idx, intern in interns_df.iterrows():
    intern_id = intern["Intern_ID"]
    target_role = intern["Target_Role"]
    industry_cat = (
        str(intern["Industry_Category"]).upper().replace("-", " ")
    )
    intern_skills = set(intern["parsed_skills"])

    intern_vec = tfidf.transform([intern["skill_text"]])

    similarities = cosine_similarity(intern_vec, kmeans.cluster_centers_)[0]
    best_cluster = int(np.argmax(similarities))

    matching_jobs = jobs_df[jobs_df["cat_clean"] == industry_cat]
    all_category_job_skills = [
        s for sublist in matching_jobs["parsed_skills"] for s in sublist
    ]

    skill_freq = Counter(all_category_job_skills)

    missing_top_skills = [
        skill
        for skill, count in skill_freq.most_common()
        if skill not in intern_skills
    ][:5]

    matched_skills = list(
        intern_skills.intersection(set(all_category_job_skills))
    )

    results.append({
        "Intern_ID": intern_id,
        "Target_Role": target_role,
        "Industry_Category": intern["Industry_Category"],
        "Matched_Cluster": best_cluster,
        "Current_Skill_Count": len(intern_skills),
        "Matched_Skills": ", ".join(matched_skills[:5]),
        "Top_Skill_Gaps": ", ".join(missing_top_skills),
        "Recommended_Training": ", ".join(
            [f"Course: {s.title()}" for s in missing_top_skills[:3]]
        ),
    })

  results_df = pd.DataFrame(results)
  output_filename = "skill_gap_analysis_output.csv"
  results_df.to_csv(output_filename, index=False)

  print(f"Analysis completed! Output saved to '{output_filename}'")
  print(results_df.head(5).to_string())


if __name__ == "__main__":
  main()