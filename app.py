# app.py (v4.2 - Dynamic Salary Display Fix)

import streamlit as st
import pandas as pd
import numpy as np
import json
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random # Needed for salary multiplier randomness

# --- Page Config ---
st.set_page_config(
    page_title="Vocational & Tech Training Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants for Salary Calculation (Copied from generator) ---
tier1_cities = ["Bangalore", "Hyderabad", "Pune", "Gurgaon", "Noida", "Mumbai", "Chennai", "Delhi"]
tier2_cities = ["Ahmedabad", "Kolkata", "Jaipur", "Mohali", "Coimbatore", "Kochi", "Thiruvananthapuram", "Indore", "Vadodara", "Visakhapatnam", "Mysore", "Nagpur", "Chandigarh"]
high_cost_states = ["Maharashtra", "Delhi NCR", "Karnataka", "Telangana", "Tamil Nadu", "Haryana"]
mid_cost_states = ["Gujarat", "Punjab", "Kerala", "West Bengal", "Andhra Pradesh", "Goa"]
low_cost_states = ["Bihar", "Uttar Pradesh", "Madhya Pradesh", "Rajasthan", "Odisha", "Jharkhand", "Chhattisgarh", "Assam", "Himachal Pradesh", "Uttarakhand"]

salary_bands_v4 = {
    "voc_low": {"min": 9000, "max": 16000}, "voc_medium": {"min": 14000, "max": 25000},
    "voc_high": {"min": 20000, "max": 35000},
    "tech_entry": {"min": 25000, "max": 50000}, "tech_mid": {"min": 45000, "max": 90000},
    "tech_specialized": {"min": 60000, "max": 120000},
    "tech_senior": {"min": 100000, "max": 180000}, "tech_lead": {"min": 150000, "max": 250000}
}

complexity_mapping = { # Map codes to readable labels
    'voc_low': 'Vocational - Basic', 'voc_medium': 'Vocational - Intermediate',
    'voc_high': 'Vocational - Advanced', 'tech_entry': 'Tech - Entry Level',
    'tech_mid': 'Tech - Mid Level', 'tech_specialized': 'Tech - Specialized',
    'tech_senior': 'Tech - Senior Level', 'tech_lead': 'Tech - Leadership'
}

# --- Helper Functions for Salary Calculation (Copied from generator) ---
def get_city_tier(city):
    """Determines approximate city tier based on predefined lists."""
    if city in tier1_cities: return 1
    elif city in tier2_cities: return 2
    else: return 3

def get_location_multiplier_v4_1(city, state, job_complexity):
    """Returns a more distinct salary multiplier based on location and job type."""
    is_tech = job_complexity.startswith("tech_")
    base_multiplier = 1.0
    city_tier = get_city_tier(city)

    if city_tier == 1:
        if is_tech: base_multiplier = random.uniform(1.40, 1.85)
        else: base_multiplier = random.uniform(1.10, 1.25)
    elif city_tier == 2:
        if is_tech: base_multiplier = random.uniform(1.05, 1.35)
        else: base_multiplier = random.uniform(1.0, 1.10)
    else: # Tier 3
        if is_tech: base_multiplier = random.uniform(0.80, 1.10)
        else: base_multiplier = random.uniform(0.90, 1.0)

    state_adjustment = 1.0
    if state in high_cost_states: state_adjustment = random.uniform(1.03, 1.10) if city_tier != 1 else random.uniform(1.0, 1.05)
    elif state in mid_cost_states: state_adjustment = random.uniform(0.98, 1.04)
    elif state in low_cost_states: state_adjustment = random.uniform(0.90, 0.98)

    final_multiplier = base_multiplier * state_adjustment
    final_multiplier = max(0.75, min(final_multiplier, 2.0)) # Clamp
    return final_multiplier

# --- Caching Functions (No changes needed here) ---
@st.cache_data
def load_data(csv_path):
    # ... (Keep existing load_data function from previous app.py version) ...
    # It should load the correct CSV ("synthetic_vocational_tech_data_20k_v4_1.csv")
    # And parse skill lists
    print(f"Attempting to load data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path) # Load without specific na_values initially
        def parse_skill_list(skill_str):
            if not isinstance(skill_str, str): return []
            try: parsed = json.loads(skill_str); return parsed if isinstance(parsed, list) else []
            except:
                 try: parsed = ast.literal_eval(skill_str); return parsed if isinstance(parsed, list) else []
                 except: return []
        df['RequiredSkills_list'] = df['RequiredSkills'].apply(parse_skill_list)
        df['SkillsTaught_list'] = df['SkillsTaught'].apply(parse_skill_list)
        st.success(f"Data loaded and parsed for {len(df)} rows.")
        print(f"Data loaded. Columns: {df.columns.tolist()}")
        if 'JobComplexity' not in df.columns or 'EstimatedMinSalary' not in df.columns:
             st.warning("Warning: Expected columns 'JobComplexity'/'EstimatedMinSalary' not found.")
        return df
    except FileNotFoundError:
        st.error(f"Fatal Error: The file '{csv_path}' was not found.")
        return None
    except Exception as e:
        st.error(f"Fatal Error during data loading/parsing: {e}")
        return None


@st.cache_resource
def create_skill_vectors(_df):
    # ... (Keep existing create_skill_vectors function from previous app.py version) ...
    # Make sure it does NOT have .astype(str)
    if _df is None or 'SkillsTaught_list' not in _df or 'RequiredSkills_list' not in _df:
         st.warning("DataFrame unavailable, cannot create skill vectors.")
         return None, None, None
    print("Creating TF-IDF skill vectors...")
    try:
        def identity_tokenizer(tokens_list): return tokens_list if isinstance(tokens_list, list) else []
        tfidf_vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
        tfidf_matrix_programs = tfidf_vectorizer.fit_transform(_df['SkillsTaught_list'])
        tfidf_matrix_jobs = tfidf_vectorizer.transform(_df['RequiredSkills_list'])
        if not tfidf_vectorizer.vocabulary_:
             st.error("TF-IDF Error: Vocabulary empty. Check skill list parsing/content.")
             return None, None, None
        st.success("TF-IDF Vectorizer fitted and matrices created.")
        print(f"TF-IDF matrices created. Programs: {tfidf_matrix_programs.shape}, Jobs: {tfidf_matrix_jobs.shape}")
        return tfidf_vectorizer, tfidf_matrix_programs, tfidf_matrix_jobs
    except ValueError as ve: st.error(f"Fatal Error creating TF-IDF vectors (ValueError): {ve}"); return None, None, None
    except Exception as e: st.error(f"Fatal Error creating TF-IDF vectors: {e}"); return None, None, None


# --- Recommendation Function (No logic change needed) ---
# (Keep existing recommend_programs function from previous app.py version)
def recommend_programs(job_title, target_city, target_state, df, tfidf_jobs, tfidf_programs, top_n=5):
    """Recommends top_n programs using Streamlit feedback elements."""
    print(f"Searching: Job='{job_title}', City='{target_city}', State='{target_state}'")
    job_indices = df.index[df['JobTitle'] == job_title].tolist()
    if not job_indices: st.error(f"Error: Job Title '{job_title}' not found."); return pd.DataFrame(), []
    target_job_index = job_indices[0]
    if target_job_index >= tfidf_jobs.shape[0]: st.error(f"Error: Index mismatch for job '{job_title}'."); return pd.DataFrame(), []
    target_job_vector = tfidf_jobs[target_job_index]
    required_skills_list = df.loc[target_job_index, 'RequiredSkills_list']

    program_indices = df.index[(df['ProgramCity'] == target_city) & (df['ProgramState'] == target_state)].tolist()
    if not program_indices: st.info(f"Job '{job_title}' found, but no training programs listed in {target_city}, {target_state}."); return pd.DataFrame(), required_skills_list

    valid_program_indices = [idx for idx in program_indices if 0 <= idx < tfidf_programs.shape[0]]
    if not valid_program_indices: st.warning("Program indices mismatch TF-IDF matrix."); return pd.DataFrame(), required_skills_list
    filtered_program_vectors = tfidf_programs[valid_program_indices]

    try: cosine_similarities = cosine_similarity(target_job_vector.reshape(1, -1), filtered_program_vectors)
    except Exception as e: st.error(f"Error calculating similarity: {e}"); return pd.DataFrame(), required_skills_list
    similarity_scores = cosine_similarities[0]

    program_similarity_map = list(zip(valid_program_indices, similarity_scores))
    sorted_programs = sorted(program_similarity_map, key=lambda x: x[1], reverse=True)

    actual_top_n = min(top_n, len(sorted_programs))
    if actual_top_n == 0: st.info("No recommendations based on similarity scores."); return pd.DataFrame(), required_skills_list
    top_program_indices = [idx for idx, score in sorted_programs[:actual_top_n]]

    recommendations_df = df.loc[top_program_indices].copy()
    score_map = dict(sorted_programs[:actual_top_n])
    recommendations_df['SimilarityScore'] = recommendations_df.index.map(score_map)
    recommendations_df = recommendations_df[recommendations_df['SimilarityScore'] > 0.01] # Min threshold

    if not recommendations_df.empty: st.success(f"Found {len(recommendations_df)} relevant recommendation(s).")
    else: st.info("Found programs, but none met minimum similarity threshold.")
    return recommendations_df, required_skills_list

# --- Streamlit App UI ---

st.title("🎓 Vocational & Tech Training Recommender (India)")
st.markdown("""
Select your desired **job role**, **state**, and **city** to get recommendations for
training programs or relevant educational pathways. Recommendations are based on skill matching.
""") # Updated note

# --- Load data and prepare vectors ---
csv_path = "Vocational_progrram.csv" # <-- *** ENSURE CORRECT FILENAME ***
df_loaded = load_data(csv_path)

if df_loaded is not None:
    vectorizer, tfidf_progs, tfidf_jobs = create_skill_vectors(df_loaded)

    if vectorizer is not None and tfidf_progs is not None and tfidf_jobs is not None:

        # --- User Input Fields (Sidebar) ---
        st.sidebar.header("Configure Your Search:")
        job_options = [""] + sorted(df_loaded['JobTitle'].unique())
        selected_job = st.sidebar.selectbox("1. Select Target Job Title:", options=job_options, index=0, help="Choose the job role.")
        state_options = [""] + sorted(df_loaded['JobState'].unique())
        selected_state = st.sidebar.selectbox("2. Select Target State:", options=state_options, index=0, help="Choose the state.")
        city_options = [""]
        if selected_state:
            cities_in_state = sorted(df_loaded[df_loaded['JobState'] == selected_state]['JobCity'].unique())
            city_options.extend(cities_in_state)
            city_disabled = False; city_help = "Choose the city."
        else: city_disabled = True; city_help = "Select state first."
        selected_city = st.sidebar.selectbox("3. Select Target City:", options=city_options, index=0, help=city_help, disabled=city_disabled)
        num_recs = st.sidebar.slider("4. Max Recommendations:", min_value=1, max_value=10, value=5, help="Adjust number of results.")
        st.sidebar.markdown("---")
        submit_button = st.sidebar.button("🔍 Find Training Options")

        # --- Display Area ---
        st.subheader("Search Results & Job Information:")

        if submit_button:
            if selected_job and selected_city and selected_state:

                # --- *** Display DYNAMICALLY CALCULATED Job Info *** ---
                job_info_rows = df_loaded[df_loaded['JobTitle'] == selected_job]
                if not job_info_rows.empty:
                    # Get base complexity and salary band from the first matching row
                    job_info = job_info_rows.iloc[0]
                    job_complexity = job_info.get('JobComplexity', 'voc_medium') # Default if column missing
                    base_salary_info = salary_bands_v4.get(job_complexity, salary_bands_v4["voc_medium"])
                    base_min = base_salary_info["min"]
                    base_max = base_salary_info["max"]

                    # Calculate multiplier based on USER'S SELECTED location
                    loc_multiplier = get_location_multiplier_v4_1(selected_city, selected_state, job_complexity)

                    # Calculate estimated salary for THIS specific query context
                    # Add a slight random element for display variation, consistent with generation
                    display_variation = random.uniform(0.98, 1.02) # Small variation for display
                    est_min_salary_disp = int(base_min * loc_multiplier * display_variation / 500) * 500
                    est_max_salary_disp = int(base_max * loc_multiplier * display_variation / 500) * 500
                    if est_min_salary_disp >= est_max_salary_disp: est_min_salary_disp = int(est_max_salary_disp * 0.8 / 500) * 500
                    est_min_salary_disp = max(8500, est_min_salary_disp)
                    est_max_salary_disp = max(est_min_salary_disp + 2500, est_max_salary_disp)

                    # Display Calculated Info
                    st.markdown(f"**Details for '{selected_job}' in {selected_city}, {selected_state}:**")
                    col1, col2 = st.columns(2)
                    complexity_label = complexity_mapping.get(job_complexity, str(job_complexity).capitalize())
                    with col1: st.metric(label="Typical Role Level", value=complexity_label)
                    salary_str = f"₹{est_min_salary_disp:,.0f} - ₹{est_max_salary_disp:,.0f} / month (Est.)"
                    with col2: st.metric(label="Estimated Salary Range (Location Adjusted)", value=salary_str, help="Salary is approximate, synthetic, and varies.")
                    st.markdown("---")
                else:
                    st.warning(f"Could not retrieve specific details for job '{selected_job}', but proceeding with search.")
                    st.markdown("---")

                # --- Call Recommendation Function ---
                recommendations_df, job_skills = recommend_programs(
                    selected_job, selected_city, selected_state,
                    df_loaded, tfidf_jobs, tfidf_progs, top_n=num_recs
                )

                # --- Display Results ---
                if not recommendations_df.empty:
                    st.markdown(f"**Top {len(recommendations_df)} Recommended Training Options:**")
                    display_cols = ['RecommendedProgramName', 'TrainingProvider', 'ProgramCity', 'ProgramState', 'ProgramDurationMonths', 'ProgramCertification', 'SimilarityScore']
                    display_df = recommendations_df[display_cols].rename(columns={
                        'RecommendedProgramName': 'Program/Pathway Name', 'TrainingProvider': 'Provider/Type',
                        'ProgramCity': 'City', 'ProgramState': 'State', 'ProgramDurationMonths': 'Duration (Months)',
                        'ProgramCertification': 'Certification/Outcome', 'SimilarityScore': 'Skill Match Score'
                    })
                    st.dataframe(display_df.style.format({'Skill Match Score': "{:.3f}"}), use_container_width=True)
                    with st.expander("View Required Skills for Selected Job Role"):
                        if job_skills:
                             st.markdown(f"**Key Skills Generally Required for '{selected_job}':**")
                             for skill in job_skills: st.markdown(f"- {skill}")
                        else: st.write("Could not retrieve required skills details.")
                # else: Messages handled inside function

            else:
                st.warning("⚠️ Please select a Job Title, State, AND City before clicking 'Find Training Options'.")
        else:
             st.info("Select your preferences in the sidebar and click 'Find Training Options' to get recommendations.")

    else: st.error("Could not prepare skill vectors. Check data loading/TF-IDF.")
else: st.error("Application cannot start - data loading issues.")

# Footer
st.markdown("---")
st.caption("Vocational & Tech Training Recommender | ML Project ")