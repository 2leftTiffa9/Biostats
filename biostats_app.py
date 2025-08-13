import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import datetime
import bcrypt
import io
import base64
import math
from collections import Counter
from scipy.stats import chi2, norm, t, ttest_rel, f_oneway
from itertools import combinations

# --- Inject custom CSS for styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    color: #333; /* Default text color for general visibility */
}

h1, h2, h3, h4, h5, h6, .css-10trblm {
    color: #004d40;
    font-weight: 600;
}

.stButton>button {
    background-color: #00796b;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 24px;
    font-weight: 600;
    transition: background-color 0.3s;
}

.stButton>button:hover {
    background-color: #004d40;
}

.stTextInput>div>div>input {
    border-radius: 8px;
}

.stExpander {
    background-color: #e0f2f1;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
}

/* FIX: Ensure content inside expander is visible */
.stExpander>.stExpander.stExpanderDetails {
    background-color: #f5f5f5;
    color: #333; /* Set a dark color for the text */
}

.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: #e0f2f1;
    border-radius: 8px 8px 0 0;
    gap: 10px;
    padding: 10px;
    color: #004d40;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background-color: #00796b;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ------------------ DATABASE FUNCTIONS ------------------

def get_connection():
    """Establishes a connection to the SQLite database."""
    return sqlite3.connect('biostats_app.db')

def init_db():
    """Initializes the database and creates required tables if they don't exist."""
    conn = get_connection()
    c = conn.cursor()

    # Notes table
    c.execute('''
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')

    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL
        )
    ''')

    # Research table with BLOB storage
    c.execute('''
        CREATE TABLE IF NOT EXISTS research (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            authors TEXT,
            year INTEGER,
            study_design TEXT,
            keywords TEXT,
            abstract TEXT,
            file_name TEXT,
            file_data BLOB,
            uploaded_by INTEGER,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (uploaded_by) REFERENCES users(id)
        )
    ''')

    conn.commit()
    conn.close()

# ------------------ USER FUNCTIONS ------------------

def create_user(username, password):
    """Creates a new user with hashed password."""
    conn = get_connection()
    c = conn.cursor()
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    try:
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    """Validates user login."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        user_id, password_hash = result
        if bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8')):
            return user_id
    return None

# ------------------ NOTES FUNCTIONS ------------------

def save_note(title, content):
    conn = get_connection()
    c = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    c.execute("INSERT INTO notes (title, content, timestamp) VALUES (?, ?, ?)", (title, content, timestamp))
    conn.commit()
    conn.close()

def get_notes():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, title, content, timestamp FROM notes ORDER BY timestamp DESC")
    notes = c.fetchall()
    conn.close()
    return notes

def delete_note(note_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    conn.commit()
    conn.close()

# ------------------ RESEARCH REPOSITORY FUNCTIONS ------------------

def save_research(title, authors, year, study_design, keywords, abstract, file_name, file_data, uploaded_by):
    conn = get_connection()
    c = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    c.execute("""
        INSERT INTO research 
        (title, authors, year, study_design, keywords, abstract, file_name, file_data, uploaded_by, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (title, authors, year, study_design, keywords, abstract, file_name, file_data, uploaded_by, timestamp))
    conn.commit()
    conn.close()

def get_all_research(search_query=None, author_filter=None, year_filter=None):
    conn = get_connection()
    c = conn.cursor()
    query = "SELECT id, title, authors, year, study_design, keywords, abstract, file_name, file_data, timestamp FROM research WHERE 1=1"
    params = []
    if search_query:
        query += " AND (title LIKE ? OR authors LIKE ? OR keywords LIKE ?)"
        params.extend([f'%{search_query}%', f'%{search_query}%', f'%{search_query}%'])
    if author_filter:
        query += " AND authors LIKE ?"
        params.append(f'%{author_filter}%')
    if year_filter:
        query += " AND year = ?"
        params.append(year_filter)
    query += " ORDER BY timestamp DESC"
    c.execute(query, tuple(params))
    results = c.fetchall()
    conn.close()
    return results

def delete_research(research_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM research WHERE id = ?", (research_id,))
    conn.commit()
    conn.close()

def download_file_button(file_name, file_data):
    b64 = base64.b64encode(file_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">📥 Download {file_name}</a>'
    return href

# ------------------ STATISTICAL & EPIDEMIOLOGICAL FUNCTIONS ------------------

def calculate_mean(data):
    if not data: return 0
    return sum(data) / len(data)

def calculate_median(data):
    if not data: return 0
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 1: return sorted_data[n // 2]
    else:
        mid1 = sorted_data[n // 2 - 1]
        mid2 = sorted_data[n // 2]
        return (mid1 + mid2) / 2

def calculate_mode(data):
    if not data: return ["N/A"]
    counts = Counter(data)
    max_count = max(counts.values())
    modes = [key for key, value in counts.items() if value == max_count]
    if max_count == 1 and len(modes) == len(data): return ["No mode"]
    return modes

def calculate_quartiles(data):
    if len(data) < 4: return None, None
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 0:
        lower_half = sorted_data[:n // 2]
        upper_half = sorted_data[n // 2:]
    else:
        lower_half = sorted_data[:n // 2]
        upper_half = sorted_data[n // 2 + 1:]
    q1 = calculate_median(lower_half)
    q3 = calculate_median(upper_half)
    return q1, q3

def calculate_interquartile_range(data):
    q1, q3 = calculate_quartiles(data)
    if q1 is None or q3 is None: return 0
    return q3 - q1

def calculate_variance(data):
    if len(data) < 2: return 0
    mean = calculate_mean(data)
    sum_of_squares = sum([(x - mean) ** 2 for x in data])
    return sum_of_squares / (len(data) - 1)

def calculate_standard_deviation(data):
    variance = calculate_variance(data)
    return math.sqrt(variance)

def calculate_coefficient_of_variation(data):
    mean = calculate_mean(data)
    std_dev = calculate_standard_deviation(data)
    if mean == 0: return 0
    return (std_dev / mean) * 100

def calculate_prevalence(cases, population):
    if population == 0: return 0
    return (cases / population) * 1000

def calculate_incidence_rate(new_cases, person_time):
    if person_time == 0: return 0
    return (new_cases / person_time) * 1000

def calculate_odds_ratio(a, b, c, d):
    if b * c == 0: return float('inf')
    return (a * d) / (b * c)

def calculate_relative_risk(a, b, c, d):
    risk_exposed = a / (a + b) if (a + b) != 0 else 0
    risk_unexposed = c / (c + d) if (c + d) != 0 else 0
    if risk_unexposed == 0: return float('inf')
    return risk_exposed / risk_unexposed

def calculate_risk_difference(a, b, c, d):
    risk_exposed = a / (a + b) if (a + b) != 0 else 0
    risk_unexposed = c / (c + d) if (c + d) != 0 else 0
    return risk_exposed - risk_unexposed

def calculate_attributable_fraction_exposed(rr):
    if rr == 0: return 0
    return (rr - 1) / rr

def calculate_population_attributable_fraction(p_e, rr):
    if p_e * (rr - 1) + 1 == 0: return 0
    return (p_e * (rr - 1)) / (p_e * (rr - 1) + 1)

def calculate_attributable_risk_case_control(or_value):
    if or_value == 0: return 0
    return (or_value - 1) / or_value

def calculate_chi_squared_from_table(observed_table):
    observed = np.array(observed_table)
    if observed.ndim != 2 or observed.size == 0:
        return 0, 1.0, 0
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)
    grand_total = observed.sum()
    if grand_total == 0: return 0, 1.0, 0
    expected = np.outer(row_totals, col_totals) / grand_total
    if np.any(expected < 5):
        st.warning("Warning: One or more expected counts are less than 5. The chi-squared test may not be appropriate.")
    chi_sq_stat = np.sum((observed - expected)**2 / expected)
    rows, cols = observed.shape
    df = (rows - 1) * (cols - 1)
    if df <= 0: return 0, 1.0, 0
    p_value = 1 - chi2.cdf(chi_sq_stat, df)
    return chi_sq_stat, p_value, df

def calculate_z_test_proportion(x, n, p0):
    if n == 0 or p0 < 0 or p0 > 1: return 0, 1.0
    p_hat = x / n
    standard_error = math.sqrt((p0 * (1 - p0)) / n)
    if standard_error == 0: return 0, 1.0
    z_stat = (p_hat - p0) / standard_error
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    return z_stat, p_value

def calculate_z_ci(mean, std_dev, n, confidence_level=0.95):
    if n == 0 or std_dev == 0: return 0, 0
    alpha = 1 - confidence_level
    z_critical = norm.ppf(1 - alpha / 2)
    margin_of_error = z_critical * (std_dev / math.sqrt(n))
    return mean - margin_of_error, mean + margin_of_error

def calculate_t_ci(mean, std_dev, n, confidence_level=0.95):
    if n <= 1 or std_dev == 0: return 0, 0
    alpha = 1 - confidence_level
    df = n - 1
    t_critical = t.ppf(1 - alpha / 2, df)
    margin_of_error = t_critical * (std_dev / math.sqrt(n))
    return mean - margin_of_error, mean + margin_of_error

def calculate_z_test_mean(sample_mean, pop_mean, pop_std_dev, n):
    if pop_std_dev == 0 or n == 0: return 0, 1.0
    z_stat = (sample_mean - pop_mean) / (pop_std_dev / math.sqrt(n))
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    return z_stat, p_value

def calculate_t_test_mean(sample_mean, pop_mean, sample_std_dev, n):
    if sample_std_dev == 0 or n <= 1: return 0, 1.0
    t_stat = (sample_mean - pop_mean) / (sample_std_dev / math.sqrt(n))
    df = n - 1
    p_value = 2 * (1 - t.cdf(abs(t_stat), df))
    return t_stat, p_value

def calculate_z_test_two_proportions(x1, n1, x2, n2):
    p1_hat = x1 / n1 if n1 > 0 else 0
    p2_hat = x2 / n2 if n2 > 0 else 0
    p_pooled = (x1 + x2) / (n1 + n2)
    if p_pooled == 0 or p_pooled == 1:
        st.warning("Pooled proportion is 0 or 1, standard error cannot be calculated.")
        return 0, 1.0
    se_pooled = math.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    if se_pooled == 0:
        return 0, 1.0
    z_stat = (p1_hat - p2_hat) / se_pooled
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    return z_stat, p_value
    
def calculate_paired_t_test(sample1, sample2):
    """Performs a paired t-test on two related samples."""
    if len(sample1) != len(sample2):
        return None, None
    t_stat, p_value = ttest_rel(sample1, sample2)
    return t_stat, p_value

def calculate_anova(*samples):
    """Performs a one-way ANOVA on two or more samples."""
    if len(samples) < 2:
        return None, None
    f_stat, p_value = f_oneway(*samples)
    return f_stat, p_value
    
def calculate_binomial_probability(n, k, p):
    """Calculates the probability of exactly k successes in n trials."""
    if p < 0 or p > 1:
        st.error("Probability 'p' must be between 0 and 1.")
        return 0
    try:
        nCk = math.comb(n, k)
        probability = nCk * (p ** k) * ((1 - p) ** (n - k))
        return probability
    except ValueError:
        st.error("Invalid input for binomial calculation. n must be >= k and non-negative.")
        return 0

def calculate_normal_probability(mu, sigma, x1, x2=None):
    """Calculates normal distribution probability based on Z-scores."""
    if sigma <= 0:
        st.error("Standard deviation must be a positive number.")
        return 0
    
    if x2 is None: # Single value (P(X<x) or P(X>x))
        z_score = (x1 - mu) / sigma
        prob_less = norm.cdf(z_score)
        prob_greater = 1 - prob_less
        return z_score, prob_less, prob_greater
    else: # Two values (P(x1 < X < x2))
        z1 = (x1 - mu) / sigma
        z2 = (x2 - mu) / sigma
        prob_between = norm.cdf(z2) - norm.cdf(z1)
        return z1, z2, prob_between

def calculate_poisson_probability(lambda_val, k):
    """Calculates the Poisson probability of exactly k events."""
    if lambda_val < 0 or k < 0:
        st.error("Lambda and k must be non-negative.")
        return 0
    
    try:
        probability = (lambda_val**k * math.exp(-lambda_val)) / math.factorial(k)
        return probability
    except ValueError:
        st.error("Invalid input for Poisson calculation.")
        return 0

# ------------------ UI PAGES ------------------

def home_page():
    st.title("Biostatistics & Epidemiology Tool")
    st.markdown("""
        Welcome to the Biostatistics & Epidemiology Tool. This application is designed to help you
        with various statistical and epidemiological calculations. Use the navigation
        sidebar to access different sections of the tool.

        ### What you can do with this app:

        * **Descriptive Statistics:** Analyze a dataset by calculating key metrics like mean, median, mode, variance, and more.
        * **Epidemiological Measures:** Compute essential public health metrics such as prevalence, incidence, odds ratio, and relative risk.
        * **Hypothesis Testing:** Perform significance tests like the Chi-Squared test, Z/T-tests for means and proportions, and more advanced tests like Paired T-test and ANOVA.
        * **Probability Distributions:** Calculate probabilities for the Binomial, Normal, and Poisson distributions.
        * **Research Notes & Community Health Notes:** Manage your own research notes and upload, search, and preview community health notes.

        Use the navigation on the left to get started.
    """)

def descriptive_stats_page():
    st.header("📊 Descriptive Statistics")
    st.write("Enter a list of numbers (separated by commas or spaces) to calculate descriptive statistics.")
    user_input_stats = st.text_area("Enter your data here:", "15, 20, 25, 30, 35, 40")
    if st.button("Calculate Statistics"):
        if user_input_stats:
            try:
                data_list = [float(x.strip()) for x in user_input_stats.replace(',', ' ').split()]
                if data_list:
                    st.subheader("Results")
                    mean_value = calculate_mean(data_list)
                    median_value = calculate_median(data_list)
                    mode_values = calculate_mode(data_list)
                    variance_value = calculate_variance(data_list)
                    std_dev_value = calculate_standard_deviation(data_list)
                    q1, q3 = calculate_quartiles(data_list)
                    iqr_value = calculate_interquartile_range(data_list)
                    cv_value = calculate_coefficient_of_variation(data_list)
                    
                    st.write(f"**Mean:** {mean_value:.2f}")
                    with st.expander("Show Mean Formula & Explanation"):
                        st.latex(r''' \bar{x} = \frac{\sum x}{n} ''' )
                        st.markdown(f"""
                            **Explanation:**
                            * The mean is the sum of all data points divided by the number of data points.
                            * Sum of data: {sum(data_list)}
                            * Number of data points ($ n $): {len(data_list)}
                            * Calculation: {sum(data_list)} / {len(data_list)} = {mean_value:.2f}
                        """)
                    
                    st.write(f"**Median:** {median_value:.2f}")
                    with st.expander("Show Median Explanation"):
                        st.markdown(f"""
                            **Explanation:**
                            * The median is the middle value of a sorted dataset.
                            * Sorted data: {sorted(data_list)}
                            * Since there are {len(data_list)} data points (an even number), the median is the average of the two middle values.
                            * The middle values are {sorted(data_list)[len(data_list)//2 - 1]} and {sorted(data_list)[len(data_list)//2]}.
                            * Calculation: ({sorted(data_list)[len(data_list)//2 - 1]} + {sorted(data_list)[len(data_list)//2]}) / 2 = {median_value:.2f}
                        """)

                    st.write(f"**Mode(s):** {', '.join(map(str, mode_values))}")
                    with st.expander("Show Mode Explanation"):
                        st.markdown("""
                            **Explanation:**
                            * The mode is the value that appears most frequently in a dataset.
                        """)
                        
                    st.write(f"**Variance:** {variance_value:.2f}")
                    with st.expander("Show Variance Formula & Explanation"):
                        st.latex(r''' s^2 = \frac{\sum (x_i - \bar{x})^2}{n-1} ''' )
                        st.markdown(f"""
                            **Explanation:**
                            * Variance measures how spread out the data is. It's the average of the squared differences from the mean.
                            * First, we find the difference between each data point and the mean ($\bar{{x}}={mean_value:.2f}$), square it, and sum them up.
                            * Sum of squared differences: {sum([(x - mean_value) ** 2 for x in data_list]):.2f}
                            * Then we divide by the number of data points minus one ($n-1$).
                            * Calculation: {sum([(x - mean_value) ** 2 for x in data_list]):.2f} / ({len(data_list)}-1) = {variance_value:.2f}
                        """)

                    st.write(f"**Standard Deviation:** {std_dev_value:.2f}")
                    with st.expander("Show Standard Deviation Formula & Explanation"):
                        st.latex(r''' s = \sqrt{s^2} ''' )
                        st.markdown(f"""
                            **Explanation:**
                            * Standard deviation is the square root of the variance. It tells us the typical distance of data points from the mean.
                            * Calculation: $\sqrt{{{variance_value:.2f}}} = {std_dev_value:.2f}$
                        """)

                    if q1 is not None and q3 is not None: 
                        st.write(f"**Q1 (25th Percentile):** {q1:.2f}")
                        st.write(f"**Q3 (75th Percentile):** {q3:.2f}")
                        st.write(f"**Interquartile Range (IQR):** {iqr_value:.2f}")
                        with st.expander("Show Quartile & IQR Explanation"):
                            st.markdown(f"""
                                **Explanation:**
                                * Quartiles divide the sorted data into four equal parts.
                                * Q1 is the median of the lower half of the data.
                                * Q3 is the median of the upper half of the data.
                                * The Interquartile Range (IQR) is the difference between the third and first quartiles: $IQR = Q3 - Q1$.
                                * Calculation: {q3:.2f} - {q1:.2f} = {iqr_value:.2f}
                            """)
                    
                    st.write(f"**Coefficient of Variation (CV):** {cv_value:.2f}%")
                    with st.expander("Show CV Formula & Explanation"):
                        st.latex(r''' CV = \frac{s}{\bar{x}} \times 100\% ''' )
                        st.markdown(f"""
                            **Explanation:**
                            * The Coefficient of Variation measures the relative variability. It's useful for comparing the dispersion of two different datasets.
                            * Calculation: ({std_dev_value:.2f} / {mean_value:.2f}) × 100% = {cv_value:.2f}%
                        """)

                else: st.warning("Please enter some numbers.")
            except ValueError: st.error("Invalid input. Please make sure you are entering numbers only.")

def epidemiological_measures_page():
    st.header("📈 Epidemiological Measures")
    
    st.subheader("Measures of Frequency")
    st.markdown("---")
    
    st.subheader("Prevalence")
    col1, col2 = st.columns(2)
    with col1: prevalence_cases = st.number_input("Number of Existing Cases:", min_value=0, value=100, key='pc')
    with col2: prevalence_pop = st.number_input("Total Population:", min_value=1, value=10000, key='pp')
    if st.button("Calculate Prevalence", key="prevalence_button"):
        if prevalence_pop > 0:
            prevalence_rate = calculate_prevalence(prevalence_cases, prevalence_pop)
            st.write(f"**Prevalence:** {prevalence_rate:.2f} per 1,000 population")
            with st.expander("Show Prevalence Formula & Explanation"):
                st.latex(r''' \text{Prevalence} = \frac{\text{Number of Existing Cases}}{\text{Total Population}} \times 1000 ''' )
                st.markdown(f"""
                    **Explanation:**
                    * Prevalence measures the proportion of a population with a disease at a specific time.
                    * Calculation: ({prevalence_cases} / {prevalence_pop}) × 1000 = {prevalence_rate:.2f} per 1,000.
                """)
        else: st.error("Total population cannot be zero.")
    
    st.markdown("---")
    st.subheader("Incidence Rate")
    col3, col4 = st.columns(2)
    with col3: incidence_cases = st.number_input("Number of New Cases:", min_value=0, value=50, key='ic')
    with col4: incidence_person_time = st.number_input("Person-Time at Risk:", min_value=1, value=5000, key='ipt')
    if st.button("Calculate Incidence", key="incidence_button"):
        if incidence_person_time > 0:
            incidence_rate = calculate_incidence_rate(incidence_cases, incidence_person_time)
            st.write(f"**Incidence Rate:** {incidence_rate:.2f} per 1,000 person-time at risk")
            with st.expander("Show Incidence Rate Formula & Explanation"):
                st.latex(r''' \text{Incidence Rate} = \frac{\text{Number of New Cases}}{\text{Person-Time at Risk}} \times 1000 ''' )
                st.markdown(f"""
                    **Explanation:**
                    * Incidence rate measures the speed at which new cases occur in a population.
                    * Calculation: ({incidence_cases} / {incidence_person_time}) × 1000 = {incidence_rate:.2f} per 1,000 person-time.
                """)
        else: st.error("Person-time at risk cannot be zero.")

    st.markdown("---")
    st.subheader("Measures of Association")
    st.markdown("---")

    st.write("Calculate measures of association from a 2x2 table.")
    st.subheader("2x2 Table")
    st.markdown(
        """
        | | Outcome Present | Outcome Absent |
        | :--- | :---: | :---: |
        | **Exposed** | a | b |
        | **Unexposed** | c | d |
        """
    )
    
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a: a = st.number_input("a", min_value=0, value=40, label_visibility="hidden", key='a')
    with col_b: b = st.number_input("b", min_value=0, value=60, label_visibility="hidden", key='b')
    with col_c: c = st.number_input("c", min_value=0, value=20, label_visibility="hidden", key='c')
    with col_d: d = st.number_input("d", min_value=0, value=80, label_visibility="hidden", key='d')
    
    st.markdown("---")
    
    p_e = st.number_input("Prevalence of Exposure in the Population (0-1):", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    if st.button("Calculate All Association Measures"):
        if (a + b) == 0 or (c + d) == 0: st.error("The sum of exposed or unexposed groups cannot be zero for Relative Risk.")
        else:
            odds_ratio = calculate_odds_ratio(a, b, c, d)
            relative_risk = calculate_relative_risk(a, b, c, d)
            risk_difference = calculate_risk_difference(a, b, c, d)
            if relative_risk == 1: attributable_fraction_exposed = 0
            else: attributable_fraction_exposed = calculate_attributable_fraction_exposed(relative_risk)
            if odds_ratio == 1: attributable_risk_case_control = 0
            else: attributable_risk_case_control = calculate_attributable_risk_case_control(odds_ratio)
            if relative_risk == 1: population_attributable_fraction = 0
            else: population_attributable_fraction = calculate_population_attributable_fraction(p_e, relative_risk)
            
            st.subheader("Results")
            st.write(f"**Odds Ratio (OR):** {odds_ratio:.2f}")
            with st.expander("Show Odds Ratio Formula & Explanation"):
                st.latex(r''' OR = \frac{ad}{bc} ''' )
                st.markdown(f"""
                    **Explanation:**
                    * The Odds Ratio is a measure of association between an exposure and an outcome in case-control studies.
                    * Calculation: ({a} × {d}) / ({b} × {c}) = {odds_ratio:.2f}
                """)
                
            st.write(f"**Relative Risk (RR):** {relative_risk:.2f}")
            with st.expander("Show Relative Risk Formula & Explanation"):
                st.latex(r''' RR = \frac{a / (a+b)}{c / (c+d)} ''' )
                st.markdown(f"""
                    **Explanation:**
                    * Relative Risk measures the ratio of the probability of an event occurring in an exposed group versus a non-exposed group. Used in cohort studies.
                    * Risk in Exposed: {a} / ({a} + {b}) = {a/(a+b):.2f}
                    * Risk in Unexposed: {c} / ({c} + {d}) = {c/(c+d):.2f}
                    * Calculation: {a/(a+b):.2f} / {c/(c+d):.2f} = {relative_risk:.2f}
                """)
                
            st.write(f"**Risk Difference (RD):** {risk_difference:.2f}")
            with st.expander("Show Risk Difference Formula & Explanation"):
                st.latex(r''' RD = \frac{a}{a+b} - \frac{c}{c+d} ''' )
                st.markdown(f"""
                    **Explanation:**
                    * Risk Difference is the difference in the proportion of disease in the exposed and non-exposed groups. It tells us the excess risk attributable to the exposure.
                    * Calculation: ({a}/({a}+{b})) - ({c}/({c}+{d})) = {risk_difference:.2f}
                """)
                
            st.write(f"**Attributable Fraction in the Exposed (AF_e):** {attributable_fraction_exposed:.2f} ({attributable_fraction_exposed*100:.2f}%)")
            with st.expander("Show AF_e Formula & Explanation"):
                st.latex(r''' AF_e = \frac{RR - 1}{RR} ''' )
                st.markdown(f"""
                    **Explanation:**
                    * AF_e is the proportion of risk in the exposed group that is attributable to the exposure itself.
                    * Calculation: ({relative_risk:.2f} - 1) / {relative_risk:.2f} = {attributable_fraction_exposed:.2f}
                """)
                
            st.write(f"**Population Attributable Fraction (PAF):** {population_attributable_fraction:.2f} ({population_attributable_fraction*100:.2f}%)")
            with st.expander("Show PAF Formula & Explanation"):
                st.latex(r''' PAF = \frac{P_e(RR - 1)}{P_e(RR - 1) + 1} ''' )
                st.markdown(f"""
                    **Explanation:**
                    * PAF is the proportion of disease in the entire population that is attributable to the exposure.
                    * Calculation: ({p_e} × ({relative_risk:.2f}-1)) / ({p_e} × ({relative_risk:.2f}-1) + 1) = {population_attributable_fraction:.2f}
                """)
                
            st.write(f"**Attributable Risk for Case-Control (using OR):** {attributable_risk_case_control:.2f} ({attributable_risk_case_control*100:.2f}%)")
            with st.expander("Show Attributable Risk (OR) Formula & Explanation"):
                st.latex(r''' AR_{cc} = \frac{OR - 1}{OR} ''' )
                st.markdown(f"""
                    **Explanation:**
                    * Similar to AF_e, but used with the Odds Ratio from a case-control study. It estimates the proportion of disease among exposed cases attributable to the exposure.
                    * Calculation: ({odds_ratio:.2f} - 1) / {odds_ratio:.2f} = {attributable_risk_case_control:.2f}
                """)

def hypothesis_testing_page():
    st.header("📑 Hypothesis Testing")
    st.subheader("Tests for Independence & Proportions")
    
    st.markdown("---")

    st.subheader("Chi-Squared Test for a R x C Table")
    st.write("Test for independence between two categorical variables. Enter your observed values as a table. Use spaces or commas to separate columns and new lines to separate rows.")
    st.markdown("Example for a 2x3 table: `10, 20, 30\n15, 25, 35`")
    
    chi_sq_input = st.text_area("Observed values table:", "40, 60\n20, 80", key="chi_sq_input")
    
    if st.button("Calculate Chi-Squared Test", key="chi_sq_button"):
        try:
            rows = chi_sq_input.strip().split('\n')
            table = [[float(val.strip()) for val in row.split(',')] for row in rows]
            
            chi_sq_stat, p_value, df = calculate_chi_squared_from_table(table)

            st.subheader("Results")
            st.write(f"**Chi-Squared Statistic ($X^2$):** {chi_sq_stat:.2f}")
            st.write(f"**Degrees of Freedom (df):** {df}")
            st.write(f"**P-value:** {p_value:.4f}")
            
            with st.expander("Show Chi-Squared Formula & Explanation"):
                st.latex(r''' X^2 = \sum \frac{(O-E)^2}{E} ''' )
                st.markdown(f"""
                    **Explanation:**
                    * The Chi-Squared test compares observed counts ($O$) to expected counts ($E$) if there were no association between the variables.
                    * First, the expected counts for each cell are calculated based on row and column totals.
                    * The difference between observed and expected counts is squared and divided by the expected count for each cell.
                    * These values are then summed to get the Chi-Squared statistic.
                    * The degrees of freedom (df) are calculated as: $df = (rows-1) \times (cols-1) = (2-1) \times (2-1) = 1$.
                    * Finally, the p-value is determined by comparing the $X^2$ statistic to the chi-squared distribution with the calculated degrees of freedom.
                """)
            
            if p_value < 0.05:
                st.success("The p-value is less than the significance level of 0.05. We reject the null hypothesis. There is a statistically significant association between the variables.")
            else:
                st.info("The p-value is greater than or equal to the significance level of 0.05. We fail to reject the null hypothesis. There is no statistically significant association between the variables.")
        except (ValueError, IndexError):
            st.error("Invalid input format. Please ensure you are entering numbers in a valid table format.")
    
    st.markdown("---")
    st.subheader("Z-test for Two Proportions")
    st.write("Test if the proportions of two independent populations are significantly different.")
    
    st.markdown("**Sample 1**")
    col_x1, col_n1 = st.columns(2)
    with col_x1: x1 = st.number_input("Number of successes (x1):", min_value=0, value=250, key='x1')
    with col_n1: n1 = st.number_input("Sample size (n1):", min_value=1, value=500, key='n1')
    
    st.markdown("**Sample 2**")
    col_x2, col_n2 = st.columns(2)
    with col_x2: x2 = st.number_input("Number of successes (x2):", min_value=0, value=200, key='x2')
    with col_n2: n2 = st.number_input("Sample size (n2):", min_value=1, value=500, key='n2')

    if st.button("Calculate Two-Proportion Z-test"):
        if x1 > n1 or x2 > n2:
            st.error("Number of successes cannot be greater than the sample size.")
        else:
            z_stat, p_value = calculate_z_test_two_proportions(x1, n1, x2, n2)
            st.subheader("Results")
            st.write(f"**Z-statistic:** {z_stat:.2f}")
            st.write(f"**P-value:** {p_value:.4f}")
            
            with st.expander("Show Two-Proportion Z-test Formula & Explanation"):
                st.latex(r''' Z = \frac{(\hat{p_1} - \hat{p_2})}{\sqrt{\hat{p}(1-\hat{p})(\frac{1}{n_1} + \frac{1}{n_2})}} ''' )
                st.markdown(f"""
                    **Explanation:**
                    * This test checks if there is a significant difference between two population proportions.
                    * The formula uses the pooled proportion ($\hat{{p}}$) to calculate the standard error.
                    * The test statistic (Z) measures how many standard errors the difference between the two sample proportions is from zero.
                    * Calculation: (({x1}/{n1}) - ({x2}/{n2})) / $\sqrt{{{((x1+x2)/(n1+n2))}}} \times (1-{((x1+x2)/(n1+n2))}) \times (1/{n1} + 1/{n2})$ = {z_stat:.2f}
                    * The p-value is calculated from the Z-statistic using a standard normal distribution.
                """)
            
            if p_value < 0.05:
                st.success("The p-value is less than 0.05. We reject the null hypothesis. The two population proportions are significantly different.")
            else:
                st.info("The p-value is greater than or equal to 0.05. We fail to reject the null hypothesis. The two population proportions are not significantly different.")
    
    st.markdown("---")
    st.subheader("Tests for the Mean")
    
    st.subheader("Confidence Intervals for the Mean")
    st.write("Calculate the confidence interval for the population mean.")
    mean_ci = st.number_input("Sample Mean:", value=100.0, key="mean_ci")
    std_dev_ci = st.number_input("Sample/Population Standard Deviation:", value=15.0, key="std_dev_ci")
    n_ci = st.number_input("Sample Size (n):", min_value=1, value=30, key="n_ci")
    confidence_level = st.slider("Confidence Level:", min_value=0.50, max_value=0.99, value=0.95, step=0.01)
    
    if st.button("Calculate Confidence Interval", key="ci_button"):
        if n_ci < 20: 
            st.write("**Using T-distribution (n < 20)**")
            lower, upper = calculate_t_ci(mean_ci, std_dev_ci, n_ci, confidence_level)
            st.write(f"**Confidence Interval ({confidence_level*100:.0f}%):** [{lower:.2f}, {upper:.2f}]")
            with st.expander("Show T-CI Formula & Explanation"):
                st.latex(r''' \text{CI} = \bar{x} \pm t_{\alpha/2, n-1} (\frac{s}{\sqrt{n}}) ''' )
                st.markdown(f"""
                    **Explanation:**
                    * The confidence interval provides a range of plausible values for the population mean.
                    * For small sample sizes ($n < 20$), the t-distribution is used.
                    * The margin of error is calculated using the t-critical value, standard deviation, and sample size.
                    * The interval is then Mean ± Margin of Error.
                    * Your interval is: [{lower:.2f}, {upper:.2f}]
                """)
        else:
            st.write("**Using Z-distribution (n >= 20)**")
            lower, upper = calculate_z_ci(mean_ci, std_dev_ci, n_ci, confidence_level)
            st.write(f"**Confidence Interval ({confidence_level*100:.0f}%):** [{lower:.2f}, {upper:.2f}]")
            with st.expander("Show Z-CI Formula & Explanation"):
                st.latex(r''' \text{CI} = \bar{x} \pm Z_{\alpha/2} (\frac{\sigma}{\sqrt{n}}) ''' )
                st.markdown(f"""
                    **Explanation:**
                    * For larger sample sizes ($n \ge 20$), the Z-distribution is used.
                    * The margin of error is calculated using the Z-critical value, standard deviation, and sample size.
                    * The interval is then Mean ± Margin of Error.
                    * Your interval is: [{lower:.2f}, {upper:.2f}]
                """)

    st.markdown("---")

    st.subheader("Significance Test for the Mean")
    st.write("Test if a sample mean is significantly different from a hypothesized population mean.")
    sample_mean_test = st.number_input("Sample Mean:", value=102.0, key="sample_mean_test")
    pop_mean_test = st.number_input("Hypothesized Population Mean ($\mu_0$):", value=100.0, key="pop_mean_test")
    std_dev_test = st.number_input("Sample Standard Deviation (s):", value=15.0, key="std_dev_test")
    n_test = st.number_input("Sample Size (n):", min_value=2, value=25, key="n_test")

    if st.button("Calculate Mean Significance Test", key="mean_test_button"):
        if n_test < 20:
            st.write("**Using T-test (n < 20)**")
            t_stat, p_value = calculate_t_test_mean(sample_mean_test, pop_mean_test, std_dev_test, n_test)
            st.write(f"**T-statistic:** {t_stat:.2f}")
            st.write(f"**Degrees of Freedom (df):** {n_test-1}")
            st.write(f"**P-value:** {p_value:.4f}")
            with st.expander("Show T-test Formula & Explanation"):
                st.latex(r''' t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}} ''' )
                st.markdown(f"""
                    **Explanation:**
                    * The T-test is used to determine if the means of two groups are statistically different from each other.
                    * This one-sample t-test compares your sample mean to a hypothesized population mean.
                    * Calculation: ({sample_mean_test:.2f} - {pop_mean_test:.2f}) / ({std_dev_test:.2f} / $\sqrt{{{n_test}}}$) = {t_stat:.2f}
                """)
        else:
            st.write("**Using Z-test (n >= 20)**")
            z_stat, p_value = calculate_z_test_mean(sample_mean_test, pop_mean_test, std_dev_test, n_test)
            st.write(f"**Z-statistic:** {z_stat:.2f}")
            st.write(f"**P-value:** {p_value:.4f}")
            with st.expander("Show Z-test Formula & Explanation"):
                st.latex(r''' Z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}} ''' )
                st.markdown(f"""
                    **Explanation:**
                    * The Z-test is used for larger samples ($n \ge 20$) to compare a sample mean to a known population mean.
                    * Calculation: ({sample_mean_test:.2f} - {pop_mean_test:.2f}) / ({std_dev_test:.2f} / $\sqrt{{{n_test}}}$) = {z_stat:.2f}
                """)
        
        if p_value < 0.05:
            st.success("The p-value is less than 0.05. We reject the null hypothesis. The sample mean is significantly different from the hypothesized population mean.")
        else:
            st.info("The p-value is greater than or equal to 0.05. We fail to reject the null hypothesis. The sample mean is not significantly different from the hypothesized population mean.")

    st.markdown("---")
    st.subheader("Paired T-test")
    st.write("Compares the means of two related samples (e.g., before and after a treatment).")
    st.markdown("Enter data as a comma-separated list.")
    sample1_input = st.text_input("Sample 1 (e.g., 'before' data):", "10, 12, 15, 11, 13", key="paired_t_sample1")
    sample2_input = st.text_input("Sample 2 (e.g., 'after' data):", "12, 14, 16, 12, 15", key="paired_t_sample2")
    
    if st.button("Calculate Paired T-test"):
        try:
            sample1 = [float(x.strip()) for x in sample1_input.replace(',', ' ').split()]
            sample2 = [float(x.strip()) for x in sample2_input.replace(',', ' ').split()]
            if len(sample1) != len(sample2):
                st.error("The two samples must have the same number of data points.")
            elif len(sample1) < 2:
                st.error("Samples must have at least 2 data points.")
            else:
                t_stat, p_value = calculate_paired_t_test(sample1, sample2)
                st.subheader("Results")
                st.write(f"**T-statistic:** {t_stat:.4f}")
                st.write(f"**P-value:** {p_value:.4f}")
                
                with st.expander("Show Paired T-test Explanation"):
                    st.markdown("""
                        **Explanation:**
                        * The paired t-test is used to compare the means of two measurements taken from the same individuals, items, or related units.
                        * The null hypothesis is that the mean difference between the two samples is zero.
                        * The p-value indicates the probability of observing a difference as extreme as the one in your data, assuming the null hypothesis is true.
                    """)
                
                if p_value < 0.05:
                    st.success("The p-value is less than 0.05. We reject the null hypothesis. There is a statistically significant difference between the two samples.")
                else:
                    st.info("The p-value is greater than or equal to 0.05. We fail to reject the null hypothesis. There is no statistically significant difference between the two samples.")
        except ValueError:
            st.error("Invalid input. Please enter a comma-separated list of numbers.")

    st.markdown("---")
    st.subheader("One-way ANOVA")
    st.write("Compares the means of three or more independent groups.")
    st.markdown("Enter data for each group on a new line. Separate values within a group with commas.")
    anova_input = st.text_area("Group Data (e.g., Group 1: 10, 12, 14\nGroup 2: 15, 17, 19\nGroup 3: 20, 22, 24):", 
                               "10, 12, 14\n15, 17, 19\n20, 22, 24", key="anova_data")
    
    if st.button("Calculate ANOVA"):
        try:
            groups_data = anova_input.strip().split('\n')
            samples = []
            for group_str in groups_data:
                sample = [float(x.strip()) for x in group_str.replace(',', ' ').split()]
                if sample:
                    samples.append(sample)

            if len(samples) < 2:
                st.error("You need to enter data for at least two groups.")
            elif any(len(s) < 2 for s in samples):
                st.error("Each group must have at least two data points.")
            else:
                f_stat, p_value = calculate_anova(*samples)
                st.subheader("Results")
                st.write(f"**F-statistic:** {f_stat:.4f}")
                st.write(f"**P-value:** {p_value:.4f}")
                
                with st.expander("Show ANOVA Explanation"):
                    st.markdown("""
                        **Explanation:**
                        * ANOVA tests the null hypothesis that the means of two or more groups are equal.
                        * The F-statistic is the ratio of the variance between the groups to the variance within the groups. A larger F-statistic suggests a larger difference between the group means.
                        * The p-value indicates the probability of obtaining an F-statistic as extreme as the one calculated, assuming the group means are all equal.
                    """)

                if p_value < 0.05:
                    st.success("The p-value is less than 0.05. We reject the null hypothesis. There is a statistically significant difference between the means of at least two of the groups.")
                else:
                    st.info("The p-value is greater than or equal to 0.05. We fail to reject the null hypothesis. There is no statistically significant difference between the means of the groups.")
        except ValueError:
            st.error("Invalid input. Please ensure you are entering numbers in the correct format.")


def probability_distributions_page():
    st.header("🎲 Probability Distributions")

    st.subheader("Binomial Distribution")
    st.write("Calculate probabilities for a fixed number of independent trials with two outcomes.")
    
    st.markdown("---")
    
    st.subheader("Parameters")
    col_n, col_p = st.columns(2)
    with col_n: n_binom = st.number_input("Number of trials (n):", min_value=1, value=10, key="n_binom")
    with col_p: p_binom = st.number_input("Probability of success ($\pi$):", min_value=0.0, max_value=1.0, value=0.22, step=0.01, key="p_binom")
    
    st.markdown("---")
    
    st.subheader("Probability Calculation")
    k_binom = st.number_input("Number of successes (k):", min_value=0, max_value=n_binom, value=5, key="k_binom")
    calculation_type = st.radio(
        "Select Probability Type",
        ('P(X = k)', 'P(X ≤ k)', 'P(X ≥ k)')
    )
    
    if st.button("Calculate Binomial Probability", key="binom_calc_btn"):
        if k_binom > n_binom:
            st.error("Number of successes (k) cannot be greater than the number of trials (n).")
        else:
            if calculation_type == 'P(X = k)':
                prob = calculate_binomial_probability(n_binom, k_binom, p_binom)
                st.write(f"**P(X = {k_binom}):** {prob:.4f} or {prob*100:.2f}%")
                with st.expander("Show P(X=k) Formula & Explanation"):
                    st.latex(r''' P(X=k) = \binom{n}{k} p^k (1-p)^{n-k} ''' )
                    st.markdown(f"""
                        **Explanation:**
                        * This formula calculates the probability of getting exactly $k$ successes in $n$ independent trials.
                        * The term $\binom{{{n_binom}}}{{{k_binom}}}$ is the binomial coefficient, which is the number of ways to choose $k$ successes from $n$ trials.
                        * Calculation: {math.comb(n_binom, k_binom)} × {p_binom}^{k_binom} × (1 - {p_binom})^{n_binom-k_binom} = {prob:.4f}
                    """)
            
            elif calculation_type == 'P(X ≤ k)':
                cumulative_prob = 0
                for k in range(k_binom + 1):
                    cumulative_prob += calculate_binomial_probability(n_binom, k, p_binom)
                st.write(f"**P(X ≤ {k_binom}):** {cumulative_prob:.4f} or {cumulative_prob*100:.2f}%")
            
            elif calculation_type == 'P(X ≥ k)':
                if k_binom == 0:
                    prob = 1.0
                else:
                    cumulative_prob_less_than_k = 0
                    for k in range(k_binom):
                        cumulative_prob_less_than_k += calculate_binomial_probability(n_binom, k, p_binom)
                    prob = 1 - cumulative_prob_less_than_k
                st.write(f"**P(X ≥ {k_binom}):** {prob:.4f} or {prob*100:.2f}%")
    
    st.info("To calculate the probability of 'at least one' success, select **P(X ≥ k)** and set **k = 1**.")

    st.markdown("---")
    
    st.subheader("Mean and Variance")
    if st.button("Show Mean and Variance", key="binom_stats_btn"):
        mean_binom = n_binom * p_binom
        variance_binom = n_binom * p_binom * (1 - p_binom)
        st.write(f"**Mean ($\mu$):** {mean_binom:.2f}")
        st.write(f"**Variance ($\sigma^2$):** {variance_binom:.2f}")
        with st.expander("Show Mean & Variance Formulas"):
            st.latex(r''' \mu = np ''' )
            st.latex(r''' \sigma^2 = np(1-p) ''' )
            st.markdown(f"""
                **Explanation:**
                * Mean = {n_binom} × {p_binom} = {mean_binom:.2f}
                * Variance = {n_binom} × {p_binom} × (1 - {p_binom}) = {variance_binom:.2f}
            """)

    st.subheader("Normal Distribution")
    st.write("Calculate probabilities for a normal distribution.")

    st.markdown("---")
    st.subheader("Parameters")
    col_mu, col_sigma = st.columns(2)
    with col_mu: mu_norm = st.number_input("Mean ($\mu$):", value=100.0, key="mu_norm")
    with col_sigma: sigma_norm = st.number_input("Standard Deviation ($\sigma$):", min_value=0.01, value=15.0, key="sigma_norm")
    
    st.markdown("---")
    st.subheader("Probability Calculation")
    prob_type_norm = st.radio(
        "Select Probability Type",
        ('P(X < x)', 'P(X > x)', 'P(x₁ < X < x₂)'),
        key="norm_calc_type"
    )

    if prob_type_norm == 'P(x₁ < X < x₂)':
        col_x1_norm, col_x2_norm = st.columns(2)
        with col_x1_norm: x1_norm = st.number_input("Lower Value (x₁):", value=85.0, key="x1_norm")
        with col_x2_norm: x2_norm = st.number_input("Upper Value (x₂):", value=115.0, key="x2_norm")
        if st.button("Calculate Normal Probability", key="normal_calc_btn_two_values"):
            if x1_norm >= x2_norm:
                st.error("The lower value (x₁) must be less than the upper value (x₂).")
            else:
                z1, z2, prob = calculate_normal_probability(mu_norm, sigma_norm, x1_norm, x2_norm)
                st.write(f"**P({x1_norm} < X < {x2_norm}):** {prob:.4f} or {prob*100:.2f}%")
                with st.expander("Show Normal Distribution Formula & Explanation"):
                    st.latex(r''' Z = \frac{x - \mu}{\sigma} ''' )
                    st.markdown(f"""
                        **Explanation:**
                        * The Normal distribution is used for continuous data. We first standardize the values ($x_1$ and $x_2$) to Z-scores.
                        * Calculation for $Z_1$: ({x1_norm} - {mu_norm}) / {sigma_norm} = {z1:.2f}
                        * Calculation for $Z_2$: ({x2_norm} - {mu_norm}) / {sigma_norm} = {z2:.2f}
                        * The probability is the area under the standard normal curve between $Z_1$ and $Z_2$, which is calculated using the cumulative distribution function (CDF).
                    """)

    else: # For P(X < x) and P(X > x)
        x_norm = st.number_input("Value (x):", value=100.0, key="x_norm")
        if st.button("Calculate Normal Probability", key="normal_calc_btn_one_value"):
            z_score, prob_less, prob_greater = calculate_normal_probability(mu_norm, sigma_norm, x_norm)
            if prob_type_norm == 'P(X < x)':
                st.write(f"**P(X < {x_norm}):** {prob_less:.4f} or {prob_less*100:.2f}%")
                with st.expander("Show Normal Distribution Formula & Explanation"):
                    st.latex(r''' Z = \frac{x - \mu}{\sigma} ''' )
                    st.markdown(f"""
                        **Explanation:**
                        * We first standardize the value $x$ to a Z-score.
                        * Calculation: ({x_norm} - {mu_norm}) / {sigma_norm} = {z_score:.2f}
                        * The probability is the area under the standard normal curve to the left of the Z-score, which is calculated using the cumulative distribution function (CDF).
                    """)
            else:
                st.write(f"**P(X > {x_norm}):** {prob_greater:.4f} or {prob_greater*100:.2f}%")
                with st.expander("Show Normal Distribution Formula & Explanation"):
                    st.latex(r''' Z = \frac{x - \mu}{\sigma} ''' )
                    st.markdown(f"""
                        **Explanation:**
                        * We first standardize the value $x$ to a Z-score.
                        * Calculation: ({x_norm} - {mu_norm}) / {sigma_norm} = {z_score:.2f}
                        * The probability is the area under the standard normal curve to the right of the Z-score, which is calculated as $1 - \text{{CDF}}(Z)$.
                    """)
    
    st.subheader("Poisson Distribution")
    st.write("Calculate probabilities for a Poisson distribution.")

    st.markdown("---")
    st.subheader("Parameters")
    lambda_poisson = st.number_input("Average Rate of Events ($\lambda$):", min_value=0.0, value=3.0, step=0.1, key="lambda_poisson")

    st.markdown("---")
    st.subheader("Probability Calculation")
    k_poisson = st.number_input("Number of Events (k):", min_value=0, value=2, key="k_poisson")
    calculation_type_poisson = st.radio(
        "Select Probability Type",
        ('P(X = k)', 'P(X ≤ k)', 'P(X > k)'),
        key="poisson_calc_type"
    )
    
    if st.button("Calculate Poisson Probability"):
        if lambda_poisson <= 0 and k_poisson > 0:
            st.warning("With an average rate of 0, the probability of more than 0 events is 0.")
        
        if calculation_type_poisson == 'P(X = k)':
            prob = calculate_poisson_probability(lambda_poisson, k_poisson)
            st.write(f"**P(X = {k_poisson}):** {prob:.4f} or {prob*100:.2f}%")
            with st.expander("Show P(X=k) Formula & Explanation"):
                st.latex(r''' P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!} ''' )
                st.markdown(f"""
                    **Explanation:**
                    * The Poisson distribution models the number of events in a fixed interval of time or space.
                    * The formula uses the average rate ($\lambda$) and the number of events ($k$).
                    * Calculation: ({lambda_poisson:.2f}^{k_poisson} × e^{{-lambda_poisson:.2f}}) / {math.factorial(k_poisson)}! = {prob:.4f}
                """)
        
        elif calculation_type_poisson == 'P(X ≤ k)':
            cumulative_prob = 0
            for k in range(k_poisson + 1):
                cumulative_prob += calculate_poisson_probability(lambda_poisson, k)
            st.write(f"**P(X ≤ {k_poisson}):** {cumulative_prob:.4f} or {cumulative_prob*100:.2f}%")

        elif calculation_type_poisson == 'P(X > k)':
            if k_poisson < 0:
                st.error("The number of events k cannot be negative.")
            else:
                cumulative_prob_less_than_or_equal_to_k = 0
                for k in range(k_poisson + 1):
                    cumulative_prob_less_than_or_equal_to_k += calculate_poisson_probability(lambda_poisson, k)
                prob_greater = 1 - cumulative_prob_less_than_or_equal_to_k
                st.write(f"**P(X > {k_poisson}):** {prob_greater:.4f} or {prob_greater*100:.2f}%")
    
    st.markdown("---")
    st.subheader("Mean and Variance")
    if st.button("Show Mean and Variance", key="poisson_stats_btn"):
        st.write(f"**Mean ($\mu$):** {lambda_poisson:.2f}")
        st.write(f"**Variance ($\sigma^2$):** {lambda_poisson:.2f}")
        with st.expander("Show Mean & Variance Formulas"):
            st.latex(r''' \mu = \lambda ''' )
            st.latex(r''' \sigma^2 = \lambda ''' )
            st.markdown(f"""
                **Explanation:**
                * For a Poisson distribution, the mean and the variance are both equal to the average rate of events ($\lambda$).
            """)

def research_notes_page():
    st.header("📝 Research Notes")
    if st.session_state.user_id == "guest":
        st.info("You must be logged in to create and manage notes.")
        return
        
    st.write("Here you can create and manage your research notes.")
    
    # Input for new notes
    st.subheader("Create a New Note")
    note_title = st.text_input("Note Title")
    note_content = st.text_area("Note Content", height=200)
    
    if st.button("Save Note"):
        if note_title and note_content:
            save_note(note_title, note_content)
            st.success("Note saved successfully!")
        else:
            st.warning("Please enter both a title and content for your note.")

    st.markdown("---")

    # Display existing notes
    st.subheader("Your Saved Notes")
    notes = get_notes()
    if notes:
        for note in notes:
            note_id, title, content, timestamp = note
            with st.expander(f"**{title}** - *{timestamp.split('T')[0]}*"):
                st.markdown(content)
                if st.button(f"Delete Note", key=f"delete_note_{note_id}"):
                    delete_note(note_id)
                    st.success("Note deleted.")
                    st.experimental_rerun()
    else:
        st.info("No notes found. Start by creating a new one above.")

def community_health_notes_page():
    st.header("📚 Community Health Notes")
    st.write("Upload, search, and preview community health notes.")

    # Upload Form
    is_guest = st.session_state.user_id == "guest"
    if is_guest:
        st.info("You must be logged in to upload documents.")
        
    with st.expander("➕ Upload New Document"):
        title = st.text_input("Title", disabled=is_guest)
        authors = st.text_input("Authors", disabled=is_guest)
        year = st.number_input("Year", min_value=1900, max_value=datetime.datetime.now().year, value=datetime.datetime.now().year, disabled=is_guest)
        study_design = st.selectbox("Study Design", ["Cohort", "Case-Control", "Cross-Sectional", "RCT", "Other"], disabled=is_guest)
        keywords = st.text_input("Keywords (comma-separated)", disabled=is_guest)
        abstract = st.text_area("Abstract", height=150, disabled=is_guest)
        file = st.file_uploader("Upload File (PDF or Word)", type=["pdf", "docx"], disabled=is_guest)
        
        if st.button("Save Document", disabled=is_guest):
            if title and file:
                file_data = file.read()
                save_research(title, authors, year, study_design, keywords, abstract, file.name, file_data, st.session_state.user_id)
                st.success("✅ Document saved successfully!")
            else:
                st.warning("⚠️ Title and file are required.")

    st.markdown("---")

    # Search & Filters
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        search_query = st.text_input("🔍 Search (title, author, keyword)")
    with col2:
        author_filter = st.text_input("Filter by Author")
    with col3:
        year_filter = st.number_input("Filter by Year", min_value=0, max_value=datetime.datetime.now().year, step=1)

    if year_filter == 0:
        year_filter = None

    # Results
    results = get_all_research(search_query, author_filter, year_filter)

    if results:
        for res in results:
            rid, title, authors, year, design, keywords, abstract, file_name, file_data, ts = res
            st.subheader(f"{title} ({year})")
            st.markdown(f"**Authors:** {authors}")
            st.markdown(f"**Design:** {design}")
            st.markdown(f"**Keywords:** {keywords}")
            st.markdown(f"**Abstract:** {abstract}")

            # File preview if PDF
            if file_name.lower().endswith(".pdf"):
                st.markdown("**Preview:**")
                pdf_b64 = base64.b64encode(file_data).decode("utf-8")
                pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_b64}" width="700" height="400" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)

            # Download link
            st.markdown(download_file_button(file_name, file_data), unsafe_allow_html=True)

            # Delete button (only for logged-in users, not guests)
            if not is_guest:
                if st.button("🗑️ Delete", key=f"del_{rid}"):
                    delete_research(rid)
                    st.success("Deleted successfully.")
                    st.experimental_rerun()

            st.markdown("---")
    else:
        st.info("No documents found. Start by uploading one above.")

# ------------------ MAIN APP LOGIC ------------------

def main():
    st.set_page_config(page_title="Biostats & Epidemiology App", layout="wide", initial_sidebar_state="collapsed")
    init_db()

    if "user_id" not in st.session_state:
        st.session_state.user_id = None
        st.session_state.page = "Home"

    # Login/Register/Continue
    if not st.session_state.user_id:
        st.sidebar.title("Login / Register")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Login"):
                user_id = authenticate_user(username, password)
                if user_id:
                    st.session_state.user_id = user_id
                    st.session_state.page = "Home"
                    st.success("Logged in successfully!")
                    st.experimental_rerun()
                else:
                    st.error("Invalid credentials.")
        with col2:
            if st.button("Register"):
                if create_user(username, password):
                    st.success("User registered successfully! Please log in.")
                else:
                    st.error("Username already exists.")

        st.sidebar.markdown("---")
        if st.sidebar.button("Continue to App"):
            st.session_state.user_id = "guest"
            st.session_state.page = "Home"
            st.success("Continuing as a guest. Note: Some features like saving notes and uploading research are disabled.")
            st.experimental_rerun()
    else:
        st.sidebar.title("Navigation")
        page_from_sidebar = st.sidebar.radio("Go to", [
            "Home",
            "Descriptive Statistics",
            "Epidemiological Measures",
            "Hypothesis Testing",
            "Probability Distributions",
            "Research Notes",
            "Community Health Notes"
        ], index=["Home", "Descriptive Statistics", "Epidemiological Measures", "Hypothesis Testing", "Probability Distributions", "Research Notes", "Community Health Notes"].index(st.session_state.page))

        # This line is changed to fix the double-click issue
        st.session_state.page = page_from_sidebar
        
        # Header with search bar
        st.title("Biostatistics & Epidemiology Tool")
        
        if st.session_state.page == "Home":
            home_page()
        elif st.session_state.page == "Descriptive Statistics":
            descriptive_stats_page()
        elif st.session_state.page == "Epidemiological Measures":
            epidemiological_measures_page()
        elif st.session_state.page == "Hypothesis Testing":
            hypothesis_testing_page()
        elif st.session_state.page == "Probability Distributions":
            probability_distributions_page()
        elif st.session_state.page == "Research Notes":
            research_notes_page()
        elif st.session_state.page == "Community Health Notes":
            community_health_notes_page()

        if st.sidebar.button("Logout"):
            st.session_state.user_id = None
            st.experimental_rerun()


if __name__ == "__main__":
    main()