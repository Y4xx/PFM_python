import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import re
from datetime import datetime
import hashlib
import time
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import textwrap
import uuid  # For generating unique keys

# Enhanced page configuration with light theme
st.set_page_config(
    page_title="Research Papers Explorer",
    layout="wide",
    page_icon="🔬",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern light design
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --accent-color: #F18F01;
        --background-color: #FFFFFF;
        --surface-color: #F8F9FA;
        --text-primary: #2C3E50;
        --text-secondary: #5D6D7E;
        --border-color: #E5E8EB;
        --success-color: #27AE60;
        --warning-color: #F39C12;
        --error-color: #E74C3C;
    }
    
    /* Force light mode */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-primary);
        width: 100%;
        text-color: black;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(46, 134, 171, 0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--surface-color);
        border-right: 1px solid var(--border-color);
    }
    
    .sidebar-header {
        color: white;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        display: flex;
        height: 100%;
        
    }
    
    .sidebar-header h2 {
        margin: 0;
        font-size: 2rem;
        font-weight: 600;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 4px solid var(--primary-color);
    }
    
    .user-message {
        background: linear-gradient(135deg, #EBF3FD, #F0F7FF);
        border-left-color: var(--primary-color);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #F8F9FA, #FFFFFF);
        border-left-color: var(--secondary-color);
    }
    
    /* Statistics cards */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 15px rgba(0,0,0,0.08);
        border: 1px solid var(--border-color);
        margin: 0.5rem 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 25px rgba(0,0,0,0.12);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0;
    }
    
    .stat-label {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(46, 134, 171, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(46, 134, 171, 0.4);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid var(--border-color);
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(46, 134, 171, 0.1);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div > select {
        border-radius: 10px;
        border: 2px solid var(--border-color);
        padding: 0.75rem 1rem;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background-color: var(--primary-color);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: var(--surface-color);
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }
    
    /* Metric styling */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid var(--border-color);
        text-align: center;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: var(--primary-color);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: rgba(39, 174, 96, 0.1);
        border: 1px solid var(--success-color);
        border-radius: 8px;
    }
    
    .stError {
        background-color: rgba(231, 76, 60, 0.1);
        border: 1px solid var(--error-color);
        border-radius: 8px;
    }
    
    .stWarning {
        background-color: rgba(243, 156, 18, 0.1);
        border: 1px solid var(--warning-color);
        border-radius: 8px;
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 15px rgba(0,0,0,0.08);
        border: 1px solid var(--border-color);
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        background: var(--surface-color);
        padding: 2rem;
        border-radius: 10px;
        margin-top: 3rem;
        text-align: center;
        border: 1px solid var(--border-color);
    }
    
    /* Article cards */
    .article-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 15px rgba(0,0,0,0.08);
        border: 1px solid var(--border-color);
        margin: 1rem 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .article-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 25px rgba(0,0,0,0.12);
    }
    
    .article-title {
        color: var(--primary-color);
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .article-meta {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .stat-number {
            font-size: 2rem;
        }
        
        .chat-message {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Download necessary resources
try:
    nltk.download('stopwords', quiet=True)
    STOPWORDS = set(stopwords.words('english'))
except:
    STOPWORDS = set()

# Load resources
@st.cache_resource(ttl=3600)
def load_resources():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Load FAISS index
    faiss_path = os.path.join(BASE_DIR, "faiss_index.bin")
    index = faiss.read_index(faiss_path)
    
    # Load article IDs
    ids_path = os.path.join(BASE_DIR, "article_ids.pkl")
    with open(ids_path, "rb") as f:
        article_ids = pickle.load(f)
    
    # Load language model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load authors list
    try:
        conn = sqlite3.connect(os.path.join(BASE_DIR, "arxiv_database.db"))
        authors_df = pd.read_sql_query("SELECT DISTINCT author_name FROM authors", conn)
        conn.close()
        authors = authors_df['author_name'].dropna().astype(str).tolist()
    except:
        authors = []
    
    return index, article_ids, model, authors

index, article_ids, model, authors = load_resources()

# Create new SQLite connection
def get_db_connection():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "arxiv_database.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return conn

# Cache for searches
search_cache = {}

def cached_semantic_search(query, filters, k=100):
    """Search with query signature-based caching"""
    query_signature = hashlib.md5((query + str(filters)).encode()).hexdigest()
    if query_signature in search_cache:
        return search_cache[query_signature]
    
    # New search
    query_embedding = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_embedding, k)
    result_ids = [article_ids[i] for i in indices[0]]
    scores = distances[0]
    
    # Retrieve article details
    conn = get_db_connection()
    placeholders = ', '.join(['?'] * len(result_ids))
    query_sql = f"""
        SELECT article_id, title, authors, abstract, year, categories, doi
        FROM articles
        WHERE article_id IN ({placeholders})
    """
    results_df = pd.read_sql_query(query_sql, conn, params=result_ids)
    conn.close()
    
    # Add relevance scores
    scores_df = pd.DataFrame({
        'article_id': result_ids,
        'similarity_score': scores
    })
    
    results_df = pd.merge(
        results_df, 
        scores_df, 
        on='article_id', 
        how='inner'
    ).sort_values('similarity_score', ascending=False)
    
    # Apply filters
    year_range = filters.get('year_range')
    if year_range:
        results_df = results_df[
            (results_df['year'] >= year_range[0]) & 
            (results_df['year'] <= year_range[1])
        ]
    
    domain = filters.get('domain')
    if domain:
        results_df = results_df[
            results_df['categories'].str.contains(domain, na=False)
        ]
    
    search_cache[query_signature] = results_df
    return results_df

# Advanced contextual analysis
def contextual_query_understanding(query):
    """Understand user intent and extract key entities"""
    # Check cache first
    if "nlp_analysis" in st.session_state and st.session_state.nlp_analysis.get(query):
        return st.session_state.nlp_analysis[query]
    
    # Initialize entities
    entities = {
        "domains": [],
        "time_periods": [],
        "authors": [],
    }
    
    # Time period detection
    current_year = datetime.now().year
    time_patterns = [
        (r"since (\d{4})", lambda m: [int(m.group(1)), current_year]),
        (r"last (\d+) years", lambda m: [current_year - int(m.group(1)), current_year]),
        (r"from (\d{4}) to (\d{4})", lambda m: [int(m.group(1)), int(m.group(2))]),
        (r"(\d{4})-(\d{4})", lambda m: [int(m.group(1)), int(m.group(2))]),
        (r"in (\d{4})", lambda m: [int(m.group(1))]),
    ]
    
    for pattern, handler in time_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            entities["time_periods"] = handler(match)
            break
    
    # Author detection
    for author in authors:
        if author.lower() in query.lower():
            entities["authors"].append(author)
    
    # Domain detection
    if "cs." in query.lower():
        for token in query.split():
            if token.lower().startswith("cs."):
                entities["domains"].append(token)
    
    # Intent detection
    intent = "articles"
    intent_keywords = {
        "authors": ["researcher", "scientist", "professor", "expert", "author"],
        "trends": ["trend", "evolution", "growth", "over time", "publication pattern"],
        "collaborations": ["collaborator", "partner", "co-author", "work with", "network"],
        "topics": ["topic", "theme", "research area", "focus", "subject"],
        "articles": ["paper", "article", "publication", "study", "research"]
    }
    
    # Determine intent based on keywords
    for intent_type, keywords in intent_keywords.items():
        if any(keyword in query.lower() for keyword in keywords):
            intent = intent_type
            break
    
    analysis = {
        "intent": intent,
        "entities": entities,
        "original_query": query
    }
    
    # Cache analysis
    if "nlp_analysis" not in st.session_state:
        st.session_state.nlp_analysis = {}
    st.session_state.nlp_analysis[query] = analysis
    
    return analysis

# Research functions
def get_top_authors(domain=None, year_range=None, min_papers=3, limit=10):
    """Get top authors with minimum paper threshold"""
    conn = get_db_connection()
    try:
        conditions = []
        params = []
        
        if domain:
            conditions.append("a.categories LIKE ?")
            params.append(f'%{domain}%')
        
        if year_range:
            conditions.append("a.year BETWEEN ? AND ?")
            params.extend([year_range[0], year_range[1]])
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
            SELECT au.author_name, COUNT(DISTINCT aa.article_id) as paper_count
            FROM article_author aa
            JOIN authors au ON aa.author_id = au.author_id
            JOIN articles a ON aa.article_id = a.article_id
            {where_clause}
            GROUP BY au.author_name
            HAVING paper_count >= ?
            ORDER BY paper_count DESC
            LIMIT ?
        """
        params.extend([min_papers, limit])
        
        return pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_publication_trends(domain=None, start_year=2020, end_year=None):
    """Get accurate publication trends"""
    if end_year is None:
        end_year = datetime.now().year
        
    conn = get_db_connection()
    try:
        # Create complete year range
        all_years = pd.DataFrame({'year': range(start_year, end_year + 1)})
        
        conditions = ["year BETWEEN ? AND ?"]
        params = [start_year, end_year]
        
        if domain:
            conditions.append("categories LIKE ?")
            params.append(f'%{domain}%')
        
        where_clause = "WHERE " + " AND ".join(conditions)
        
        query = f"""
            SELECT year, COUNT(*) as paper_count
            FROM articles
            {where_clause}
            GROUP BY year
        """
        
        results = pd.read_sql_query(query, conn, params=params)
        
        # Merge with complete year range
        results = pd.merge(all_years, results, on='year', how='left').fillna(0)
        return results.sort_values('year')
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_collaborators(author_name, domain=None):
    """Get collaborators for an author"""
    conn = get_db_connection()
    try:
        # Find closest matching author name
        author_lower = author_name.lower()
        best_match = next((a for a in authors if author_lower in a.lower()), author_name)
        
        conditions = ["a1.author_name = ?", "a2.author_name != ?"]
        params = [best_match, best_match]
        
        if domain:
            conditions.append("a.categories LIKE ?")
            params.append(f'%{domain}%')
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
            SELECT a2.author_name, COUNT(*) as collab_count
            FROM article_author aa1
            JOIN article_author aa2 ON aa1.article_id = aa2.article_id
            JOIN authors a1 ON aa1.author_id = a1.author_id
            JOIN authors a2 ON aa2.author_id = a2.author_id
            JOIN articles a ON aa1.article_id = a.article_id
            WHERE {where_clause}
            GROUP BY a2.author_name
            ORDER BY collab_count DESC
            LIMIT 10
        """
        
        return pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

def generate_topic_wordcloud(articles_df):
    """Generate word cloud of research topics"""
    if articles_df.empty:
        return None
        
    text = " ".join(articles_df['title'] + " " + articles_df['abstract'])
    if not text.strip():
        return None
        
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
        stopwords=STOPWORDS, 
        max_words=100,
        colormap='viridis'
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    fig.patch.set_facecolor('white')
    return fig

# Enhanced plotting functions with better styling
def create_styled_bar_chart(data, x_col, y_col, title, color_col=None):
    """Create a styled bar chart with modern design"""
    fig = px.bar(
        data, 
        x=x_col, 
        y=y_col,
        title=title,
        color=color_col if color_col else y_col,
        color_continuous_scale='Blues',
        template='plotly_white'
    )
    
    fig.update_layout(
        title_font_size=20,
        title_font_color='#2C3E50',
        font_family='Arial, sans-serif',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.update_traces(
        marker_line_color='rgba(46, 134, 171, 0.8)',
        marker_line_width=1.5,
        hovertemplate='<b>%{x}</b><br>%{y}<extra></extra>'
    )
    
    return fig

def create_styled_line_chart(data, x_col, y_col, title):
    """Create a styled line chart with modern design"""
    fig = px.line(
        data, 
        x=x_col, 
        y=y_col,
        title=title,
        markers=True,
        template='plotly_white'
    )
    
    fig.update_layout(
        title_font_size=20,
        title_font_color='#2C3E50',
        font_family='Arial, sans-serif',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.update_traces(
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=8, color='#A23B72'),
        hovertemplate='<b>%{x}</b><br>%{y}<extra></extra>'
    )
    
    return fig

# Generate intelligent responses
def generate_intelligent_response(analysis, results):
    """Generate contextual responses in natural language"""
    response = []
    visualizations = []
    query = analysis["original_query"]
    entities = analysis["entities"]
    
    # Custom response based on intent
    if analysis["intent"] == "authors":
        domain = entities.get("domains", [None])[0] if entities.get("domains") else None
        
        if results.empty:
            response.append("🔍 **No researchers found**")
            response.append("💡 Try broadening your criteria or checking the spelling.")
        else:
            domain_info = f" in {domain}" if domain else ""
            response.append(f"👤 **Top researchers{domain_info}:**")
            
            # Display top authors in cards
            for i, row in results.head(5).iterrows():
                response.append(f"**{i+1}.** {row['author_name']} - **{row['paper_count']} publications**")
            
            # Enhanced bar chart visualization
            if not results.empty:
                top_authors = results.head(10)
                fig = create_styled_bar_chart(
                    top_authors, 
                    'author_name', 
                    'paper_count',
                    f'Top Researchers{domain_info}',
                    'paper_count'
                )
                visualizations.append(("Publication Count", fig))
    
    elif analysis["intent"] == "trends":
        # Determine time period
        time_periods = entities.get('time_periods', [])
        if not time_periods:
            start_year = 2020
            end_year = datetime.now().year
        elif len(time_periods) == 1:
            start_year = time_periods[0]
            end_year = datetime.now().year
        else:
            start_year = min(time_periods)
            end_year = max(time_periods)
        
        domain = entities.get("domains", [None])[0] if entities.get("domains") else None
        
        results = get_publication_trends(domain=domain, start_year=start_year, end_year=end_year)
        
        if results.empty:
            response.append("📊 **No publication trends found**")
            response.append("💡 Try with a different domain or time period.")
        else:
            domain_info = f" in {domain}" if domain else ""
            total_papers = results['paper_count'].sum()
            avg_papers = results['paper_count'].mean()
            
            response.append(f"📈 **Publication trends{domain_info} ({start_year}-{end_year}):**")
            response.append(f"📊 **Total papers:** {total_papers:,}")
            response.append(f"📊 **Average per year:** {avg_papers:.1f}")
            
            # Enhanced line chart visualization
            if not results.empty:
                fig = create_styled_line_chart(
                    results, 
                    'year', 
                    'paper_count',
                    f'Publication Trends{domain_info}'
                )
                visualizations.append(("Publication Trends", fig))
    
    elif analysis["intent"] == "topics":
        domain = entities.get("domains", [None])[0] if entities.get("domains") else None
        
        if results.empty:
            response.append("🔍 **No articles found for topic analysis**")
        else:
            domain_info = f" in {domain}" if domain else ""
            response.append(f"🌐 **Research topics analysis{domain_info}:**")
            response.append(f"📊 **Analyzed {len(results)} articles**")
            
            # Word cloud visualization
            wordcloud_fig = generate_topic_wordcloud(results)
            if wordcloud_fig:
                visualizations.append(("Research Topics", wordcloud_fig))
    
    elif analysis["intent"] == "collaborations":
        author_name = entities.get("authors", [""])[0]
        
        domain = entities.get("domains", [None])[0] if entities.get("domains") else None
        
        if not author_name:
            response.append("🔍 **Author name required**")
            response.append("💡 Example: 'Who collaborates with Yann LeCun?'")
        else:
            collaborators = get_collaborators(author_name, domain=domain)
            
            if collaborators.empty:
                response.append(f"🔍 **No collaborators found for {author_name}**")
                response.append("💡 Try a different spelling or full name.")
            else:
                response.append(f"🤝 **{author_name}'s frequent collaborators:**")
                
                # List collaborators
                for i, row in collaborators.head(5).iterrows():
                    response.append(f"**{i+1}.** {row['author_name']} - **{row['collab_count']} joint papers**")
                
                # Enhanced bar chart visualization
                if not collaborators.empty:
                    fig = create_styled_bar_chart(
                        collaborators, 
                        'author_name', 
                        'collab_count',
                        f'Collaborators of {author_name}',
                        'collab_count'
                    )
                    visualizations.append(("Collaboration Network", fig))
    
    else:  # Default article search
        domain = entities.get("domains", [None])[0] if entities.get("domains") else None
        time_periods = entities.get('time_periods', [])
        
        if results.empty:
            response.append("🔍 **No articles found**")
            response.append("💡 Try broadening your search terms or adding synonyms.")
        else:
            domain_info = f" in {domain}" if domain else ""
            year_info = f" since {min(time_periods)}" if time_periods else ""
            
            response.append(f"📚 **Found {len(results)} relevant articles**{domain_info}{year_info}")
            
            # Display top articles in enhanced cards
            for i, row in results.head(3).iterrows():
                arxiv_link = f"https://arxiv.org/abs/{row['article_id']}"
                
                # Shorten long titles
                title = row['title']
                short_title = title if len(title) <= 80 else f"{title[:77]}..."
                
                response.append("---")
                response.append(f"### 📄 [{short_title}]({arxiv_link})")
                response.append(f"**👥 Authors:** {row['authors']}")
                response.append(f"**📅 Year:** {row['year']} | **🎯 Relevance:** {row['similarity_score']:.3f}")
                
                # DOI link if available
                if row['doi'] and row['doi'] != 'None':
                    doi_link = f"https://doi.org/{row['doi']}"
                    response.append(f"**🔗 DOI:** [Full paper]({doi_link})")
                
                # Abstract with expander
                if row['abstract']:
                    with st.expander("📖 View Abstract", expanded=False):
                        st.markdown(f"<div style='text-align: justify; line-height: 1.6;'>{row['abstract']}</div>", unsafe_allow_html=True)
            
            # Enhanced relevance score visualization
            if len(results) > 1:
                scores_df = results.head(10)[['title', 'similarity_score']].copy()
                scores_df['title_short'] = scores_df['title'].apply(lambda x: x[:50] + "..." if len(x) > 50 else x)
                
                fig = px.bar(
                    scores_df, 
                    x='similarity_score', 
                    y='title_short',
                    orientation='h',
                    title='Most Relevant Articles',
                    color='similarity_score',
                    color_continuous_scale='RdYlBu_r',
                    template='plotly_white'
                )
                
                fig.update_layout(
                    height=600,
                    showlegend=False,
                    title_font_size=20,
                    title_font_color='#2C3E50',
                    font_family='Arial, sans-serif',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=50, r=50, t=80, b=50)
                )
                
                fig.update_traces(
                    hovertemplate='<b>%{y}</b><br>Relevance: %{x:.3f}<extra></extra>'
                )
                
                visualizations.append(("Article Relevance", fig))

    return response, visualizations

# Enhanced UI Functions
def display_database_stats():
    """Display database statistics in a modern card layout"""
    try:
        conn = get_db_connection()
        
        # Get total articles
        total_articles = pd.read_sql_query("SELECT COUNT(*) as count FROM articles", conn)['count'].iloc[0]
        
        # Get total authors
        total_authors = pd.read_sql_query("SELECT COUNT(DISTINCT author_name) as count FROM authors", conn)['count'].iloc[0]
        
        # Get year range
        year_range = pd.read_sql_query("SELECT MIN(year) as min_year, MAX(year) as max_year FROM articles", conn)
        min_year = year_range['min_year'].iloc[0]
        max_year = year_range['max_year'].iloc[0]
        
        # Get most popular category
        category_stats = pd.read_sql_query("""
            SELECT categories, COUNT(*) as count 
            FROM articles 
            WHERE categories IS NOT NULL 
            GROUP BY categories 
            ORDER BY count DESC 
            LIMIT 1
        """, conn)
        
        conn.close()
        
        # Display stats in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{total_articles:,}</div>
                <div class="stat-label">Total Articles</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{total_authors:,}</div>
                <div class="stat-label">Unique Authors</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{max_year - min_year + 1}</div>
                <div class="stat-label">Years Covered</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            years_span = f"{min_year}-{max_year}"
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number" style="font-size: 1.5rem;">{years_span}</div>
                <div class="stat-label">Time Range</div>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error loading database statistics: {str(e)}")

def display_search_suggestions():
    """Display search suggestions with examples"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #F8F9FA, #E9ECEF); padding: 1.5rem; border-radius: 12px; margin: 1rem 0; border: 1px solid #E5E8EB;">
        <h4 style="color: #2C3E50; margin-bottom: 1rem;">💡 Search Examples</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem;">
            <div>
                <strong>🔍 Article Search:</strong><br>
                <em>"machine learning transformers"</em><br>
                <em>"computer vision attention mechanism"</em>
            </div>
            <div>
                <strong>👤 Author Research:</strong><br>
                <em>"top researchers in NLP"</em><br>
                <em>"who are the experts in computer vision"</em>
            </div>
            <div>
                <strong>📊 Trends Analysis:</strong><br>
                <em>"publication trends in AI since 2020"</em><br>
                <em>"research growth in machine learning"</em>
            </div>
            <div>
                <strong>🤝 Collaborations:</strong><br>
                <em>"who collaborates with Geoffrey Hinton"</em><br>
                <em>"research partners in deep learning"</em>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# User interface
# st.markdown("""
# <div class="main-header">
#     <h1>📚 Research Papers Explorer</h1>
#     <p>Your intelligent companion for scientific research exploration</p>
# </div>
# """, unsafe_allow_html=True)

# Display database statistics
display_database_stats()

# Initialize conversation history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.nlp_analysis = {}  # Cache for NLP analysis

# Enhanced sidebar with better styling
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2>Search Filters</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Year filter with enhanced styling
    st.markdown("""
    <h3 style="color: white;">Publication Period</h3>
    """, unsafe_allow_html=True)
    
    min_year, max_year = 2020, datetime.now().year
    selected_years = st.slider(
        "Select year range", 
        min_value=min_year, 
        max_value=max_year, 
        value=(min_year, max_year),
        help="Filter articles by publication year"
    )
    
    # Domain filter with enhanced styling
    st.markdown("<h3 style='color: white;'>Research Domain</h3>", unsafe_allow_html=True)
    try:
        conn = get_db_connection()
        categories = pd.read_sql_query("SELECT DISTINCT categories FROM articles", conn)['categories']
        conn.close()
        
        all_categories = set()
        for cat_list in categories:
            if cat_list:
                all_categories.update(cat_list.split())
        
        cs_categories = sorted([cat for cat in all_categories if cat.startswith('cs.')])
        
        selected_category = st.selectbox(
            "Choose a domain", 
            ["All domains"] + cs_categories,
            help="Filter by specific research domain"
        )
        
        if selected_category == "All domains":
            selected_category = None
    except:
        selected_category = None
    
    # Advanced options
    st.markdown("<h3 style='color: white;'>Advanced Options</h3>", unsafe_allow_html=True)
    max_results = st.slider(
        "Max results", 
        min_value=10, 
        max_value=100, 
        value=50,
        help="Maximum number of results to return"
    )
    
    # # Clear cache button
    # if st.button("🗑️ Clear Cache", help="Clear search cache for fresh results"):
    #     search_cache.clear()
    #     st.session_state.nlp_analysis = {}
    #     st.success("Cache cleared successfully!")
    
    # Quick stats in sidebar
    st.markdown("<h3 style='color: white;'>Quick Stats</h3>", unsafe_allow_html=True)
    try:
        conn = get_db_connection()
        recent_papers = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM articles WHERE year >= 2023", 
            conn
        )['count'].iloc[0]
        conn.close()
        
        st.metric("Recent Papers (2023+)", f"{recent_papers:,}")
    except:
        pass

# Display search suggestions if no conversation yet
if not st.session_state.chat_history:
    display_search_suggestions()

# Display conversation history with enhanced styling
for msg_idx, msg in enumerate(st.session_state.chat_history):
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.2rem; margin-right: 0.5rem;">🧑‍💻</span>
                <strong style="color: #2E86AB;">You asked:</strong>
            </div>
            <div style="font-size: 1.1rem;">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <span style="font-size: 1.2rem; margin-right: 0.5rem;">🔬</span>
                <strong style="color: #A23B72;">Research Assistant:</strong>
            </div>
            <div style="line-height: 1.6;">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display visualizations with enhanced containers
        if "visualizations" in msg:
            for viz_idx, (title, viz) in enumerate(msg["visualizations"]):
                # Generate unique key for each visualization
                unique_key = f"viz_{msg_idx}_{viz_idx}_{uuid.uuid4().hex[:8]}"
                
                st.markdown(f"""
                <div class="chart-container">
                    <h4 style="color: #2C3E50; margin-bottom: 1rem;">📊 {title}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                if isinstance(viz, plt.Figure):
                    st.pyplot(viz, use_container_width=True)
                elif hasattr(viz, 'show'):
                    st.plotly_chart(viz, use_container_width=True, key=unique_key)

# Enhanced chat input with better styling
st.markdown("""
<div style="margin: 2rem 0 1rem 0;">
    <h3 style="color: #2C3E50; margin-bottom: 1rem;">💬 Ask your research question</h3>
</div>
""", unsafe_allow_html=True)

# Handle user input
if prompt := st.chat_input("Search for papers, authors, trends, or collaborations...", key="main_input"):
    # Add question to history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.spinner("🔍 Analyzing your query and searching the database..."):
        start_time = time.time()
        
        # Advanced contextual analysis
        analysis = contextual_query_understanding(prompt)
        
        # Apply filters
        filters = {
            "year_range": selected_years,
            "domain": selected_category,
        }
        
        # Handle different intents
        results = pd.DataFrame()
        visualizations = []
        
        try:
            if analysis["intent"] == "authors":
                results = get_top_authors(
                    domain=filters['domain'],
                    year_range=selected_years,
                    limit=max_results
                )
                
            elif analysis["intent"] == "trends":
                # Use extracted dates or default filters
                time_periods = analysis['entities'].get('time_periods', [])
                if not time_periods:
                    start_year = selected_years[0]
                    end_year = selected_years[1]
                elif len(time_periods) == 1:
                    start_year = time_periods[0]
                    end_year = selected_years[1]
                else:
                    start_year = min(time_periods)
                    end_year = max(time_periods)
                
                results = get_publication_trends(
                    domain=filters['domain'],
                    start_year=start_year,
                    end_year=end_year
                )
                
            elif analysis["intent"] == "topics":
                results = cached_semantic_search(analysis["original_query"], filters, k=max_results)
                
            elif analysis["intent"] == "collaborations":
                # Don't do semantic search for collaborators
                results = pd.DataFrame()
                
            else:  # Default article search
                results = cached_semantic_search(prompt, filters, k=max_results)
                
        except Exception as e:
            st.error(f"❌ Search error: {str(e)}")
        
        # Generate response
        try:
            response_lines, visualizations = generate_intelligent_response(analysis, results)
            response_text = "\n\n".join(response_lines)
        except Exception as e:
            response_text = f"⚠️ Error generating response: {str(e)}"
            visualizations = []
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Add performance metrics
        performance_info = f"""
        
        ⏱️ **Performance:** Response generated in {response_time:.2f}s  
        🔍 **Query Type:** {analysis['intent'].title()}  
        📊 **Results Found:** {len(results) if not results.empty else 0}
        """
        
        response_text += performance_info
        
        # Create assistant message
        assistant_msg = {
            "role": "assistant", 
            "content": response_text,
            "visualizations": visualizations
        }
        
        # Add to history
        st.session_state.chat_history.append(assistant_msg)
        
        # Show success message
        st.success("✅ Query processed successfully!")
        
        # Reload interface
        st.rerun()

# Enhanced footer
st.markdown("""
<div class="footer">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h4 style="color: #2C3E50; margin: 0;">📚 Research Papers Explorer</h4>
            <p style="color: #5D6D7E; margin: 0.5rem 0 0 0;">
                Powered by advanced NLP and semantic search technology
            </p>
        </div>
        <div style="text-align: right;">
            <p style="color: #5D6D7E; margin: 0; font-size: 0.9rem;">
                © 2025 Research Assistant<br>
                <strong>Yassine OUJAMA & Yassir M'SAAD</strong>
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
