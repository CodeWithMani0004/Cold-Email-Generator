import streamlit as st

# ✅ MUST be first Streamlit command
st.set_page_config(
    layout="wide",
    page_icon="🤑",
    page_title="Email Generator"
)

import pandas as pd
from langchain_community.document_loaders import WebBaseLoader
from chains import Chain
from portfolio import Portfolio

# --- Custom CSS for style ---
st.markdown("""
    <style>
        .main { background-color: #f7f7f7; }
        h1 { color: #ff4b4b; }
        .stTextInput > div > div > input {
            border: 2px solid #ff4b4b;
            border-radius: 8px;
        }
        .stButton > button {
            background-color: #ff4b4b;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
        }
        .stButton > button:hover {
            background-color: #d63c3c;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

def create_stream_app(llm, portfolio):
    st.title("🚀 Cold Email Generator")
    st.markdown("**Enter a job listing URL and let the AI craft tailored cold emails.**")

    col1, col2 = st.columns([3, 1])
    with col1:
        url_input = st.text_input("🌐 Job Listing URL:", 
            value="https://jobdetails.nestle.com/job/Esplugues-Llobregat-Technology-Expert-R&D-Information-Technology-B-08950/1204832601/?feedId=256801"
        )
    with col2:
        submit_button = st.button("🔥 Generate Emails")

    if submit_button:
        with st.spinner("🧠 Thinking... extracting jobs & crafting emails..."):
            try:
                loader = WebBaseLoader([url_input])
                data = loader.load().pop().page_content
                portfolio.load_portfolio()
                jobs = llm.extract_jobs(data)

                email_records = []

                for job in jobs:
                    skills = job.get('skills', [])
                    links = portfolio.query_links(skills)
                    email = llm.write_mail(job, links)

                    st.subheader(f"💼 {job.get('title', 'Untitled Role')}")
                    st.markdown(f"**Skills Required:** {', '.join(skills) if skills else 'N/A'}")
                    st.code(email, language='markdown')

                    email_records.append({
                        "Job Title": job.get('title', ''),
                        "Skills": ", ".join(skills),
                        "Email": email
                    })

                if email_records:
                    df = pd.DataFrame(email_records)
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download All Emails as CSV",
                        data=csv,
                        file_name="generated_emails.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"❌ An Error Occurred: {e}")
                st.info("💡 Please check if the URL is accessible and contains job information.")

if __name__ == "__main__":
    try:
        chain = Chain()
        portfolio = Portfolio()
        create_stream_app(chain, portfolio)
    except Exception as e:
        st.error(f"❌ Failed to initialize application: {e}")
        st.info("💡 Please check if all required files are present and properly configured.")
