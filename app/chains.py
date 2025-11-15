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
                    # Get job details with fallbacks
                    title = job.get('title') or job.get('role') or 'Untitled Role'
                    skills = job.get('skills', [])
                    
                    # Handle skills formatting
                    if isinstance(skills, str):
                        skills = [s.strip() for s in skills.split(',') if s.strip()]
                    elif not isinstance(skills, list):
                        skills = []
                    
                    # Only query portfolio if skills exist
                    if skills:
                        links = portfolio.query_links(skills)
                    else:
                        links = []
                        st.warning(f"⚠️ No skills found for {title}")
                    
                    email = llm.write_mail(job, links)

                    st.subheader(f"💼 {title}")
                    st.markdown(f"**Skills Required:** {', '.join(skills) if skills else 'N/A'}")
                    st.code(email, language='markdown')

                    email_records.append({
                        "Job Title": title,
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
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Please set it in .env file or environment variables.")
        
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            groq_api_key=api_key
        )

    def extract_jobs(self,cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### Scraped text from website:
            {page_data} 
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: 'role', 'experience', 'skills' and 'description'.
            For skills, provide them as a comma-separated string.
            Only return the valid JSON.
            ### NO PREAMBLE only give me the json object
            """
        )       

        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke({"page_data": cleaned_text})

        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big...")
        return res if isinstance(res,list) else [res]
    
    def write_mail(self,job,links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION
            {job_description}

            ### INSTRUCTION:
            You are Manideep, a business development executive at Anarch. Anarch is an AI and software solution company focused on seamless integration of business processes through automated tools. Over our experience, we have empowered numerous enterprises with tailored solutions, process optimization, cost reduction and heightened overall efficiency.
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability in fulfilling their needs.
            Also showcase the most relevant ones from the following links from Anarch's portfolio: {link_list}
            Remember you are Manideep, BDE at Anarch.
            Do not provide a preamble.

            ### NO PREAMBLE only give me the email
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description":str(job),"link_list":links})
        return res.content
    
if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))

