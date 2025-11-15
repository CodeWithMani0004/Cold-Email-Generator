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