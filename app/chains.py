import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.1-8b-instant",temperature=0,groq_api_key="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")

    def extract_jobs(self,cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### Scrap text from website:
            {page_data} 
            ###INSTRCUTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job boostings and return them adjacent form it containing the following keys: 'role', 'experience', 'skills' and 'description'.
            Only return the valid JSON.
            ### NO PREAMBLE only give me the json object
            """
        )       

        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke({cleaned_text})

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
            You are Manideep, a business development executive at Anarch. Anarch is an ai and software Solution company is Seamless integration of business processes through automated tools Over our experience. We have overpowered numerous enterprises With tailored solutions Process optimization Cost reduction and heightened overall efficiency.
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability in fulfilling their needs.
            Also at the most relevant Ones from the following links Showcase Anarch's portfolio: {link_list}
            Remember you are manideep BDE at Anarch.
            do not provide a preamble.

            ### NO PREAMBLE only give me the email
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description":str(job),"link_list":links})
        return res.content
    
if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))