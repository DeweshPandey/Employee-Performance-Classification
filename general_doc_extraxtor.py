import os
from dotenv import load_dotenv


load_dotenv()

from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# need to use metadata

INDEX_NAME = "resume-vec-index"

def run_llm(query: str, metadata_filter: dict = None):
    
    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
    
    docsearch = PineconeVectorStore( index_name = INDEX_NAME, embedding = embeddings)
    
    retriever  = docsearch.as_retriever(  search_kwargs={'filter': metadata_filter} )
    
    
    # retrieval using LCEL chain 
    # query ="""You are an expert in analyzing resumes specifically for sales officer roles.
    # Extract all relevant resume details such as  Summary, Education, Work Experience, Skills, Certifications, Projects, and Languages.
    # Use the following format to provide your output. 
    # If any specific information is not available, return "Not Available" for that field. Include a relevance score (0 to 1) for each extracted section based on the embedding match confidence.

    # Format:
    # {
    #     "Summary": "<Extracted Summary>",
    #     "Education": "<Extracted Education Details or 'Not Available'>",
    #     "Work Experience": "<Extracted Work Experience Details or 'Not Available'>",
    #     "Skills": "<Extracted Skills or 'Not Available'>",
    #     "Certifications": "<Extracted Certifications or 'Not Available'>",
    #     "Projects": "<Extracted Projects or 'Not Available'>",
    #     "Languages": "<Extracted Languages or 'Not Available'>"
    # }

    # Instructions:
    # 1. Retrieve the most relevant information from the resume document(s) based on the query.
    # 2. Organize the output strictly in the given format.
    # 3. Assign a relevance score (0 to 1) for each field based on how well it matches the query embeddings.
    # 4. Provide a brief explanation for low scores (<0.5) if applicable.
    # 5. Use concise language, and maintain consistent structure.


    # """
    # metadata_filter =  {"source":{"$eq": "EMP0001"} }
    # relevant_docs = docsearch.similarity_search(query= query,  filter= metadata_filter , k=20)
    # chain = retrieval_qa_chat_prompt | chat 
    # res=chain.invoke(input = { 'context' : relevant_docs , 'input': query })
    # print(res.content)
        
    
    chat = ChatOpenAI( model = "gpt-4o" ,verbose = True , temperature = 0 )
    
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat" )
    # ans= retrieval_qa_chat_prompt.invoke(input = { 'context' : relevant_docs , 'input': query })
    

    stuff_documents_chain = create_stuff_documents_chain( chat, retrieval_qa_chat_prompt)
    
    qa = create_retrieval_chain(
        retriever = retriever, combine_docs_chain = stuff_documents_chain
    )
    
    result = qa.invoke(input = {"input" : query})
    
    new_result = {
        "query" : result["input"],
        "result" : result["answer"],
        "source_documents" : result["context"]
    }
    
    return new_result

if __name__ == "__main__":
    
    query ="""You are an expert in analyzing resumes specifically for sales officer roles.
    Extract all relevant resume details such as  Summary, Education, Work Experience, Skills, Certifications, Projects, and Languages.
    Use the following format to provide your output. 
    If any specific information is not available, return "Not Available" for that field. Include a relevance score (0 to 1) for each extracted section based on the embedding match confidence.

    Format:
    {
        "Summary": "<Extracted Summary>",
        "Education": "<Extracted Education Details or 'Not Available'>",
        "Work Experience": "<Extracted Work Experience Details or 'Not Available'>",
        "Skills": "<Extracted Skills or 'Not Available'>",
        "Certifications": "<Extracted Certifications or 'Not Available'>",
        "Projects": "<Extracted Projects or 'Not Available'>",
        "Languages": "<Extracted Languages or 'Not Available'>"
    }

    Instructions:
    1. Retrieve the most relevant information from the resume document(s) based on the query.
    2. Organize the output strictly in the given format.
    3. Assign a relevance score (0 to 1) for each field based on how well it matches the query embeddings.
    4. Provide a brief explanation for low scores (<0.5) if applicable.
    5. Use concise language, and maintain consistent structure.


    """
    
    # query = """
        
    #     You are an expert in analyzing resumes specifically for sales officer roles. 
    #     Evaluate the relevance of the following sections in resumes based on their alignment with sales officer criteria:
    #     Summary, Education, Work Experience, Skills, Certifications, Projects, and Languages.

    #     Instructions:
    #     1. Assign a relevance score (0 to 1) for each section based on the strength of its alignment with sales officer requirements use word embeddings , such as:
    #     - Sales-related experience (e.g., hitting targets, client management).
    #     - Sales skills (e.g., CRM tools, negotiation, communication).
    #     - Certifications or training.
    #     - Relevant education or projects demonstrating sales potential.
    #     2. Provide a brief explanation for low scores (<0.5).
    #     3. Return the output with only the section names and their respective scores.

    #     Output Format:
    #     {
    #         "Summary": <Relevance Score>,
    #         "Education": <Relevance Score>,
    #         "Work Experience": <Relevance Score>,
    #         "Skills": <Relevance Score>,
    #         "Certifications": <Relevance Score>,
    #         "Projects": <Relevance Score>,
    #         "Languages": <Relevance Score>
    #     }
    # """
    
    file_list = os.listdir("D:\Downloads\Datasets\ResumeClassifier_Piramal\CV train")
    for file_name in file_list[:1]:
        
        CandidateID = file_name[:7]
        
        metadata =  {
            "source":{"$eq": CandidateID}
            }
        # print(metadata)
        res  = run_llm( query =query , metadata_filter= metadata)
        
        file_path = f"D:\Downloads\Datasets\ResumeClassifier_Piramal\Parser_CV_train\{CandidateID}.txt"
        with open(file_path, "w") as file:
            print(res["result"])
            file.write(res["result"])
    
        