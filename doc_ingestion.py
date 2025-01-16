import os
import docx2txt
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec



load_dotenv()

embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')


def pinecone_document_exists(CandidateID,index):

    # Query to check if the document exists
    query_response = index.query(
        vector=[0]*1536,  # Empty vector since we're only filtering by metadata
        top_k=1,    # Only need to check if it exists
        filter={"source": CandidateID},
        include_metadata=True,
    )
    
    return query_response['matches']


def ingest_docs():
    
    chunk_cnt = 0
    
    dir_path = "D:\Downloads\Datasets\ResumeClassifier_Piramal\CV train"
    
    file_list = os.listdir(dir_path)
    
        
    pc = Pinecone( api_key=os.environ.get("PINECONE_API_KEY") )
    
    index_name = "resume-vec-index"
    index = pc.Index(index_name)
    
    vector_store = PineconeVectorStore(index=index, embedding=OpenAIEmbeddings())
  

    for file_name in file_list[:-1]: 
        
        CandidateID = file_name[:7]
        # Check the response
        if not pinecone_document_exists(CandidateID, index):
            print("")
            print(f"Document '{CandidateID}' already exists in the index.")
            print("")
            
        else:
            
            # print(f"Document '{file_name[:7]}' does not exist in the index.")
            
            loader = Docx2txtLoader(f"{dir_path}\{file_name}")
    
            raw_document = loader.load()   

            text_splitter = RecursiveCharacterTextSplitter( chunk_size= 600, chunk_overlap = 60 )
            
            documents = text_splitter.split_documents(raw_document)
            
            
            for doc in documents:
                # if doc.metadata["source"] ==file_name[:-5]:
                #     continue
                doc.metadata.update({"source": CandidateID})
            
            # print(documents)
            chunk_cnt +=  len(documents)
            print(f"Going to add {len(documents)} to Pinecone")
            
            ids = [CandidateID + str(i) for i in range(len(documents)) ]
            
            vector_store.add_documents(documents=documents, ids = ids)

            # # from_documents is a static method
            # PineconeVectorStore.from_documents(
            #     document, embeddings, index_name = index_name, 
            # )
            
            print(f"****Loading {file_name} to vectorstore done*****")
            
    return chunk_cnt 


if __name__ == '__main__':
    
    total_chunks = ingest_docs()
    print(total_chunks)
    