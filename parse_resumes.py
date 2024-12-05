# %%
import os
import time
import shutil  # For moving files
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader

# %%
debug = True

# %%
root = "/app/"
doc_folder = root + "resumes/"
#vector_db_path = root + "/data-ingestion-local/"
vector_db_path = root + "resumes_vector_db/"
embed_model_name = "all-MiniLM-L6-v2"

# %%
def process_resumes(doc_folder):
    pdfs = [pdf for pdf in os.listdir(doc_folder) if pdf.endswith(".pdf")]
    if len(pdfs) == 0:
        if debug:
            print("No PDF files found in the folder. Exiting...")
        return -1
    else:
        if debug:
            print(pdfs)
        doc_container = []
        archive_dir = os.path.join(doc_folder, "_archive")
        for pdf in pdfs:
            if os.path.exists(os.path.join(archive_dir, "processed_files.txt")):
                with open(os.path.join(archive_dir, "processed_files.txt"), "r") as f:
                    processed_files = f.read().splitlines()
            else:
                processed_files = []

            if pdf in processed_files:
                if debug:
                    print(f"File {pdf} has already been processed. Skipping...")
                if not os.path.exists(archive_dir):
                    os.makedirs(archive_dir)
                shutil.move(os.path.join(doc_folder, pdf), archive_dir)   
                continue
            else:
                if debug:
                    print(f"Processing file {pdf}...")
                loader = PyPDFLoader(os.path.join(doc_folder, pdf),
                                extract_images=False)
                docsRaw = loader.load()
                for doc in docsRaw:
                    doc_container.append(doc)
                # Store the processed file name in a text file
                metadata_folder = os.path.join(doc_folder, "_metadata")
                if not os.path.exists(metadata_folder):
                    os.makedirs(metadata_folder)
                with open(os.path.join(metadata_folder, "processed_files.txt"), "a") as f:
                    f.write(pdf + "\n")
                
                # Move the processed file to archive_dir folder
                
                if not os.path.exists(archive_dir):
                    os.makedirs(archive_dir)
                shutil.move(os.path.join(doc_folder, pdf), archive_dir)
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        docs_split = splitter.split_documents(documents=doc_container)
        if debug:
            print(docs_split[0])
            print("\n--- Document Chunks Information ---", end="\n")
            print(f"Number of document chunks: {len(docs_split)}", end="\n\n")
        return docs_split

# %%
def create_vector_db(docs_split, vector_db_path):
    if docs_split == -1:
        print("No documents to process.")
    else:
        embedF = HuggingFaceEmbeddings(model_name = embed_model_name)
        if debug:    
            print("[INFO] Started embedding", end="\n")
        start = time.time()
        vectorDB = Chroma.from_documents(documents=docs_split,
                                        embedding=embedF,
                                        persist_directory=vector_db_path)
        end = time.time()
        if debug:
            print(f"Time taken to embed: {end - start} seconds")
        print("Vector Database created successfully")

# %%
def add_to_existing_vector_database(docs_split, vector_db_path):
    if docs_split == -1:
        print("No documents to process.")
    else:
        if debug:
            print("[INFO] Started embedding and adding data to existing database", end="\n")
        embedF = HuggingFaceEmbeddings(model_name=embed_model_name)
        vectorDB = Chroma(persist_directory=vector_db_path, embedding_function=embedF)
        vectorDB.add_documents(docs_split)
        vectorDB.persist()

# %%
docs_split = process_resumes(doc_folder)
if not os.path.exists(vector_db_path):
    if debug:
        print("Vector Database does not exist")
    create_vector_db(docs_split, vector_db_path)
else:
    if debug:
        print("Vector Database Already Exists")
    add_to_existing_vector_database(docs_split, vector_db_path)


