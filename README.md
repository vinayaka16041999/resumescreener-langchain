# Resume Screener with LangChain

This project is a Resume Screener application that leverages LangChain for processing and analyzing resumes. The application is designed to watch a directory for new resume files, process them, and provide a conversational interface for querying and shortlisting candidates based on their resumes.

## Project Structure

- `resume_screener.py`: The main Streamlit application for interacting with the resume screener.
- `filewatcher.py`: Watches a directory for new resume files and processes them.
- `parse_resumes.py`: Contains functions for processing resumes and managing the vector database.
- `start.sh`: Script to start the Python and Streamlit applications.
- `docker/Dockerfile`: Dockerfile for containerizing the application.
- `requirements.txt`: List of dependencies required for the project.

## Features

- **File Watching**: Automatically detects new resume files in a specified directory.
- **Resume Processing**: Extracts text from PDF resumes and splits them into chunks for embedding.
- **Vector Database**: Creates and updates a vector database for efficient resume retrieval.
- **Conversational Interface**: Provides a Streamlit-based UI for querying and shortlisting candidates.

## Setup and Installation

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/yourusername/resumescreener-langchain.git
    cd resumescreener-langchain
    ```

2. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Set Up Environment Variables**:
    Create a `.env` file and add your environment variables, such as API keys.

4. **Run the Application**:
    ```sh
    ./start.sh
    ```

## Docker Setup

1. **Build the Docker Image**:
    ```sh
    docker build -t resumescreener-langchain .
    ```

2. **Run the Docker Container**:
    ```sh
    docker run -p 8501:8501 resumescreener-langchain
    ```

## How It Works

1. **File Watching**:
    - `filewatcher.py` uses `watchdog` to monitor the `/app/resumes/` directory for new PDF files.
    - When a new file is detected, it triggers the `process_resumes` function in `parse_resumes.py`.

2. **Resume Processing**:
    - `parse_resumes.py` extracts text from PDF files using `PyPDFLoader`.
    - The text is split into chunks using `RecursiveCharacterTextSplitter`.
    - The chunks are embedded using `HuggingFaceEmbeddings` and stored in a Chroma vector database.

3. **Vector Database Management**:
    - If the vector database does not exist, it is created using `create_vector_db`.
    - If the vector database exists, new documents are added using `add_to_existing_vector_database`.

4. **Conversational Interface**:
    - `resume_screener.py` provides a Streamlit UI for interacting with the resume screener.
    - Users can upload resumes, ask questions, and get responses based on the processed resumes.
    - The application uses LangChain's `ConversationalRetrievalChain` to handle user queries and retrieve relevant information from the vector database.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.
