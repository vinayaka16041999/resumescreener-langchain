from parse_resumes import *
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class DocumentHandler(FileSystemEventHandler):
    def on_created(self, event):
        # Check if the created file is a document (e.g., .docx, .pdf, .txt)
        if not event.is_directory and event.src_path.endswith('.pdf'):
            print(f"New document added: {event.src_path}")
            #process_document(event.src_path)
            docs_split = process_resumes(doc_folder)
            if not os.path.exists(vector_db_path):
                if debug:
                    print("Vector Database does not exist")
                create_vector_db(docs_split, vector_db_path)
            else:
                if debug:
                    print("Vector Database Already Exists")
                add_to_existing_vector_database(docs_split, vector_db_path)

if __name__ == "__main__":
    # Directory to watch
    path_to_watch = "/app/resumes/"

    # Initialize the event handler and observer
    event_handler = DocumentHandler()
    observer = Observer()
    observer.schedule(event_handler, path=path_to_watch, recursive=False)

    if debug:
        print(f"Watching directory: {path_to_watch}")
    observer.start()