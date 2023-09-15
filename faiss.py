from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import pandas as pd

class FAISSDatabase:
    def __init__(self, model_name, csv_file_path):
        self.embedding_function = SentenceTransformerEmbeddings(model_name=model_name)
        self.df = pd.read_csv(csv_file_path)
        self.list_of_documents = []

    def create_documents_list(self):
        # Iterate over the items in the "text" column and add them to the list
        for index, row in self.df.iterrows():
            text_value = row['text']
            self.list_of_documents.append(Document(page_content=text_value))

    def build_index(self):
        self.db = FAISS.from_documents(self.list_of_documents, self.embedding_function)

    def similarity_search(self, query_text):
        results_with_scores = self.db.similarity_search_with_score(query_text)
        return results_with_scores

    def save_index(self, path_with_index_name):
        self.db.save_local(path_with_index_name)

    @classmethod
    def load_index(cls, index_name, model_name):
        embedding_function = SentenceTransformerEmbeddings(model_name=model_name)
        loaded_db = FAISS.load_local(index_name, embedding_function)
        return loaded_db

if __name__ == "__main__":
    # Example usage
    db = FAISSDatabase(model_name="google/flan-t5-base", csv_file_path='/content/sample.csv')
    db.create_documents_list()
    db.build_index()

    query_text = "This is a Ray Fish"
    results = db.similarity_search(query_text)
    for doc, score in results:
        print(f"Content: {doc.page_content}, Score: {score}")

    # Saving and loading the index
    db.save_index("faiss_index")
    loaded_db = FAISSDatabase.load_index("faiss_index", model_name="google/flan-t5-base")
