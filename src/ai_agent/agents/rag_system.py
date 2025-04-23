"""
RAG System - Retrieval Augmented Generation for enhancing responses with external knowledge
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import json
from pathlib import Path
import os

class RAGSystem:
    def __init__(self):
        """Initialize the RAG system."""
        self.llm = Ollama(model="deepseek")
        self.embeddings = OllamaEmbeddings(model="deepseek")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vector_store = None
        
    def index_scraped_content(self, content, metadata=None):
        """
        Index scraped content for retrieval.
        
        Args:
            content (str): Text content to index
            metadata (dict, optional): Metadata about the content
        
        Returns:
            bool: Success status
        """
        try:
            # Split the content into chunks
            if metadata is None:
                metadata = {}
            
            documents = self.text_splitter.create_documents(
                texts=[content],
                metadatas=[metadata]
            )
            
            # Create or update vector store
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
            else:
                self.vector_store.add_documents(documents)
                
            return True
        except Exception as e:
            print(f"Error indexing content: {str(e)}")
            return False
    
    def index_json_data(self, json_path=None):
        """
        Index data from JSON file for retrieval.
        
        Args:
            json_path (str, optional): Path to JSON file
        
        Returns:
            bool: Success status
        """
        try:
            if json_path is None:
                # Use default path
                json_path = Path(__file__).parent / "assets" / "generated_data.json"
                
            # Check if file exists
            if not os.path.exists(json_path):
                print(f"JSON file not found: {json_path}")
                return False
                
            # Load JSON data
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract content to index
            texts = []
            
            # Add sections
            if 'sections' in data:
                for section, content in data['sections'].items():
                    texts.append(f"Section {section}: {content}")
            
            # Add tweet content
            if 'tweets' in data and len(data['tweets']) > 1:  # Skip header row
                for tweet in data['tweets'][1:]:  # Skip header row
                    if len(tweet) >= 4:  # Ensure we have the tweet text
                        texts.append(f"Tweet from {tweet[0]} on {tweet[1]}: {tweet[3]}")
            
            # Add details content
            if 'details' in data:
                for detail in data['details']:
                    texts.append(f"Detail for {detail.get('Date', 'Unknown date')}: "
                                f"Sentiment: {detail.get('Sentiment', 'N/A')} "
                                f"Elements: {detail.get('Elements', 'N/A')} "
                                f"Impact: {detail.get('Impact', 'N/A')} "
                                f"Requests: {detail.get('Requests', 'N/A')} "
                                f"Summary: {detail.get('Summary', 'N/A')}")
            
            # Index the content
            for text in texts:
                self.index_scraped_content(text)
                
            return True
        except Exception as e:
            print(f"Error indexing JSON data: {str(e)}")
            return False
    
    def get_retriever(self):
        """Get the retriever for the vector store."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Index content first.")
            
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
    
    def query(self, question):
        """
        Query the RAG system with a question.
        
        Args:
            question (str): Question to ask
        
        Returns:
            str: Response from the RAG system
        """
        if self.vector_store is None:
            return "No indexed content available for retrieval."
            
        retriever = self.get_retriever()
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        result = qa_chain({"query": question})
        
        return result["result"]
