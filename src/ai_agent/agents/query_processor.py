"""
Query Processor - Rephrases and classifies user queries
"""
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

class QueryProcessor:
    def __init__(self):
        """Initialize the query processor with LLM models."""
        # Initialize DeepSeek LLM via Ollama
        self.llm = Ollama(model="deepseek")
        
        # Create rephraser chain
        rephraser_template = """
        You are a helpful assistant that rephrases user queries to make them clearer and more structured.
        Please rephrase the following query in a clear, concise manner that would be suitable for an AI system to process.
        
        Original query: {query}
        
        Rephrased query:
        """
        
        self.rephraser_prompt = PromptTemplate(
            input_variables=["query"],
            template=rephraser_template
        )
        
        self.rephraser_chain = LLMChain(
            llm=self.llm,
            prompt=self.rephraser_prompt
        )
        
        # Create classifier chain
        classifier_template = """
        You are a helpful assistant that classifies user queries into specific categories.
        Please classify the following query into one of these categories:
        1. report_generation - If a sentiment report is to be generated about a topic
        2. general_question - If the user is asking a general question that doesn't require a report
        
        Return ONLY "report_generation" or "general_question" as your answer.
        
        User query: {query}
        
        Classification:
        """
        
        self.classifier_prompt = PromptTemplate(
            input_variables=["query"],
            template=classifier_template
        )
        
        self.classifier_chain = LLMChain(
            llm=self.llm,
            prompt=self.classifier_prompt
        )
    
    def rephrase_query(self, query):
        """Rephrase the user query to make it clearer."""
        result = self.rephraser_chain.run(query=query)
        return result.strip()
    
    def classify_query(self, query):
        """Classify the user query as report_generation or general_question."""
        result = self.classifier_chain.run(query=query)
        # Ensure we get a clean classification
        result = result.strip().lower()
        
        if "report_generation" in result:
            return "report_generation"
        else:
            return "general_question"
