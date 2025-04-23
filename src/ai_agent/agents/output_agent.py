"""
Output Agent - Formats responses from other agents for user presentation
"""
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from pathlib import Path
import os

class OutputAgent:
    def __init__(self):
        """Initialize the output agent for formatting responses."""
        # Initialize DeepSeek LLM via Ollama
        self.llm = Ollama(model="deepseek")
        
        # Create formatter chain
        formatter_template = """
        You are a helpful assistant that formats responses in a clear, concise manner.
        Your goal is to present information in a user-friendly way.
        
        Original response:
        {response}
        
        Additional context (if available):
        {context}
        
        Please format the above response to be clear, helpful and well-structured for the user.
        If a report was generated, make sure to mention the path where the user can find it.
        
        Formatted response:
        """
        
        self.formatter_prompt = PromptTemplate(
            input_variables=["response", "context"],
            template=formatter_template
        )
        
        self.formatter_chain = LLMChain(
            llm=self.llm,
            prompt=self.formatter_prompt
        )
    
    def format_general_response(self, response):
        """Format response from general agent."""
        return self.formatter_chain.run(
            response=response, 
            context=""
        )
    
    def format_report_response(self, response_data):
        """Format response from report agent."""
        # Extract information
        result = response_data.get("result", "")
        report_generated = response_data.get("report_generated", False)
        report_path = response_data.get("report_path", None)
        
        # Create context about report generation
        context = ""
        if report_generated and report_path:
            # Convert to relative path for better readability
            try:
                report_rel_path = os.path.relpath(report_path)
                context = f"A report was successfully generated and saved to: {report_rel_path}"
            except:
                context = f"A report was successfully generated and saved to: {report_path}"
        
        return self.formatter_chain.run(
            response=result, 
            context=context
        )
