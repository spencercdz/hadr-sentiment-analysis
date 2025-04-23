"""
General Agent - Handles general user questions using web search and Wikipedia
"""
from langchain.agents import initialize_agent, AgentType
from langchain_community.llms import Ollama
from tool_registry import ToolRegistry

class GeneralAgent:
    def __init__(self):
        """Initialize the general agent with search tools."""
        # Initialize DeepSeek LLM via Ollama
        self.llm = Ollama(model="deepseek")
        
        # Get search tools
        tool_registry = ToolRegistry()
        self.tools = tool_registry.get_search_tools()
        
        # Initialize ReAct agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.REACT_DOCSTORE,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    def process_query(self, query):
        """
        Process a general question using the agent.
        
        Args:
            query (str): User's question
            
        Returns:
            str: Agent's response
        """
        try:
            # Prepare system message to instruct the agent
            system_message = """
            You are a helpful assistant that answers general questions.
            Use the tools available to you to find information and provide accurate responses.
            Make your responses comprehensive but concise.
            """
            
            # Run the agent
            result = self.agent.run(
                f"{system_message}\n\nQuestion: {query}"
            )
            
            return result
            
        except Exception as e:
            # Handle any errors gracefully
            error_msg = str(e)
            if "Agent stopped due to iteration limit" in error_msg:
                return "I apologize, but I couldn't find a complete answer within the allowed steps. Here's what I found so far:\n\n" + str(self.agent.intermediate_steps)
            else:
                return f"I encountered an error while processing your question: {error_msg}. Please try rephrasing your question."
