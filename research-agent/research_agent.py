"""
Research Agent - Intelligent research and information analysis agent for SolnAI.
Built on AutoGen v0.4.7 with Claude 3.7 Sonnet integration.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
import asyncio

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_core.model_context import BufferedChatCompletionContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResearchAgent:
    """
    Research Agent for conducting in-depth research and information analysis.
    
    This agent leverages AutoGen v0.4.7 and Claude 3.7 Sonnet to:
    1. Conduct comprehensive web research on a given topic
    2. Analyze and synthesize information from multiple sources
    3. Generate well-structured research reports
    4. Answer complex questions with detailed explanations
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Research Agent.
        
        Args:
            api_key: Anthropic API key (optional, will use environment variable if not provided)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or pass it to the constructor.")
        
        # Initialize the model client
        self.model_client = AnthropicChatCompletionClient(
            model="claude-3-7-sonnet",
            api_key=self.api_key,
            temperature=0.7,
        )
        
        # Initialize the research assistant agent
        self.researcher = AssistantAgent(
            name="researcher",
            model_client=self.model_client,
            system_message="""You are an expert research assistant with exceptional skills in information gathering, 
            analysis, and synthesis. Your task is to conduct thorough research on given topics and provide 
            comprehensive, accurate, and well-structured information.

            Follow these guidelines:
            1. Conduct comprehensive research on the given topic
            2. Analyze information from multiple perspectives
            3. Synthesize findings into coherent, well-structured responses
            4. Provide detailed explanations with logical reasoning
            5. Cite sources or explain the basis of your knowledge
            6. Acknowledge limitations or gaps in available information
            7. Maintain objectivity and avoid biases
            8. Use appropriate formatting for readability
            
            When using tools:
            - Use web search to find the most recent and relevant information
            - Extract key information from search results
            - Analyze data to identify patterns and insights
            - Synthesize findings into a coherent response
            """,
            model_context=BufferedChatCompletionContext(buffer_size=20),  # Keep a buffer of 20 messages
            tools=[self._web_search, self._extract_data, self._analyze_data]
        )
        
        # Initialize the user proxy agent
        self.user_proxy = UserProxyAgent(
            name="user",
            human_input_mode="NEVER",  # Don't ask for human input
            code_execution_config={"work_dir": "workspace"}
        )
    
    async def _web_search(self, query: str) -> str:
        """
        Simulate web search to find information.
        
        Args:
            query: Search query
            
        Returns:
            Search results as a string
        """
        logger.info(f"Searching for: {query}")
        # In a real implementation, this would connect to a search API
        return f"Search results for '{query}': This is a simulated search result. In a real implementation, this would return actual search results from a search engine API."
    
    async def _extract_data(self, text: str, extraction_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured data from text.
        
        Args:
            text: Text to extract data from
            extraction_pattern: Pattern defining what to extract
            
        Returns:
            Dict with extracted data
        """
        logger.info("Extracting data using pattern")
        # In a real implementation, this would use regex or other extraction methods
        return {
            "success": True,
            "data": {
                "extracted": "This is simulated extracted data. In a real implementation, this would extract structured data based on the provided pattern."
            }
        }
    
    async def _analyze_data(self, data: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        """
        Analyze data to extract insights.
        
        Args:
            data: Data to analyze
            analysis_type: Type of analysis to perform
            
        Returns:
            Dict with analysis results
        """
        logger.info(f"Analyzing data with analysis type: {analysis_type}")
        # In a real implementation, this would perform actual data analysis
        return {
            "success": True,
            "analysis": f"This is a simulated analysis of type '{analysis_type}'. In a real implementation, this would perform actual data analysis based on the provided data and analysis type."
        }
    
    async def research(self, topic: str) -> Dict[str, Any]:
        """
        Conduct research on a given topic.
        
        Args:
            topic: Research topic
            
        Returns:
            Dict with research results
        """
        logger.info(f"Starting research on topic: {topic}")
        
        # Create the research request
        research_request = TextMessage(content=f"Conduct comprehensive research on the following topic: {topic}", source="user")
        
        # Get the research response
        response = await self.researcher.on_messages(
            [research_request],
            cancellation_token=CancellationToken(),
        )
        
        # Extract the research results
        research_results = {
            "topic": topic,
            "response": response.chat_message.content,
            "inner_messages": [str(msg) for msg in response.inner_messages]
        }
        
        logger.info(f"Research completed for topic: {topic}")
        return research_results
    
    async def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a complex question with detailed research.
        
        Args:
            question: Question to answer
            
        Returns:
            Dict with answer and supporting information
        """
        logger.info(f"Answering question: {question}")
        
        # Create the question request
        question_request = TextMessage(content=f"Answer the following question with detailed research: {question}", source="user")
        
        # Get the answer response
        response = await self.researcher.on_messages(
            [question_request],
            cancellation_token=CancellationToken(),
        )
        
        # Extract the answer
        answer_results = {
            "question": question,
            "answer": response.chat_message.content,
            "inner_messages": [str(msg) for msg in response.inner_messages]
        }
        
        logger.info(f"Question answered: {question}")
        return answer_results
    
    async def generate_report(self, topic: str, report_format: str = "markdown") -> Dict[str, Any]:
        """
        Generate a structured research report on a given topic.
        
        Args:
            topic: Report topic
            report_format: Format of the report (markdown, json, etc.)
            
        Returns:
            Dict with report and metadata
        """
        logger.info(f"Generating report on topic: {topic} in format: {report_format}")
        
        # Create the report request
        report_request = TextMessage(
            content=f"Generate a comprehensive research report on '{topic}'. Format the report in {report_format}.", 
            source="user"
        )
        
        # Get the report response
        response = await self.researcher.on_messages(
            [report_request],
            cancellation_token=CancellationToken(),
        )
        
        # Extract the report
        report_results = {
            "topic": topic,
            "format": report_format,
            "report": response.chat_message.content,
            "inner_messages": [str(msg) for msg in response.inner_messages]
        }
        
        logger.info(f"Report generated for topic: {topic}")
        return report_results
    
    async def run_interactive_session(self) -> None:
        """
        Run an interactive research session where the user can ask questions.
        """
        logger.info("Starting interactive research session")
        
        # Create a console for interaction
        await Console(
            self.researcher.on_messages_stream(
                [TextMessage(content="I'm ready to help with your research. What would you like to know?", source="user")],
                cancellation_token=CancellationToken(),
            ),
            output_stats=True,
        )


# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = ResearchAgent()
    
    # Run a research task
    asyncio.run(agent.research("The impact of artificial intelligence on healthcare"))
