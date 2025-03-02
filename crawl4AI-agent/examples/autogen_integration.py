"""
Example of integrating Crawl4AI with AutoGen v0.4.7.

This example demonstrates how to use the Crawl4AI agent in a multi-agent system
with AutoGen v0.4.7, including integration with the Claude wrapper.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path to import crawl4ai
sys.path.append(str(Path(__file__).parent.parent))

from crawl4ai import Crawl4AIAgent
from autogen import UserProxyAgent, ConversableAgent, GroupChat, GroupChatManager
from soln_ai.llm.claude_wrapper import ClaudeWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the example."""
    # Initialize Claude wrapper
    claude = ClaudeWrapper()
    
    # Test Claude connection
    test_result = claude.test_connection()
    if test_result["success"]:
        logger.info(f"Claude connection successful! Latency: {test_result['latency_seconds']}s")
    else:
        logger.error(f"Claude connection failed: {test_result['error']}")
        return
    
    # Get LLM config for AutoGen
    llm_config = claude.get_llm_config()
    
    # Create the web crawler agent
    crawler = Crawl4AIAgent(
        name="WebCrawler",
        system_message="""You are a specialized web crawler agent. You can navigate websites, 
        extract information, and summarize content. Use your tools to help the team with research tasks.
        Always respect website terms of service and robots.txt.""",
        llm_config=llm_config
    )
    
    # Create a research agent
    researcher = ConversableAgent(
        name="Researcher",
        system_message="""You are a research specialist. You analyze information provided by the WebCrawler
        and identify key insights, patterns, and facts. Ask the WebCrawler to find specific information when needed.""",
        llm_config=llm_config
    )
    
    # Create a writer agent
    writer = ConversableAgent(
        name="Writer",
        system_message="""You are a content writer. You take research findings and create well-structured,
        engaging content. Ask for specific information from the Researcher or WebCrawler if needed.""",
        llm_config=llm_config
    )
    
    # Create a user proxy agent
    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="TERMINATE",
        code_execution_config={"work_dir": "workspace"}
    )
    
    # Create a group chat
    groupchat = GroupChat(
        agents=[user_proxy, crawler, researcher, writer],
        messages=[],
        max_round=12
    )
    
    # Create a group chat manager
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config
    )
    
    # Start the conversation
    user_proxy.initiate_chat(
        manager,
        message="""I need a comprehensive report on the latest developments in AI safety. 
        Please research recent publications, extract key findings, and create a structured report.
        Focus on developments from the last year."""
    )

if __name__ == "__main__":
    main()
