"""
Code Executor Agent - Intelligent code execution and debugging agent for SolnAI.
Built on AutoGen v0.4.7 with Claude 3.7 Sonnet integration.
"""

import os
import sys
import logging
import tempfile
import subprocess
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import asyncio

from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
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

class SolnAICodeExecutorAgent:
    """
    Code Executor Agent for executing, debugging, and optimizing code.
    
    This agent leverages AutoGen v0.4.7 and Claude 3.7 Sonnet to:
    1. Execute code in a safe environment
    2. Debug and fix code issues
    3. Optimize code for better performance
    4. Explain code functionality
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        work_dir: Optional[str] = None,
        supported_languages: Optional[List[str]] = None
    ):
        """
        Initialize the Code Executor Agent.
        
        Args:
            api_key: Anthropic API key (optional, will use environment variable if not provided)
            work_dir: Working directory for code execution (optional, will use a temp directory if not provided)
            supported_languages: List of supported programming languages (optional)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or pass it to the constructor.")
        
        # Set up working directory
        self.work_dir = work_dir or tempfile.mkdtemp()
        os.makedirs(self.work_dir, exist_ok=True)
        logger.info(f"Working directory: {self.work_dir}")
        
        # Set supported languages
        self.supported_languages = supported_languages or ["python", "javascript", "bash", "typescript"]
        
        # Initialize the model client
        self.model_client = AnthropicChatCompletionClient(
            model="claude-3-7-sonnet",
            api_key=self.api_key,
            temperature=0.2,  # Lower temperature for more deterministic code generation
        )
        
        # Initialize the code assistant agent
        self.code_assistant = AssistantAgent(
            name="code_assistant",
            model_client=self.model_client,
            system_message=f"""You are an expert code developer and debugger. Your task is to write, execute, 
            debug, and optimize code based on user requirements.

            Follow these guidelines:
            1. Write clean, efficient, and well-documented code
            2. Follow best practices for the specific programming language
            3. Provide clear explanations of your code
            4. Debug issues thoroughly and suggest fixes
            5. Optimize code for better performance when needed
            
            You can work with the following languages: {', '.join(self.supported_languages)}
            
            When writing code:
            - Include proper error handling
            - Add comments to explain complex logic
            - Follow standard coding conventions
            - Consider edge cases
            - Write modular and reusable code
            """,
            model_context=BufferedChatCompletionContext(buffer_size=20),  # Keep a buffer of 20 messages
        )
        
        # Initialize the code executor agent
        self.executor = CodeExecutorAgent(
            name="executor",
            work_dir=self.work_dir,
            system_message="I am a code executor. I execute code and return the results.",
        )
    
    async def execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Execute code and return the results.
        
        Args:
            code: Code to execute
            language: Programming language of the code
            
        Returns:
            Dict with execution results
        """
        if language not in self.supported_languages:
            return {
                "success": False,
                "error": f"Unsupported language: {language}. Supported languages: {', '.join(self.supported_languages)}"
            }
        
        logger.info(f"Executing {language} code")
        
        # Create a message with the code to execute
        code_message = TextMessage(
            content=f"```{language}\n{code}\n```\nPlease execute this code and return the results.",
            source="user"
        )
        
        # Execute the code
        response = await self.executor.on_messages(
            [code_message],
            cancellation_token=CancellationToken(),
        )
        
        # Extract the execution results
        execution_results = {
            "language": language,
            "code": code,
            "output": response.chat_message.content,
            "success": "error" not in response.chat_message.content.lower(),
        }
        
        return execution_results
    
    async def debug_code(self, code: str, error_message: str, language: str = "python") -> Dict[str, Any]:
        """
        Debug code and suggest fixes.
        
        Args:
            code: Code to debug
            error_message: Error message from previous execution
            language: Programming language of the code
            
        Returns:
            Dict with debugging results
        """
        if language not in self.supported_languages:
            return {
                "success": False,
                "error": f"Unsupported language: {language}. Supported languages: {', '.join(self.supported_languages)}"
            }
        
        logger.info(f"Debugging {language} code")
        
        # Create a message with the code to debug
        debug_message = TextMessage(
            content=f"""I have the following {language} code that's producing an error:
            
```{language}
{code}
```

Error message:
```
{error_message}
```

Please debug this code, explain what's wrong, and provide a fixed version.""",
            source="user"
        )
        
        # Get debugging suggestions
        response = await self.code_assistant.on_messages(
            [debug_message],
            cancellation_token=CancellationToken(),
        )
        
        # Extract the debugging results
        debugging_results = {
            "language": language,
            "original_code": code,
            "error_message": error_message,
            "analysis": response.chat_message.content,
        }
        
        # Try to extract the fixed code
        fixed_code = self._extract_code_block(response.chat_message.content, language)
        if fixed_code:
            debugging_results["fixed_code"] = fixed_code
        
        return debugging_results
    
    async def optimize_code(self, code: str, language: str = "python", optimization_goal: str = "performance") -> Dict[str, Any]:
        """
        Optimize code for better performance or readability.
        
        Args:
            code: Code to optimize
            language: Programming language of the code
            optimization_goal: Goal of optimization (performance, readability, memory)
            
        Returns:
            Dict with optimization results
        """
        if language not in self.supported_languages:
            return {
                "success": False,
                "error": f"Unsupported language: {language}. Supported languages: {', '.join(self.supported_languages)}"
            }
        
        logger.info(f"Optimizing {language} code for {optimization_goal}")
        
        # Create a message with the code to optimize
        optimize_message = TextMessage(
            content=f"""I have the following {language} code that I want to optimize for {optimization_goal}:
            
```{language}
{code}
```

Please optimize this code for {optimization_goal}, explain your optimizations, and provide the optimized version.""",
            source="user"
        )
        
        # Get optimization suggestions
        response = await self.code_assistant.on_messages(
            [optimize_message],
            cancellation_token=CancellationToken(),
        )
        
        # Extract the optimization results
        optimization_results = {
            "language": language,
            "original_code": code,
            "optimization_goal": optimization_goal,
            "analysis": response.chat_message.content,
        }
        
        # Try to extract the optimized code
        optimized_code = self._extract_code_block(response.chat_message.content, language)
        if optimized_code:
            optimization_results["optimized_code"] = optimized_code
        
        return optimization_results
    
    async def explain_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Explain code functionality in detail.
        
        Args:
            code: Code to explain
            language: Programming language of the code
            
        Returns:
            Dict with explanation results
        """
        if language not in self.supported_languages:
            return {
                "success": False,
                "error": f"Unsupported language: {language}. Supported languages: {', '.join(self.supported_languages)}"
            }
        
        logger.info(f"Explaining {language} code")
        
        # Create a message with the code to explain
        explain_message = TextMessage(
            content=f"""I have the following {language} code that I want to understand better:
            
```{language}
{code}
```

Please explain this code in detail, including:
1. Overall purpose
2. How it works
3. Key components and their functions
4. Any potential issues or limitations
5. Suggestions for improvement""",
            source="user"
        )
        
        # Get explanation
        response = await self.code_assistant.on_messages(
            [explain_message],
            cancellation_token=CancellationToken(),
        )
        
        # Extract the explanation results
        explanation_results = {
            "language": language,
            "code": code,
            "explanation": response.chat_message.content,
        }
        
        return explanation_results
    
    async def generate_code(self, requirements: str, language: str = "python") -> Dict[str, Any]:
        """
        Generate code based on requirements.
        
        Args:
            requirements: Requirements for the code
            language: Programming language to use
            
        Returns:
            Dict with generated code
        """
        if language not in self.supported_languages:
            return {
                "success": False,
                "error": f"Unsupported language: {language}. Supported languages: {', '.join(self.supported_languages)}"
            }
        
        logger.info(f"Generating {language} code based on requirements")
        
        # Create a message with the requirements
        generate_message = TextMessage(
            content=f"""I need {language} code that meets the following requirements:
            
{requirements}

Please write clean, efficient, and well-documented code that meets these requirements.""",
            source="user"
        )
        
        # Generate code
        response = await self.code_assistant.on_messages(
            [generate_message],
            cancellation_token=CancellationToken(),
        )
        
        # Extract the generated code
        generation_results = {
            "language": language,
            "requirements": requirements,
            "response": response.chat_message.content,
        }
        
        # Try to extract the generated code
        generated_code = self._extract_code_block(response.chat_message.content, language)
        if generated_code:
            generation_results["generated_code"] = generated_code
        
        return generation_results
    
    async def run_interactive_session(self) -> None:
        """
        Run an interactive code development session.
        """
        logger.info("Starting interactive code development session")
        
        # Create a console for interaction
        await Console(
            self.code_assistant.on_messages_stream(
                [TextMessage(content="I'm ready to help with your code. What would you like to do?", source="user")],
                cancellation_token=CancellationToken(),
            ),
            output_stats=True,
        )
    
    def _extract_code_block(self, text: str, language: str) -> Optional[str]:
        """
        Extract code block from text.
        
        Args:
            text: Text containing code block
            language: Programming language of the code
            
        Returns:
            Extracted code or None if not found
        """
        import re
        
        # Pattern to match code blocks with the specified language
        pattern = f"```(?:{language})?\\n(.*?)\\n```"
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        return None


# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = SolnAICodeExecutorAgent()
    
    # Example code to execute
    code = """
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    
    return fib

# Test the function
result = fibonacci(10)
print(f"Fibonacci sequence: {result}")
"""
    
    # Execute the code
    asyncio.run(agent.execute_code(code, "python"))
