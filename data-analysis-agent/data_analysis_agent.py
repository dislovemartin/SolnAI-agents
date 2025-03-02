"""
Data Analysis Agent - Intelligent data analysis and visualization agent for SolnAI.
Built on AutoGen v0.4.7 with Claude 3.7 Sonnet integration.
"""

import os
import sys
import logging
import json
import tempfile
from typing import Dict, List, Any, Optional, Union, Tuple
import asyncio
import base64
from io import BytesIO

# Import AutoGen components
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

class DataAnalysisAgent:
    """
    Data Analysis Agent for analyzing and visualizing data.
    
    This agent leverages AutoGen v0.4.7 and Claude 3.7 Sonnet to:
    1. Analyze various data formats (CSV, JSON, Excel, etc.)
    2. Generate statistical insights
    3. Create visualizations
    4. Provide data-driven recommendations
    5. Generate reports
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        work_dir: Optional[str] = None,
        supported_formats: Optional[List[str]] = None
    ):
        """
        Initialize the Data Analysis Agent.
        
        Args:
            api_key: Anthropic API key (optional, will use environment variable if not provided)
            work_dir: Working directory for data files (optional, will use a temp directory if not provided)
            supported_formats: List of supported data formats (optional)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or pass it to the constructor.")
        
        # Set up working directory
        self.work_dir = work_dir or tempfile.mkdtemp()
        os.makedirs(self.work_dir, exist_ok=True)
        logger.info(f"Working directory: {self.work_dir}")
        
        # Set supported formats
        self.supported_formats = supported_formats or ["csv", "json", "excel", "parquet", "sql"]
        
        # Initialize the model client
        self.model_client = AnthropicChatCompletionClient(
            model="claude-3-7-sonnet",
            api_key=self.api_key,
            temperature=0.2,  # Lower temperature for more deterministic analysis
        )
        
        # Initialize the data analysis assistant agent
        self.data_assistant = AssistantAgent(
            name="data_analyst",
            model_client=self.model_client,
            system_message=f"""You are an expert data analyst specializing in data analysis, statistics, and visualization.
            Your task is to analyze data, generate insights, create visualizations, and provide recommendations.

            Follow these guidelines:
            1. Thoroughly analyze data to identify patterns, trends, and outliers
            2. Generate statistical insights using appropriate methods
            3. Create clear and informative visualizations
            4. Provide data-driven recommendations
            5. Explain your analysis in a clear, concise manner
            
            You can work with the following data formats: {', '.join(self.supported_formats)}
            
            When analyzing data:
            - Consider data quality issues (missing values, outliers, etc.)
            - Use appropriate statistical methods based on data types
            - Create visualizations that best represent the data
            - Provide actionable insights and recommendations
            - Consider the context and purpose of the analysis
            """,
            model_context=BufferedChatCompletionContext(buffer_size=20),  # Keep a buffer of 20 messages
        )
        
        # Initialize the user proxy agent for executing code
        self.user_proxy = UserProxyAgent(
            name="data_executor",
            human_input_mode="NEVER",
            code_execution_config={
                "work_dir": self.work_dir,
                "use_docker": False,  # Set to True to use Docker for isolation
                "timeout": 60,  # Timeout for code execution in seconds
            },
        )
    
    async def analyze_data(self, data_path: str, analysis_type: str = "general") -> Dict[str, Any]:
        """
        Analyze data and generate insights.
        
        Args:
            data_path: Path to the data file
            analysis_type: Type of analysis to perform (general, statistical, exploratory, etc.)
            
        Returns:
            Dict with analysis results
        """
        logger.info(f"Analyzing data: {data_path}, analysis type: {analysis_type}")
        
        # Check if the file exists
        if not os.path.exists(data_path):
            return {
                "success": False,
                "error": f"Data file not found: {data_path}"
            }
        
        # Check if the file format is supported
        file_extension = data_path.split('.')[-1].lower()
        if file_extension not in self.supported_formats:
            return {
                "success": False,
                "error": f"Unsupported file format: {file_extension}. Supported formats: {', '.join(self.supported_formats)}"
            }
        
        # Create a message with the data analysis request
        analysis_message = TextMessage(
            content=f"""I need to analyze the data in the file: {data_path}
            
            Analysis type: {analysis_type}
            
            Please perform the following tasks:
            1. Load and explore the data
            2. Clean the data if necessary (handle missing values, outliers, etc.)
            3. Perform appropriate statistical analysis
            4. Generate insights and identify patterns
            5. Create visualizations to illustrate key findings
            6. Provide a summary of the analysis and recommendations
            
            Please write and execute Python code to perform this analysis.""",
            source="user"
        )
        
        # Start a conversation between the data assistant and user proxy
        await self.user_proxy.initiate_chat(
            self.data_assistant,
            message=analysis_message,
        )
        
        # Extract the analysis results from the conversation
        analysis_results = {
            "data_path": data_path,
            "analysis_type": analysis_type,
            "conversation": self.user_proxy.chat_messages[self.data_assistant],
        }
        
        # Try to extract the final analysis summary
        for message in reversed(self.user_proxy.chat_messages[self.data_assistant]):
            if isinstance(message, TextMessage) and "summary" in message.content.lower():
                analysis_results["summary"] = message.content
                break
        
        return analysis_results
    
    async def create_visualization(self, data_path: str, visualization_type: str, columns: List[str] = None) -> Dict[str, Any]:
        """
        Create a visualization based on the data.
        
        Args:
            data_path: Path to the data file
            visualization_type: Type of visualization to create (bar, line, scatter, pie, etc.)
            columns: List of columns to include in the visualization
            
        Returns:
            Dict with visualization results
        """
        logger.info(f"Creating visualization: {visualization_type} for data: {data_path}")
        
        # Check if the file exists
        if not os.path.exists(data_path):
            return {
                "success": False,
                "error": f"Data file not found: {data_path}"
            }
        
        # Create a message with the visualization request
        columns_str = f"columns: {', '.join(columns)}" if columns else "Use appropriate columns based on the data"
        visualization_message = TextMessage(
            content=f"""I need to create a {visualization_type} visualization for the data in the file: {data_path}
            
            {columns_str}
            
            Please perform the following tasks:
            1. Load the data
            2. Clean and prepare the data for visualization
            3. Create a {visualization_type} visualization
            4. Make sure the visualization is clear, informative, and visually appealing
            5. Save the visualization as an image file
            6. Display the visualization
            
            Please write and execute Python code to create this visualization.""",
            source="user"
        )
        
        # Start a conversation between the data assistant and user proxy
        await self.user_proxy.initiate_chat(
            self.data_assistant,
            message=visualization_message,
        )
        
        # Extract the visualization results from the conversation
        visualization_results = {
            "data_path": data_path,
            "visualization_type": visualization_type,
            "columns": columns,
            "conversation": self.user_proxy.chat_messages[self.data_assistant],
        }
        
        # Try to find the path to the saved visualization
        import re
        for message in reversed(self.user_proxy.chat_messages[self.data_assistant]):
            if isinstance(message, TextMessage):
                # Look for file paths in the message
                file_paths = re.findall(r'saved (?:to|as) [\'"](.+?\.(?:png|jpg|jpeg|svg|pdf))[\'"]', message.content, re.IGNORECASE)
                if file_paths:
                    visualization_results["visualization_path"] = file_paths[0]
                    
                    # If the file exists, encode it as base64
                    if os.path.exists(file_paths[0]):
                        with open(file_paths[0], "rb") as image_file:
                            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                            visualization_results["visualization_base64"] = encoded_image
                    
                    break
        
        return visualization_results
    
    async def generate_report(self, data_path: str, report_format: str = "markdown") -> Dict[str, Any]:
        """
        Generate a comprehensive data analysis report.
        
        Args:
            data_path: Path to the data file
            report_format: Format of the report (markdown, html, pdf)
            
        Returns:
            Dict with report results
        """
        logger.info(f"Generating {report_format} report for data: {data_path}")
        
        # Check if the file exists
        if not os.path.exists(data_path):
            return {
                "success": False,
                "error": f"Data file not found: {data_path}"
            }
        
        # Create a message with the report generation request
        report_message = TextMessage(
            content=f"""I need to generate a comprehensive data analysis report for the data in the file: {data_path}
            
            Report format: {report_format}
            
            Please perform the following tasks:
            1. Load and explore the data
            2. Clean the data if necessary
            3. Perform exploratory data analysis
            4. Generate descriptive statistics
            5. Create appropriate visualizations
            6. Identify key insights and patterns
            7. Provide recommendations based on the analysis
            8. Generate a well-structured report in {report_format} format
            
            Please write and execute Python code to generate this report.""",
            source="user"
        )
        
        # Start a conversation between the data assistant and user proxy
        await self.user_proxy.initiate_chat(
            self.data_assistant,
            message=report_message,
        )
        
        # Extract the report results from the conversation
        report_results = {
            "data_path": data_path,
            "report_format": report_format,
            "conversation": self.user_proxy.chat_messages[self.data_assistant],
        }
        
        # Try to find the path to the saved report
        import re
        for message in reversed(self.user_proxy.chat_messages[self.data_assistant]):
            if isinstance(message, TextMessage):
                # Look for file paths in the message
                file_paths = re.findall(r'saved (?:to|as) [\'"](.+?\.(?:md|html|pdf))[\'"]', message.content, re.IGNORECASE)
                if file_paths:
                    report_results["report_path"] = file_paths[0]
                    
                    # If the file exists, read its content
                    if os.path.exists(file_paths[0]):
                        with open(file_paths[0], "r") as report_file:
                            report_results["report_content"] = report_file.read()
                    
                    break
        
        return report_results
    
    async def predict(self, data_path: str, target_column: str, features: List[str] = None, model_type: str = "auto") -> Dict[str, Any]:
        """
        Train a predictive model and make predictions.
        
        Args:
            data_path: Path to the data file
            target_column: Column to predict
            features: List of feature columns to use for prediction
            model_type: Type of model to use (auto, linear, tree, ensemble, etc.)
            
        Returns:
            Dict with prediction results
        """
        logger.info(f"Training predictive model for target: {target_column}, data: {data_path}")
        
        # Check if the file exists
        if not os.path.exists(data_path):
            return {
                "success": False,
                "error": f"Data file not found: {data_path}"
            }
        
        # Create a message with the prediction request
        features_str = f"features: {', '.join(features)}" if features else "Use appropriate features based on the data"
        prediction_message = TextMessage(
            content=f"""I need to train a predictive model for the data in the file: {data_path}
            
            Target column: {target_column}
            {features_str}
            Model type: {model_type}
            
            Please perform the following tasks:
            1. Load and explore the data
            2. Clean and prepare the data for modeling
            3. Split the data into training and testing sets
            4. Train a {model_type} model
            5. Evaluate the model performance
            6. Make predictions on the test set
            7. Provide a summary of the model and its performance
            
            Please write and execute Python code to train this predictive model.""",
            source="user"
        )
        
        # Start a conversation between the data assistant and user proxy
        await self.user_proxy.initiate_chat(
            self.data_assistant,
            message=prediction_message,
        )
        
        # Extract the prediction results from the conversation
        prediction_results = {
            "data_path": data_path,
            "target_column": target_column,
            "features": features,
            "model_type": model_type,
            "conversation": self.user_proxy.chat_messages[self.data_assistant],
        }
        
        # Try to extract the model performance metrics
        import re
        for message in reversed(self.user_proxy.chat_messages[self.data_assistant]):
            if isinstance(message, TextMessage):
                # Look for common metrics in the message
                metrics = {}
                
                # Try to find accuracy
                accuracy_match = re.search(r'accuracy(?:\s+score)?(?:\s*:|\s+is|\s+=)\s+(0\.\d+|\d+\.\d+|\d+%)', message.content, re.IGNORECASE)
                if accuracy_match:
                    metrics["accuracy"] = accuracy_match.group(1)
                
                # Try to find R-squared
                r2_match = re.search(r'r(?:-|\s)?squared(?:\s+score)?(?:\s*:|\s+is|\s+=)\s+(0\.\d+|\d+\.\d+|\d+%)', message.content, re.IGNORECASE)
                if r2_match:
                    metrics["r_squared"] = r2_match.group(1)
                
                # Try to find RMSE
                rmse_match = re.search(r'rmse(?:\s+score)?(?:\s*:|\s+is|\s+=)\s+(\d+\.\d+|\d+)', message.content, re.IGNORECASE)
                if rmse_match:
                    metrics["rmse"] = rmse_match.group(1)
                
                if metrics:
                    prediction_results["metrics"] = metrics
                    break
        
        return prediction_results
    
    async def run_interactive_session(self) -> None:
        """
        Run an interactive data analysis session.
        """
        logger.info("Starting interactive data analysis session")
        
        # Create a console for interaction
        await Console(
            self.data_assistant.on_messages_stream(
                [TextMessage(content="I'm ready to help with your data analysis. What would you like to do?", source="user")],
                cancellation_token=CancellationToken(),
            ),
            output_stats=True,
        )


# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = DataAnalysisAgent()
    
    # Example: Run an interactive session
    asyncio.run(agent.run_interactive_session())
