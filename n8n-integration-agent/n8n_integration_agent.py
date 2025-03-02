"""
N8N Integration Agent - Intelligent agent for integrating SolnAI with N8N workflows.
Built on AutoGen v0.4.7 with Claude 3.7 Sonnet integration.
"""

import os
import sys
import logging
import json
import requests
import time
import hashlib
import datetime
import statistics
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set
import asyncio

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

class N8NIntegrationAgent:
    """
    N8N Integration Agent for connecting SolnAI with N8N workflows.
    
    This agent leverages AutoGen v0.4.7 and Claude 3.7 Sonnet to:
    1. Create and manage N8N workflows
    2. Trigger N8N workflows from SolnAI
    3. Process data from N8N workflows
    4. Generate workflow templates based on natural language descriptions
    5. Debug and optimize N8N workflows
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        n8n_api_url: Optional[str] = None,
        n8n_api_key: Optional[str] = None,
        enable_caching: bool = True,
        cache_ttl: int = 3600,  # Default: 1 hour
        performance_tracking: bool = True  # Enable performance tracking by default
    ):
        """
        Initialize the N8N Integration Agent.
        
        Args:
            api_key: Anthropic API key (optional, will use environment variable if not provided)
            n8n_api_url: URL of the N8N API (optional, will use environment variable if not provided)
            n8n_api_key: API key for N8N (optional, will use environment variable if not provided)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or pass it to the constructor.")
        
        # Set up N8N API connection
        self.n8n_api_url = n8n_api_url or os.environ.get("N8N_API_URL")
        self.n8n_api_key = n8n_api_key or os.environ.get("N8N_API_KEY")
        
        if not self.n8n_api_url:
            logger.warning("N8N API URL not provided. Some functionality may be limited.")
        
        if not self.n8n_api_key:
            logger.warning("N8N API key not provided. Some functionality may be limited.")
        
        # Add caching support
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self.cache = {}  # Simple in-memory cache
        
        # Add performance tracking and optimization
        self.performance_tracking = performance_tracking
        self.optimization_suggestions = set()  # Store unique optimization suggestions
        self.performance_metrics = {}  # Store performance metrics by workflow ID
        self.execution_times = {}  # Store execution times for performance analysis
        
        # Initialize the model client
        self.model_client = AnthropicChatCompletionClient(
            model="claude-3-7-sonnet",
            api_key=self.api_key,
            temperature=0.2,  # Lower temperature for more deterministic results
        )
        
        # Initialize the N8N assistant agent
        self.n8n_assistant = AssistantAgent(
            name="n8n_assistant",
            model_client=self.model_client,
            system_message="""You are an expert in N8N workflow automation and integration.
            Your task is to help users create, manage, and optimize N8N workflows that integrate with SolnAI.

            Follow these guidelines:
            1. Create well-structured N8N workflows based on user requirements
            2. Integrate SolnAI agents with N8N workflows
            3. Optimize workflows for efficiency and reliability
            4. Debug workflow issues
            5. Provide clear explanations of workflow functionality
            
            When working with N8N:
            - Follow N8N best practices for workflow design
            - Use appropriate nodes for different tasks
            - Ensure proper error handling
            - Consider security and authentication
            - Document workflows clearly
            """,
            model_context=BufferedChatCompletionContext(buffer_size=20),  # Keep a buffer of 20 messages
        )
        
        # Initialize best practices knowledge base
        self.best_practices = {
            "parallel_execution": [
                "Use Split In Batches nodes for parallel processing",
                "Minimize dependencies between parallel branches",
                "Use Merge nodes to consolidate results"
            ],
            "caching": [
                "Cache results of expensive operations",
                "Implement conditional execution to skip redundant operations",
                "Use function nodes for custom caching logic"
            ],
            "api_optimization": [
                "Batch API requests when possible",
                "Use pagination for large datasets",
                "Implement retry logic for unreliable APIs",
                "Cache API responses when appropriate"
            ],
            "data_transformation": [
                "Use Set node for efficient data mapping",
                "Process data in batches for large datasets",
                "Use Function nodes for complex transformations"
            ],
            "error_handling": [
                "Implement error handling on all critical nodes",
                "Use Error Trigger nodes for recovery workflows",
                "Log errors with appropriate context"
            ]
        }
        
        # Initialize the user proxy agent
        self.user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            code_execution_config={
                "work_dir": ".",
                "use_docker": False,
            },
        )
    
    async def analyze_workflow_performance(self, workflow_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the performance characteristics of an N8N workflow.
        
        This method examines the workflow structure to identify:
        - Potential bottlenecks
        - Excessive API calls
        - Inefficient data processing
        - Missing error handling
        - Opportunities for parallelization
        
        Args:
            workflow_json: JSON representation of the N8N workflow
            
        Returns:
            Dict with analysis results and optimization suggestions
        """
        logger.info(f"Analyzing workflow performance")
        
        # Extract basic workflow metrics
        node_count = len(workflow_json.get("nodes", []))
        connection_count = len(workflow_json.get("connections", {}).get("main", []))
        
        # Categorize nodes by type
        node_types = {}
        api_nodes = []
        transform_nodes = []
        trigger_nodes = []
        error_handling_nodes = []
        
        for node in workflow_json.get("nodes", []):
            node_type = node.get("type", "")
            node_types[node_type] = node_types.get(node_type, 0) + 1
            
            # Identify API nodes (HTTP Request, API, etc.)
            if "http" in node_type.lower() or "api" in node_type.lower():
                api_nodes.append(node)
            
            # Identify transformation nodes (Set, Function, etc.)
            if node_type in ["Set", "Function", "FunctionItem", "Move Binary Data"]:
                transform_nodes.append(node)
            
            # Identify trigger nodes
            if "trigger" in node_type.lower():
                trigger_nodes.append(node)
            
            # Identify error handling nodes
            if "error" in node_type.lower():
                error_handling_nodes.append(node)
        
        # Create a message with the workflow analysis request
        workflow_str = json.dumps(workflow_json, indent=2)
        analysis_message = TextMessage(
            content=f"""I need to analyze the performance of an N8N workflow.

Here is the workflow JSON:
```json
{workflow_str}
```

Please analyze this workflow for performance issues and optimization opportunities. Focus on:
1. Identifying potential bottlenecks
2. Finding excessive or redundant API calls
3. Detecting inefficient data processing
4. Locating missing error handling
5. Discovering opportunities for parallelization

Provide specific optimization suggestions with explanations.""",
            source="user"
        )
        
        # Get workflow analysis response
        response = await self.n8n_assistant.on_messages(
            [analysis_message],
            cancellation_token=CancellationToken(),
        )
        
        # Extract optimization suggestions
        suggestions = self._extract_optimization_suggestions(response.chat_message.content)
        
        # Add suggestions to the set of all suggestions
        self.optimization_suggestions.update(suggestions)
        
        # Create the analysis result
        result = {
            "workflow_metrics": {
                "node_count": node_count,
                "connection_count": connection_count,
                "node_types": node_types,
                "api_node_count": len(api_nodes),
                "transform_node_count": len(transform_nodes),
                "trigger_node_count": len(trigger_nodes),
                "error_handling_node_count": len(error_handling_nodes),
            },
            "performance_analysis": response.chat_message.content,
            "optimization_suggestions": list(suggestions),
        }
        
        return result
    
    async def optimize_for_parallel_execution(self, workflow_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize an N8N workflow for parallel execution.
        
        This method analyzes the workflow to identify tasks that can be executed in parallel,
        and restructures the workflow to take advantage of N8N's parallel processing capabilities.
        
        Args:
            workflow_json: JSON representation of the N8N workflow
            
        Returns:
            Dict with the parallelized workflow
        """
        logger.info(f"Optimizing workflow for parallel execution")
        
        # Create a message with the parallelization request
        workflow_str = json.dumps(workflow_json, indent=2)
        parallelization_message = TextMessage(
            content=f"""I need to optimize an N8N workflow for parallel execution.

Here is the current workflow JSON:
```json
{workflow_str}
```

Please restructure this workflow to maximize parallel execution by:
1. Identifying independent operations that can run in parallel
2. Using Split In Batches nodes to process data in parallel
3. Adding Merge nodes to consolidate results
4. Ensuring proper error handling for parallel branches

Provide the optimized workflow JSON and explain the parallelization strategy.""",
            source="user"
        )
        
        # Get parallelization response
        response = await self.n8n_assistant.on_messages(
            [parallelization_message],
            cancellation_token=CancellationToken(),
        )
        
        # Extract the parallelized workflow JSON from the response
        parallelized_workflow_json = self._extract_json(response.chat_message.content)
        
        result = {
            "original_workflow": workflow_json,
            "explanation": response.chat_message.content,
        }
        
        if parallelized_workflow_json:
            result["parallelized_workflow"] = parallelized_workflow_json
        
        return result
    
    async def implement_workflow_caching(self, workflow_json: Dict[str, Any], cache_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Implement intelligent caching strategies in an N8N workflow.
        
        This method modifies the workflow to add caching for appropriate operations,
        reducing redundant API calls and improving performance.
        
        Args:
            workflow_json: JSON representation of the N8N workflow
            cache_config: Optional configuration for caching behavior
            
        Returns:
            Dict with the cached workflow
        """
        logger.info(f"Implementing caching in workflow")
        
        # Set default cache configuration if not provided
        if cache_config is None:
            cache_config = {
                "ttl": self.cache_ttl,
                "cache_api_calls": True,
                "cache_expensive_operations": True,
            }
        
        # Create a message with the caching implementation request
        workflow_str = json.dumps(workflow_json, indent=2)
        cache_config_str = json.dumps(cache_config, indent=2)
        
        caching_message = TextMessage(
            content=f"""I need to implement caching in an N8N workflow to improve performance.

Here is the current workflow JSON:
```json
{workflow_str}
```

Caching configuration:
```json
{cache_config_str}
```

Please modify this workflow to implement caching for:
1. API calls that may be repeated
2. Expensive data processing operations
3. Operations with stable inputs that produce stable outputs

Use Function nodes to implement the caching logic, with appropriate TTL values.
Provide the modified workflow JSON with caching implemented and explain your caching strategy.""",
            source="user"
        )
        
        # Get caching implementation response
        response = await self.n8n_assistant.on_messages(
            [caching_message],
            cancellation_token=CancellationToken(),
        )
        
        # Extract the cached workflow JSON from the response
        cached_workflow_json = self._extract_json(response.chat_message.content)
        
        result = {
            "original_workflow": workflow_json,
            "cache_config": cache_config,
            "explanation": response.chat_message.content,
        }
        
        if cached_workflow_json:
            result["cached_workflow"] = cached_workflow_json
        
        return result
    
    async def batch_process_workflow(self, workflow_json: Dict[str, Any], batch_size: int = 10) -> Dict[str, Any]:
        """
        Optimize a workflow for processing large datasets using batching.
        
        This method modifies the workflow to implement batch processing strategies,
        improving performance when dealing with large datasets.
        
        Args:
            workflow_json: JSON representation of the N8N workflow
            batch_size: Size of batches for processing
            
        Returns:
            Dict with the batched workflow
        """
        logger.info(f"Optimizing workflow for batch processing with batch size {batch_size}")
        
        # Create a message with the batch processing request
        workflow_str = json.dumps(workflow_json, indent=2)
        batching_message = TextMessage(
            content=f"""I need to optimize an N8N workflow for batch processing of large datasets.

Here is the current workflow JSON:
```json
{workflow_str}
```

Please modify this workflow to implement batch processing with a batch size of {batch_size} by:
1. Adding Split In Batches nodes at appropriate points
2. Configuring parallel processing for the batches
3. Adding Merge nodes to consolidate batch results
4. Ensuring proper error handling for batch processing

Provide the modified workflow JSON and explain your batch processing strategy.""",
            source="user"
        )
        
        # Get batch processing response
        response = await self.n8n_assistant.on_messages(
            [batching_message],
            cancellation_token=CancellationToken(),
        )
        
        # Extract the batched workflow JSON from the response
        batched_workflow_json = self._extract_json(response.chat_message.content)
        
        result = {
            "original_workflow": workflow_json,
            "batch_size": batch_size,
            "explanation": response.chat_message.content,
        }
        
        if batched_workflow_json:
            result["batched_workflow"] = batched_workflow_json
        
        return result
    
    async def optimize_data_transformation(self, workflow_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize data transformation operations in an N8N workflow.
        
        This method identifies and improves inefficient data transformation patterns,
        reducing execution time and resource usage.
        
        Args:
            workflow_json: JSON representation of the N8N workflow
            
        Returns:
            Dict with the optimized workflow
        """
        logger.info(f"Optimizing data transformations in workflow")
        
        # Create a message with the data transformation optimization request
        workflow_str = json.dumps(workflow_json, indent=2)
        transformation_message = TextMessage(
            content=f"""I need to optimize data transformation operations in an N8N workflow.

Here is the current workflow JSON:
```json
{workflow_str}
```

Please analyze and optimize the data transformation operations by:
1. Identifying inefficient transformation patterns
2. Replacing multiple transformations with more efficient alternatives
3. Optimizing JSON and string manipulations
4. Reducing unnecessary data copying or manipulation
5. Using Set nodes instead of Function nodes where appropriate

Provide the optimized workflow JSON and explain your optimization strategy.""",
            source="user"
        )
        
        # Get data transformation optimization response
        response = await self.n8n_assistant.on_messages(
            [transformation_message],
            cancellation_token=CancellationToken(),
        )
        
        # Extract the optimized workflow JSON from the response
        optimized_workflow_json = self._extract_json(response.chat_message.content)
        
        result = {
            "original_workflow": workflow_json,
            "explanation": response.chat_message.content,
        }
        
        if optimized_workflow_json:
            result["optimized_workflow"] = optimized_workflow_json
        
        return result
    
    async def reduce_api_calls(self, workflow_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize an N8N workflow by reducing redundant API calls.
        
        This method identifies patterns of redundant API calls and restructures the workflow
        to minimize the number of external calls while maintaining functionality.
        
        Args:
            workflow_json: JSON representation of the N8N workflow
            
        Returns:
            Dict with the optimized workflow
        """
        logger.info(f"Optimizing workflow to reduce API calls")
        
        # Create a message with the API call reduction request
        workflow_str = json.dumps(workflow_json, indent=2)
        api_optimization_message = TextMessage(
            content=f"""I need to optimize an N8N workflow by reducing redundant API calls.

Here is the current workflow JSON:
```json
{workflow_str}
```

Please analyze and optimize the workflow to reduce API calls by:
1. Identifying redundant or repeated API calls
2. Implementing caching for API responses
3. Batching multiple single-item API calls into bulk operations
4. Using pagination efficiently for large datasets
5. Implementing conditional execution to avoid unnecessary API calls

Provide the optimized workflow JSON and explain your API call reduction strategy.""",
            source="user"
        )
        
        # Get API call reduction response
        response = await self.n8n_assistant.on_messages(
            [api_optimization_message],
            cancellation_token=CancellationToken(),
        )
        
        # Extract the optimized workflow JSON from the response
        optimized_workflow_json = self._extract_json(response.chat_message.content)
        
        result = {
            "original_workflow": workflow_json,
            "explanation": response.chat_message.content,
        }
        
        if optimized_workflow_json:
            result["optimized_workflow"] = optimized_workflow_json
        
        return result
    
    async def create_workflow(self, workflow_description: str) -> Dict[str, Any]:
        """
        Create an N8N workflow based on a natural language description.
        
        Args:
            workflow_description: Natural language description of the workflow
            
        Returns:
            Dict with the created workflow
        """
        logger.info(f"Creating N8N workflow from description")
        
        # Check cache if caching is enabled
        if self.enable_caching:
            cache_key = self.generate_cache_key("create_workflow", workflow_description)
            cached_result = self.get_cached_result(cache_key)
            if cached_result is not None:
                logger.info(f"Using cached workflow for description")
                return cached_result
        
        # Create a message with the workflow creation request
        workflow_message = TextMessage(
            content=f"""I need to create an N8N workflow based on the following description:

{workflow_description}

Please create a complete N8N workflow JSON that implements this functionality.
The workflow should follow N8N best practices and include proper error handling.
Please provide the workflow JSON and a brief explanation of how it works.""",
            source="user"
        )
        
        # Get workflow creation response
        response = await self.n8n_assistant.on_messages(
            [workflow_message],
            cancellation_token=CancellationToken(),
        )
        
        # Extract the workflow JSON from the response
        workflow_json = self._extract_json(response.chat_message.content)
        
        result = {
            "description": workflow_description,
            "explanation": response.chat_message.content,
        }
        
        if workflow_json:
            result["workflow_json"] = workflow_json
            
            # Deploy the workflow if N8N API is configured
            if self.n8n_api_url and self.n8n_api_key:
                deployment_result = self._deploy_workflow(workflow_json)
                result["deployment"] = deployment_result
                
            # If the workflow was successfully created, analyze its performance
            if self.performance_tracking:
                try:
                    performance_analysis = await self.analyze_workflow_performance(workflow_json)
                    result["performance_analysis"] = performance_analysis
                except Exception as e:
                    logger.error(f"Error analyzing workflow performance: {e}")
                
            # Cache the result if caching is enabled
            if self.enable_caching:
                self.set_cached_result(cache_key, result)
        
        return result
    
    async def optimize_workflow(self, workflow_json: Dict[str, Any], optimization_goal: str = "performance") -> Dict[str, Any]:
        """
        Optimize an existing N8N workflow.
        
        Args:
            workflow_json: JSON representation of the N8N workflow
            optimization_goal: Goal of the optimization (performance, reliability, readability)
            
        Returns:
            Dict with the optimized workflow
        """
        logger.info(f"Optimizing N8N workflow for {optimization_goal}")
        
        # Check cache if caching is enabled
        if self.enable_caching:
            # Create a stable representation of the workflow JSON for caching
            workflow_str_for_cache = json.dumps(workflow_json, sort_keys=True)
            cache_key = self.generate_cache_key("optimize_workflow", workflow_str_for_cache, optimization_goal)
            cached_result = self.get_cached_result(cache_key)
            if cached_result is not None:
                logger.info(f"Using cached optimization result for {optimization_goal}")
                return cached_result
        
        # First, analyze the workflow performance if performance tracking is enabled
        if self.performance_tracking:
            try:
                performance_analysis = await self.analyze_workflow_performance(workflow_json)
                # Use the performance analysis to guide optimization
                optimization_suggestions = performance_analysis.get("optimization_suggestions", [])
            except Exception as e:
                logger.error(f"Error analyzing workflow performance: {e}")
                optimization_suggestions = []
        else:
            optimization_suggestions = []
        
        # Choose the appropriate optimization method based on the goal
        if optimization_goal == "performance":
            # For performance optimization, try parallel execution first
            try:
                parallel_result = await self.optimize_for_parallel_execution(workflow_json)
                if "parallelized_workflow" in parallel_result:
                    workflow_json = parallel_result["parallelized_workflow"]
            except Exception as e:
                logger.error(f"Error optimizing for parallel execution: {e}")
            
            # Then try to reduce API calls
            try:
                api_result = await self.reduce_api_calls(workflow_json)
                if "optimized_workflow" in api_result:
                    workflow_json = api_result["optimized_workflow"]
            except Exception as e:
                logger.error(f"Error reducing API calls: {e}")
            
            # Finally, optimize data transformations
            try:
                transform_result = await self.optimize_data_transformation(workflow_json)
                if "optimized_workflow" in transform_result:
                    workflow_json = transform_result["optimized_workflow"]
            except Exception as e:
                logger.error(f"Error optimizing data transformations: {e}")
        
        # Create a message with the workflow optimization request
        workflow_str = json.dumps(workflow_json, indent=2)
        suggestions_str = "\n".join([f"- {s}" for s in optimization_suggestions[:5]]) if optimization_suggestions else ""
        
        # Prepare the suggestions part separately to avoid nested f-strings
        suggestions_part = ""
        if suggestions_str:
            suggestions_part = f"Based on performance analysis, consider these suggestions:\n{suggestions_str}\n"
            
        optimization_message = TextMessage(
            content=f"""I have an N8N workflow that I want to optimize for {optimization_goal}.

Here is the current workflow JSON:
```json
{workflow_str}
```

{suggestions_part}Please optimize this workflow for {optimization_goal}. Consider the following:
1. Removing unnecessary nodes
2. Improving error handling
3. Optimizing data flow
4. Reducing API calls
5. Improving readability and maintainability

Please provide the optimized workflow JSON and explain the changes you made.""",
            source="user"
        )
        
        # Get workflow optimization response
        response = await self.n8n_assistant.on_messages(
            [optimization_message],
            cancellation_token=CancellationToken(),
        )
        
        # Extract the optimized workflow JSON from the response
        optimized_workflow_json = self._extract_json(response.chat_message.content)
        
        result = {
            "original_workflow": workflow_json,
            "optimization_goal": optimization_goal,
            "explanation": response.chat_message.content,
        }
        
        if optimized_workflow_json:
            result["optimized_workflow"] = optimized_workflow_json
            
            # If caching is enabled, cache the result
            if self.enable_caching:
                self.set_cached_result(cache_key, result)
        
        return result
    
    async def debug_workflow(self, workflow_json: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """
        Debug an N8N workflow that is producing errors.
        
        Args:
            workflow_json: JSON representation of the N8N workflow
            error_message: Error message from the workflow execution
            
        Returns:
            Dict with the debugged workflow
        """
        logger.info(f"Debugging N8N workflow")
        
        # Check cache if caching is enabled
        if self.enable_caching:
            # Create a stable representation of the workflow JSON and error for caching
            workflow_str_for_cache = json.dumps(workflow_json, sort_keys=True)
            cache_key = self.generate_cache_key("debug_workflow", workflow_str_for_cache, error_message)
            cached_result = self.get_cached_result(cache_key)
            if cached_result is not None:
                logger.info(f"Using cached debug result")
                return cached_result
        
        # Create a message with the workflow debugging request
        workflow_str = json.dumps(workflow_json, indent=2)
        debug_message = TextMessage(
            content=f"""I have an N8N workflow that is producing the following error:

Error message:
```
{error_message}
```

Here is the current workflow JSON:
```json
{workflow_str}
```

Please debug this workflow and fix the issues causing the error.
Provide the fixed workflow JSON and explain what was causing the error and how you fixed it.""",
            source="user"
        )
        
        # Get workflow debugging response
        response = await self.n8n_assistant.on_messages(
            [debug_message],
            cancellation_token=CancellationToken(),
        )
        
        # Extract the fixed workflow JSON from the response
        fixed_workflow_json = self._extract_json(response.chat_message.content)
        
        result = {
            "original_workflow": workflow_json,
            "error_message": error_message,
            "explanation": response.chat_message.content,
        }
        
        if fixed_workflow_json:
            result["fixed_workflow"] = fixed_workflow_json
            
            # If the workflow was successfully fixed, analyze its performance
            if self.performance_tracking and fixed_workflow_json:
                try:
                    performance_analysis = await self.analyze_workflow_performance(fixed_workflow_json)
                    result["performance_analysis"] = performance_analysis
                except Exception as e:
                    logger.error(f"Error analyzing fixed workflow performance: {e}")
            
            # If caching is enabled, cache the result
            if self.enable_caching:
                self.set_cached_result(cache_key, result)
        
        return result
    
    async def integrate_solnai_agent(self, workflow_json: Dict[str, Any], agent_type: str, integration_point: str) -> Dict[str, Any]:
        """
        Integrate a SolnAI agent into an N8N workflow.
        
        Args:
            workflow_json: JSON representation of the N8N workflow
            agent_type: Type of SolnAI agent to integrate (research, code_executor, data_analysis, etc.)
            integration_point: Description of where in the workflow to integrate the agent
            
        Returns:
            Dict with the integrated workflow
        """
        logger.info(f"Integrating {agent_type} agent into N8N workflow")
        
        # Check cache if caching is enabled
        if self.enable_caching:
            # Create a stable representation of the workflow JSON and integration parameters for caching
            workflow_str_for_cache = json.dumps(workflow_json, sort_keys=True)
            cache_key = self.generate_cache_key("integrate_solnai_agent", workflow_str_for_cache, agent_type, integration_point)
            cached_result = self.get_cached_result(cache_key)
            if cached_result is not None:
                logger.info(f"Using cached integration result for {agent_type} agent")
                return cached_result
        
        # Create a message with the agent integration request
        workflow_str = json.dumps(workflow_json, indent=2)
        integration_message = TextMessage(
            content=f"""I have an N8N workflow that I want to integrate with a SolnAI {agent_type} agent.

Here is the current workflow JSON:
```json
{workflow_str}
```

I want to integrate the agent at this point in the workflow: {integration_point}

Please modify the workflow to integrate the SolnAI {agent_type} agent.
The integration should:
1. Send data from the workflow to the agent
2. Wait for the agent to process the data
3. Receive the results from the agent
4. Continue the workflow with the agent's results

Please provide the modified workflow JSON and explain how the integration works.""",
            source="user"
        )
        
        # Get agent integration response
        response = await self.n8n_assistant.on_messages(
            [integration_message],
            cancellation_token=CancellationToken(),
        )
        
        # Extract the integrated workflow JSON from the response
        integrated_workflow_json = self._extract_json(response.chat_message.content)
        
        result = {
            "original_workflow": workflow_json,
            "agent_type": agent_type,
            "integration_point": integration_point,
            "explanation": response.chat_message.content,
        }
        
        if integrated_workflow_json:
            result["integrated_workflow"] = integrated_workflow_json
            
            # If the workflow was successfully integrated, analyze its performance
            if self.performance_tracking and integrated_workflow_json:
                try:
                    performance_analysis = await self.analyze_workflow_performance(integrated_workflow_json)
                    result["performance_analysis"] = performance_analysis
                except Exception as e:
                    logger.error(f"Error analyzing integrated workflow performance: {e}")
            
            # If caching is enabled, cache the result
            if self.enable_caching:
                self.set_cached_result(cache_key, result)
        
        return result
    
    async def generate_workflow_template(self, template_type: str, customizations: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a template N8N workflow for a specific use case.
        
        Args:
            template_type: Type of template to generate (data_processing, api_integration, email_automation, etc.)
            customizations: Dict of customizations to apply to the template
            
        Returns:
            Dict with the generated template workflow
        """
        logger.info(f"Generating {template_type} workflow template")
        
        # Check cache if caching is enabled
        if self.enable_caching:
            # Create a stable representation of the template parameters for caching
            customizations_str_for_cache = json.dumps(customizations, sort_keys=True) if customizations else ""
            cache_key = self.generate_cache_key("generate_workflow_template", template_type, customizations_str_for_cache)
            cached_result = self.get_cached_result(cache_key)
            if cached_result is not None:
                logger.info(f"Using cached template for {template_type}")
                return cached_result
        
        # Create customizations string
        customizations_str = ""
        if customizations:
            customizations_str = "with the following customizations:\n"
            for key, value in customizations.items():
                customizations_str += f"- {key}: {value}\n"
        
        # Create a message with the template generation request
        template_message = TextMessage(
            content=f"""I need to generate an N8N workflow template for {template_type} {customizations_str}

Please create a complete N8N workflow JSON that serves as a template for this use case.
The template should:
1. Follow N8N best practices
2. Include proper error handling
3. Be well-documented with comments
4. Be easily customizable

Please provide the template workflow JSON and a brief explanation of how it works and how it can be customized.""",
            source="user"
        )
        
        # Get template generation response
        response = await self.n8n_assistant.on_messages(
            [template_message],
            cancellation_token=CancellationToken(),
        )
        
        # Extract the template workflow JSON from the response
        template_workflow_json = self._extract_json(response.chat_message.content)
        
        result = {
            "template_type": template_type,
            "customizations": customizations,
            "explanation": response.chat_message.content,
        }
        
        if template_workflow_json:
            result["template_workflow"] = template_workflow_json
            
            # If the template was successfully generated, analyze its performance
            if self.performance_tracking and template_workflow_json:
                try:
                    performance_analysis = await self.analyze_workflow_performance(template_workflow_json)
                    result["performance_analysis"] = performance_analysis
                except Exception as e:
                    logger.error(f"Error analyzing template workflow performance: {e}")
            
            # If caching is enabled, cache the result
            if self.enable_caching:
                self.set_cached_result(cache_key, result)
        
        return result
    
    def clear_cache(self, pattern: Optional[str] = None) -> Dict[str, Any]:
        """
        Clear the cache, optionally filtering by a pattern.
        
        Args:
            pattern: Optional pattern to match cache keys against
            
        Returns:
            Dict with information about the cleared cache
        """
        if not self.enable_caching:
            return {"status": "error", "message": "Caching is not enabled"}
        
        cache_size_before = len(self.cache)
        keys_to_remove = []
        
        if pattern:
            # Remove only keys matching the pattern
            for key in self.cache.keys():
                if pattern in key:
                    keys_to_remove.append(key)
        else:
            # Remove all keys
            keys_to_remove = list(self.cache.keys())
        
        # Remove the keys
        for key in keys_to_remove:
            del self.cache[key]
        
        return {
            "status": "success",
            "cache_size_before": cache_size_before,
            "cache_size_after": len(self.cache),
            "keys_removed": len(keys_to_remove),
            "pattern": pattern
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current cache.
        
        Returns:
            Dict with cache statistics
        """
        if not self.enable_caching:
            return {"status": "error", "message": "Caching is not enabled"}
        
        # Group cache items by type
        cache_types = {}
        expired_count = 0
        current_time = time.time()
        
        for key, item in self.cache.items():
            # Check if the item has expired
            expiration_time = item["timestamp"] + item["ttl"]
            if current_time > expiration_time:
                expired_count += 1
                continue
            
            # Extract the cache type from the key (first part before the hash)
            cache_type = key.split("_")[0] if "_" in key else "unknown"
            if cache_type not in cache_types:
                cache_types[cache_type] = 0
            cache_types[cache_type] += 1
        
        return {
            "status": "success",
            "total_items": len(self.cache),
            "expired_items": expired_count,
            "active_items": len(self.cache) - expired_count,
            "cache_types": cache_types,
            "cache_size_bytes": sys.getsizeof(self.cache)
        }
    
    def export_optimization_suggestions(self) -> Dict[str, Any]:
        """
        Export all collected optimization suggestions.
        
        Returns:
            Dict with optimization suggestions
        """
        if not self.performance_tracking:
            return {"status": "error", "message": "Performance tracking is not enabled"}
        
        # Group suggestions by category
        categorized_suggestions = {
            "performance": [],
            "reliability": [],
            "security": [],
            "maintainability": [],
            "other": []
        }
        
        # Keywords for categorization
        category_keywords = {
            "performance": ["performance", "speed", "efficient", "fast", "slow", "optimize", "bottleneck", "cache", "parallel"],
            "reliability": ["reliability", "error", "exception", "fail", "crash", "robust", "handle", "recover"],
            "security": ["security", "secure", "auth", "encrypt", "protect", "vulnerability", "risk"],
            "maintainability": ["maintainability", "readable", "clean", "document", "comment", "structure"]
        }
        
        # Categorize each suggestion
        for suggestion in self.optimization_suggestions:
            suggestion_lower = suggestion.lower()
            assigned = False
            
            for category, keywords in category_keywords.items():
                if any(keyword in suggestion_lower for keyword in keywords):
                    categorized_suggestions[category].append(suggestion)
                    assigned = True
                    break
            
            if not assigned:
                categorized_suggestions["other"].append(suggestion)
        
        return {
            "status": "success",
            "total_suggestions": len(self.optimization_suggestions),
            "categorized_suggestions": categorized_suggestions,
            "all_suggestions": list(self.optimization_suggestions)
        }
    
    def export_performance_analysis(self, workflow_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Export a comprehensive performance analysis for a workflow.
        
        Args:
            workflow_json: JSON representation of the N8N workflow
            
        Returns:
            Dict with performance analysis
        """
        if not self.performance_tracking:
            return {"status": "error", "message": "Performance tracking is not enabled"}
        
        try:
            # Extract basic workflow metrics
            node_count = len(workflow_json.get("nodes", []))
            connection_count = len(workflow_json.get("connections", {}).get("main", []))
            
            # Categorize nodes by type
            node_types = {}
            api_nodes = []
            transform_nodes = []
            trigger_nodes = []
            error_handling_nodes = []
            
            for node in workflow_json.get("nodes", []):
                node_type = node.get("type", "")
                node_types[node_type] = node_types.get(node_type, 0) + 1
                
                # Identify API nodes (HTTP Request, API, etc.)
                if "http" in node_type.lower() or "api" in node_type.lower():
                    api_nodes.append(node)
                
                # Identify transformation nodes (Set, Function, etc.)
                if node_type in ["Set", "Function", "FunctionItem", "Move Binary Data"]:
                    transform_nodes.append(node)
                
                # Identify trigger nodes
                if "trigger" in node_type.lower():
                    trigger_nodes.append(node)
                
                # Identify error handling nodes
                if "error" in node_type.lower():
                    error_handling_nodes.append(node)
            
            # Calculate complexity metrics
            complexity_score = node_count * 0.5 + connection_count * 0.3 + len(api_nodes) * 1.0
            error_handling_ratio = len(error_handling_nodes) / node_count if node_count > 0 else 0
            
            # Determine performance risk areas
            performance_risks = []
            
            if len(api_nodes) > 5:
                performance_risks.append("High number of API calls may cause performance issues")
            
            if error_handling_ratio < 0.1:
                performance_risks.append("Low error handling coverage may cause reliability issues")
            
            if node_count > 20 and len(transform_nodes) / node_count > 0.5:
                performance_risks.append("High proportion of transformation nodes may indicate inefficient data processing")
            
            # Create the analysis result
            result = {
                "status": "success",
                "workflow_metrics": {
                    "node_count": node_count,
                    "connection_count": connection_count,
                    "node_types": node_types,
                    "api_node_count": len(api_nodes),
                    "transform_node_count": len(transform_nodes),
                    "trigger_node_count": len(trigger_nodes),
                    "error_handling_node_count": len(error_handling_nodes),
                },
                "complexity_analysis": {
                    "complexity_score": complexity_score,
                    "error_handling_ratio": error_handling_ratio,
                    "complexity_level": "High" if complexity_score > 20 else "Medium" if complexity_score > 10 else "Low"
                },
                "performance_risks": performance_risks,
                "optimization_suggestions": list(self.optimization_suggestions)
            }
            
            return result
        except Exception as e:
            logger.error(f"Error exporting performance analysis: {e}")
            return {"status": "error", "message": str(e)}
    
    async def benchmark_workflow(self, workflow_json: Dict[str, Any], iterations: int = 5, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Benchmark a workflow's performance by executing it multiple times and measuring performance metrics.
        
        Args:
            workflow_json: JSON representation of the N8N workflow
            iterations: Number of times to execute the workflow
            input_data: Optional input data for the workflow
            
        Returns:
            Dict with benchmark results
        """
        if not self.performance_tracking:
            return {"status": "error", "message": "Performance tracking is not enabled"}
        
        if not self.n8n_api_url or not self.n8n_api_key:
            return {"status": "error", "message": "N8N API connection not configured"}
        
        try:
            # First, import the workflow to N8N
            import_response = self._import_workflow(workflow_json)
            if not import_response or "id" not in import_response:
                return {"status": "error", "message": "Failed to import workflow for benchmarking"}
            
            workflow_id = import_response["id"]
            execution_times = []
            memory_usage = []
            success_count = 0
            error_count = 0
            errors = []
            
            logger.info(f"Benchmarking workflow {workflow_id} with {iterations} iterations")
            
            # Execute the workflow multiple times
            for i in range(iterations):
                start_time = time.time()
                
                try:
                    # Execute the workflow
                    execution_result = self._execute_workflow(workflow_id, input_data)
                    
                    # Record execution time
                    end_time = time.time()
                    execution_time = end_time - start_time
                    execution_times.append(execution_time)
                    
                    # Check if execution was successful
                    if execution_result and "finished" in execution_result and execution_result["finished"]:
                        success_count += 1
                        
                        # Try to extract memory usage if available
                        if "data" in execution_result and "memoryUsage" in execution_result["data"]:
                            memory_usage.append(execution_result["data"]["memoryUsage"])
                    else:
                        error_count += 1
                        errors.append(f"Execution {i+1} failed: {execution_result}")
                except Exception as e:
                    error_count += 1
                    errors.append(f"Execution {i+1} error: {str(e)}")
                
                # Add a small delay between executions
                await asyncio.sleep(0.5)
            
            # Calculate statistics
            avg_execution_time = statistics.mean(execution_times) if execution_times else 0
            median_execution_time = statistics.median(execution_times) if execution_times else 0
            min_execution_time = min(execution_times) if execution_times else 0
            max_execution_time = max(execution_times) if execution_times else 0
            std_dev_execution_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            
            avg_memory_usage = statistics.mean(memory_usage) if memory_usage else 0
            
            # Clean up - delete the imported workflow
            self._delete_workflow(workflow_id)
            
            # Prepare benchmark results
            benchmark_results = {
                "status": "success",
                "workflow_id": workflow_id,
                "iterations": iterations,
                "successful_executions": success_count,
                "failed_executions": error_count,
                "execution_times": {
                    "average": avg_execution_time,
                    "median": median_execution_time,
                    "min": min_execution_time,
                    "max": max_execution_time,
                    "standard_deviation": std_dev_execution_time,
                    "raw_times": execution_times
                },
                "memory_usage": {
                    "average": avg_memory_usage,
                    "raw_usage": memory_usage
                },
                "errors": errors,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Store benchmark results in performance metrics
            self.performance_metrics[workflow_id] = benchmark_results
            
            return benchmark_results
        except Exception as e:
            logger.error(f"Error benchmarking workflow: {e}")
            return {"status": "error", "message": str(e)}
    
    def _import_workflow(self, workflow_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Import a workflow into N8N.
        
        Args:
            workflow_json: JSON representation of the N8N workflow
            
        Returns:
            Dict with the imported workflow details
        """
        if not self.n8n_api_url or not self.n8n_api_key:
            logger.error("N8N API connection not configured")
            return {}
        
        try:
            headers = {
                "X-N8N-API-KEY": self.n8n_api_key,
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.n8n_api_url}/workflows",
                headers=headers,
                json=workflow_json
            )
            
            if response.status_code in (200, 201):
                return response.json()
            else:
                logger.error(f"Failed to import workflow: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            logger.error(f"Error importing workflow: {e}")
            return {}
    
    def _execute_workflow(self, workflow_id: str, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a workflow in N8N.
        
        Args:
            workflow_id: ID of the workflow to execute
            input_data: Optional input data for the workflow
            
        Returns:
            Dict with the execution result
        """
        if not self.n8n_api_url or not self.n8n_api_key:
            logger.error("N8N API connection not configured")
            return {}
        
        try:
            headers = {
                "X-N8N-API-KEY": self.n8n_api_key,
                "Content-Type": "application/json"
            }
            
            payload = {}
            if input_data:
                payload["data"] = input_data
            
            response = requests.post(
                f"{self.n8n_api_url}/workflows/{workflow_id}/execute",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to execute workflow: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            return {}
    
    def _delete_workflow(self, workflow_id: str) -> bool:
        """
        Delete a workflow from N8N.
        
        Args:
            workflow_id: ID of the workflow to delete
            
        Returns:
            Boolean indicating success or failure
        """
        if not self.n8n_api_url or not self.n8n_api_key:
            logger.error("N8N API connection not configured")
            return False
        
        try:
            headers = {
                "X-N8N-API-KEY": self.n8n_api_key
            }
            
            response = requests.delete(
                f"{self.n8n_api_url}/workflows/{workflow_id}",
                headers=headers
            )
            
            return response.status_code in (200, 204)
        except Exception as e:
            logger.error(f"Error deleting workflow: {e}")
            return False
    
    async def run_interactive_session(self) -> None:
        """
        Run an interactive N8N integration session.
        """
        logger.info("Starting interactive N8N integration session")
        
        # Create a console for interaction
        await Console(
            self.n8n_assistant.on_messages_stream(
                [TextMessage(content="I'm ready to help with N8N workflow integration. What would you like to do?", source="user")],
                cancellation_token=CancellationToken(),
            ),
            output_stats=True,
        )
    
    def _extract_optimization_suggestions(self, text: str) -> set[str]:
        """
        Extract optimization suggestions from the text response.
        
        Args:
            text: Text containing optimization suggestions
            
        Returns:
            Set of optimization suggestions
        """
        import re
        
        suggestions = set()
        
        # Look for numbered lists (1. Something)
        numbered_suggestions = re.findall(r'\d+\.\s*([^\n]+)', text)
        for suggestion in numbered_suggestions:
            suggestion = suggestion.strip()
            if suggestion and len(suggestion) > 10:  # Avoid very short phrases
                suggestions.add(suggestion)
        
        # Look for bullet points
        bullet_suggestions = re.findall(r'[•\-*]\s*([^\n]+)', text)
        for suggestion in bullet_suggestions:
            suggestion = suggestion.strip()
            if suggestion and len(suggestion) > 10:  # Avoid very short phrases
                suggestions.add(suggestion)
        
        # Look for suggestions in paragraphs containing keywords
        keywords = ["optimize", "improve", "efficiency", "performance", "bottleneck", "cache", "parallel"]
        paragraphs = text.split('\n\n')
        for paragraph in paragraphs:
            if any(keyword in paragraph.lower() for keyword in keywords):
                # Clean up the paragraph
                clean_paragraph = re.sub(r'\s+', ' ', paragraph).strip()
                if clean_paragraph and len(clean_paragraph) > 20:  # Longer to ensure it's a real suggestion
                    suggestions.add(clean_paragraph)
        
        return suggestions
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """
        Get a cached result if available and not expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        if not self.enable_caching:
            return None
        
        if key not in self.cache:
            return None
        
        cached_item = self.cache[key]
        expiration_time = cached_item["timestamp"] + cached_item["ttl"]
        
        # Check if the cached item has expired
        if time.time() > expiration_time:
            # Remove the expired item
            del self.cache[key]
            return None
        
        return cached_item["value"]
    
    def set_cached_result(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Cache a result for future use.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (optional, defaults to self.cache_ttl)
        """
        if not self.enable_caching:
            return
        
        ttl = ttl if ttl is not None else self.cache_ttl
        
        self.cache[key] = {
            "value": value,
            "timestamp": time.time(),
            "ttl": ttl
        }
    
    def generate_cache_key(self, *args: Any, **kwargs: Any) -> str:
        """
        Generate a deterministic cache key from the inputs.
        
        Args:
            *args: Positional arguments to include in the key
            **kwargs: Keyword arguments to include in the key
            
        Returns:
            A string cache key
        """
        # Convert args and kwargs to a stable string representation
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        
        # Join all parts and create a hash
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def run_with_cache(self, func: Callable, *args: Any, ttl: Optional[int] = None, **kwargs: Any) -> Any:
        """
        Run a function with caching support.
        
        Args:
            func: Function to run
            *args: Positional arguments for the function
            ttl: Time-to-live for the cached result (optional)
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result (from cache if available)
        """
        if not self.enable_caching:
            return await func(*args, **kwargs)
        
        # Generate a cache key for this function call
        cache_key = self.generate_cache_key(func.__name__, *args, **kwargs)
        
        # Check if we have a cached result
        cached_result = self.get_cached_result(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for {func.__name__}")
            return cached_result
        
        # No cached result, run the function
        logger.info(f"Cache miss for {func.__name__}")
        result = await func(*args, **kwargs)
        
        # Cache the result
        self.set_cached_result(cache_key, result, ttl)
        
        return result
    
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from text.
        
        Args:
            text: Text containing JSON
            
        Returns:
            Extracted JSON as dict or None if not found
        """
        import re
        
        # Find JSON blocks in the text
        json_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)```', text)
        
        if not json_blocks:
            # Try to find JSON without code blocks
            json_blocks = re.findall(r'(\{\s*"[^"]+"\s*:[\s\S]*\})', text)
        
        for block in json_blocks:
            try:
                # Try to parse the JSON
                json_obj = json.loads(block.strip())
                return json_obj
            except json.JSONDecodeError:
                continue
        
        return None
    
    def _deploy_workflow(self, workflow_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deploy a workflow to N8N.
        
        Args:
            workflow_json: JSON representation of the N8N workflow
            
        Returns:
            Dict with deployment results
        """
        if not self.n8n_api_url or not self.n8n_api_key:
            return {
                "success": False,
                "error": "N8N API URL or API key not configured"
            }
        
        try:
            # Set up the request headers
            headers = {
                "Content-Type": "application/json",
                "X-N8N-API-KEY": self.n8n_api_key
            }
            
            # Create the workflow
            create_url = f"{self.n8n_api_url}/workflows"
            response = requests.post(create_url, json=workflow_json, headers=headers)
            
            if response.status_code == 200:
                workflow_data = response.json()
                return {
                    "success": True,
                    "workflow_id": workflow_data.get("id"),
                    "workflow_name": workflow_data.get("name"),
                    "workflow_url": f"{self.n8n_api_url}/workflow/{workflow_data.get('id')}"
                }
            else:
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "error": response.text
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = N8NIntegrationAgent()
    
    # Example: Run an interactive session
    asyncio.run(agent.run_interactive_session())
