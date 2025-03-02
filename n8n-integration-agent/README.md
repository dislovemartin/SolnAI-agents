# N8N Integration Agent for SolnAI

## Overview

The N8N Integration Agent is an intelligent assistant built on AutoGen v0.4.7 that specializes in integrating SolnAI with N8N workflows. It leverages Claude 3.7 Sonnet to provide high-quality workflow automation capabilities within the SolnAI ecosystem.

## Features

- **Workflow Creation**: Creates N8N workflows based on natural language descriptions
- **Workflow Optimization**: Optimizes existing workflows for performance, reliability, and readability
- **Workflow Debugging**: Identifies and fixes issues in workflows that are producing errors
- **SolnAI Agent Integration**: Integrates SolnAI agents into N8N workflows
- **Template Generation**: Generates template workflows for common use cases
- **Interactive Sessions**: Supports interactive workflow development sessions

## Architecture

The N8N Integration Agent is built using the following components:

- **AutoGen v0.4.7**: Provides the multi-agent framework
- **Claude 3.7 Sonnet**: Powers the workflow understanding and generation
- **AssistantAgent**: Handles the core workflow development functionality
- **UserProxyAgent**: Manages user interactions and code execution
- **N8N API Integration**: Connects with the N8N API for workflow deployment

## Installation

### Prerequisites

- Python 3.8 or higher
- AutoGen v0.4.7
- Anthropic API key
- N8N instance with API access (optional, for deployment)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/solnai-app.git
   cd solnai-app/SolnAI-agents/n8n-integration-agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   ```bash
   export ANTHROPIC_API_KEY="your_anthropic_api_key"
   export N8N_API_URL="https://your-n8n-instance/api/v1"
   export N8N_API_KEY="your_n8n_api_key"
   ```

## Usage

### Creating a Workflow

```python
import asyncio
from n8n_integration_agent import N8NIntegrationAgent

async def main():
    # Initialize the agent
    agent = N8NIntegrationAgent()
    
    # Create a workflow based on a description
    workflow_description = """
    Create a workflow that monitors a Gmail inbox for new emails with attachments,
    extracts the attachments, processes them using the SolnAI Research Agent,
    and sends a summary of the findings back to the sender.
    """
    
    result = await agent.create_workflow(workflow_description)
    
    # Print the workflow JSON
    if "workflow_json" in result:
        print(json.dumps(result["workflow_json"], indent=2))

# Run the example
asyncio.run(main())
```

### Optimizing a Workflow

```python
import asyncio
import json
from n8n_integration_agent import N8NIntegrationAgent

async def main():
    # Initialize the agent
    agent = N8NIntegrationAgent()
    
    # Load an existing workflow
    with open("path/to/workflow.json", "r") as f:
        workflow_json = json.load(f)
    
    # Optimize the workflow for performance
    result = await agent.optimize_workflow(
        workflow_json=workflow_json,
        optimization_goal="performance"
    )
    
    # Print the optimized workflow JSON
    if "optimized_workflow" in result:
        print(json.dumps(result["optimized_workflow"], indent=2))

# Run the example
asyncio.run(main())
```

### Debugging a Workflow

```python
import asyncio
import json
from n8n_integration_agent import N8NIntegrationAgent

async def main():
    # Initialize the agent
    agent = N8NIntegrationAgent()
    
    # Load a workflow with errors
    with open("path/to/workflow.json", "r") as f:
        workflow_json = json.load(f)
    
    # Debug the workflow
    error_message = "Error: Cannot read property 'data' of undefined"
    result = await agent.debug_workflow(
        workflow_json=workflow_json,
        error_message=error_message
    )
    
    # Print the fixed workflow JSON
    if "fixed_workflow" in result:
        print(json.dumps(result["fixed_workflow"], indent=2))

# Run the example
asyncio.run(main())
```

### Integrating a SolnAI Agent

```python
import asyncio
import json
from n8n_integration_agent import N8NIntegrationAgent

async def main():
    # Initialize the agent
    agent = N8NIntegrationAgent()
    
    # Load an existing workflow
    with open("path/to/workflow.json", "r") as f:
        workflow_json = json.load(f)
    
    # Integrate a Research Agent
    result = await agent.integrate_solnai_agent(
        workflow_json=workflow_json,
        agent_type="research",
        integration_point="After the HTTP Request node to analyze the API response"
    )
    
    # Print the integrated workflow JSON
    if "integrated_workflow" in result:
        print(json.dumps(result["integrated_workflow"], indent=2))

# Run the example
asyncio.run(main())
```

### Generating a Template Workflow

```python
import asyncio
from n8n_integration_agent import N8NIntegrationAgent

async def main():
    # Initialize the agent
    agent = N8NIntegrationAgent()
    
    # Generate a template workflow for data processing
    customizations = {
        "data_source": "CSV file",
        "processing_steps": ["Filter", "Transform", "Aggregate"],
        "output_destination": "Google Sheets"
    }
    
    result = await agent.generate_workflow_template(
        template_type="data_processing",
        customizations=customizations
    )
    
    # Print the template workflow JSON
    if "template_workflow" in result:
        print(json.dumps(result["template_workflow"], indent=2))

# Run the example
asyncio.run(main())
```

### Interactive Session

```python
import asyncio
from n8n_integration_agent import N8NIntegrationAgent

async def main():
    # Initialize the agent
    agent = N8NIntegrationAgent()
    
    # Start an interactive workflow development session
    await agent.run_interactive_session()

# Run the example
asyncio.run(main())
```

## API Reference

### `N8NIntegrationAgent`

The main class that provides N8N integration capabilities.

#### Methods

- `create_workflow(workflow_description: str) -> Dict[str, Any]`: Creates an N8N workflow based on a natural language description
- `optimize_workflow(workflow_json: Dict[str, Any], optimization_goal: str = "performance") -> Dict[str, Any]`: Optimizes an existing N8N workflow
- `debug_workflow(workflow_json: Dict[str, Any], error_message: str) -> Dict[str, Any]`: Debugs an N8N workflow that is producing errors
- `integrate_solnai_agent(workflow_json: Dict[str, Any], agent_type: str, integration_point: str) -> Dict[str, Any]`: Integrates a SolnAI agent into an N8N workflow
- `generate_workflow_template(template_type: str, customizations: Dict[str, Any] = None) -> Dict[str, Any]`: Generates a template N8N workflow for a specific use case
- `run_interactive_session() -> None`: Runs an interactive N8N integration session

## N8N Integration

The N8N Integration Agent can connect with N8N in the following ways:

1. **Workflow Creation**: Generate workflow JSON that can be imported into N8N
2. **Direct API Integration**: Deploy workflows directly to N8N using the N8N API
3. **SolnAI Agent Integration**: Connect SolnAI agents with N8N workflows
4. **Workflow Optimization**: Improve existing N8N workflows

## Integration with SolnAI

The N8N Integration Agent is designed to integrate seamlessly with other SolnAI agents:

- Can use the Research Agent to gather information for workflows
- Works with the Code Executor Agent for custom code execution in workflows
- Integrates with the Data Analysis Agent for data processing in workflows
- Supports the SolnAI message format for interoperability

## Limitations

- The current implementation requires an N8N instance with API access for deployment
- Complex workflows may require manual adjustments
- Performance depends on the quality of the Claude 3.7 Sonnet model and its API availability

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
