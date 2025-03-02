# Research Agent for SolnAI

## Overview

The Research Agent is an intelligent assistant built on AutoGen v0.4.7 that specializes in conducting comprehensive research, analyzing information, and generating well-structured reports. It leverages Claude 3.7 Sonnet to provide high-quality research capabilities within the SolnAI ecosystem.

## Features

- **Comprehensive Research**: Conducts in-depth research on any given topic
- **Information Analysis**: Analyzes and synthesizes information from multiple sources
- **Report Generation**: Creates well-structured research reports in various formats
- **Question Answering**: Provides detailed answers to complex questions
- **Interactive Sessions**: Supports interactive research sessions

## Architecture

The Research Agent is built using the following components:

- **AutoGen v0.4.7**: Provides the multi-agent framework
- **Claude 3.7 Sonnet**: Powers the natural language understanding and generation
- **AssistantAgent**: Handles the core research functionality
- **UserProxyAgent**: Manages user interactions
- **BufferedChatCompletionContext**: Maintains conversation context efficiently

## Installation

### Prerequisites

- Python 3.8 or higher
- AutoGen v0.4.7
- Anthropic API key

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/solnai-app.git
   cd solnai-app/SolnAI-agents/research-agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   ```bash
   export ANTHROPIC_API_KEY="your_anthropic_api_key"
   ```

## Usage

### Basic Research

```python
import asyncio
from research_agent import ResearchAgent

async def main():
    # Initialize the agent
    agent = ResearchAgent()
    
    # Conduct research on a topic
    results = await agent.research("The impact of artificial intelligence on healthcare")
    
    # Print the research results
    print(results["response"])

# Run the example
asyncio.run(main())
```

### Generating Reports

```python
import asyncio
from research_agent import ResearchAgent

async def main():
    # Initialize the agent
    agent = ResearchAgent()
    
    # Generate a research report
    report = await agent.generate_report(
        topic="Renewable energy trends in 2025",
        report_format="markdown"
    )
    
    # Print the report
    print(report["report"])

# Run the example
asyncio.run(main())
```

### Interactive Session

```python
import asyncio
from research_agent import ResearchAgent

async def main():
    # Initialize the agent
    agent = ResearchAgent()
    
    # Start an interactive research session
    await agent.run_interactive_session()

# Run the example
asyncio.run(main())
```

## API Reference

### `ResearchAgent`

The main class that provides research capabilities.

#### Methods

- `research(topic: str) -> Dict[str, Any]`: Conducts research on a given topic
- `answer_question(question: str) -> Dict[str, Any]`: Answers a complex question with detailed research
- `generate_report(topic: str, report_format: str = "markdown") -> Dict[str, Any]`: Generates a structured research report
- `run_interactive_session() -> None`: Runs an interactive research session

## Integration with SolnAI

The Research Agent is designed to integrate seamlessly with other SolnAI agents:

- Can be used by the Crawl4AI agent to analyze crawled data
- Works with other agents in multi-agent workflows
- Supports the SolnAI message format for interoperability

## Limitations

- The current implementation simulates web search and data extraction tools
- Real-world implementation would require integration with actual search APIs and data extraction tools
- Performance depends on the quality of the Claude 3.7 Sonnet model and its API availability

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
