# Crawl4AI Agent for SolnAI

## Overview

Crawl4AI is a specialized agent for SolnAI that performs intelligent web crawling and data extraction. Built on AutoGen v0.4.7, this agent can autonomously navigate websites, extract structured data, and integrate the information with other SolnAI agents.

## Features

- **Intelligent Web Crawling**: Autonomously navigates websites following semantic relevance rather than just link structure
- **Selective Data Extraction**: Identifies and extracts relevant information based on context and user requirements
- **Content Summarization**: Generates concise summaries of crawled content using Claude 3.7 Sonnet
- **Multi-format Support**: Handles various content types including HTML, PDF, images (via OCR), and JavaScript-rendered content
- **Rate Limiting & Politeness**: Built-in mechanisms to respect robots.txt, rate limits, and ethical crawling practices
- **Integration with SolnAI**: Seamlessly works with other agents in the SolnAI ecosystem

## Architecture

The Crawl4AI agent consists of several components:

1. **Crawler Engine**: Core component for web navigation and content retrieval
2. **Content Processor**: Extracts and processes different content types
3. **LLM Integration**: Uses Claude 3.7 Sonnet for content understanding and summarization
4. **Memory Store**: Maintains context and history of crawled content
5. **API Interface**: Exposes functionality to other SolnAI agents

## Installation

```bash
# From the SolnAI-agents directory
cd crawl4AI-agent
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with the following variables:

```
ANTHROPIC_API_KEY=your_api_key_here
CRAWL4AI_RATE_LIMIT=5  # requests per second
CRAWL4AI_MAX_DEPTH=3   # maximum crawl depth
CRAWL4AI_USER_AGENT="Crawl4AI Bot (https://solnai.com/bot)"
```

## Usage

### Basic Usage

```python
from crawl4ai import Crawl4AIAgent
from autogen import UserProxyAgent

# Initialize the agent
crawler = Crawl4AIAgent(
    name="web_crawler",
    system_message="You are a helpful web crawler that finds and extracts information.",
    llm_config={
        "model": "claude-3-7-sonnet",
        "temperature": 0.7
    }
)

# Create a user proxy
user = UserProxyAgent(
    name="user",
    human_input_mode="TERMINATE",
    code_execution_config={"work_dir": "crawl_workspace"}
)

# Start a conversation
user.initiate_chat(
    crawler,
    message="Crawl https://example.com and extract all product information."
)
```

### Integration with SolnAI

```python
from soln_ai.agents import TeamManager
from crawl4ai import Crawl4AIAgent
from soln_ai.llm.claude_wrapper import ClaudeWrapper

# Initialize Claude wrapper
claude = ClaudeWrapper()

# Create the crawler agent
crawler = Crawl4AIAgent(
    name="web_crawler",
    system_message="You are a helpful web crawler that finds and extracts information.",
    llm_config=claude.get_llm_config()
)

# Add to a team
team = TeamManager()
team.add_agent(crawler)
team.add_agent(research_agent)
team.add_agent(writer_agent)

# Start the team task
team.initiate_task("Research the latest AI trends and write a report.")
```

## API Reference

### Crawl4AIAgent Class

```python
class Crawl4AIAgent(ConversableAgent):
    """
    Agent for web crawling and data extraction.
    """
    
    def __init__(self, name, system_message, llm_config, **kwargs):
        """Initialize the Crawl4AI agent."""
        super().__init__(name=name, system_message=system_message, llm_config=llm_config, **kwargs)
        
        # Register tools
        self.register_tool(self.crawl_url)
        self.register_tool(self.extract_data)
        self.register_tool(self.summarize_content)
        
    def crawl_url(self, url, max_depth=1, follow_links=True):
        """Crawl a URL and its linked pages up to max_depth."""
        # Implementation details...
        
    def extract_data(self, html_content, extraction_pattern):
        """Extract structured data from HTML content."""
        # Implementation details...
        
    def summarize_content(self, content, max_length=500):
        """Generate a summary of the content."""
        # Implementation details...
```

## Advanced Features

### Custom Extraction Patterns

You can define custom extraction patterns to target specific information:

```python
extraction_pattern = {
    "products": {
        "selector": ".product-item",
        "fields": {
            "name": ".product-name",
            "price": ".product-price",
            "description": ".product-description"
        }
    }
}

results = crawler.extract_data(html_content, extraction_pattern)
```

### Content Filtering

Filter crawled content based on relevance:

```python
crawler.set_content_filter(
    relevance_threshold=0.7,
    required_keywords=["AI", "machine learning"],
    excluded_sections=[".sidebar", ".comments"]
)
```

## Limitations

- JavaScript-heavy sites may require additional configuration
- Some websites may block automated crawling
- Processing large volumes of content may require significant resources
- OCR functionality for images has limited accuracy

## Contributing

Contributions to Crawl4AI are welcome! Please see our [contributing guidelines](../../CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
