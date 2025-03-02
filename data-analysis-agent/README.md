# Data Analysis Agent for SolnAI

## Overview

The Data Analysis Agent is an intelligent assistant built on AutoGen v0.4.7 that specializes in analyzing data, generating insights, creating visualizations, and providing data-driven recommendations. It leverages Claude 3.7 Sonnet to provide high-quality data analysis capabilities within the SolnAI ecosystem.

## Features

- **Data Analysis**: Analyzes various data formats (CSV, JSON, Excel, etc.)
- **Statistical Insights**: Generates statistical insights from data
- **Visualization**: Creates clear and informative visualizations
- **Predictive Modeling**: Trains predictive models and makes predictions
- **Report Generation**: Creates comprehensive data analysis reports
- **Interactive Sessions**: Supports interactive data analysis sessions

## Architecture

The Data Analysis Agent is built using the following components:

- **AutoGen v0.4.7**: Provides the multi-agent framework
- **Claude 3.7 Sonnet**: Powers the data understanding and analysis
- **AssistantAgent**: Handles the core data analysis functionality
- **UserProxyAgent**: Executes code for data processing and visualization
- **BufferedChatCompletionContext**: Maintains conversation context efficiently

## Installation

### Prerequisites

- Python 3.8 or higher
- AutoGen v0.4.7
- Anthropic API key
- Data analysis libraries (pandas, numpy, matplotlib, seaborn, scikit-learn)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/solnai-app.git
   cd solnai-app/SolnAI-agents/data-analysis-agent
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

### Basic Data Analysis

```python
import asyncio
from data_analysis_agent import DataAnalysisAgent

async def main():
    # Initialize the agent
    agent = DataAnalysisAgent()
    
    # Analyze a dataset
    results = await agent.analyze_data(
        data_path="/path/to/your/data.csv",
        analysis_type="exploratory"
    )
    
    # Print the analysis summary
    if "summary" in results:
        print(results["summary"])

# Run the example
asyncio.run(main())
```

### Creating Visualizations

```python
import asyncio
from data_analysis_agent import DataAnalysisAgent

async def main():
    # Initialize the agent
    agent = DataAnalysisAgent()
    
    # Create a visualization
    results = await agent.create_visualization(
        data_path="/path/to/your/data.csv",
        visualization_type="bar",
        columns=["category", "value"]
    )
    
    # Print the path to the saved visualization
    if "visualization_path" in results:
        print(f"Visualization saved to: {results['visualization_path']}")

# Run the example
asyncio.run(main())
```

### Generating Reports

```python
import asyncio
from data_analysis_agent import DataAnalysisAgent

async def main():
    # Initialize the agent
    agent = DataAnalysisAgent()
    
    # Generate a data analysis report
    results = await agent.generate_report(
        data_path="/path/to/your/data.csv",
        report_format="markdown"
    )
    
    # Print the report content
    if "report_content" in results:
        print(results["report_content"])

# Run the example
asyncio.run(main())
```

### Predictive Modeling

```python
import asyncio
from data_analysis_agent import DataAnalysisAgent

async def main():
    # Initialize the agent
    agent = DataAnalysisAgent()
    
    # Train a predictive model
    results = await agent.predict(
        data_path="/path/to/your/data.csv",
        target_column="target",
        features=["feature1", "feature2", "feature3"],
        model_type="random_forest"
    )
    
    # Print the model metrics
    if "metrics" in results:
        print("Model Performance Metrics:")
        for metric, value in results["metrics"].items():
            print(f"{metric}: {value}")

# Run the example
asyncio.run(main())
```

### Interactive Session

```python
import asyncio
from data_analysis_agent import DataAnalysisAgent

async def main():
    # Initialize the agent
    agent = DataAnalysisAgent()
    
    # Start an interactive data analysis session
    await agent.run_interactive_session()

# Run the example
asyncio.run(main())
```

## API Reference

### `DataAnalysisAgent`

The main class that provides data analysis capabilities.

#### Methods

- `analyze_data(data_path: str, analysis_type: str = "general") -> Dict[str, Any]`: Analyzes data and generates insights
- `create_visualization(data_path: str, visualization_type: str, columns: List[str] = None) -> Dict[str, Any]`: Creates visualizations based on data
- `generate_report(data_path: str, report_format: str = "markdown") -> Dict[str, Any]`: Generates comprehensive data analysis reports
- `predict(data_path: str, target_column: str, features: List[str] = None, model_type: str = "auto") -> Dict[str, Any]`: Trains predictive models and makes predictions
- `run_interactive_session() -> None`: Runs an interactive data analysis session

## Supported Data Formats

The Data Analysis Agent supports the following data formats by default:

- CSV
- JSON
- Excel
- Parquet
- SQL

Additional formats can be added by specifying them in the `supported_formats` parameter during initialization.

## Integration with SolnAI

The Data Analysis Agent is designed to integrate seamlessly with other SolnAI agents:

- Can be used by the Research Agent to analyze research data
- Works with the Code Executor Agent for complex data processing tasks
- Supports the SolnAI message format for interoperability

## Limitations

- The current implementation executes code in the local environment
- Performance depends on the quality of the Claude 3.7 Sonnet model and its API availability
- Complex data analysis may require multiple iterations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
