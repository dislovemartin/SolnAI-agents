# Code Executor Agent for SolnAI

## Overview

The Code Executor Agent is an intelligent assistant built on AutoGen v0.4.7 that specializes in executing, debugging, optimizing, and explaining code. It leverages Claude 3.7 Sonnet to provide high-quality code development capabilities within the SolnAI ecosystem.

## Features

- **Code Execution**: Safely executes code in various programming languages
- **Debugging**: Identifies and fixes code issues with detailed explanations
- **Optimization**: Improves code for better performance, readability, or memory usage
- **Code Explanation**: Provides detailed explanations of code functionality
- **Code Generation**: Creates code based on specific requirements
- **Interactive Sessions**: Supports interactive code development sessions

## Architecture

The Code Executor Agent is built using the following components:

- **AutoGen v0.4.7**: Provides the multi-agent framework
- **Claude 3.7 Sonnet**: Powers the code understanding and generation
- **AssistantAgent**: Handles code analysis, debugging, and optimization
- **CodeExecutorAgent**: Manages safe code execution
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
   cd solnai-app/SolnAI-agents/code-executor-agent
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

### Executing Code

```python
import asyncio
from code_executor_agent import SolnAICodeExecutorAgent

async def main():
    # Initialize the agent
    agent = SolnAICodeExecutorAgent()
    
    # Code to execute
    python_code = """
    def greet(name):
        return f"Hello, {name}!"
    
    result = greet("World")
    print(result)
    """
    
    # Execute the code
    result = await agent.execute_code(python_code, language="python")
    
    # Print the execution result
    print(result["output"])

# Run the example
asyncio.run(main())
```

### Debugging Code

```python
import asyncio
from code_executor_agent import SolnAICodeExecutorAgent

async def main():
    # Initialize the agent
    agent = SolnAICodeExecutorAgent()
    
    # Code with a bug
    buggy_code = """
    def divide(a, b):
        return a / b
    
    result = divide(10, 0)
    print(result)
    """
    
    # Error message from execution
    error_message = "ZeroDivisionError: division by zero"
    
    # Debug the code
    result = await agent.debug_code(buggy_code, error_message, language="python")
    
    # Print the debugging analysis
    print(result["analysis"])
    
    # Print the fixed code if available
    if "fixed_code" in result:
        print("\nFixed code:")
        print(result["fixed_code"])

# Run the example
asyncio.run(main())
```

### Optimizing Code

```python
import asyncio
from code_executor_agent import SolnAICodeExecutorAgent

async def main():
    # Initialize the agent
    agent = SolnAICodeExecutorAgent()
    
    # Code to optimize
    code_to_optimize = """
    def fibonacci(n):
        if n <= 0:
            return []
        
        fib = []
        for i in range(n):
            if i <= 1:
                fib.append(i)
            else:
                fib.append(fib[i-1] + fib[i-2])
        
        return fib
    
    result = fibonacci(20)
    print(result)
    """
    
    # Optimize the code for performance
    result = await agent.optimize_code(
        code_to_optimize, 
        language="python", 
        optimization_goal="performance"
    )
    
    # Print the optimization analysis
    print(result["analysis"])
    
    # Print the optimized code if available
    if "optimized_code" in result:
        print("\nOptimized code:")
        print(result["optimized_code"])

# Run the example
asyncio.run(main())
```

### Interactive Session

```python
import asyncio
from code_executor_agent import SolnAICodeExecutorAgent

async def main():
    # Initialize the agent
    agent = SolnAICodeExecutorAgent()
    
    # Start an interactive code development session
    await agent.run_interactive_session()

# Run the example
asyncio.run(main())
```

## API Reference

### `SolnAICodeExecutorAgent`

The main class that provides code execution and development capabilities.

#### Methods

- `execute_code(code: str, language: str = "python") -> Dict[str, Any]`: Executes code and returns the results
- `debug_code(code: str, error_message: str, language: str = "python") -> Dict[str, Any]`: Debugs code and suggests fixes
- `optimize_code(code: str, language: str = "python", optimization_goal: str = "performance") -> Dict[str, Any]`: Optimizes code for a specific goal
- `explain_code(code: str, language: str = "python") -> Dict[str, Any]`: Explains code functionality in detail
- `generate_code(requirements: str, language: str = "python") -> Dict[str, Any]`: Generates code based on requirements
- `run_interactive_session() -> None`: Runs an interactive code development session

## Supported Languages

The Code Executor Agent supports the following programming languages by default:

- Python
- JavaScript
- TypeScript
- Bash

Additional languages can be added by specifying them in the `supported_languages` parameter during initialization.

## Integration with SolnAI

The Code Executor Agent is designed to integrate seamlessly with other SolnAI agents:

- Can be used by the Research Agent to execute code for data analysis
- Works with Crawl4AI agent to process and analyze crawled data
- Supports multi-agent workflows for complex code development tasks

## Security Considerations

The Code Executor Agent executes code in a controlled environment, but there are still security considerations to keep in mind:

- Code execution is performed in a sandboxed environment
- Avoid executing untrusted code that may have malicious intent
- Set appropriate resource limits to prevent excessive resource consumption
- Consider using Docker containers for additional isolation

## Limitations

- The current implementation executes code in the local environment
- Performance depends on the quality of the Claude 3.7 Sonnet model and its API availability
- Complex debugging may require multiple iterations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
