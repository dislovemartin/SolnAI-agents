# Enhanced Migration Script for SolnAI Agents to AutoGen v0.4.7
---

"""
SolnAI Agents Migration to AutoGen v0.4.7 - Enhanced Rules File

Version: 1.0

Date: 2025-02-20

Description:

This script provides a comprehensive, step-by-step guide to migrate SolnAI agents 

to be compatible with AutoGen v0.4.7. It updates agent structures, configuration 

loading, message handling, multi-agent collaboration, and LLM integration (with 

Claude 3.7 Sonnet and MCP servers). Enhanced error handling, detailed logging, 

and unit test stubs are included to ensure a smooth migration process.


Prerequisites:

  - Clone the SolnAI-agents repository.

  - Install AutoGen v0.4.7: pip install pyautogen==0.4.7

  - Set API keys: ANTHROPIC_API_KEY, MCP_API_KEY, MCP_BASE_URL.

  - Use Python 3.8 or higher.

  - Install dependencies: pip install anthropic requests


Usage:

  Run this script to perform the migration. A report is saved as 'migration_report.json'

  and logs are written to 'migration.log'.

"""

```python
import os

import sys

import json

import time

import re

import logging

import pkg_resources

import random

import traceback

import math

from pathlib import Path

import functools

from typing import List, Dict, Any, Optional, Tuple


# -----------------------------------------------------------------------------

# Global Variables

# -----------------------------------------------------------------------------

CONFIG_PATH = "./config"

AGENT_CONFIG_PATH = os.path.join(CONFIG_PATH, "agents")

LLM_CONFIG_PATH = os.path.join(CONFIG_PATH, "llm")

AUTOGEN_CONFIG_PATH = os.path.join(CONFIG_PATH, "autogen_config.json")

TARGET_VERSION = "0.4.7"


logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s [%(levelname)s] %(message)s",

    handlers=[

        logging.FileHandler("migration.log"),

        logging.StreamHandler(sys.stdout)

    ]

)


# -----------------------------------------------------------------------------

# Version Validation

# -----------------------------------------------------------------------------

def validate_autogen_version() -> Tuple[bool, str]:

    """

    Verify that the installed AutoGen version is compatible with v0.4.7.


    Returns:

        tuple: (is_compatible: bool, message: str)

    """

    try:

        import autogen  # Ensure AutoGen is installed

        version = pkg_resources.get_distribution("pyautogen").version

        logging.info(f"Found AutoGen version: {version}")

        parts = version.split('.')

        if len(parts) < 3:

            parts.append('0')

        major, minor, patch = parts[0:3]

        patch = patch.split('-')[0]  # Remove any suffixes


        if version.startswith(TARGET_VERSION):

            logging.info(f"AutoGen version {version} matches target version {TARGET_VERSION}")

            return True, f"Compatible version detected: {version}"

        if int(major) == 0 and int(minor) == 4:

            if int(patch) == 7:

                return True, f"Compatible version detected: {version}"

            elif int(patch) > 7:

                msg = (f"AutoGen version {version} is newer than target {TARGET_VERSION}. "

                       f"Most features should work, but verify version-specific behavior.")

                logging.warning(msg)

                return True, msg

            else:

                msg = (f"AutoGen version {version} is older than target {TARGET_VERSION}. "

                       f"Consider upgrading to avoid issues.")

                logging.warning(msg)

                return False, msg

        else:

            msg = f"Incompatible version: {version}. Required: {TARGET_VERSION}"

            logging.error(msg)

            return False, msg

    except Exception as e:

        msg = f"AutoGen not installed or error checking version: {e}"

        logging.error(msg)

        return False, msg


# -----------------------------------------------------------------------------

# Directory and Config Setup

# -----------------------------------------------------------------------------

def create_directory_structure():

    """

    Create necessary directories for agents, configuration, examples, and tests.

    """

    dirs = [

        "soln_ai/agents", "soln_ai/utils", "soln_ai/llm",

        "soln_ai/roles", "soln_ai/protocols",

        "config", "examples", "tests"

    ]

    for d in dirs:

        Path(d).mkdir(parents=True, exist_ok=True)

        logging.info(f"Created directory: {d}")


def write_config_list_for_v047():

    """

    Write the AutoGen configuration file for v0.4.7.

    """

    config = {

        "config_list": [

            {"model": "claude-3-7-sonnet", "api_key": "${ANTHROPIC_API_KEY}"},

            {"model": "default", "api_key": "${MCP_API_KEY}", "base_url": "${MCP_BASE_URL}"}

        ]

    }

    with open(AUTOGEN_CONFIG_PATH, "w") as f:

        json.dump(config, f, indent=2)

    logging.info(f"Configuration file created at {AUTOGEN_CONFIG_PATH}")


# -----------------------------------------------------------------------------

# Agent Structure Transformation

# -----------------------------------------------------------------------------

def rule_convert_agent_class_definitions():

    """

    Update agent class definitions to use AutoGen v0.4.7 base classes.

    """

    transformation_stats = {"files_processed": 0, "class_updates": 0, "imports_updated": 0, "errors": []}

    class_mapping = {

        "BaseAgent": "ConversableAgent",  # v0.4.7 uses ConversableAgent

        "UserAgent": "UserProxyAgent"

    }

    import_replacements = [

        (r'from\s+soln_ai\.agents\s+import\s+([\w\s,]+)', r'from autogen import \g<1>'),

        (r'from\s+soln_ai\.utils\s+import\s+AgentConfig', r'from autogen.config import config_list_from_json')

    ]

    agent_files = list(Path("soln_ai/agents").glob("**/*.py"))

    transformation_stats["files_processed"] = len(agent_files)

    for file_path in agent_files:

        try:

            with open(file_path, "r") as f:

                content = f.read()

            original_content = content

            for pattern, replacement in import_replacements:

                content = re.sub(pattern, replacement, content)

                transformation_stats["imports_updated"] += 1

            for old, new in class_mapping.items():

                pattern = rf'class\s+(\w+)\s*\(\s*{old}\s*\):'

                matches = re.findall(pattern, content)

                transformation_stats["class_updates"] += len(matches)

                content = re.sub(pattern, f'class \\1({new}):', content)

            if content != original_content:

                with open(file_path, "w") as f:

                    f.write(content)

                logging.info(f"Updated {file_path}")

        except Exception as e:

            msg = f"Error processing {file_path}: {e}"

            logging.error(msg)

            transformation_stats["errors"].append(msg)

    logging.info(f"Agent class transformation stats: {transformation_stats}")

    return transformation_stats


def rule_update_agent_configuration_loading():

    """

    Update configuration loading to use AutoGen v0.4.7's config format.

    """

    from pathlib import Path

    files = list(Path(".").glob("**/config*.py")) + list(Path(".").glob("**/settings*.py"))

    for file_path in files:

        with open(file_path, "r") as f:

            content = f.read()

        content = content.replace("load_config", "config_list_from_json")

        with open(file_path, "w") as f:

            f.write(content)

        logging.info(f"Updated configuration loading in {file_path}")

    return {"files_updated": len(files)}


def rule_adapt_agent_state_management():

    """

    Update agent state management for AutoGen v0.4.7.

    """

    from pathlib import Path

    state_template = """

    def _save_state(self):

        \"\"\"Save agent state to a file.\"\"\"

        import json

        with open(f"{self.name}_state.json", "w") as f:

            json.dump(self._state, f)

    

    def _load_state(self):

        \"\"\"Load agent state from a file.\"\"\"

        import json

        try:

            with open(f"{self.name}_state.json", "r") as f:

                self._state = json.load(f)

        except Exception:

            self._state = {}

    """

    stats = {"files_processed": 0, "state_methods_added": 0, "errors": []}

    files = list(Path("soln_ai/agents").glob("**/*.py"))

    stats["files_processed"] = len(files)

    for file_path in files:

        try:

            with open(file_path, "r") as f:

                content = f.read()

            if "def _save_state" not in content:

                content += "\n" + state_template

                stats["state_methods_added"] += 1

                with open(file_path, "w") as f:

                    f.write(content)

                logging.info(f"Updated state management in {file_path}")

        except Exception as e:

            msg = f"Error updating state in {file_path}: {e}"

            logging.error(msg)

            stats["errors"].append(msg)

    return stats


# -----------------------------------------------------------------------------

# AutoGen v0.4.7 Compatibility Rules

# -----------------------------------------------------------------------------

def rule_update_message_handling_v047():

    """

    Update message handling methods to be compatible with AutoGen v0.4.7.

    """

    from pathlib import Path

    import re

    files = list(Path("soln_ai/agents").glob("**/*.py"))

    for file_path in files:

        with open(file_path, "r") as f:

            content = f.read()

        content = re.sub(r'self\.send_message\(', 'self.send(', content)

        content = re.sub(r'self\.receive_message\(', 'self.receive(', content)

        with open(file_path, "w") as f:

            f.write(content)

        logging.info(f"Updated message handling in {file_path}")

    return {"files_updated": len(files)}


def rule_adapt_function_calling_v047():

    """

    Adapt function calling to AutoGen v0.4.7 standards.

    """

    from pathlib import Path

    import re

    stats = {"files_processed": 0, "patterns_replaced": 0, "errors": []}

    files = list(Path("soln_ai/agents").glob("**/*.py"))

    stats["files_processed"] = len(files)

    patterns = [

        (r'self\.register_function\(', r'self.register_reply('),

        (r'self\.call_function\(', r'self.generate_reply(')

    ]

    for file_path in files:

        try:

            with open(file_path, "r") as f:

                content = f.read()

            original = content

            for pattern, replacement in patterns:

                new_content = re.sub(pattern, replacement, content)

                if new_content != content:

                    stats["patterns_replaced"] += 1

                    content = new_content

            if content != original:

                with open(file_path, "w") as f:

                    f.write(content)

                logging.info(f"Updated function calling in {file_path}")

        except Exception as e:

            msg = f"Error updating function calling in {file_path}: {e}"

            logging.error(msg)

            stats["errors"].append(msg)

    return stats


def rule_update_agent_initialization_v047():

    """

    Update agent __init__ methods to include required parameters for AutoGen v0.4.7.

    """

    from pathlib import Path

    import re

    stats = {"files_processed": 0, "inits_updated": 0, "errors": []}

    files = list(Path("soln_ai/agents").glob("**/*.py"))

    stats["files_processed"] = len(files)

    init_pattern = re.compile(r"def __init__\s*\(\s*self\s*,\s*(.*?)\):", re.DOTALL)

    for file_path in files:

        try:

            with open(file_path, "r") as f:

                content = f.read()

            original = content

            match = init_pattern.search(content)

            if match and "system_message" not in match.group(1):

                new_params = match.group(1) + ', system_message="I am an AI assistant."'

                content = content.replace(match.group(0), f"def __init__(self, {new_params}):")

                stats["inits_updated"] += 1

            if content != original:

                with open(file_path, "w") as f:

                    f.write(content)

                logging.info(f"Updated __init__ in {file_path}")

        except Exception as e:

            msg = f"Error updating __init__ in {file_path}: {e}"

            logging.error(msg)

            stats["errors"].append(msg)

    return stats


# -----------------------------------------------------------------------------

# Multi-Agent Collaboration Rules (v0.4.7 Specific)

# -----------------------------------------------------------------------------

def rule_implement_conversation_management_v047():

    """

    Implement conversation management suitable for AutoGen v0.4.7.

    

    Since v0.4.7 lacks GroupChat, we provide a manual coordination example.

    """

    from pathlib import Path

    conv_dir = Path("examples/conversation_management")

    conv_dir.mkdir(parents=True, exist_ok=True)

    conv_file = conv_dir / "conversation_example.py"

    example_code = r'''import os

import autogen

from autogen import ConversableAgent, UserProxyAgent, config_list_from_json

from soln_ai.llm.claude_wrapper import ClaudeWrapper

import logging


logging.basicConfig(level=logging.INFO)


# Initialize ClaudeWrapper for v0.4.7

claude_config = {"model": "claude-3-7-sonnet", "temperature": 0.7, "max_tokens": 4096}

claude = ClaudeWrapper(claude_config)

llm_config = claude.get_llm_config()


# Create agents

coder = ConversableAgent(name="Coder", system_message="I write code.", llm_config=llm_config)

reviewer = ConversableAgent(name="Reviewer", system_message="I review code.", llm_config=llm_config)

user_proxy = UserProxyAgent(name="User", human_input_mode="TERMINATE")


# Simulate conversation:

user_proxy.initiate_chat(coder, message="Task: Write a Fibonacci function.")

code_response = user_proxy.last_message(coder)

user_proxy.initiate_chat(reviewer, message=f"Review this code:\n\n{code_response}")

review_response = user_proxy.last_message(reviewer)

print("Final review response:", review_response)

'''

    with open(conv_file, "w") as f:

        f.write(example_code)

    

    # Create a conversation management utility class

    utils_dir = Path("soln_ai/utils")

    utils_dir.mkdir(parents=True, exist_ok=True)

    

    conv_utils_file = utils_dir / "conversation_manager.py"

    conv_utils_code = '''

from typing import List, Dict, Any, Optional

import logging

from autogen import ConversableAgent, UserProxyAgent


class ConversationManager:

    """

    Utility class for managing conversations between agents in AutoGen v0.4.7.

    Since v0.4.7 doesn't have GroupChat, this class provides alternate patterns.

    """

    

    def __init__(self, user_proxy: UserProxyAgent):

        """Initialize with a user proxy to coordinate conversations."""

        self.user_proxy = user_proxy

        self.agents = {}

        self.history = {}

        

    def add_agent(self, agent: ConversableAgent, role: str):

        """Add an agent with a specific role."""

        self.agents[role] = agent

        self.history[role] = []

        logging.info(f"Added agent '{agent.name}' with role '{role}'")

        

    def initiate_conversation(self, role: str, message: str) -> str:

        """Start conversation with an agent by role."""

        if role not in self.agents:

            raise ValueError(f"No agent with role '{role}'")

            

        agent = self.agents[role]

        self.user_proxy.initiate_chat(agent, message=message)

        

        response = self.user_proxy.last_message(agent)

        self.history[role].append({"from": "user", "message": message, "response": response})

        

        return response

        

    def continue_conversation(self, from_role: str, to_role: str, 

                            context: Optional[str] = None) -> str:

        """Continue conversation between agents."""

        if from_role not in self.agents or to_role not in self.agents:

            raise ValueError(f"Invalid roles: '{from_role}' or '{to_role}'")

            

        from_agent = self.agents[from_role]

        to_agent = self.agents[to_role]

        

        # Get last response from the first agent

        last_message = self.user_proxy.last_message(from_agent)

        

        # Create prompt with optional context

        prompt = last_message

        if context:

            prompt = f"{context}\\n\\n{last_message}"

            

        # Send to the second agent

        self.user_proxy.initiate_chat(to_agent, message=prompt)

        

        response = self.user_proxy.last_message(to_agent)

        self.history[to_role].append({

            "from": from_role, 

            "message": prompt, 

            "response": response

        })

        

        return response

        

    def get_conversation_summary(self) -> Dict[str, List[Dict[str, str]]]:

        """Get complete conversation history."""

        return self.history

'''

    

    with open(conv_utils_file, "w") as f:

        f.write(conv_utils_code)

    

    logging.info(f"Created conversation management example at {conv_file}")

    logging.info(f"Created conversation management utilities at {conv_utils_file}")

    

    return {

        "files_created": 2,

        "message": "Conversation management implemented for v0.4.7"

    }


def rule_implement_collaboration_protocols_v047():

    """

    Implement collaboration protocols (task decomposition, consensus, aggregation)

    suitable for AutoGen v0.4.7.

    """

    from pathlib import Path

    

    # Create protocols directory

    protocols_dir = Path("soln_ai/protocols")

    protocols_dir.mkdir(parents=True, exist_ok=True)

    

    # Create __init__.py to make it a proper package

    init_file = protocols_dir / "__init__.py"

    with open(init_file, "w") as f:

        f.write("# Collaboration protocols for multi-agent systems\n")

    

    # Create the task decomposition protocol

    task_decomp_file = protocols_dir / "task_decomposition.py"

    task_decomp_code = '''

import logging

from typing import List, Dict, Any, Callable, Optional

from autogen import ConversableAgent, UserProxyAgent


class TaskDecompositionProtocol:

    """

    Task decomposition protocol for AutoGen v0.4.7.

    

    This protocol helps break down complex tasks into smaller subtasks

    that can be handled by specialized agents.

    """

    

    def __init__(self, planner: ConversableAgent, user_proxy: UserProxyAgent):

        """

        Initialize the task decomposition protocol.

        

        Args:

            planner: Agent responsible for breaking down tasks

            user_proxy: User proxy agent to coordinate the process

        """

        self.planner = planner

        self.user_proxy = user_proxy

        self.subtasks = []

        

    def decompose_task(self, task: str, max_subtasks: int = 5) -> List[str]:

        """

        Decompose a complex task into subtasks.

        

        Args:

            task: The complex task to decompose

            max_subtasks: Maximum number of subtasks to create

            

        Returns:

            List of subtask descriptions

        """

        prompt = f"""

        Break down the following complex task into {max_subtasks} or fewer subtasks.

        For each subtask, provide a clear description of what needs to be done.

        Format your response as a numbered list of subtasks.

        

        TASK: {task}

        """

        

        # Ask planner to decompose the task

        self.user_proxy.initiate_chat(self.planner, message=prompt)

        response = self.user_proxy.last_message(self.planner)

        

        # Parse subtasks from response

        subtasks = self._parse_subtasks(response)

        if not subtasks:

            logging.warning(f"Failed to parse subtasks from response: {response}")

            return []

            

        self.subtasks = subtasks

        logging.info(f"Task decomposed into {len(subtasks)} subtasks")

        return subtasks

    

    def _parse_subtasks(self, response: str) -> List[str]:

        """

        Parse subtasks from the planner's response.

        

        Args:

            response: Planner's response text

            

        Returns:

            List of subtask descriptions

        """

        subtasks = []

        

        # Split by newlines and look for numbered list items

        lines = response.strip().split("\\n")

        for line in lines:

            line = line.strip()

            

            # Check for numbered list format (1. Task or 1) Task)

            if (line and 

                (line[0].isdigit() or 

                 (len(line) > 2 and line[0].isdigit() and line[1] in ['.', ')']))):

                

                # Extract the task part after the number

                parts = line.split(".", 1)

                if len(parts) > 1:

                    subtasks.append(parts[1].strip())

                else:

                    parts = line.split(")", 1)

                    if len(parts) > 1:

                        subtasks.append(parts[1].strip())

        

        return subtasks

    

    def assign_subtasks(self, 

                        agents: Dict[str, ConversableAgent], 

                        assignment_strategy: Optional[Callable] = None) -> Dict[str, List[str]]:

        """

        Assign subtasks to specialized agents.

        

        Args:

            agents: Dictionary of agent name to agent object

            assignment_strategy: Optional function to determine assignments

            

        Returns:

            Dictionary of agent name to list of assigned subtasks

        """

        if not self.subtasks:

            logging.warning("No subtasks to assign")

            return {}

            

        if assignment_strategy:

            # Use custom assignment strategy

            assignments = assignment_strategy(self.subtasks, agents)

        else:

            # Default round-robin assignment

            assignments = {}

            agent_names = list(agents.keys())

            for i, subtask in enumerate(self.subtasks):

                agent_name = agent_names[i % len(agent_names)]

                if agent_name not in assignments:

                    assignments[agent_name] = []

                assignments[agent_name].append(subtask)

        

        logging.info(f"Subtasks assigned to {len(assignments)} agents")

        return assignments

'''

    

    with open(task_decomp_file, "w") as f:

        f.write(task_decomp_code)

    

    # Create the consensus protocol

    consensus_file = protocols_dir / "consensus.py"

    consensus_code = '''

import logging

from typing import List, Dict, Any, Optional, Tuple

from autogen import ConversableAgent, UserProxyAgent


class ConsensusProtocol:

    """

    Consensus building protocol for AutoGen v0.4.7.

    

    This protocol helps multiple agents reach agreement on a decision,

    solution, or approach to a problem.

    """

    

    def __init__(self, 

                user_proxy: UserProxyAgent, 

                required_agreement: float = 0.7):

        """

        Initialize the consensus protocol.

        

        Args:

            user_proxy: User proxy agent to coordinate the process

            required_agreement: Fraction of agents required to reach consensus

                                (0.7 = 70% agreement needed)

        """

        self.user_proxy = user_proxy

        self.required_agreement = required_agreement

        

    def reach_consensus(self, 

                       agents: List[ConversableAgent], 

                       question: str,

                       options: Optional[List[str]] = None,

                       max_rounds: int = 3) -> Tuple[str, float]:

        """

        Attempt to reach consensus among agents on a question.

        

        Args:

            agents: List of agents participating in consensus

            question: The question or decision to reach consensus on

            options: Optional list of predetermined options

            max_rounds: Maximum number of discussion rounds

            

        Returns:

            Tuple of (consensus answer, agreement level)

        """

        if not agents:

            raise ValueError("At least one agent is required")

            

        logging.info(f"Starting consensus protocol with {len(agents)} agents")

        

        # If options provided, format them into the prompt

        options_text = ""

        if options:

            options_text = "\\nOptions:\\n" + "\\n".join([f"- {opt}" for opt in options])

        

        # Round 1: Initial responses

        responses = []

        for agent in agents:

            prompt = f"""

            Please provide your answer to the following question:

            

            QUESTION: {question}

            {options_text}

            

            Provide a clear, concise answer. If you're choosing from options, specify which one.

            """

            self.user_proxy.initiate_chat(agent, message=prompt)

            response = self.user_proxy.last_message(agent)

            responses.append(response)

        

        # Check if we already have consensus

        consensus, agreement = self._analyze_consensus(responses, options)

        if agreement >= self.required_agreement:

            logging.info(f"Consensus reached in first round: {consensus} ({agreement:.0%} agreement)")

            return consensus, agreement

            

        # Additional rounds if needed

        for round_num in range(2, max_rounds + 1):

            logging.info(f"Consensus round {round_num}: current agreement {agreement:.0%}")

            

            # Share all perspectives with each agent

            new_responses = []

            perspectives = "\\n\\n".join([f"Agent {i+1}: {resp}" for i, resp in enumerate(responses)])

            

            for i, agent in enumerate(agents):

                prompt = f"""

                We're trying to reach consensus on this question:

                

                QUESTION: {question}

                {options_text}

                

                Here are all perspectives so far:

                {perspectives}

                

                Your previous answer was: {responses[i]}

                

                Based on all perspectives, please provide your updated answer.

                You may keep your answer or revise it to help reach consensus.

                """

                self.user_proxy.initiate_chat(agent, message=prompt)

                response = self.user_proxy.last_message(agent)

                new_responses.append(response)

            

            # Update responses for next round

            responses = new_responses

            

            # Check if we've reached consensus

            consensus, agreement = self._analyze_consensus(responses, options)

            if agreement >= self.required_agreement:

                logging.info(f"Consensus reached in round {round_num}: {consensus} ({agreement:.0%} agreement)")

                return consensus, agreement

        

        # If we get here, we didn't reach the required agreement

        logging.warning(f"Failed to reach required consensus after {max_rounds} rounds. " 

                      f"Best agreement: {agreement:.0%} on '{consensus}'")

        return consensus, agreement

    

    def _analyze_consensus(self, 

                         responses: List[str], 

                         options: Optional[List[str]]) -> Tuple[str, float]:

        """

        Analyze responses to determine consensus level.

        

        Args:

            responses: List of agent responses

            options: Optional list of predetermined options

            

        Returns:

            Tuple of (most common response, agreement level)

        """

        if not responses:

            return "", 0.0

            

        # If we have predetermined options, try to match responses to them

        if options:

            # Map responses to options based on keyword matching

            option_count = {option: 0 for option in options}

            for response in responses:

                best_match = None

                most_words = 0

                

                for option in options:

                    # Count how many words from the option appear in the response

                    option_words = set(option.lower().split())

                    response_words = set(response.lower().split())

                    common_words = option_words.intersection(response_words)

                    

                    if len(common_words) > most_words:

                        most_words = len(common_words)

                        best_match = option

                

                if best_match and most_words > 0:

                    option_count[best_match] += 1

            

            # Find most common option

            most_common = max(option_count.items(), key=lambda x: x[1])

            option, count = most_common

            

            agreement = count / len(responses)

            return option, agreement

        

        # Without predetermined options, we'll use exact matching to count identical responses

        response_count = {}

        for response in responses:

            # Use a simplified version of the response (lowercase, whitespace normalized)

            simple_resp = " ".join(response.lower().split())

            if simple_resp in response_count:

                response_count[simple_resp] += 1

            else:

                response_count[simple_resp] = 1

        

        # Find most common response

        most_common = max(response_count.items(), key=lambda x: x[1])

        response, count = most_common

        

        agreement = count / len(responses)

        return response, agreement

'''

    

    with open(consensus_file, "w") as f:

        f.write(consensus_code)

    

    # Create the result aggregation protocol

    aggregation_file = protocols_dir / "result_aggregation.py"

    aggregation_code = '''

import logging

from typing import List, Dict, Any, Optional, Callable, Union

from autogen import ConversableAgent, UserProxyAgent


class ResultAggregationProtocol:

    """

    Result aggregation protocol for AutoGen v0.4.7.

    

    This protocol helps combine results from multiple agents into

    a coherent final output.

    """

    

    def __init__(self, 

                aggregator: ConversableAgent,

                user_proxy: UserProxyAgent):

        """

        Initialize the result aggregation protocol.

        

        Args:

            aggregator: Agent responsible for aggregating results

            user_proxy: User proxy agent to coordinate the process

        """

        self.aggregator = aggregator

        self.user_proxy = user_proxy

        

    def aggregate_results(self, 

                        results: Dict[str, str],

                        task_description: str,

                        aggregation_type: str = "summary") -> str:

        """

        Aggregate results from multiple agents.

        

        Args:

            results: Dictionary of agent name to result

            task_description: Description of the original task

            aggregation_type: Type of aggregation to perform 

                             (summary, consensus, best)

            

        Returns:

            Aggregated result

        """

        if not results:

            logging.warning("No results to aggregate")

            return ""

            

        # Format results for the aggregator

        formatted_results = "\\n\\n".join([f"--- {agent} ---\\n{result}" 

                                         for agent, result in results.items()])

        

        prompt = f"""

        You are aggregating results from multiple agents working on this task:

        

        TASK: {task_description}

        

        Below are the results from each agent:

        

        {formatted_results}

        

        Your job is to create a {aggregation_type} that combines these results.

        

        If 'summary', provide a coherent synthesis of all inputs.

        If 'consensus', identify points of agreement and disagreement.

        If 'best', select the best response and explain why.

        

        AGGREGATED RESULT:

        """

        

        # Ask aggregator to combine results

        self.user_proxy.initiate_chat(self.aggregator, message=prompt)

        aggregated_result = self.user_proxy.last_message(self.aggregator)

        

        logging.info(f"Results aggregated using '{aggregation_type}' approach")

        return aggregated_result

        

    def iterative_refinement(self,

                            initial_result: str,

                            critics: List[ConversableAgent],

                            task_description: str,

                            max_iterations: int = 3) -> str:

        """

        Refine a result through iterative feedback from critics.

        

        Args:

            initial_result: The initial result to refine

            critics: List of agents that will provide feedback

            task_description: Description of the original task

            max_iterations: Maximum number of refinement iterations

            

        Returns:

            Refined result

        """

        current_result = initial_result

        

        for iteration in range(max_iterations):

            logging.info(f"Refinement iteration {iteration+1}/{max_iterations}")

            

            # Collect feedback from critics

            all_feedback = []

            for i, critic in enumerate(critics):

                prompt = f"""

                Review the following result for this task:

                

                TASK: {task_description}

                

                CURRENT RESULT:

                {current_result}

                

                Please provide specific feedback on how to improve this result.

                Focus on issues like accuracy, completeness, clarity, and coherence.

                Be constructive and specific in your feedback.

                """

                

                self.user_proxy.initiate_chat(critic, message=prompt)

                feedback = self.user_proxy.last_message(critic)

                all_feedback.append(feedback)

            

            # Combine feedback

            combined_feedback = "\\n\\n".join([f"Critic {i+1}: {feedback}" 

                                            for i, feedback in enumerate(all_feedback)])

            

            # Ask aggregator to refine based on feedback

            refine_prompt = f"""

            Refine the following result based on critic feedback:

            

            TASK: {task_description}

            

            CURRENT RESULT:

            {current_result}

            

            CRITIC FEEDBACK:

            {combined_feedback}

            

            Please provide an improved version that addresses the feedback.

            REFINED RESULT:

            """

            

            self.user_proxy.initiate_chat(self.aggregator, message=refine_prompt)

            refined_result = self.user_proxy.last_message(self.aggregator)

            

            # Check if there are significant changes

            if self._similarity(current_result, refined_result) > 0.9:

                logging.info(f"Result stabilized after {iteration+1} iterations")

                return refined_result

                

            current_result = refined_result

        

        logging.info(f"Completed {max_iterations} refinement iterations")

        return current_result

    

    def _similarity(self, text1: str, text2: str) -> float:

        """

        Calculate a simple similarity score between two texts.

        

        Args:

            text1: First text

            text2: Second text

            

        Returns:

            Similarity score between 0 and 1

        """

        # Simple word-based Jaccard similarity

        words1 = set(text1.lower().split())

        words2 = set(text2.lower().split())

        

        if not words1 and not words2:

            return 1.0

            

        intersection = len(words1.intersection(words2))

        union = len(words1.union(words2))

        

        return intersection / union

'''

    

    with open(aggregation_file, "w") as f:

        f.write(aggregation_code)

    

    # Create an example

    examples_dir = Path("examples/collaboration")

    examples_dir.mkdir(parents=True, exist_ok=True)

    

    example_file = examples_dir / "multi_agent_collaboration.py"

    example_code = '''

import os

import logging

from autogen import ConversableAgent, UserProxyAgent

from soln_ai.llm.llm_factory import LLMFactory, LLMProvider

from soln_ai.protocols.task_decomposition import TaskDecompositionProtocol

from soln_ai.protocols.consensus import ConsensusProtocol

from soln_ai.protocols.result_aggregation import ResultAggregationProtocol


logging.basicConfig(level=logging.INFO)


# Initialize LLM factory and get an LLM

factory = LLMFactory()

llm = factory.get_llm(LLMProvider.CLAUDE)  # Falls back to MCP if Claude unavailable

llm_config = llm.get_llm_config()


# Create specialized agents

planner = ConversableAgent(

    name="Planner",

    system_message="You break down complex problems into steps and create plans.",

    llm_config=llm_config

)


coder = ConversableAgent(

    name="Coder",

    system_message="You write clean, efficient code to solve problems.",

    llm_config=llm_config

)


reviewer = ConversableAgent(

    name="Reviewer",

    system_message="You review solutions to ensure they're correct and optimal.",

    llm_config=llm_config

)


researcher = ConversableAgent(

    name="Researcher",

    system_message="You gather information and provide facts about topics.",

    llm_config=llm_config

)


aggregator = ConversableAgent(

    name="Aggregator",

    system_message="You combine and synthesize information from multiple sources.",

    llm_config=llm_config

)


# Create user proxy agent

user_proxy = UserProxyAgent(

    name="User",

    human_input_mode="TERMINATE",

    code_execution_config={"work_dir": "workspace"}

)


# Define the complex task

task = "Create a Python function that can download stock price data for a given ticker symbol, visualize it with matplotlib, and calculate moving averages. The function should handle errors gracefully."


# Step 1: Task Decomposition

decomposer = TaskDecompositionProtocol(planner, user_proxy)

subtasks = decomposer.decompose_task(task)

print(f"\\nTask decomposed into {len(subtasks)} subtasks:")

for i, subtask in enumerate(subtasks):

    print(f"  {i+1}. {subtask}")


# Step 2: Assign subtasks to agents

agents = {

    "Coder": coder,

    "Researcher": researcher,

    "Reviewer": reviewer

}

assignments = decomposer.assign_subtasks(agents)


# Step 3: Execute subtasks and collect results

results = {}

for agent_name, assigned_tasks in assignments.items():

    agent = agents[agent_name]

    agent_results = []

    

    for subtask in assigned_tasks:

        user_proxy.initiate_chat(agent, message=f"Complete this subtask: {subtask}")

        result = user_proxy.last_message(agent)

        agent_results.append(result)

    

    # Combine results for this agent

    results[agent_name] = "\\n\\n".join(agent_results)


# Step 4: Aggregate results

aggregation = ResultAggregationProtocol(aggregator, user_proxy)

final_result = aggregation.aggregate_results(results, task)

print("\\n--- FINAL AGGREGATED RESULT ---\\n")

print(final_result)


# Step 5: Iterative refinement

critics = [planner, reviewer]

refined_result = aggregation.iterative_refinement(final_result, critics, task)

print("\\n--- REFINED FINAL RESULT ---\\n")

print(refined_result)


# (Optional) Step 6: Consensus on solution approach

consensus = ConsensusProtocol(user_proxy)

solution_options = [

    "Use pandas_datareader for stock data",

    "Use yfinance for stock data",

    "Use Alpha Vantage API for stock data"

]

best_approach, agreement = consensus.reach_consensus(

    [planner, coder, researcher], 

    "What's the best library to use for downloading stock data?",

    solution_options

)

print(f"\\nConsensus reached: {best_approach} with {agreement:.0%} agreement")

'''

    

    with open(example_file, "w") as f:

        f.write(example_code)

    

    logging.info(f"Created collaboration protocols in {protocols_dir}")

    logging.info(f"Created collaboration example at {example_file}")

    

    return {

        "status": "implemented",

        "files_created": 4,

        "protocols": ["task_decomposition", "consensus", "result_aggregation"],

        "message": "Collaboration protocols implemented for AutoGen v0.4.7"

    }


def rule_define_specialized_agent_roles_v047():

    """

    Define specialized agent roles using a factory pattern for AutoGen v0.4.7.

    """

    from pathlib import Path

    from enum import Enum, auto

    

    # Create roles directory

    roles_dir = Path("soln_ai/roles")

    roles_dir.mkdir(parents=True, exist_ok=True)

    

    # Create __init__.py to make it a proper package

    init_file = roles_dir / "__init__.py"

    with open(init_file, "w") as f:

        f.write("# Specialized agent roles for AutoGen v0.4.7\n")

    

    # Create the roles definitions

    roles_file = roles_dir / "role_definitions.py"

    roles_code = '''

from enum import Enum, auto

from typing import Dict, Any


class AgentRole(Enum):

    """Enum defining specialized agent roles."""

    CODER = auto()

    PLANNER = auto()

    REVIEWER = auto()

    RESEARCHER = auto()

    CRITIC = auto()

    SUMMARIZER = auto()

    COORDINATOR = auto()

    CREATIVE = auto()

    ENGINEER = auto()

    TEACHER = auto()


# System messages for each role

ROLE_SYSTEM_MESSAGES = {

    AgentRole.CODER: """You are an expert programmer focused on writing clean, efficient, and well-documented code.

Your strengths include:

- Writing concise and readable code in multiple languages

- Implementing efficient algorithms and data structures

- Following coding best practices and standards

- Writing clear comments and documentation

- Handling error cases appropriately""",

    

    AgentRole.PLANNER: """You are a strategic planner who breaks down complex problems into clear, actionable steps.

Your strengths include:

- Analyzing requirements and constraints thoroughly

- Creating structured plans with logical sequences

- Anticipating potential challenges and dependencies

- Prioritizing tasks effectively

- Keeping track of the big picture while handling details""",

    

    AgentRole.REVIEWER: """You are a detail-oriented reviewer who evaluates solutions for correctness, efficiency, and quality.

Your strengths include:

- Spotting bugs, edge cases, and potential issues

- Suggesting improvements to existing solutions

- Evaluating performance and scalability

- Ensuring code meets requirements and standards

- Providing constructive, specific feedback""",

    

    AgentRole.RESEARCHER: """You are a thorough researcher who gathers and synthesizes information from multiple sources.

Your strengths include:

- Finding relevant information efficiently

- Evaluating the credibility of sources

- Summarizing complex topics concisely

- Identifying trends and patterns in data

- Presenting balanced perspectives on controversial topics""",

    

    AgentRole.CRITIC: """You are an analytical critic who evaluates ideas and identifies potential weaknesses.

Your strengths include:

- Finding logical flaws or inconsistencies

- Questioning assumptions

- Identifying missing information or perspectives

- Suggesting constructive improvements

- Thinking adversarially to strengthen ideas""",

    

    AgentRole.SUMMARIZER: """You are a skilled summarizer who condenses complex information while retaining key points.

Your strengths include:

- Identifying the most important information

- Creating concise summaries at appropriate detail levels

- Organizing information logically

- Maintaining neutrality and accuracy

- Adapting summaries for different audiences""",

    

    AgentRole.COORDINATOR: """You are a collaborative coordinator who manages interactions between multiple agents.

Your strengths include:

- Tracking progress across parallel workstreams

- Ensuring all participants understand their responsibilities

- Identifying and resolving bottlenecks or conflicts

- Maintaining focus on goals and deadlines

- Synthesizing input from multiple sources""",

    

    AgentRole.CREATIVE: """You are a creative thinker who generates innovative ideas and approaches.

Your strengths include:

- Thinking outside conventional boundaries

- Connecting disparate concepts in novel ways

- Generating multiple diverse solutions

- Taking appropriate creative risks

- Balancing creativity with practicality""",

    

    AgentRole.ENGINEER: """You are a practical engineer who designs and builds robust systems.

Your strengths include:

- Creating practical, working solutions to problems

- Designing robust, scalable architectures

- Balancing theoretical ideals with practical constraints

- Testing and validating designs thoroughly

- Documenting systems clearly for users and maintainers""",

    

    AgentRole.TEACHER: """You are a patient teacher who explains complex concepts clearly and accessibly.

Your strengths include:

- Breaking down complex topics into understandable parts

- Adapting explanations to the audience's knowledge level

- Providing helpful examples and analogies

- Answering questions thoroughly and clearly

- Encouraging deeper understanding rather than memorization"""

}


# Configuration parameters for each role

ROLE_DEFAULT_CONFIGS = {

    AgentRole.CODER: {

        "temperature": 0.2,

        "top_p": 0.9,

        "max_tokens": 4096

    },

    AgentRole.PLANNER: {

        "temperature": 0.3,

        "top_p": 0.9,

        "max_tokens": 3072

    },

    AgentRole.REVIEWER: {

        "temperature": 0.1,

        "top_p": 0.9,

        "max_tokens": 3072

    },

    AgentRole.RESEARCHER: {

        "temperature": 0.2,

        "top_p": 0.9,

        "max_tokens": 4096

    },

    AgentRole.CRITIC: {

        "temperature": 0.3,

        "top_p": 0.9,

        "max_tokens": 2048

    },

    AgentRole.SUMMARIZER: {

        "temperature": 0.1,

        "top_p": 0.9,

        "max_tokens": 2048

    },

    AgentRole.COORDINATOR: {

        "temperature": 0.4,

        "top_p": 0.9,

        "max_tokens": 3072

    },

    AgentRole.CREATIVE: {

        "temperature": 0.8,

        "top_p": 0.95,

        "max_tokens": 4096

    },

    AgentRole.ENGINEER: {

        "temperature": 0.3,

        "top_p": 0.9,

        "max_tokens": 4096

    },

    AgentRole.TEACHER: {

        "temperature": 0.4,

        "top_p": 0.9,

        "max_tokens": 3072

    }

}

'''

    

    with open(roles_file, "w") as f:

        f.write(roles_code)

    

    # Create the agent factory

    factory_file = roles_dir / "agent_factory.py"

    factory_code = '''

import logging

from typing import Dict, Any, Optional, Union, List

from enum import Enum

from autogen import ConversableAgent, UserProxyAgent


from soln_ai.roles.role_definitions import AgentRole, ROLE_SYSTEM_MESSAGES, ROLE_DEFAULT_CONFIGS

from soln_ai.llm.llm_factory import LLMFactory, LLMProvider


class SpecializedAgentFactory:

    """

    Factory for creating specialized agents with predefined roles.

    

    This factory simplifies creation of agents with specific roles

    and appropriate configuration for each role.

    """

    

    def __init__(self, llm_factory: Optional[LLMFactory] = None):

        """

        Initialize the specialized agent factory.

        

        Args:

            llm_factory: LLMFactory instance or None to create a new one

        """

        self.llm_factory = llm_factory or LLMFactory()

        

    def create_agent(self, 

                    role: Union[str, AgentRole],

                    name: Optional[str] = None,

                    llm_provider: Optional[Union[str, LLMProvider]] = None,

                    custom_config: Optional[Dict[str, Any]] = None) -> ConversableAgent:

        """

        Create a specialized agent with a specific role.

        

        Args:

            role: Predefined role from AgentRole enum or role name string

            name: Optional custom name (defaults to role name)

            llm_provider: Optional LLM provider specification

            custom_config: Optional custom configuration to override defaults

            

        Returns:

            ConversableAgent with the specified role

        """

        # Convert string role to enum if needed

        if isinstance(role, str):

            try:

                role = AgentRole[role.upper()]

            except KeyError:

                valid_roles = [r.name for r in AgentRole]

                raise ValueError(f"Invalid role: {role}. Valid roles: {valid_roles}")

        

        # Set default name based on role if not provided

        if name is None:

            name = role.name.capitalize()

        

        # Get system message for the role

        system_message = ROLE_SYSTEM_MESSAGES[role]

        

        # Combine role default config with custom config

        role_config = ROLE_DEFAULT_CONFIGS[role].copy()

        if custom_config:

            role_config.update(custom_config)

        

        # Get LLM with appropriate configuration

        llm = self.llm_factory.get_llm(llm_provider, config=role_config)

        llm_config = llm.get_llm_config()

        

        # Create and return the specialized agent

        agent = ConversableAgent(

            name=name,

            system_message=system_message,

            llm_config=llm_config

        )

        

        logging.info(f"Created {role.name} agent named '{name}'")

        return agent

    

    def create_user_proxy(self, 

                        name: str = "User",

                        human_input_mode: str = "TERMINATE",

                        code_execution_config: Optional[Dict[str, Any]] = None) -> UserProxyAgent:

        """

        Create a user proxy agent.

        

        Args:

            name: Name for the user proxy

            human_input_mode: Human input interaction mode

            code_execution_config: Configuration for code execution

            

        Returns:

            UserProxyAgent

        """

        # Set default code execution config if not provided

        if code_execution_config is None:

            code_execution_config = {"work_dir": "workspace"}

        

        # Create and return the user proxy

        user_proxy = UserProxyAgent(

            name=name,

            human_input_mode=human_input_mode,

            code_execution_config=code_execution_config

        )

        

        logging.info(f"Created UserProxyAgent named '{name}'")

        return user_proxy

    

    def create_team(self, 

                   roles: List[Union[str, AgentRole]],

                   llm_provider: Optional[Union[str, LLMProvider]] = None) -> Dict[str, ConversableAgent]:

        """

        Create a team of specialized agents for different roles.

        

        Args:

            roles: List of roles for the team

            llm_provider: Optional LLM provider for all agents

            

        Returns:

            Dictionary of agent name to agent object

        """

        team = {}

        

        for role in roles:

            agent = self.create_agent(role, llm_provider=llm_provider)

            team[agent.name] = agent

            

        logging.info(f"Created team with {len(team)} specialized agents")

        return team

'''

    

    with open(factory_file, "w") as f:

        f.write(factory_code)

    

    # Create an example

    examples_dir = Path("examples/specialized_agents")

    examples_dir.mkdir(parents=True, exist_ok=True)

    

    example_file = examples_dir / "specialized_agents_example.py"

    example_code = '''

import os

import logging

from soln_ai.roles.agent_factory import SpecializedAgentFactory

from soln_ai.roles.role_definitions import AgentRole

from soln_ai.llm.llm_factory import LLMFactory, LLMProvider


logging.basicConfig(level=logging.INFO)


# Initialize factories

llm_factory = LLMFactory()

agent_factory = SpecializedAgentFactory(llm_factory)


# Create specialized agents for different roles

coder = agent_factory.create_agent(

    role=AgentRole.CODER,

    name="PythonCoder",

    llm_provider=LLMProvider.CLAUDE,

    custom_config={"temperature": 0.1}  # Lower temperature for more precise code

)


planner = agent_factory.create_agent(

    role=AgentRole.PLANNER,

    llm_provider=LLMProvider.MCP

)


# Create a user proxy

user = agent_factory.create_user_proxy(

    human_input_mode="TERMINATE",

    code_execution_config={"work_dir": "workspace"}

)


# Define a coding task

task = "Create a Python function that checks if a string is a palindrome"


# First ask the planner to break down the task

user.initiate_chat(

    planner,

    message=f"Create a plan for this task: {task}"

)

plan = user.last_message(planner)


# Then ask the coder to implement the solution

user.initiate_chat(

    coder,

    message=f"Implement this plan: {plan}\\n\\nTask: {task}"

)

implementation = user.last_message(coder)


# Print the implementation

print("\\n--- FINAL IMPLEMENTATION ---\\n")

print(implementation)


# Create a team of specialists

print("\\n--- CREATING SPECIALIST TEAM ---\\n")

team = agent_factory.create_team([

    AgentRole.RESEARCHER,

    AgentRole.CRITIC,

    AgentRole.TEACHER

])


print(f"Created team with {len(team)} specialists:")

for name, agent in team.items():

    print(f"  - {name}: {agent.system_message.split('\\n')[0]}")

'''

    

    with open(example_file, "w") as f:

        f.write(example_code)

    

    logging.info(f"Created role definitions at {roles_file}")

    logging.info(f"Created agent factory at {factory_file}")

    logging.info(f"Created specialized agents example at {example_file}")

    

    return {

        "status": "implemented",

        "files_created": 3,

        "roles_defined": ["CODER", "PLANNER", "REVIEWER", "RESEARCHER", "CRITIC", 

                         "SUMMARIZER", "COORDINATOR", "CREATIVE", "ENGINEER", "TEACHER"],

        "message": "Specialized agent roles defined for AutoGen v0.4.7"

    }


# -----------------------------------------------------------------------------

# LLM Integration Rules for v0.4.7

# -----------------------------------------------------------------------------

def rule_integrate_claude_api_v047():

    """

    Integrate Claude 3.7 Sonnet using a v0.4.7-compatible wrapper.

    """

    from pathlib import Path

    

    # Create llm directory if it doesn't exist

    llm_dir = Path("soln_ai/llm")

    llm_dir.mkdir(parents=True, exist_ok=True)

    

    # Create __init__.py to make it a proper package

    init_file = llm_dir / "__init__.py"

    if not init_file.exists():

        with open(init_file, "w") as f:

            f.write("# LLM integration package\n")

    

    # Create the ClaudeWrapper class file

    claude_file = llm_dir / "claude_wrapper.py"

    claude_wrapper_code = '''

import os

import time

import logging

import random

import json

from typing import Dict, List, Any, Optional, Union


class ClaudeWrapper:

    """

    Claude 3.7 Sonnet wrapper compatible with AutoGen v0.4.7.

    

    This wrapper provides:

    - Message format conversion between AutoGen and Claude

    - Error handling with exponential backoff and retries

    - Appropriate configuration for AutoGen v0.4.7

    """

    

    def __init__(self, config: Dict[str, Any] = None):

        """

        Initialize the Claude wrapper with configuration.

        

        Args:

            config: Configuration dict with optional keys:

                - model: Claude model name (default: "claude-3-7-sonnet")

                - temperature: Sampling temperature (default: 0.7)

                - max_tokens: Maximum tokens to generate (default: 4096)

                - top_p: Top-p sampling parameter (default: 0.9)

        """

        # Set default configuration

        self.default_config = {

            "model": "claude-3-7-sonnet",

            "temperature": 0.7,

            "max_tokens": 4096,

            "top_p": 0.9,

        }

        

        # Merge with provided config

        self.config = {**self.default_config, **(config or {})}

        

        # Check for API key

        self.api_key = os.environ.get("ANTHROPIC_API_KEY")

        if not self.api_key:

            logging.warning("ANTHROPIC_API_KEY not found in environment variables")

        

        # Initialize Anthropic client

        try:

            import anthropic

            self.client = anthropic.Anthropic(api_key=self.api_key)

            logging.info(f"Claude wrapper initialized with model {self.config['model']}")

        except ImportError:

            logging.error("Required package 'anthropic' not found. Please install with: pip install anthropic")

            raise

        except Exception as e:

            logging.error(f"Failed to initialize Anthropic client: {str(e)}")

            raise

        

        # Retry parameters

        self.max_retries = 3

        self.base_retry_delay = 2  # seconds

    

    def get_llm_config(self) -> Dict[str, Any]:

        """

        Get LLM configuration dictionary compatible with AutoGen v0.4.7.

        

        Returns:

            Dict with configuration for AutoGen

        """

        # AutoGen v0.4.7 configuration format

        return {

            "model": self.config["model"],

            "api_key": self.api_key,

            "temperature": self.config["temperature"],

            "max_tokens": self.config["max_tokens"]

        }

    

    def complete(self, messages: List[Dict[str, str]]) -> Dict[str, str]:

        """

        Generate a completion from Claude API with error handling and retries.

        

        Args:

            messages: List of message dicts with 'role' and 'content' keys

            

        Returns:

            Dict with 'role' and 'content' keys containing the response

            

        Raises:

            Exception: If all retries fail

        """

        if not messages:

            raise ValueError("Messages list cannot be empty")

        

        # Validate messages format

        for msg in messages:

            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:

                raise ValueError(f"Invalid message format. Expected dict with 'role' and 'content' keys. Got: {msg}")

        

        # Apply retries with exponential backoff

        for attempt in range(self.max_retries):

            try:

                # Convert to Claude format

                claude_messages = self._convert_to_claude_format(messages)

                

                # Call Claude API

                response = self.client.messages.create(

                    model=self.config["model"],

                    messages=claude_messages,

                    max_tokens=self.config["max_tokens"],

                    temperature=self.config["temperature"],

                    top_p=self.config["top_p"]

                )

                

                # Extract and return content

                return {

                    "role": "assistant",

                    "content": response.content[0].text

                }

                

            except Exception as e:

                logging.warning(f"Claude API error (attempt {attempt+1}): {str(e)}")

                

                # If this was the last attempt, re-raise the exception

                if attempt == self.max_retries - 1:

                    logging.error(f"Claude API failed after {self.max_retries} attempts: {str(e)}")

                    raise

                

                # Otherwise, wait with exponential backoff and jitter

                retry_delay = self.base_retry_delay * (2 ** attempt)

                jitter = random.uniform(0, 0.1 * retry_delay)

                wait_time = retry_delay + jitter

                logging.info(f"Retrying in {wait_time:.2f} seconds...")

                time.sleep(wait_time)

    

    def _convert_to_claude_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:

        """

        Convert AutoGen message format to Claude format.

        

        Claude expects:

        - 'role' can be 'user' or 'assistant'

        - System messages should be incorporated into the user message

        

        Args:

            messages: List of message dicts in AutoGen format

            

        Returns:

            List of message dicts in Claude format

        """

        claude_messages = []

        system_messages = []

        

        # First, extract any system messages

        for msg in messages:

            if msg["role"].lower() == "system":

                system_messages.append(msg["content"])

            else:

                # Map 'user' and 'assistant' roles directly

                claude_messages.append({

                    "role": msg["role"].lower(),

                    "content": msg["content"]

                })

        

        # If we have system messages, prepend them to the first user message

        if system_messages and claude_messages:

            # Find the first user message

            for i, msg in enumerate(claude_messages):

                if msg["role"] == "user":

                    # Prepend system instructions to this user message

                    system_content = "\\n\\n".join(system_messages)

                    msg["content"] = f"<system>\\n{system_content}\\n</system>\\n\\n{msg['content']}"

                    break

            # If no user messages, add a new one with the system content

            else:

                system_content = "\\n\\n".join(system_messages)

                claude_messages.insert(0, {

                    "role": "user",

                    "content": f"<system>\\n{system_content}\\n</system>"

                })

        

        return claude_messages

    

    def test_connection(self) -> Dict[str, Any]:

        """

        Test connection to Claude API.

        

        Returns:

            Dict with test results

        """

        try:

            start_time = time.time()

            response = self.complete([{"role": "user", "content": "Hello, this is a test message. Please respond with 'Claude connection successful.'"}])

            elapsed = time.time() - start_time

            

            return {

                "success": True,

                "response": response["content"],

                "latency_seconds": round(elapsed, 2)

            }

        except Exception as e:

            return {

                "success": False,

                "error": str(e)

            }

'''

    

    with open(claude_file, "w") as f:

        f.write(claude_wrapper_code)

    

    # Create an example usage file

    examples_dir = Path("examples/llm_integration")

    examples_dir.mkdir(parents=True, exist_ok=True)

    

    claude_example_file = examples_dir / "claude_example.py"

    claude_example_code = '''

import os

import logging

from soln_ai.llm.claude_wrapper import ClaudeWrapper

from autogen import ConversableAgent, UserProxyAgent


logging.basicConfig(level=logging.INFO)


# Initialize Claude wrapper with custom configuration

claude = ClaudeWrapper({

    "model": "claude-3-7-sonnet",

    "temperature": 0.7,

    "max_tokens": 2048

})


# Test the connection

test_result = claude.test_connection()

if test_result["success"]:

    logging.info(f"Claude connection successful! Latency: {test_result['latency_seconds']}s")

else:

    logging.error(f"Claude connection failed: {test_result['error']}")

    exit(1)


# Get the LLM configuration for AutoGen

llm_config = claude.get_llm_config()


# Create a conversable agent using Claude

assistant = ConversableAgent(

    name="Claude-Assistant",

    system_message="You are a helpful AI assistant powered by Claude 3.7 Sonnet.",

    llm_config=llm_config

)


# Create a user proxy agent

user = UserProxyAgent(

    name="User",

    human_input_mode="TERMINATE",

    code_execution_config={"work_dir": "workspace"}

)


# Start a conversation

user.initiate_chat(

    assistant,

    message="Explain how you're now integrated with AutoGen v0.4.7. Keep it brief."

)

'''

    

    with open(claude_example_file, "w") as f:

        f.write(claude_example_code)

    

    logging.info(f"Created Claude wrapper at {claude_file}")

    logging.info(f"Created Claude example at {claude_example_file}")

    

    return {

        "status": "implemented",

        "files_created": 2,

        "message": "Claude 3.7 Sonnet integration implemented for AutoGen v0.4.7"

    }


def rule_integrate_mcp_server_v047():

    """

    Integrate MCP server using a v0.4.7-compatible wrapper.

    """

    from pathlib import Path

    

    # Create llm directory if it doesn't exist

    llm_dir = Path("soln_ai/llm")

    llm_dir.mkdir(parents=True, exist_ok=True)

    

    # Create the MCPWrapper class file

    mcp_file = llm_dir / "mcp_wrapper.py"

    mcp_wrapper_code = '''

import os

import time

import logging

import random

import json

import requests

from typing import Dict, List, Any, Optional, Union


class MCPServerError(Exception):

    """Exception raised for MCP server errors."""

    pass


class MCPAuthenticationError(Exception):

    """Exception raised for MCP authentication errors."""

    pass


class MCPWrapper:

    """

    MCP server wrapper compatible with AutoGen v0.4.7.

    

    This wrapper provides:

    - Connection to MCP server endpoints

    - Message format conversion between AutoGen and MCP server

    - Error handling with exponential backoff and retries

    - Appropriate configuration for AutoGen v0.4.7

    """

    

    def __init__(self, config: Dict[str, Any] = None):

        """

        Initialize the MCP wrapper with configuration.

        

        Args:

            config: Configuration dict with optional keys:

                - base_url: MCP server URL (default: from MCP_BASE_URL env var)

                - model: Model to use (default: "default")

                - temperature: Sampling temperature (default: 0.7)

                - max_tokens: Maximum tokens to generate (default: 2048)

                - top_p: Top-p sampling parameter (default: 0.9)

        """

        # Set default configuration

        self.default_config = {

            "base_url": os.environ.get("MCP_BASE_URL", "https://api.mcp-server.com/v1"),

            "model": "default",

            "temperature": 0.7,

            "max_tokens": 2048,

            "top_p": 0.9,

        }

        

        # Merge with provided config

        self.config = {**self.default_config, **(config or {})}

        

        # Get API key from environment

        self.api_key = os.environ.get("MCP_API_KEY")

        if not self.api_key:

            logging.warning("MCP_API_KEY not found in environment variables")

        

        # Validate base URL

        if not self.config["base_url"]:

            raise ValueError("MCP base URL not provided in config or MCP_BASE_URL environment variable")

            

        logging.info(f"MCP wrapper initialized with base URL: {self.config['base_url']}")

        

        # Retry parameters

        self.max_retries = 3

        self.base_retry_delay = 2  # seconds

    

    def get_llm_config(self) -> Dict[str, Any]:

        """

        Get LLM configuration dictionary compatible with AutoGen v0.4.7.

        

        Returns:

            Dict with configuration for AutoGen

        """

        # AutoGen v0.4.7 configuration format

        return {

            "model": self.config["model"],

            "api_key": self.api_key,

            "base_url": self.config["base_url"],

            "temperature": self.config["temperature"],

            "max_tokens": self.config["max_tokens"]

        }

    

    def complete(self, messages: List[Dict[str, str]]) -> Dict[str, str]:

        """

        Generate a completion from MCP server with error handling and retries.

        

        Args:

            messages: List of message dicts with 'role' and 'content' keys

            

        Returns:

            Dict with 'role' and 'content' keys containing the response

            

        Raises:

            MCPServerError: If server returns an error

            MCPAuthenticationError: If authentication fails

            Exception: For other errors

        """

        if not messages:

            raise ValueError("Messages list cannot be empty")

        

        # Validate messages format

        for msg in messages:

            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:

                raise ValueError(f"Invalid message format. Expected dict with 'role' and 'content' keys. Got: {msg}")

        

        # Prepare request headers

        headers = {

            "Content-Type": "application/json",

            "Authorization": f"Bearer {self.api_key}"

        }

        

        # Prepare request payload

        payload = {

            "model": self.config["model"],

            "messages": messages,

            "temperature": self.config["temperature"],

            "max_tokens": self.config["max_tokens"],

            "top_p": self.config["top_p"]

        }

        

        # Apply retries with exponential backoff

        for attempt in range(self.max_retries):

            try:

                # Send request to MCP server

                response = requests.post(

                    f"{self.config['base_url']}/chat/completions",

                    headers=headers,

                    json=payload,

                    timeout=30  # 30 second timeout

                )

                

                # Check for HTTP errors

                response.raise_for_status()

                

                # Parse response

                result = response.json()

                

                # Extract content based on expected MCP response format

                # (This may need adjustment based on actual MCP API response structure)

                if "choices" in result and len(result["choices"]) > 0:

                    content = result["choices"][0]["message"]["content"]

                    return {

                        "role": "assistant",

                        "content": content

                    }

                else:

                    raise MCPServerError(f"Invalid response format from MCP server: {result}")

                

            except requests.exceptions.HTTPError as e:

                # Handle different HTTP error codes

                status_code = e.response.status_code

                error_msg = str(e)

                

                # Handle authentication errors

                if status_code in (401, 403):

                    raise MCPAuthenticationError(f"MCP authentication failed: {error_msg}")

                    

                # Handle rate limiting

                if status_code == 429:

                    if attempt < self.max_retries - 1:

                        retry_delay = self.base_retry_delay * (2 ** attempt)

                        jitter = random.uniform(0, 0.1 * retry_delay)

                        wait_time = retry_delay + jitter

                        logging.warning(f"Rate limit reached. Retrying in {wait_time:.2f} seconds...")

                        time.sleep(wait_time)

                        continue

                    else:

                        raise MCPServerError(f"Rate limit exceeded and max retries reached: {error_msg}")

                

                # Handle server errors

                if 500 <= status_code < 600:

                    if attempt < self.max_retries - 1:

                        retry_delay = self.base_retry_delay * (2 ** attempt)

                        jitter = random.uniform(0, 0.1 * retry_delay)

                        wait_time = retry_delay + jitter

                        logging.warning(f"MCP server error. Retrying in {wait_time:.2f} seconds...")

                        time.sleep(wait_time)

                        continue

                    else:

                        raise MCPServerError(f"MCP server error after max retries: {error_msg}")

                

                # Other HTTP errors

                raise MCPServerError(f"MCP API request failed: {error_msg}")

                

            except requests.exceptions.RequestException as e:

                # Network-related errors

                if attempt < self.max_retries - 1:

                    retry_delay = self.base_retry_delay * (2 ** attempt)

                    logging.warning(f"Network error: {str(e)}. Retrying in {retry_delay} seconds...")

                    time.sleep(retry_delay)

                    continue

                else:

                    raise MCPServerError(f"Network error after max retries: {str(e)}")

                    

            except json.JSONDecodeError:

                # Invalid JSON response

                raise MCPServerError("Invalid JSON response from MCP server")

                

            except Exception as e:

                # Other unexpected errors

                logging.error(f"Unexpected error in MCP API call: {str(e)}")

                raise

    

    def test_connection(self) -> Dict[str, Any]:

        """

        Test connection to MCP server.

        

        Returns:

            Dict with test results

        """

        try:

            start_time = time.time()

            response = self.complete([{"role": "user", "content": "Hello, this is a test message. Please respond with 'MCP connection successful.'"}])

            elapsed = time.time() - start_time

            

            return {

                "success": True,

                "response": response["content"],

                "latency_seconds": round(elapsed, 2)

            }

        except Exception as e:

            return {

                "success": False,

                "error": str(e)

            }

'''

    

    with open(mcp_file, "w") as f:

        f.write(mcp_wrapper_code)

    

    # Create an example usage file

    examples_dir = Path("examples/llm_integration")

    examples_dir.mkdir(parents=True, exist_ok=True)

    

    mcp_example_file = examples_dir / "mcp_example.py"

    mcp_example_code = '''

import os

import logging

from soln_ai.llm.mcp_wrapper import MCPWrapper

from autogen import ConversableAgent, UserProxyAgent


logging.basicConfig(level=logging.INFO)


# Initialize MCP wrapper with custom configuration

mcp = MCPWrapper({

    "model": "default",

    "temperature": 0.7,

    "max_tokens": 2048

})


# Test the connection

test_result = mcp.test_connection()

if test_result["success"]:

    logging.info(f"MCP connection successful! Latency: {test_result['latency_seconds']}s")

else:

    logging.error(f"MCP connection failed: {test_result['error']}")

    exit(1)


# Get the LLM configuration for AutoGen

llm_config = mcp.get_llm_config()


# Create a conversable agent using MCP

assistant = ConversableAgent(

    name="MCP-Assistant",

    system_message="You are a helpful AI assistant powered by MCP.",

    llm_config=llm_config

)


# Create a user proxy agent

user = UserProxyAgent(

    name="User",

    human_input_mode="TERMINATE",

    code_execution_config={"work_dir": "workspace"}

)


# Start a conversation

user.initiate_chat(

    assistant,

    message="Explain how you're now integrated with AutoGen v0.4.7. Keep it brief."

)

'''

    

    with open(mcp_example_file, "w") as f:

        f.write(mcp_example_code)

    

    logging.info(f"Created MCP wrapper at {mcp_file}")

    logging.info(f"Created MCP example at {mcp_example_file}")

    

    return {

        "status": "implemented",

        "files_created": 2,

        "message": "MCP server integration implemented for AutoGen v0.4.7"

    }


def rule_create_llm_factory_v047():

    """

    Create an LLMFactory for multiple providers (Claude and MCP) in v0.4.7.

    """

    from pathlib import Path

    from enum import Enum

    

    # Create llm directory if it doesn't exist

    llm_dir = Path("soln_ai/llm")

    llm_dir.mkdir(parents=True, exist_ok=True)

    

    # Create the LLMFactory class file

    factory_file = llm_dir / "llm_factory.py"

    factory_code = '''

import os

import logging

from typing import Dict, Any, Optional, List, Tuple, Union

from enum import Enum


class LLMProvider(Enum):

    """Enum for supported LLM providers."""

    CLAUDE = "claude"

    MCP = "mcp"


class LLMFactory:

    """

    Factory class for LLM provider selection and configuration.

    

    This factory:

    - Provides a unified interface for different LLM providers

    - Handles selection of appropriate provider based on requirements

    - Manages fallbacks if primary provider is unavailable

    - Generates consistent AutoGen v0.4.7 configurations

    """

    

    def __init__(self):

        """Initialize the LLM factory."""

        self._providers = {}

        self._initialized_providers = set()

        self._register_providers()

        

    def _register_providers(self):

        """Register available LLM providers."""

        # Initialize provider classes mapping

        self._providers = {

            LLMProvider.CLAUDE: {

                "class": "ClaudeWrapper",

                "module": "soln_ai.llm.claude_wrapper",

                "env_var": "ANTHROPIC_API_KEY"

            },

            LLMProvider.MCP: {

                "class": "MCPWrapper",

                "module": "soln_ai.llm.mcp_wrapper",

                "env_var": "MCP_API_KEY"

            }

        }

        

        # Log available providers

        available = []

        for provider, details in self._providers.items():

            env_var = details["env_var"]

            if os.environ.get(env_var):

                available.append(provider.value)

        

        if available:

            logging.info(f"Available LLM providers: {', '.join(available)}")

        else:

            logging.warning("No LLM providers available. Please set API key environment variables.")

    

    def get_llm(self, 

               provider: Union[str, LLMProvider] = None, 

               config: Dict[str, Any] = None,

               fallback: bool = True) -> Any:

        """

        Get an LLM wrapper instance for the specified provider.

        

        Args:

            provider: LLM provider (CLAUDE or MCP) or provider name string

            config: Configuration for the LLM

            fallback: Whether to try alternate providers if first choice fails

            

        Returns:

            LLM wrapper instance

            

        Raises:

            ValueError: If provider is invalid or no providers available

            ImportError: If provider module can't be imported

        """

        # Set default configuration

        config = config or {}

        

        # If provider is specified as string, convert to enum

        if isinstance(provider, str):

            try:

                provider = LLMProvider(provider.lower())

            except ValueError:

                valid_providers = [p.value for p in LLMProvider]

                raise ValueError(f"Invalid provider: {provider}. Valid options are: {valid_providers}")

        

        # If no provider specified, select first available

        if provider is None:

            provider = self._select_default_provider()

            if provider is None:

                raise ValueError("No LLM providers available. Please set API key environment variables.")

        

        # Get provider details

        if provider not in self._providers:

            valid_providers = [p.value for p in LLMProvider]

            raise ValueError(f"Invalid provider: {provider}. Valid options are: {valid_providers}")

            

        provider_info = self._providers[provider]

        

        # Check if API key is available

        env_var = provider_info["env_var"]

        if not os.environ.get(env_var):

            if fallback:

                logging.warning(f"API key environment variable {env_var} not set for {provider.value}. Trying fallback.")

                return self._try_fallback(provider, config)

            else:

                raise ValueError(f"API key environment variable {env_var} not set for {provider.value}.")

        

        # Import the provider module

        try:

            provider_module = __import__(provider_info["module"], fromlist=[provider_info["class"]])

            wrapper_class = getattr(provider_module, provider_info["class"])

        except ImportError:

            if fallback:

                logging.warning(f"Failed to import {provider_info['module']}. Trying fallback.")

                return self._try_fallback(provider, config)

            else:

                raise ImportError(f"Failed to import {provider_info['module']}. Please ensure it's installed.")

        except AttributeError:

            if fallback:

                logging.warning(f"Class {provider_info['class']} not found in {provider_info['module']}. Trying fallback.")

                return self._try_fallback(provider, config)

            else:

                raise ImportError(f"Class {provider_info['class']} not found in {provider_info['module']}.")

        

        # Initialize and return provider

        try:

            llm = wrapper_class(config)

            self._initialized_providers.add(provider)

            return llm

        except Exception as e:

            if fallback:

                logging.warning(f"Failed to initialize {provider.value}: {str(e)}. Trying fallback.")

                return self._try_fallback(provider, config)

            else:

                raise

    

    def _select_default_provider(self) -> Optional[LLMProvider]:

        """

        Select the default provider based on environment variables.

        

        Returns:

            Default LLM provider or None if none available

        """

        # Try to find a provider with API key set

        for provider, details in self._providers.items():

            env_var = details["env_var"]

            if os.environ.get(env_var):

                return provider

        return None

    

    def _try_fallback(self, current_provider: LLMProvider, config: Dict[str, Any]) -> Any:

        """

        Try to use an alternate provider if the current one failed.

        

        Args:

            current_provider: Provider that failed

            config: Configuration for the LLM

            

        Returns:

            LLM wrapper instance from alternate provider

            

        Raises:

            ValueError: If no fallback providers available

        """

        # Get all providers except the current one

        fallback_providers = [p for p in LLMProvider if p != current_provider]

        

        # Try each fallback provider

        for provider in fallback_providers:

            try:

                return self.get_llm(provider, config, fallback=False)

            except Exception as e:

                logging.warning(f"Fallback to {provider.value} failed: {str(e)}")

                continue

        

        # If all fallbacks failed, raise error

        raise ValueError(f"No available LLM providers. Primary ({current_provider.value}) and all fallbacks failed.")

    

    def get_autogen_config(self, 

                          provider: Union[str, LLMProvider] = None, 

                          config: Dict[str, Any] = None) -> Dict[str, Any]:

        """

        Get AutoGen v0.4.7 configuration for the specified provider.

        

        Args:

            provider: LLM provider (CLAUDE or MCP) or provider name string

            config: Configuration for the LLM

            

        Returns:

            Dict with AutoGen v0.4.7 configuration

        """

        llm = self.get_llm(provider, config)

        return llm.get_llm_config()

    

    def test_providers(self) -> Dict[str, Dict[str, Any]]:

        """

        Test all available LLM providers.

        

        Returns:

            Dict with test results for each provider

        """

        results = {}

        

        for provider in LLMProvider:

            provider_name = provider.value

            try:

                llm = self.get_llm(provider, fallback=False)

                test_result = llm.test_connection()

                results[provider_name] = test_result

            except Exception as e:

                results[provider_name] = {

                    "success": False,

                    "error": str(e)

                }

        

        return results

'''

    

    with open(factory_file, "w") as f:

        f.write(factory_code)

    

    # Create an example usage file

    examples_dir = Path("examples/llm_integration")

    examples_dir.mkdir(parents=True, exist_ok=True)

    

    factory_example_file = examples_dir / "llm_factory_example.py"

    factory_example_code = '''

import os

import logging

from soln_ai.llm.llm_factory import LLMFactory, LLMProvider

from autogen import ConversableAgent, UserProxyAgent


logging.basicConfig(level=logging.INFO)


# Initialize LLM factory

factory = LLMFactory()


# Test all available providers

test_results = factory.test_providers()

print("LLM Provider Test Results:")

for provider, result in test_results.items():

    status = "✅ SUCCESS" if result.get("success", False) else "❌ FAILED"

    print(f"  - {provider}: {status}")

    if not result.get("success", False):

        print(f"    Error: {result.get('error', 'Unknown error')}")


# Get LLM with preference order (try Claude first, fall back to MCP)

try:

    llm = factory.get_llm(LLMProvider.CLAUDE, {"temperature": 0.7})

    provider_name = "Claude"

except Exception:

    llm = factory.get_llm(LLMProvider.MCP, {"temperature": 0.7})

    provider_name = "MCP"


# Get AutoGen config

llm_config = llm.get_llm_config()


# Create assistant agent

assistant = ConversableAgent(

    name=f"{provider_name}-Assistant",

    system_message=f"You are an AI assistant powered by {provider_name}.",

    llm_config=llm_config

)


# Create user proxy agent

user = UserProxyAgent(

    name="User",

    human_input_mode="TERMINATE",

    code_execution_config={"work_dir": "workspace"}

)


# Start conversation

user.initiate_chat(

    assistant,

    message=f"Which LLM provider are you using? Explain briefly how the factory pattern helps."

)

'''

    

    with open(factory_example_file, "w") as f:

        f.write(factory_example_code)

    

    logging.info(f"Created LLM factory at {factory_file}")

    logging.info(f"Created factory example at {factory_example_file}")

    

    return {

        "status": "implemented",

        "files_created": 2,

        "message": "LLM factory created for AutoGen v0.4.7"

    }


# -----------------------------------------------------------------------------

# Error Handling & Testing Rules for v0.4.7

# -----------------------------------------------------------------------------

def rule_implement_error_handling_v047():

    """

    Implement robust error handling for v0.4.7.

    """

    from pathlib import Path

    

    # Create utils directory if it doesn't exist

    utils_dir = Path("soln_ai/utils")

    utils_dir.mkdir(parents=True, exist_ok=True)

    

    # Create the error handling module

    error_file = utils_dir / "error_handling.py"

    error_code = '''

import time

import logging

import random

import traceback

import functools

from typing import Callable, Any, Dict, Optional, Type, Union, List, Tuple


class LLMError(Exception):

    """Base class for LLM-related errors."""

    pass


class APIError(LLMError):

    """Error for API failures."""

    pass


class AuthenticationError(LLMError):

    """Error for authentication failures."""

    pass


class RateLimitError(LLMError):

    """Error for rate limit issues."""

    pass


class ContextLengthError(LLMError):

    """Error for context length exceeded."""

    pass


class TimeoutError(LLMError):

    """Error for API timeouts."""

    pass


class AgentError(Exception):

    """Base class for agent-related errors."""

    pass


class MessageFormatError(AgentError):

    """Error for invalid message formats."""

    pass


class FunctionCallError(AgentError):

    """Error for function call failures."""

    pass


class RetryConfig:

    """Configuration for retry behavior."""

    

    def __init__(self,

                max_retries: int = 3,

                base_delay: float = 2.0,

                max_delay: float = 60.0,

                jitter: bool = True,

                jitter_factor: float = 0.1):

        """

        Initialize retry configuration.

        

        Args:

            max_retries: Maximum number of retry attempts

            base_delay: Base delay in seconds

            max_delay: Maximum delay in seconds

            jitter: Whether to add randomness to delay

            jitter_factor: Factor for jitter (0-1)

        """

        self.max_retries = max_retries

        self.base_delay = base_delay

        self.max_delay = max_delay

        self.jitter = jitter

        self.jitter_factor = jitter_factor


def retry(

    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,

    max_retries: int = 3,

    base_delay: float = 2.0,

    max_delay: float = 60.0,

    jitter: bool = True,

    logger: Optional[logging.Logger] = None

) -> Callable:

    """

    Retry decorator with exponential backoff.

    

    Args:

        exceptions: Exception type(s) to catch

        max_retries: Maximum number of retry attempts

        base_delay: Base delay in seconds

        max_delay: Maximum delay in seconds

        jitter: Whether to add randomness to delay

        logger: Logger to use (uses root logger if None)

    

    Returns:

        Decorated function with retry logic

    """

    def decorator(func: Callable) -> Callable:

        @functools.wraps(func)

        def wrapper(*args, **kwargs):

            log = logger or logging.getLogger()

            

            for attempt in range(max_retries + 1):

                try:

                    return func(*args, **kwargs)

                except exceptions as e:

                    if attempt == max_retries:

                        log.error(f"Failed after {max_retries} retries: {str(e)}")

                        raise

                    

                    # Calculate delay with exponential backoff

                    delay = min(base_delay * (2 ** attempt), max_delay)

                    

                    # Add jitter if enabled

                    if jitter:

                        jitter_amount = random.uniform(0, delay * 0.1)

                        delay += jitter_amount

                    

                    log.warning(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}. "

                               f"Retrying in {delay:.2f}s...")

                    time.sleep(delay)

        return wrapper

    return decorator


def fallback(

    alternatives: List[Callable],

    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,

    logger: Optional[logging.Logger] = None

) -> Callable:

    """

    Fallback decorator that tries alternative functions if primary fails.

    

    Args:

        alternatives: List of alternative functions to try

        exceptions: Exception type(s) to catch

        logger: Logger to use (uses root logger if None)

    

    Returns:

        Decorated function with fallback logic

    """

    def decorator(primary_func: Callable) -> Callable:

        @functools.wraps(primary_func)

        def wrapper(*args, **kwargs):

            log = logger or logging.getLogger()

            

            # Try primary function first

            try:

                return primary_func(*args, **kwargs)

            except exceptions as e:

                log.warning(f"Primary function failed: {str(e)}. Trying alternatives...")

                

                # Try alternatives in order

                last_error = e

                for i, alt_func in enumerate(alternatives):

                    try:

                        log.info(f"Trying alternative {i+1}/{len(alternatives)}...")

                        return alt_func(*args, **kwargs)

                    except exceptions as e:

                        log.warning(f"Alternative {i+1} failed: {str(e)}")

                        last_error = e

                

                # If all alternatives fail, raise the last error

                log.error(f"All alternatives failed. Last error: {str(last_error)}")

                raise last_error

        return wrapper

    return decorator


class ErrorHandler:

    """

    Handler for capturing and responding to errors.

    

    This class provides methods for:

    - Capturing and categorizing errors

    - Logging detailed error information

    - Implementing appropriate recovery strategies

    """

    

    def __init__(self, logger: Optional[logging.Logger] = None):

        """

        Initialize error handler.

        

        Args:

            logger: Logger to use (uses root logger if None)

        """

        self.logger = logger or logging.getLogger()

        self.error_counts = {}

        

    def handle(self, 

              error: Exception,

              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

        """

        Handle an error with appropriate strategy.

        

        Args:

            error: The exception to handle

            context: Optional context dict with relevant information

            

        Returns:

            Dict with error details and handling results

        """

        context = context or {}

        error_type = type(error).__name__

        

        # Track error counts

        if error_type in self.error_counts:

            self.error_counts[error_type] += 1

        else:

            self.error_counts[error_type] = 1

        

        # Log detailed error information

        self.logger.error(

            f"Error: {error_type}: {str(error)}\\n"

            f"Context: {context}\\n"

            f"Traceback: {traceback.format_exc()}"

        )

        

        # Apply appropriate handling strategy based on error type

        if isinstance(error, RateLimitError):

            return self._handle_rate_limit(error, context)

        elif isinstance(error, AuthenticationError):

            return self._handle_auth_error(error, context)

        elif isinstance(error, ContextLengthError):

            return self._handle_context_length(error, context)

        elif isinstance(error, TimeoutError):

            return self._handle_timeout(error, context)

        elif isinstance(error, MessageFormatError):

            return self._handle_message_format(error, context)

        elif isinstance(error, FunctionCallError):

            return self._handle_function_call(error, context)

        else:

            return self._handle_generic(error, context)

    

    def _handle_rate_limit(self, 

                         error: RateLimitError, 

                         context: Dict[str, Any]) -> Dict[str, Any]:

        """Handle rate limit errors."""

        self.logger.warning(f"Rate limit hit: {str(error)}")

        return {

            "error_type": "rate_limit",

            "recoverable": True,

            "strategy": "exponential_backoff",

            "suggested_delay": min(60, 5 * (2 ** (self.error_counts.get("RateLimitError", 1) - 1)))

        }

    

    def _handle_auth_error(self, 

                         error: AuthenticationError, 

                         context: Dict[str, Any]) -> Dict[str, Any]:

        """Handle authentication errors."""

        self.logger.error(f"Authentication failed: {str(error)}")

        return {

            "error_type": "authentication",

            "recoverable": False,

            "strategy": "check_credentials",

            "message": "Please check API key or credentials."

        }

    

    def _handle_context_length(self, 

                             error: ContextLengthError, 

                             context: Dict[str, Any]) -> Dict[str, Any]:

        """Handle context length errors."""

        self.logger.warning(f"Context length exceeded: {str(error)}")

        return {

            "error_type": "context_length",

            "recoverable": True,

            "strategy": "truncate_input",

            "message": "Input was too long. Truncating to fit model context."

        }

    

    def _handle_timeout(self, 

                      error: TimeoutError, 

                      context: Dict[str, Any]) -> Dict[str, Any]:

        """Handle timeout errors."""

        self.logger.warning(f"API timeout: {str(error)}")

        return {

            "error_type": "timeout",

            "recoverable": True,

            "strategy": "retry_with_longer_timeout",

            "suggested_timeout": context.get("timeout", 30) * 1.5

        }

    

    def _handle_message_format(self, 

                             error: MessageFormatError, 

                             context: Dict[str, Any]) -> Dict[str, Any]:

        """Handle message format errors."""

        self.logger.error(f"Invalid message format: {str(error)}")

        return {

            "error_type": "message_format",

            "recoverable": True,

            "strategy": "fix_format",

            "message": "Correcting message format to match requirements."

        }

    

    def _handle_function_call(self, 

                            error: FunctionCallError, 

                            context: Dict[str, Any]) -> Dict[str, Any]:

        """Handle function call errors."""

        self.logger.error(f"Function call failed: {str(error)}")

        return {

            "error_type": "function_call",

            "recoverable": True,

            "strategy": "retry_without_function",

            "message": "Function call failed. Falling back to text-only response."

        }

    

    def _handle_generic(self, 

                       error: Exception, 

                       context: Dict[str, Any]) -> Dict[str, Any]:

        """Handle generic errors."""

        self.logger.error(f"Unhandled error: {type(error).__name__}: {str(error)}")

        return {

            "error_type": "generic",

            "recoverable": False,

            "strategy": "report_and_raise",

            "message": f"An unexpected error occurred: {str(error)}"

        }

    

    def get_error_stats(self) -> Dict[str, int]:

        """Get statistics on errors handled."""

        return self.error_counts.copy()

'''

    

    with open(error_file, "w") as f:

        f.write(error_code)

    

    # Create example for error handling

    examples_dir = Path("examples/error_handling")

    examples_dir.mkdir(parents=True, exist_ok=True)

    

    example_file = examples_dir / "error_handling_example.py"

    example_code = '''

import os

import logging

import random

import time

from soln_ai.utils.error_handling import retry, fallback, ErrorHandler

from soln_ai.llm.llm_factory import LLMFactory, LLMProvider


# Set up logging

logger = logging.getLogger("error_example")

logger.setLevel(logging.INFO)

handler = logging.StreamHandler()

handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

logger.addHandler(handler)


# Initialize LLM factory

factory = LLMFactory()


# Create error handler

handler = ErrorHandler(logger)


# Define a function that might fail

@retry(

    exceptions=(Exception,),

    max_retries=3,

    base_delay=1.0,

    logger=logger

)

def call_llm_with_retry(prompt, llm):

    """Call LLM with retry logic for transient errors."""

    # Simulate random failures for demo

    if random.random() < 0.3:

        raise Exception("Simulated API error")

    

    # Actual LLM call

    return llm.complete([{"role": "user", "content": prompt}])


# Define alternative LLM functions for fallback

def call_claude(prompt):

    """Call Claude LLM."""

    try:

        llm = factory.get_llm(LLMProvider.CLAUDE, fallback=False)

        return llm.complete([{"role": "user", "content": prompt}])

    except Exception as e:

        logger.warning(f"Claude call failed: {str(e)}")

        raise


def call_mcp(prompt):

    """Call MCP LLM."""

    try:

        llm = factory.get_llm(LLMProvider.MCP, fallback=False)

        return llm.complete([{"role": "user", "content": prompt}])

    except Exception as e:

        logger.warning(f"MCP call failed: {str(e)}")

        raise


# Set up fallback chain

@fallback(

    alternatives=[call_claude, call_mcp],

    exceptions=(Exception,),

    logger=logger

)

def call_llm_with_fallback(prompt):

    """Call primary LLM with fallback to alternatives if it fails."""

    # This will try the preferred LLM first

    llm = factory.get_llm(LLMProvider.CLAUDE, fallback=False)

    return llm.complete([{"role": "user", "content": prompt}])


# Example usage

def main():

    print("Testing error handling utilities...")

    

    # Test retry

    print("\\n1. Testing retry logic:")

    try:

        # Get any available LLM

        llm = factory.get_llm()

        

        # Call with retry

        print("  Calling LLM with retry...")

        result = call_llm_with_retry("Hello, this is a test.", llm)

        print(f"  Success! Response: {result['content'][:50]}...")

    except Exception as e:

        print(f"  All retries failed: {str(e)}")

        

        # Handle error

        error_result = handler.handle(e, {"operation": "test_retry"})

        print(f"  Error handling result: {error_result}")

    

    # Test fallback

    print("\\n2. Testing fallback logic:")

    try:

        print("  Calling LLM with fallback chain...")

        result = call_llm_with_fallback("Hello, this is a fallback test.")

        print(f"  Success! Response: {result['content'][:50]}...")

    except Exception as e:

        print(f"  All fallbacks failed: {str(e)}")

        

        # Handle error

        error_result = handler.handle(e, {"operation": "test_fallback"})

        print(f"  Error handling result: {error_result}")

    

    # Show error statistics

    print("\\n3. Error statistics:")

    stats = handler.get_error_stats()

    for error_type, count in stats.items():

        print(f"  {error_type}: {count} occurrences")


if __name__ == "__main__":

    main()

'''

    

    with open(example_file, "w") as f:

        f.write(example_code)

    

    logging.info(f"Created error handling utilities at {error_file}")

    logging.info(f"Created error handling example at {example_file}")

    

    return {

        "status": "implemented",

        "files_created": 2,

        "message": "Error handling and utilities implemented for AutoGen v0.4.7"

    }


def rule_add_unit_tests_v047():

    """

    Add unit tests for migrated components for AutoGen v0.4.7.

    """

    from pathlib import Path

    

    # Create tests directory

    tests_dir = Path("tests")

    tests_dir.mkdir(parents=True, exist_ok=True)

    

    # Create __init__.py to make it a proper package

    init_file = tests_dir / "__init__.py"

    with open(init_file, "w") as f:

        f.write("# Test package for SolnAI with AutoGen v0.4.7\n")

    

    # Create conftest.py for pytest fixtures

    conftest_file = tests_dir / "conftest.py"

    conftest_code = '''

import os

import pytest

import logging

from typing import Dict, Any


# Fixture for test configuration

@pytest.fixture

def test_config():

    """Provide test configuration."""

    return {

        "model": "claude-3-7-sonnet",

        "temperature": 0.1,

        "max_tokens": 1024

    }


# Fixture for mock API responses

@pytest.fixture

def mock_responses():

    """Provide mock API responses for testing."""

    return {

        "claude": {

            "default": {

                "role": "assistant",

                "content": "This is a mock Claude response."

            },

            "error": Exception("Mock Claude API error")

        },

        "mcp": {

            "default": {

                "role": "assistant",

                "content": "This is a mock MCP response."

            },

            "error": Exception("Mock MCP API error")

        }

    }


# Fixture for setting up test environment

@pytest.fixture(scope="session", autouse=True)

def setup_test_env():

    """Set up the test environment."""

    # Set environment variables for testing

    os.environ["ANTHROPIC_API_KEY"] = "test_anthropic_key"

    os.environ["MCP_API_KEY"] = "test_mcp_key"

    os.environ["MCP_BASE_URL"] = "https://test-mcp-server.com/v1"

    

    # Set up logging for tests

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    )

    

    yield

    

    # Cleanup

    # (Normally this would be used for teardown, but we don't

    # need any specific cleanup for these tests)

'''

    

    with open(conftest_file, "w") as f:

        f.write(conftest_code)

    

    # Create test for Claude wrapper

    claude_test_file = tests_dir / "test_claude_wrapper.py"

    claude_test_code = '''

import pytest

import os

from unittest.mock import patch, MagicMock

from soln_ai.llm.claude_wrapper import ClaudeWrapper


class TestClaudeWrapper:

    """Tests for Claude wrapper."""

    

    def test_initialization(self, test_config):

        """Test wrapper initialization."""

        wrapper = ClaudeWrapper(test_config)

        assert wrapper.config["model"] == test_config["model"]

        assert wrapper.config["temperature"] == test_config["temperature"]

    

    def test_get_llm_config(self, test_config):

        """Test getting LLM config for AutoGen."""

        wrapper = ClaudeWrapper(test_config)

        config = wrapper.get_llm_config()

        

        # Check that the config has the expected keys for v0.4.7

        assert "model" in config

        assert "api_key" in config

        assert "temperature" in config

        assert "max_tokens" in config

        

        # Check values

        assert config["model"] == test_config["model"]

        assert config["temperature"] == test_config["temperature"]

    

    @patch("anthropic.Anthropic")

    def test_complete(self, mock_anthropic, test_config, mock_responses):

        """Test message completion."""

        # Set up mock

        mock_client = MagicMock()

        mock_anthropic.return_value = mock_client

        

        mock_response = MagicMock()

        mock_response.content = [MagicMock(text=mock_responses["claude"]["default"]["content"])]

        mock_client.messages.create.return_value = mock_response

        

        # Create wrapper and test

        wrapper = ClaudeWrapper(test_config)

        messages = [{"role": "user", "content": "Hello"}]

        

        response = wrapper.complete(messages)

        

        # Verify the response

        assert "role" in response

        assert "content" in response

        assert response["role"] == "assistant"

        assert response["content"] == mock_responses["claude"]["default"]["content"]

        

        # Verify API call

        mock_client.messages.create.assert_called_once()

    

    @patch("anthropic.Anthropic")

    def test_error_handling(self, mock_anthropic, test_config):

        """Test error handling and retries."""

        # Set up mock to raise an error

        mock_client = MagicMock()

        mock_anthropic.return_value = mock_client

        mock_client.messages.create.side_effect = Exception("API error")

        

        # Create wrapper

        wrapper = ClaudeWrapper(test_config)

        messages = [{"role": "user", "content": "Hello"}]

        

        # Test that it raises the exception after retries

        with pytest.raises(Exception):

            wrapper.complete(messages)

        

        # Verify it made retry attempts

        assert mock_client.messages.create.call_count > 1

    

    def test_invalid_message_format(self, test_config):

        """Test handling of invalid message format."""

        wrapper = ClaudeWrapper(test_config)

        

        # Test with empty messages

        with pytest.raises(ValueError):

            wrapper.complete([])

        

        # Test with invalid message structure

        with pytest.raises(ValueError):

            wrapper.complete([{"no_role": "user", "text": "Invalid"}])

    

    @patch("anthropic.Anthropic")

    def test_system_message_handling(self, mock_anthropic, test_config):

        """Test handling of system messages."""

        # Set up mock

        mock_client = MagicMock()

        mock_anthropic.return_value = mock_client

        

        mock_response = MagicMock()

        mock_response.content = [MagicMock(text="Response")]

        mock_client.messages.create.return_value = mock_response

        

        # Create wrapper

        wrapper = ClaudeWrapper(test_config)

        

        # Test with system message

        messages = [

            {"role": "system", "content": "System instruction"},

            {"role": "user", "content": "User message"}

        ]

        

        wrapper.complete(messages)

        

        # Get the messages that were sent to the API

        args, kwargs = mock_client.messages.create.call_args

        claude_messages = kwargs.get("messages", [])

        

        # Check that system message was properly handled

        # For Claude, system messages are processed specially

        assert len(claude_messages) > 0

        

        # The implementation should either:

        # 1. Include system content in the first user message, or

        # 2. Convert it to Claude's special system tag format

        # Check that it did one of these

        first_msg = claude_messages[0]

        assert first_msg["role"] == "user"  # Claude expects user first

        assert "System instruction" in first_msg["content"] or "<system>" in first_msg["content"]

'''

    

    with open(claude_test_file, "w") as f:

        f.write(claude_test_code)

    

    # Create test runner script

    test_runner_file = tests_dir / "run_tests.py"

    test_runner_code = '''

#!/usr/bin/env python3

"""

Test runner for SolnAI with AutoGen v0.4.7.


This script runs the test suite for the migrated SolnAI agents.

It can be used to verify that the migration was successful.


Usage:

  python tests/run_tests.py            # Run all tests

  python tests/run_tests.py -v         # Run with verbose output

  python tests/run_tests.py -k claude  # Run tests containing 'claude'

"""

import sys

import pytest

import os

import logging

from pathlib import Path


# Add parent directory to path so we can import modules

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():

    """Run the test suite."""

    # Configure logging

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    )

    

    # Build arguments for pytest

    pytest_args = [

        "--color=yes",  # Colored output

        "-xvs"          # Exit on first failure, verbose, no capture

    ]

    

    # Add any command line arguments

    pytest_args.extend(sys.argv[1:])

    

    # Run pytest

    logging.info(f"Running tests with args: {pytest_args}")

    result = pytest.main(pytest_args)

    

    # Return appropriate exit code

    return result


if __name__ == "__main__":

    sys.exit(main())

'''

    

    with open(test_runner_file, "w") as f:

        f.write(test_runner_code)

    os.chmod(test_runner_file, 0o755)  # Make executable

    

    logging.info(f"Created test fixtures at {conftest_file}")

    logging.info(f"Created Claude wrapper tests at {claude_test_file}")

    logging.info(f"Created test runner at {test_runner_file}")

    

    return {

        "status": "implemented",

        "files_created": 3,

        "message": "Unit tests created for AutoGen v0.4.7 migration"

    }


# -----------------------------------------------------------------------------

# Main Execution: Migration Orchestration

# -----------------------------------------------------------------------------

def migrate_agents():

    """

    Migrate SolnAI agents to AutoGen v0.4.7.


    Steps:

      1. Validate AutoGen version.

      2. Create directory structure and config.

      3. Apply transformation rules:

         - Agent structure, configuration loading, state management.

         - Message handling, function calling, initialization.

         - Multi-agent collaboration adjustments.

         - LLM integration (Claude, MCP, LLMFactory).

         - Error handling and testing enhancements.

      4. Generate a migration report.

    

    Returns:

        dict: Detailed migration report.

    """

    logging.info(f"Starting migration to AutoGen v{TARGET_VERSION}")

    start_time = time.time()

    results = {

        "success": False,

        "version_check": {},

        "stats": {"files_processed": 0, "transformations_applied": 0, "errors": []},

        "rules_applied": {},

        "duration_seconds": 0

    }

    

    # Step 1: Validate version

    compatible, version_msg = validate_autogen_version()

    results["version_check"] = {"compatible": compatible, "message": version_msg}

    if not compatible:

        logging.error("Migration aborted: incompatible AutoGen version.")

        results["success"] = False

        return results


    # Step 2: Setup directories and config

    create_directory_structure()

    write_config_list_for_v047()

    

    # Step 3: Apply transformation rules

    rule_stats = {}

    try:

        rule_stats["agent_structure"] = rule_convert_agent_class_definitions()

        rule_stats["config_loading"] = rule_update_agent_configuration_loading()

        rule_stats["state_management"] = rule_adapt_agent_state_management()

        results["stats"]["transformations_applied"] += 3

    except Exception as e:

        err = f"Agent structure rules error: {e}\n{traceback.format_exc()}"

        logging.error(err)

        results["stats"]["errors"].append(err)

    

    try:

        rule_stats["message_handling"] = rule_update_message_handling_v047()

        rule_stats["function_calling"] = rule_adapt_function_calling_v047()

        rule_stats["agent_init"] = rule_update_agent_initialization_v047()

        results["stats"]["transformations_applied"] += 3

    except Exception as e:

        err = f"Compatibility rules error: {e}\n{traceback.format_exc()}"

        logging.error(err)

        results["stats"]["errors"].append(err)

    

    try:

        rule_stats["conversation_management"] = rule_implement_conversation_management_v047()

        rule_stats["collaboration_protocols"] = rule_implement_collaboration_protocols_v047()

        rule_stats["agent_roles"] = rule_define_specialized_agent_roles_v047()

        results["stats"]["transformations_applied"] += 3

    except Exception as e:

        err = f"Collaboration rules error: {e}\n{traceback.format_exc()}"

        logging.error(err)

        results["stats"]["errors"].append(err)

    

    try:

        rule_stats["claude_integration"] = rule_integrate_claude_api_v047()

        rule_stats["mcp_integration"] = rule_integrate_mcp_server_v047()

        rule_stats["llm_factory"] = rule_create_llm_factory_v047()

        results["stats"]["transformations_applied"] += 3

    except Exception as e:

        err = f"LLM integration error: {e}\n{traceback.format_exc()}"

        logging.error(err)

        results["stats"]["errors"].append(err)

    

    try:

        rule_stats["error_handling"] = rule_implement_error_handling_v047()

        rule_stats["unit_tests"] = rule_add_unit_tests_v047()

        results["stats"]["transformations_applied"] += 2

    except Exception as e:

        err = f"Error handling/testing rules error: {e}\n{traceback.format_exc()}"

        logging.error(err)

        results["stats"]["errors"].append(err)

    

    results["rules_applied"] = rule_stats

    results["duration_seconds"] = round(time.time() - start_time, 2)

    results["success"] = len(results["stats"]["errors"]) == 0

    

    if results["success"]:

        logging.info(f"Migration completed successfully in {results['duration_seconds']} seconds.")

    else:

        logging.warning(f"Migration completed with errors in {results['duration_seconds']} seconds.")

    

    with open("migration_report.json", "w") as f:

        json.dump(results, f, indent=2)

    logging.info("Migration report saved as 'migration_report.json'")

    return results


if __name__ == "__main__":

    report = migrate_agents()

    sys.exit(0 if report["success"] else 1)
'''
---
