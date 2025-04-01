#!/bin/bash
# Agent Selector for SolnAI Multi-Agent Deployment
# This script helps generate a customized Docker Compose file with only selected agents

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SOURCE_FILE="docker-compose-multi-agent.yml"
OUTPUT_FILE="docker-compose-selected.yml"

echo -e "${YELLOW}===== SolnAI Agent Selector =====${NC}"
echo "This script will help you create a customized deployment with only the agents you need."

# Check if source file exists
if [ ! -f "$SOURCE_FILE" ]; then
    echo -e "${RED}Source file not found: $SOURCE_FILE${NC}"
    exit 1
fi

# Temporary file for building the new compose file
TMP_FILE=$(mktemp)

# Copy the header and common parts
sed -n '1,/services:/p' "$SOURCE_FILE" > "$TMP_FILE"

# Always include base services
echo -e "\n${YELLOW}Base Infrastructure${NC}"
echo -e "${BLUE}These services are required and will always be included.${NC}"
echo "- base-python"
echo "- archon"

# Extract base services sections and add to output
sed -n '/# BASE INFRASTRUCTURE/,/# RESEARCH & CONTENT/p' "$SOURCE_FILE" | 
  sed '$d' >> "$TMP_FILE"

# Define agent groups
declare -A agent_groups
agent_groups["Research & Content (GPU 0)"]="general-researcher advanced-researcher content-creator"
agent_groups["Video & Media (GPU 1)"]="youtube-summary youtube-educator streambuzz"
agent_groups["Technical & Development (GPU 2)"]="github-agent tech-stack-expert openai-sdk-agent"
agent_groups["Business & Industry (GPU 3)"]="lead-generator intelligent-invoicing website-redesign"
agent_groups["Integration & Workflow (GPU 4)"]="n8n-integration multi-tool-agent langgraph-parallel"
agent_groups["Core/Utility (GPU 5)"]="tinydm-agent ask-reddit-agent multi-page-scraper"

# Function to display agents in a group and let user select
select_from_group() {
    local group_name="$1"
    local agents="$2"
    local start_marker="$3"
    local end_marker="$4"
    
    echo -e "\n${YELLOW}$group_name${NC}"
    echo "Select which agents to include (space-separated numbers, or 'all', or 'none'):"
    
    local count=1
    local agent_list=()
    for agent in $agents; do
        echo "  $count) $agent"
        agent_list+=("$agent")
        ((count++))
    done
    
    read -p "Your selection: " selection
    
    local selected_agents=""
    if [ "$selection" = "all" ]; then
        selected_agents=$agents
    elif [ "$selection" != "none" ]; then
        for num in $selection; do
            if [ "$num" -ge 1 ] && [ "$num" -lt "$count" ]; then
                selected_agents="$selected_agents ${agent_list[$((num-1))]}"
            fi
        done
    fi
    
    if [ -n "$selected_agents" ]; then
        # Add section header
        grep -A 2 "$start_marker" "$SOURCE_FILE" >> "$TMP_FILE"
        
        # Add each selected agent
        for agent in $selected_agents; do
            echo -e "${GREEN}Adding $agent${NC}"
            sed -n "/^  $agent:/,/^  [a-z]/{/^  [a-z]/!p}" "$SOURCE_FILE" >> "$TMP_FILE"
        done
    fi
}

# Process each group
for group_name in "${!agent_groups[@]}"; do
    case "$group_name" in
        "Research & Content (GPU 0)")
            select_from_group "$group_name" "${agent_groups[$group_name]}" "# RESEARCH & CONTENT" "# VIDEO & MEDIA"
            ;;
        "Video & Media (GPU 1)")
            select_from_group "$group_name" "${agent_groups[$group_name]}" "# VIDEO & MEDIA" "# TECHNICAL & DEVELOPMENT"
            ;;
        "Technical & Development (GPU 2)")
            select_from_group "$group_name" "${agent_groups[$group_name]}" "# TECHNICAL & DEVELOPMENT" "# BUSINESS & INDUSTRY"
            ;;
        "Business & Industry (GPU 3)")
            select_from_group "$group_name" "${agent_groups[$group_name]}" "# BUSINESS & INDUSTRY" "# INTEGRATION & WORKFLOW"
            ;;
        "Integration & Workflow (GPU 4)")
            select_from_group "$group_name" "${agent_groups[$group_name]}" "# INTEGRATION & WORKFLOW" "# CORE/UTILITY"
            ;;
        "Core/Utility (GPU 5)")
            select_from_group "$group_name" "${agent_groups[$group_name]}" "# CORE/UTILITY" "networks:"
            ;;
    esac
done

# Add network and volumes configuration
sed -n '/^networks:/,$p' "$SOURCE_FILE" >> "$TMP_FILE"

# Save the final file
cp "$TMP_FILE" "$OUTPUT_FILE"
rm "$TMP_FILE"

echo -e "\n${GREEN}Customized Docker Compose file created: $OUTPUT_FILE${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Review the generated file to ensure it's correct"
echo "2. Validate with: docker-compose -f $OUTPUT_FILE config"
echo "3. Deploy to Brev using the instructions in README-MULTI-AGENT-DEPLOYMENT.md" 