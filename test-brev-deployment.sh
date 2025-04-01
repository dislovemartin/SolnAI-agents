#!/bin/bash
# Test script for verifying Brev deployment readiness

# Set colors for better output visibility
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== SolnAI Brev Deployment Test =====${NC}"
echo "This script will verify your configuration before deploying to Brev."

# Check if docker-compose is installed
echo -e "\n${YELLOW}Checking for docker-compose...${NC}"
if command -v docker-compose &> /dev/null || command -v docker compose &> /dev/null; then
    echo -e "${GREEN}✓ Docker Compose is installed${NC}"
else
    echo -e "${RED}✗ Docker Compose not found. Please install Docker Compose first.${NC}"
    exit 1
fi

# Check if docker-compose-brev.yml exists
echo -e "\n${YELLOW}Checking for docker-compose-brev.yml...${NC}"
if [ -f docker-compose-brev.yml ]; then
    echo -e "${GREEN}✓ docker-compose-brev.yml found${NC}"
else
    echo -e "${RED}✗ docker-compose-brev.yml not found. Make sure you're in the right directory.${NC}"
    exit 1
fi

# Validate the docker-compose file
echo -e "\n${YELLOW}Validating docker-compose-brev.yml...${NC}"
validation_output=$(docker-compose -f docker-compose-brev.yml config 2>&1 || docker compose -f docker-compose-brev.yml config 2>&1)
validation_exit_code=$?

if [[ $validation_exit_code -eq 0 ]]; then
    echo -e "${GREEN}✓ docker-compose-brev.yml is valid${NC}"
else
    echo -e "${RED}✗ docker-compose-brev.yml has syntax errors${NC}"
    echo "$validation_output"
    exit 1
fi

# Check for environment files
echo -e "\n${YELLOW}Checking for environment files...${NC}"
missing_env=0
for dir in Archon TinyDM-agent ask-reddit-agent; do
    if [ -f "${dir}/.env" ]; then
        echo -e "${GREEN}✓ ${dir}/.env found${NC}"
    else
        echo -e "${RED}✗ ${dir}/.env missing${NC}"
        missing_env=1
    fi
done

if [ $missing_env -eq 1 ]; then
    echo -e "\n${RED}Warning: Some environment files are missing. Create them before deployment.${NC}"
fi

# Check for Dockerfiles
echo -e "\n${YELLOW}Checking for Dockerfiles...${NC}"
missing_dockerfile=0
for dir in base_python_docker Archon TinyDM-agent ask-reddit-agent; do
    if [ -f "${dir}/Dockerfile" ]; then
        echo -e "${GREEN}✓ ${dir}/Dockerfile found${NC}"
    else
        echo -e "${RED}✗ ${dir}/Dockerfile missing${NC}"
        missing_dockerfile=1
    fi
done

if [ $missing_dockerfile -eq 1 ]; then
    echo -e "\n${RED}Warning: Some Dockerfiles are missing. Create them before deployment.${NC}"
    echo -e "${YELLOW}This is expected if you're using pre-built images from a registry.${NC}"
fi

# Verify README-BREV.md exists
echo -e "\n${YELLOW}Checking for README-BREV.md...${NC}"
if [ -f README-BREV.md ]; then
    echo -e "${GREEN}✓ README-BREV.md found${NC}"
else
    echo -e "${RED}✗ README-BREV.md not found. This contains important deployment instructions.${NC}"
fi

# Print deployment summary
echo -e "\n${YELLOW}===== Deployment Summary =====${NC}"
echo -e "Services defined:"
grep -A 1 "  [a-z]" docker-compose-brev.yml | grep -v "\-\-" | grep -v "deploy\|build\|image\|env\|ports" | sed 's/://'

echo -e "\n${GREEN}Pre-deployment check completed.${NC}"
echo -e "To deploy to Brev:"
echo -e "1. Log into your Brev account"
echo -e "2. Create a new Launchable with Docker Compose mode"
echo -e "3. Upload the docker-compose-brev.yml file"
echo -e "4. Select your desired GPU (L40S 48GB or A100 80GB recommended)"
echo -e "5. Click Deploy"
echo -e "\n${YELLOW}For more details, refer to README-BREV.md${NC}" 