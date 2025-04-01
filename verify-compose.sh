#!/bin/bash
# Verify Docker Compose file before uploading to Brev

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

FILE="docker-compose-brev-updated.yml"

echo -e "${YELLOW}Verifying $FILE for Brev deployment...${NC}"

# Check if file exists
if [ ! -f "$FILE" ]; then
    echo -e "${RED}File not found: $FILE${NC}"
    exit 1
fi

# Check for env_file references
if grep -q "env_file:" "$FILE"; then
    echo -e "${RED}ERROR: File contains env_file references that will cause deployment errors${NC}"
    grep -A 2 "env_file:" "$FILE"
    exit 1
else
    echo -e "${GREEN}✓ No env_file references found${NC}"
fi

# Check for environment variables
if grep -q "environment:" "$FILE"; then
    echo -e "${GREEN}✓ Environment variables are defined inline${NC}"
else
    echo -e "${YELLOW}WARNING: No environment variables defined${NC}"
fi

# Validate with docker-compose
echo -e "\n${YELLOW}Validating with docker-compose...${NC}"
if docker-compose -f "$FILE" config > /dev/null 2>&1; then
    echo -e "${GREEN}✓ File is valid according to docker-compose${NC}"
else
    echo -e "${RED}ERROR: File validation failed${NC}"
    docker-compose -f "$FILE" config
    exit 1
fi

echo -e "\n${GREEN}File is ready for Brev deployment${NC}"
echo -e "${YELLOW}Instructions:${NC}"
echo "1. Upload $FILE to Brev as your Docker Compose file"
echo "2. If the error persists, try creating a new Launchable in Brev"
echo "3. Remember that environment values can be updated in the Brev console" 