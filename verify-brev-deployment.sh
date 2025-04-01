#!/bin/bash
# Post-deployment verification script for Brev
# This script should be run after deploying to Brev to verify services are running correctly

# Set colors for better output visibility
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== SolnAI Brev Deployment Verification =====${NC}"
echo "This script will verify your deployed services on Brev."

# Ask for Brev instance hostname/IP
read -p "Enter your Brev instance hostname or IP: " BREV_HOST

# Verify each service endpoint
echo -e "\n${YELLOW}Verifying service endpoints...${NC}"

# Verify Archon
echo -e "\nTesting Archon service (port 8001)..."
if curl -s --connect-timeout 5 http://${BREV_HOST}:8001/health &> /dev/null; then
    echo -e "${GREEN}✓ Archon service is running${NC}"
else
    echo -e "${RED}✗ Archon service is not responding${NC}"
    echo "  Try accessing: http://${BREV_HOST}:8001/health"
fi

# Verify TinyDM Agent
echo -e "\nTesting TinyDM Agent service (port 8002)..."
if curl -s --connect-timeout 5 http://${BREV_HOST}:8002/health &> /dev/null; then
    echo -e "${GREEN}✓ TinyDM Agent service is running${NC}"
else
    echo -e "${RED}✗ TinyDM Agent service is not responding${NC}"
    echo "  Try accessing: http://${BREV_HOST}:8002/health"
fi

# Verify Ask Reddit Agent
echo -e "\nTesting Ask Reddit Agent service (port 8003)..."
if curl -s --connect-timeout 5 http://${BREV_HOST}:8003/health &> /dev/null; then
    echo -e "${GREEN}✓ Ask Reddit Agent service is running${NC}"
else
    echo -e "${RED}✗ Ask Reddit Agent service is not responding${NC}"
    echo "  Try accessing: http://${BREV_HOST}:8003/health"
fi

echo -e "\n${YELLOW}===== Additional Verification Steps =====${NC}"
echo -e "1. Login to Brev console and check container logs for any errors"
echo -e "2. Verify GPU usage with 'nvidia-smi' command on your Brev instance"
echo -e "3. Monitor memory usage to ensure services have sufficient resources"

echo -e "\n${YELLOW}===== Troubleshooting Tips =====${NC}"
echo -e "- If services fail health checks, wait a few minutes for initialization"
echo -e "- Check if ports are correctly exposed and not blocked by firewall"
echo -e "- Verify environment variables are correctly set"
echo -e "- For GPU issues, ensure NVIDIA Container Toolkit is installed on the instance"

echo -e "\n${GREEN}Verification process completed.${NC}"
echo -e "For more detailed troubleshooting, refer to README-BREV.md" 