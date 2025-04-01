# SolnAI Multi-Agent Deployment Guide

This README explains how to deploy multiple SolnAI agents to NVIDIA Brev using a functionally organized approach.

## Overview

The `docker-compose-multi-agent.yml` file organizes your agents into logical functional groups, each assigned to specific GPUs for optimal resource utilization. This structure allows you to deploy and manage multiple agents efficiently.

## Functional Groups

Agents are organized into the following groups:

1. **Base Infrastructure**
   - Core services that support other agents
   - Includes: base-python, archon

2. **Research & Content Agents** (GPU 0)
   - Focused on research and content creation
   - Includes: general-researcher, advanced-researcher, content-creator

3. **Video & Media Processing** (GPU 1)
   - Specialized in processing and analyzing video content
   - Includes: youtube-summary, youtube-educator, streambuzz

4. **Technical & Development** (GPU 2)
   - Tools for developers and technical tasks
   - Includes: github-agent, tech-stack-expert, openai-sdk-agent

5. **Business & Industry** (GPU 3)
   - Business-focused tools and industry-specific agents
   - Includes: lead-generator, intelligent-invoicing, website-redesign

6. **Integration & Workflow** (GPU 4)
   - Automation and integration focused agents
   - Includes: n8n-integration, multi-tool-agent, langgraph-parallel

7. **Core/Utility Agents** (GPU 5)
   - General utility agents that provide essential services
   - Includes: tinydm-agent, ask-reddit-agent, multi-page-scraper

## Resource Allocation

Each functional group is assigned to a specific GPU to prevent resource contention:

| Group | GPU ID | Memory Allocation |
|-------|--------|-------------------|
| Research & Content | 0 | 0.3-0.4 per agent |
| Video & Media | 1 | 0.3-0.4 per agent |
| Technical & Dev | 2 | 0.3-0.4 per agent |
| Business & Industry | 3 | 0.3-0.4 per agent |
| Integration & Workflow | 4 | 0.3-0.4 per agent |
| Core/Utility | 5 | 0.3-0.4 per agent |

## Deployment Instructions

### Prerequisites

- NVIDIA Brev account
- Access to an instance with 6+ GPUs (A100, L40S, or similar high-memory GPUs)
- Docker and Docker Compose installed locally (for testing)

### Deployment Steps

1. **Customize the Docker Compose File**
   ```bash
   # Validate the Docker Compose file locally before deployment
   docker-compose -f docker-compose-multi-agent.yml config
   ```

2. **Select Agents to Deploy**
   - You can comment out or remove agents you don't need
   - Ensure each GPU doesn't have too many agents assigned

3. **Create a New Launchable in Brev**
   - Log in to your Brev account
   - Create a new Launchable with Docker Compose mode
   - Upload the `docker-compose-multi-agent.yml` file

4. **Choose Appropriate Hardware**
   - For full deployment: A100 or L40S with 6+ GPUs 
   - For partial deployment: Adjust GPU assignments to fit available hardware

5. **Set Environment Variables**
   - Replace placeholder API keys and credentials

6. **Deploy**
   - Click "Deploy" to start the provisioning process

### Accessing Your Agents

Once deployed, access your agents using the URLs provided in the Brev console:

| Agent | Port | URL Format |
|-------|------|------------|
| archon | 8001 | http://[brev-instance]:8001 |
| tinydm-agent | 8002 | http://[brev-instance]:8002 |
| ask-reddit-agent | 8003 | http://[brev-instance]:8003 |
| general-researcher | 8010 | http://[brev-instance]:8010 |
| *and so on...* | | |

## Customization Options

### Adding More Agents

To add more agents to a functional group:

1. Follow the pattern in the Docker Compose file
2. Assign to the appropriate GPU group
3. Ensure total GPU memory fraction doesn't exceed 1.0 per GPU

### Adjusting Resource Allocation

For agents that need more resources:

```yaml
deploy:
  resources:
    limits:
      cpus: "1.0"  # Increase CPU allocation
      memory: 4G   # Increase memory allocation
```

### Reducing Deployment Size

For smaller deployments, focus on specific functional groups:

1. Remove or comment out entire sections
2. Reallocate GPU assignments to use fewer GPUs
3. Update CPU and memory allocations accordingly

## Troubleshooting

If you encounter deployment issues:

1. Check that the total GPU memory fraction per GPU doesn't exceed 1.0
2. Verify port assignments don't conflict
3. Ensure all required environment variables are provided
4. Check Brev instance has sufficient GPUs and memory

See the [Brev Troubleshooting Guide](README-BREV-TROUBLESHOOTING.md) for more details. 