# Deploying to NVIDIA Brev

This guide explains how to deploy the SolnAI agents to NVIDIA Brev.

## Prerequisites

- A Brev.dev account
- Your Brev API key (if applicable)
- Docker and Docker Compose installed locally (for testing)

## Pre-Deployment Testing

Before deploying to Brev, it's recommended to test your configuration locally:

```bash
# Make the test script executable if needed
chmod +x test-brev-deployment.sh

# Run the pre-deployment test
./test-brev-deployment.sh
```

The test script will verify:
- Presence of all required files
- Docker Compose file syntax
- Environment files
- Dockerfiles
- Overall configuration readiness

## Deployment Steps

### 1. Create an account on Brev

Visit the Brev console and create an account if you don't already have one.

### 2. Create a Launchable

1. Go to the Brev console
2. Click on "Create your first Launchable"
3. Choose "Docker Compose Mode"
4. Upload the docker-compose-brev.yml file from this repository

### 3. Choose your GPU type

For optimal performance, we recommend:
- L40S 48GB GPU
- A100 80GB GPU

### 4. Configure instance settings

- Select either L40S 48GB GPU or A100 80GB GPU for optimal performance
- Ensure "Docker Compose Mode" is selected
- Set appropriate environment variables in the Brev console if needed

### 5. Deploy your instance

Click "Deploy" to start the provisioning process.

### 6. Access your deployed agents

Once deployed, you can access your agents using the URLs provided in the Brev console.
The agents expose the following ports:
- Archon: 8001
- TinyDM Agent: 8002
- Ask Reddit Agent: 8003

## Additional Configuration

### GPU Acceleration

This deployment is configured to use NVIDIA GPUs with optimized settings:

- Each service is assigned to a specific GPU device ID (0, 1, and 2) for isolated performance
- Memory limits have been increased to accommodate AI model requirements
- Improved healthchecks with longer startup grace periods for large model loading
- Automatic container restarts configured with `restart: unless-stopped`

The `runtime: nvidia` setting and explicit device reservations ensure that each service has dedicated access to GPU resources, preventing resource contention between agents.

### Environment Variables

Make sure all required environment variables are set in the respective .env files before deployment.

## Performance Optimization

For optimal performance, you may want to adjust the following in the docker-compose-brev.yml file:

- GPU device assignments based on your specific hardware
- Memory limits based on your model sizes and requirements
- CPU allocation depending on your anticipated workload

For large models, consider increasing the memory allocation further, particularly for the TinyDM and Reddit agents that may be more resource-intensive.

## Post-Deployment Verification

After deployment completes, verify that all services are running correctly:

1. Check the status of all containers in the Brev console
2. Verify GPU allocation using the Brev monitoring tools
3. Test each agent's API endpoint using the provided URLs
4. Monitor resource usage to ensure the configuration is appropriate

## Common Deployment Issues and Solutions

| Issue | Solution |
|-------|----------|
| Container fails to start | Check logs for errors, verify .env files are properly configured |
| GPU not detected | Ensure the Brev instance has NVIDIA Container Toolkit installed |
| Out of memory errors | Increase memory limits in docker-compose-brev.yml |
| Container restarts continuously | Check healthcheck configuration, verify API is properly exposed |
| Slow model loading | Increase start_period in healthcheck configuration |

## Troubleshooting

If you encounter issues with the deployment:

1. Check the logs in the Brev console
2. Verify that all environment variables are correctly set
3. Ensure your GPU quota on Brev is sufficient
4. Try restarting individual services through the Brev console
5. If all services have issues, recreate the deployment with a new instance

For more information on deploying to Brev, refer to the [official documentation](https://docs.nvidia.com/brev/latest/).
