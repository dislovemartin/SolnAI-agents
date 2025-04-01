# Brev Deployment Troubleshooting Guide

## Common Errors and Solutions

### Error: "service X has env file"

**Error Message:**
```
Error: rpc error: code = Internal desc = service tinydm-agent has env file service ask-reddit-agent has env file service archon has env file
```

**Solution:**
This error occurs when your Docker Compose file references external environment files using the `env_file` directive, but these files aren't available on Brev during deployment.

To fix this:

1. Replace `env_file` references with inline `environment` variables:
   ```yaml
   # BEFORE:
   env_file:
     - ./Service/.env
   
   # AFTER:
   environment:
     API_KEY: your_api_key_here
     API_PORT: 8000
     # Other variables...
   ```

2. Use the `docker-compose-brev-updated.yml` file for deployment
3. Create a new Launchable in Brev to avoid caching issues with previous deployments
4. Set sensitive environment variables in the Brev console rather than hardcoding them

### Error: "mapping values are not allowed in this context"

**Error Message:**
```
Error: rpc error: code = Internal desc = yaml: mapping values are not allowed in this context
```

**Solution:**
This error indicates YAML syntax issues in your Docker Compose file.

Common causes and fixes:
1. Incorrect indentation
2. Improper formatting of empty volumes:
   ```yaml
   # INCORRECT:
   volumes:
     postgres_data:
   
   # CORRECT:
   volumes:
     postgres_data: {}
   ```
3. Misplaced colons or quotes

Use the `verify-compose.sh` script to validate your file before uploading.

### Error: "runtime option is not supported by this implementation"

**Error Message:**
```
Error: rpc error: code = Internal desc = runtime option is not supported by this implementation
```

**Solution:**
Some versions of Docker Compose don't support the `runtime` directive or it may be deprecated.

Replace:
```yaml
runtime: nvidia
```

With:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### General Troubleshooting Steps

1. **Validate your Docker Compose file locally:**
   ```bash
   docker-compose -f docker-compose-brev-updated.yml config
   ```

2. **Use the verification script:**
   ```bash
   ./verify-compose.sh
   ```

3. **Try with a new Launchable:**
   If errors persist, create a completely new Launchable in Brev instead of updating an existing one.

4. **Check Brev documentation:**
   Brev may update their platform, so check their latest documentation for any changes to the deployment process.

5. **Start simple and add complexity:**
   If all else fails, start with a minimal Docker Compose file and gradually add services and configuration.

## Additional Resources

- [Brev Documentation](https://docs.nvidia.com/brev/latest/)
- [Docker Compose Specification](https://github.com/compose-spec/compose-spec/blob/master/spec.md)
- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) 