#!/usr/bin/env python3
"""
SolnAI Integration for Website Redesign Agent.

This script provides integration with the SolnAI platform,
allowing the Website Redesign Agent to be used as a SolnAI agent.
"""

import os
import sys
import json
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the website redesign agent
from website_redesign_agent import run_website_redesign

def handle_solnai_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle a request from SolnAI.
    
    Args:
        request_data: The request data from SolnAI
        
    Returns:
        Dict with response data
    """
    try:
        logger.info(f"Received SolnAI request: {json.dumps(request_data)}")
        
        # Extract parameters from request
        website_url = request_data.get("website_url")
        if not website_url:
            return {
                "success": False,
                "error": "Missing required parameter: website_url"
            }
        
        # Run the website redesign process
        result = run_website_redesign(website_url)
        
        # Return the result
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        logger.error(f"Error handling SolnAI request: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def main():
    """Main entry point for SolnAI integration."""
    # Check if input file is provided
    if len(sys.argv) < 2:
        logger.error("Usage: python solnai_integration.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "response.json"
    
    try:
        # Read input file
        with open(input_file, "r") as f:
            request_data = json.load(f)
        
        # Process request
        response_data = handle_solnai_request(request_data)
        
        # Write response to output file
        with open(output_file, "w") as f:
            json.dump(response_data, f, indent=2)
        
        logger.info(f"Response written to {output_file}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error in SolnAI integration: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
