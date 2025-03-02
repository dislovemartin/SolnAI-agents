#!/usr/bin/env python3
"""
Run the Website Redesign Agent locally.

This script demonstrates how to use the Website Redesign Agent
directly without going through the API.
"""

import os
import json
import argparse
from dotenv import load_dotenv
from website_redesign_agent import run_website_redesign

# Load environment variables
load_dotenv()

def main():
    """Run the Website Redesign Agent locally"""
    parser = argparse.ArgumentParser(description="Run Website Redesign Agent locally")
    parser.add_argument("--url", required=True, help="URL of the website to redesign")
    parser.add_argument("--output", default="redesign_report.json", help="Output file path")
    args = parser.parse_args()
    
    print(f"Starting website redesign process for URL: {args.url}")
    print("This may take some time depending on the website size and complexity...")
    
    # Run the redesign process
    result = run_website_redesign(args.url)
    
    # Save the result to a file
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Website redesign completed. Results saved to {args.output}")

if __name__ == "__main__":
    main()
