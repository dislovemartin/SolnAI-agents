#!/usr/bin/env python3
"""
Run tests for the Website Redesign Agent.
"""

import os
import sys
import unittest

def run_tests():
    """Run all tests in the tests directory."""
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the script directory to the Python path
    sys.path.insert(0, script_dir)
    
    # Discover and run tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
