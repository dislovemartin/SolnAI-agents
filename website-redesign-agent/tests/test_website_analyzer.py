"""
Tests for the Website Analyzer tool.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.website_analyzer import WebsiteAnalyzer

class TestWebsiteAnalyzer(unittest.TestCase):
    """Test cases for the WebsiteAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = WebsiteAnalyzer()
    
    def test_is_valid_url(self):
        """Test URL validation."""
        self.assertTrue(self.analyzer._is_valid_url("https://example.com"))
        self.assertTrue(self.analyzer._is_valid_url("http://example.com/page"))
        self.assertFalse(self.analyzer._is_valid_url("example.com"))
        self.assertFalse(self.analyzer._is_valid_url("not a url"))
    
    @patch('tools.website_analyzer.requests.get')
    def test_check_robots_txt(self, mock_get):
        """Test robots.txt checking."""
        # Mock response with no disallow
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "User-agent: *\nAllow: /"
        mock_get.return_value = mock_response
        
        self.assertTrue(self.analyzer._check_robots_txt("https://example.com"))
        
        # Mock response with disallow all
        mock_response.text = "User-agent: *\nDisallow: /"
        self.assertFalse(self.analyzer._check_robots_txt("https://example.com"))
    
    @patch('tools.website_analyzer.requests.get')
    def test_analyze_website_basic(self, mock_get):
        """Test basic website analysis."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.elapsed.total_seconds.return_value = 0.5
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.content = b"<html><body>Test</body></html>"
        mock_response.text = "<html><head><title>Test Page</title></head><body>Test</body></html>"
        mock_get.return_value = mock_response
        
        # Create a simple HTML document
        html = """
        <html>
            <head>
                <title>Test Page</title>
                <meta name="description" content="Test description">
                <meta name="viewport" content="width=device-width, initial-scale=1">
            </head>
            <body>
                <h1>Test Heading</h1>
                <p>Test paragraph</p>
                <img src="test.jpg">
                <a href="https://example.com/page1">Link 1</a>
                <a href="https://example.com/page2">Link 2</a>
                <a href="https://external.com">External Link</a>
            </body>
        </html>
        """
        mock_response.text = html
        mock_response.content = html.encode('utf-8')
        
        # Test with follow_links=False to avoid recursive calls
        results = self.analyzer.analyze_website("https://example.com", follow_links=False)
        
        # Check basic results
        self.assertEqual(results["url"], "https://example.com")
        self.assertEqual(results["pages_analyzed"], 1)
        self.assertEqual(results["structure"]["pages"]["/"]["title"], "Test Page")
        self.assertEqual(results["seo"]["titles"]["https://example.com"], "Test Page")
        self.assertEqual(results["seo"]["meta_descriptions"]["https://example.com"], "Test description")
        self.assertEqual(results["seo"]["headings"]["https://example.com"]["h1"], ["Test Heading"])
        self.assertEqual(results["seo"]["images_without_alt"], 1)
        self.assertEqual(results["seo"]["internal_links"], 0)  # 0 because follow_links=False
        self.assertEqual(results["seo"]["external_links"], 0)  # 0 because follow_links=False
        self.assertTrue(results["mobile_friendly"])

if __name__ == '__main__':
    unittest.main()
