"""
Tests for the Crawl4AI agent.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add parent directory to path to import crawl4ai
sys.path.append(str(Path(__file__).parent.parent))

from crawl4ai import Crawl4AIAgent

class TestCrawl4AIAgent(unittest.TestCase):
    """Test cases for the Crawl4AI agent."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock LLM config
        self.llm_config = {
            "model": "claude-3-7-sonnet",
            "temperature": 0.7
        }
        
        # Create the agent
        self.agent = Crawl4AIAgent(
            name="test_crawler",
            system_message="Test crawler agent",
            llm_config=self.llm_config
        )
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.name, "test_crawler")
        self.assertEqual(self.agent.llm_config, self.llm_config)
        
        # Check default settings
        self.assertGreater(self.agent.rate_limit, 0)
        self.assertGreater(self.agent.max_depth, 0)
        self.assertIsNotNone(self.agent.user_agent)
    
    @patch('requests.get')
    def test_is_valid_url(self, mock_get):
        """Test URL validation."""
        # Valid URLs
        self.assertTrue(self.agent._is_valid_url("https://example.com"))
        self.assertTrue(self.agent._is_valid_url("http://test.org/page"))
        
        # Invalid URLs
        self.assertFalse(self.agent._is_valid_url("not-a-url"))
        self.assertFalse(self.agent._is_valid_url(""))
    
    @patch('requests.get')
    def test_check_robots_txt(self, mock_get):
        """Test robots.txt checking."""
        # Mock response for robots.txt
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """
        User-agent: *
        Disallow: /private/
        Allow: /
        """
        mock_get.return_value = mock_response
        
        # Test allowed URL
        self.assertTrue(self.agent._check_robots_txt("https://example.com/public"))
        
        # Test disallowed URL
        self.assertFalse(self.agent._check_robots_txt("https://example.com/private/data"))
    
    @patch('requests.get')
    def test_crawl_url(self, mock_get):
        """Test URL crawling."""
        # Mock response for HTML page
        mock_html_response = MagicMock()
        mock_html_response.status_code = 200
        mock_html_response.headers = {"Content-Type": "text/html"}
        mock_html_response.text = """
        <html>
            <head>
                <title>Test Page</title>
            </head>
            <body>
                <main>
                    <h1>Test Content</h1>
                    <p>This is a test page.</p>
                    <a href="https://example.com/page1">Link 1</a>
                    <a href="/page2">Link 2</a>
                </main>
            </body>
        </html>
        """
        
        # Mock robots.txt response
        mock_robots_response = MagicMock()
        mock_robots_response.status_code = 200
        mock_robots_response.text = "User-agent: *\nAllow: /"
        
        # Set up mock to return different responses
        def side_effect(url, **kwargs):
            if url.endswith("robots.txt"):
                return mock_robots_response
            return mock_html_response
        
        mock_get.side_effect = side_effect
        
        # Test crawling
        result = self.agent.crawl_url("https://example.com", max_depth=1)
        
        # Check results
        self.assertEqual(result["url"], "https://example.com")
        self.assertGreaterEqual(result["pages_crawled"], 1)
        self.assertIn("https://example.com", result["content"])
        self.assertEqual(result["content"]["https://example.com"]["title"], "Test Page")
        self.assertEqual(result["content"]["https://example.com"]["type"], "html")
        self.assertIn("Test Content", result["content"]["https://example.com"]["text"])
    
    def test_extract_data(self):
        """Test data extraction."""
        # HTML content
        html_content = """
        <html>
            <body>
                <div class="product-item">
                    <h2 class="product-name">Product 1</h2>
                    <span class="product-price">$10.99</span>
                    <p class="product-description">Description 1</p>
                </div>
                <div class="product-item">
                    <h2 class="product-name">Product 2</h2>
                    <span class="product-price">$20.99</span>
                    <p class="product-description">Description 2</p>
                </div>
            </body>
        </html>
        """
        
        # Extraction pattern
        extraction_pattern = {
            "products": {
                "selector": ".product-item",
                "fields": {
                    "name": ".product-name",
                    "price": ".product-price",
                    "description": ".product-description"
                }
            }
        }
        
        # Test extraction
        result = self.agent.extract_data(html_content, extraction_pattern)
        
        # Check results
        self.assertTrue(result["success"])
        self.assertIn("products", result["data"])
        self.assertEqual(len(result["data"]["products"]), 2)
        self.assertEqual(result["data"]["products"][0]["name"], "Product 1")
        self.assertEqual(result["data"]["products"][0]["price"], "$10.99")
        self.assertEqual(result["data"]["products"][1]["name"], "Product 2")
    
    @patch.object(Crawl4AIAgent, '_get_llm_response')
    def test_summarize_content(self, mock_get_llm_response):
        """Test content summarization."""
        # Mock LLM response
        mock_get_llm_response.return_value = {
            "role": "assistant",
            "content": "This is a summary of the test content."
        }
        
        # Test content
        content = "This is a long piece of test content that needs to be summarized."
        
        # Test summarization
        result = self.agent.summarize_content(content, max_length=100)
        
        # Check results
        self.assertTrue(result["success"])
        self.assertIn("summary", result)
        self.assertLessEqual(len(result["summary"]), 100)
    
    def test_set_content_filter(self):
        """Test setting content filter."""
        # Initial values
        initial_threshold = self.agent.content_filter["relevance_threshold"]
        
        # Set new values
        self.agent.set_content_filter(
            relevance_threshold=0.8,
            required_keywords=["test", "example"],
            excluded_sections=[".sidebar", ".comments"]
        )
        
        # Check updated values
        self.assertEqual(self.agent.content_filter["relevance_threshold"], 0.8)
        self.assertEqual(self.agent.content_filter["required_keywords"], ["test", "example"])
        self.assertEqual(self.agent.content_filter["excluded_sections"], [".sidebar", ".comments"])
        
        # Test bounds checking
        self.agent.set_content_filter(relevance_threshold=1.5)
        self.assertEqual(self.agent.content_filter["relevance_threshold"], 1.0)
        
        self.agent.set_content_filter(relevance_threshold=-0.5)
        self.assertEqual(self.agent.content_filter["relevance_threshold"], 0.0)

if __name__ == '__main__':
    unittest.main()
