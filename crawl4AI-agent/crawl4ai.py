"""
Crawl4AI Agent - Intelligent web crawling and data extraction agent for SolnAI.
Built on AutoGen v0.4.7 with Claude 3.7 Sonnet integration.
"""

import os
import time
import logging
import random
import json
import re
from typing import Dict, List, Any, Optional, Union
from urllib.parse import urlparse, urljoin
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import PyPDF2
from PIL import Image
import pytesseract
from dotenv import load_dotenv
from autogen import ConversableAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Crawl4AIAgent(ConversableAgent):
    """
    Agent for web crawling and data extraction.
    
    This agent can autonomously navigate websites, extract structured data,
    and integrate the information with other SolnAI agents.
    """
    
    def __init__(self, name: str, system_message: str, llm_config: Dict[str, Any], **kwargs):
        """
        Initialize the Crawl4AI agent.
        
        Args:
            name: Name of the agent
            system_message: System message for the agent
            llm_config: LLM configuration
            **kwargs: Additional arguments to pass to ConversableAgent
        """
        super().__init__(name=name, system_message=system_message, llm_config=llm_config, **kwargs)
        
        # Register tools
        self.register_tool(self.crawl_url)
        self.register_tool(self.extract_data)
        self.register_tool(self.summarize_content)
        
        # Configure crawler settings
        self.rate_limit = float(os.environ.get("CRAWL4AI_RATE_LIMIT", 5))  # requests per second
        self.max_depth = int(os.environ.get("CRAWL4AI_MAX_DEPTH", 3))
        self.user_agent = os.environ.get(
            "CRAWL4AI_USER_AGENT", 
            "Crawl4AI Bot (https://solnai.com/bot)"
        )
        
        # Initialize memory store
        self.memory = {}
        
        # Content filtering settings
        self.content_filter = {
            "relevance_threshold": 0.5,
            "required_keywords": [],
            "excluded_sections": []
        }
        
        logger.info(f"Crawl4AI agent '{name}' initialized")
    
    def crawl_url(self, url: str, max_depth: int = 1, follow_links: bool = True) -> Dict[str, Any]:
        """
        Crawl a URL and its linked pages up to max_depth.
        
        Args:
            url: The URL to crawl
            max_depth: Maximum depth to crawl (default: 1)
            follow_links: Whether to follow links (default: True)
            
        Returns:
            Dict with crawling results
        """
        logger.info(f"Crawling URL: {url} (max_depth={max_depth}, follow_links={follow_links})")
        
        # Validate URL
        if not self._is_valid_url(url):
            return {"error": f"Invalid URL: {url}"}
        
        # Check robots.txt
        if not self._check_robots_txt(url):
            return {"error": f"URL not allowed by robots.txt: {url}"}
        
        # Initialize results
        results = {
            "url": url,
            "pages_crawled": 0,
            "content": {},
            "links": [],
            "errors": []
        }
        
        # Set maximum depth
        max_depth = min(max_depth, self.max_depth)
        
        # Crawl the URL and its linked pages
        visited = set()
        to_visit = [(url, 0)]  # (url, depth)
        
        while to_visit:
            current_url, depth = to_visit.pop(0)
            
            # Skip if already visited or exceeds max depth
            if current_url in visited or depth > max_depth:
                continue
            
            # Mark as visited
            visited.add(current_url)
            
            try:
                # Respect rate limit
                time.sleep(1 / self.rate_limit)
                
                # Fetch page content
                headers = {"User-Agent": self.user_agent}
                response = requests.get(current_url, headers=headers, timeout=10)
                response.raise_for_status()
                
                # Process content based on content type
                content_type = response.headers.get("Content-Type", "").lower()
                
                if "text/html" in content_type:
                    # Process HTML content
                    soup = BeautifulSoup(response.text, "html.parser")
                    
                    # Extract page title
                    title = soup.title.string if soup.title else current_url
                    
                    # Extract main content
                    main_content = self._extract_main_content(soup)
                    
                    # Store content
                    results["content"][current_url] = {
                        "title": title,
                        "type": "html",
                        "text": main_content,
                        "metadata": {
                            "url": current_url,
                            "depth": depth,
                            "content_type": content_type
                        }
                    }
                    
                    # Extract links if follow_links is True and depth < max_depth
                    if follow_links and depth < max_depth:
                        links = self._extract_links(soup, current_url)
                        results["links"].extend(links)
                        
                        # Add links to visit queue
                        for link in links:
                            if link not in visited:
                                to_visit.append((link, depth + 1))
                
                elif "application/pdf" in content_type:
                    # Process PDF content
                    pdf_text = self._extract_pdf_content(response.content)
                    
                    # Store content
                    results["content"][current_url] = {
                        "title": os.path.basename(urlparse(current_url).path),
                        "type": "pdf",
                        "text": pdf_text,
                        "metadata": {
                            "url": current_url,
                            "depth": depth,
                            "content_type": content_type
                        }
                    }
                
                elif "image/" in content_type:
                    # Process image content with OCR
                    image_text = self._extract_image_text(response.content)
                    
                    # Store content
                    results["content"][current_url] = {
                        "title": os.path.basename(urlparse(current_url).path),
                        "type": "image",
                        "text": image_text,
                        "metadata": {
                            "url": current_url,
                            "depth": depth,
                            "content_type": content_type
                        }
                    }
                
                else:
                    # Skip unsupported content types
                    logger.warning(f"Unsupported content type: {content_type} for URL: {current_url}")
                    results["errors"].append(f"Unsupported content type: {content_type} for URL: {current_url}")
                    continue
                
                # Increment pages crawled counter
                results["pages_crawled"] += 1
                
            except Exception as e:
                logger.error(f"Error crawling URL {current_url}: {str(e)}")
                results["errors"].append(f"Error crawling URL {current_url}: {str(e)}")
        
        logger.info(f"Crawling completed: {results['pages_crawled']} pages crawled, {len(results['errors'])} errors")
        return results
    
    def extract_data(self, html_content: str, extraction_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured data from HTML content.
        
        Args:
            html_content: HTML content to extract data from
            extraction_pattern: Pattern defining what to extract
            
        Returns:
            Dict with extracted data
        """
        logger.info("Extracting data using pattern")
        
        try:
            # Parse HTML
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Initialize results
            results = {
                "success": True,
                "data": {}
            }
            
            # Extract data based on pattern
            for section_name, section_config in extraction_pattern.items():
                section_selector = section_config.get("selector", "")
                section_elements = soup.select(section_selector)
                
                section_data = []
                
                for element in section_elements:
                    item_data = {}
                    
                    # Extract fields
                    for field_name, field_selector in section_config.get("fields", {}).items():
                        field_element = element.select_one(field_selector)
                        item_data[field_name] = field_element.get_text(strip=True) if field_element else None
                    
                    section_data.append(item_data)
                
                results["data"][section_name] = section_data
            
            return results
            
        except Exception as e:
            logger.error(f"Error extracting data: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def summarize_content(self, content: str, max_length: int = 500) -> Dict[str, Any]:
        """
        Generate a summary of the content using Claude.
        
        Args:
            content: Content to summarize
            max_length: Maximum length of summary in characters
            
        Returns:
            Dict with summary
        """
        logger.info(f"Summarizing content (max_length={max_length})")
        
        try:
            # Truncate content if too long
            if len(content) > 100000:  # Limit to 100k characters to avoid token limits
                content = content[:100000] + "..."
            
            # Create prompt for summarization
            prompt = f"""
            Please summarize the following content in a concise way, highlighting the key points.
            Keep the summary under {max_length} characters.
            
            CONTENT:
            {content}
            
            SUMMARY:
            """
            
            # Use the LLM to generate summary
            messages = [{"role": "user", "content": prompt}]
            
            # Get the LLM response
            response = self._get_llm_response(messages)
            
            # Extract summary from response
            summary = response.get("content", "").strip()
            
            # Truncate if still too long
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            
            return {
                "success": True,
                "summary": summary,
                "length": len(summary)
            }
            
        except Exception as e:
            logger.error(f"Error summarizing content: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def set_content_filter(self, relevance_threshold: float = None, 
                          required_keywords: List[str] = None,
                          excluded_sections: List[str] = None) -> None:
        """
        Set content filtering parameters.
        
        Args:
            relevance_threshold: Minimum relevance score (0-1)
            required_keywords: List of keywords that must be present
            excluded_sections: List of CSS selectors to exclude
        """
        if relevance_threshold is not None:
            self.content_filter["relevance_threshold"] = max(0.0, min(1.0, relevance_threshold))
        
        if required_keywords is not None:
            self.content_filter["required_keywords"] = required_keywords
        
        if excluded_sections is not None:
            self.content_filter["excluded_sections"] = excluded_sections
        
        logger.info(f"Content filter updated: {self.content_filter}")
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if a URL is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _check_robots_txt(self, url: str) -> bool:
        """Check if a URL is allowed by robots.txt."""
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            response = requests.get(robots_url, timeout=5)
            if response.status_code != 200:
                # If robots.txt doesn't exist or can't be fetched, assume allowed
                return True
            
            # Very basic robots.txt parsing
            # A more robust implementation would use a proper parser
            lines = response.text.split('\n')
            user_agent_applies = False
            
            for line in lines:
                line = line.strip().lower()
                
                if line.startswith('user-agent:'):
                    agent = line[11:].strip()
                    if agent == '*' or self.user_agent.lower().find(agent) != -1:
                        user_agent_applies = True
                    else:
                        user_agent_applies = False
                
                if user_agent_applies and line.startswith('disallow:'):
                    disallow_path = line[9:].strip()
                    if disallow_path and parsed_url.path.startswith(disallow_path):
                        return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking robots.txt for {url}: {str(e)}")
            # If there's an error checking robots.txt, assume allowed
            return True
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract the main content from an HTML page."""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Remove excluded sections
        for selector in self.content_filter["excluded_sections"]:
            for element in soup.select(selector):
                element.extract()
        
        # Try to find main content
        main_content = None
        
        # Look for common content containers
        for selector in ["main", "article", "#content", ".content", "#main", ".main"]:
            content = soup.select_one(selector)
            if content:
                main_content = content
                break
        
        # If no main content found, use body
        if not main_content:
            main_content = soup.body
        
        # If still no content, use the whole soup
        if not main_content:
            main_content = soup
        
        # Get text
        text = main_content.get_text(separator='\n', strip=True)
        
        # Check for required keywords
        if self.content_filter["required_keywords"]:
            text_lower = text.lower()
            if not all(keyword.lower() in text_lower for keyword in self.content_filter["required_keywords"]):
                return ""
        
        return text
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract links from an HTML page."""
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            
            # Skip empty links, anchors, and javascript
            if not href or href.startswith('#') or href.startswith('javascript:'):
                continue
            
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            
            # Skip mailto and tel links
            if absolute_url.startswith(('mailto:', 'tel:')):
                continue
            
            # Ensure it's from the same domain (optional)
            # base_domain = urlparse(base_url).netloc
            # link_domain = urlparse(absolute_url).netloc
            # if link_domain != base_domain:
            #     continue
            
            links.append(absolute_url)
        
        return links
    
    def _extract_pdf_content(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF content."""
        try:
            from io import BytesIO
            
            # Create a file-like object from bytes
            pdf_file = BytesIO(pdf_bytes)
            
            # Create PDF reader
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from all pages
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text() + "\n\n"
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting PDF content: {str(e)}")
            return f"[PDF extraction error: {str(e)}]"
    
    def _extract_image_text(self, image_bytes: bytes) -> str:
        """Extract text from image using OCR."""
        try:
            from io import BytesIO
            
            # Create a file-like object from bytes
            image_file = BytesIO(image_bytes)
            
            # Open the image
            image = Image.open(image_file)
            
            # Use pytesseract for OCR
            text = pytesseract.image_to_string(image)
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting image text: {str(e)}")
            return f"[Image OCR error: {str(e)}]"
    
    def _get_llm_response(self, messages: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Get a response from the LLM.
        
        This is a helper method that uses the agent's LLM to generate a response.
        """
        # This is a simplified implementation
        # In a real implementation, this would use the agent's LLM configuration
        # to generate a response
        
        # For now, we'll just return a mock response
        return {
            "role": "assistant",
            "content": "This is a mock summary of the content."
        }


# Example usage
if __name__ == "__main__":
    # Initialize the agent
    crawler = Crawl4AIAgent(
        name="web_crawler",
        system_message="You are a helpful web crawler that finds and extracts information.",
        llm_config={
            "model": "claude-3-7-sonnet",
            "temperature": 0.7
        }
    )
    
    # Example crawl
    results = crawler.crawl_url("https://example.com", max_depth=1)
    print(json.dumps(results, indent=2))
