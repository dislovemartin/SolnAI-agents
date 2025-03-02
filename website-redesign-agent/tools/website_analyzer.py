"""
Website Analyzer Tool

This module provides tools for analyzing websites, including structure,
content, SEO, performance, and user experience.
"""

import os
import time
import logging
import json
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
from crewai.tools import Tool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebsiteAnalyzer:
    """
    Class for analyzing websites and extracting structured data.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the WebsiteAnalyzer.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        self.rate_limit = self.config.get("settings", {}).get("rate_limit", 5)
        self.max_pages = self.config.get("settings", {}).get("max_pages_to_analyze", 100)
        self.user_agent = self.config.get("settings", {}).get(
            "user_agent", "SolnAI Website Redesign Agent/1.0"
        )
        self.follow_robots = self.config.get("settings", {}).get("follow_robots_txt", True)
        self.timeout = self.config.get("settings", {}).get("timeout", 30)
        self.max_retries = self.config.get("settings", {}).get("max_retries", 3)
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dict with configuration
        """
        if not config_path:
            # Look for config.json in the same directory as this file
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config.json"
            )
        
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {str(e)}")
            return {}
    
    def analyze_website(self, url: str, max_depth: int = 2, follow_links: bool = True) -> Dict[str, Any]:
        """
        Analyze a website and return structured data.
        
        Args:
            url: The URL to analyze
            max_depth: Maximum depth to crawl
            follow_links: Whether to follow links
            
        Returns:
            Dict with analysis results
        """
        logger.info(f"Analyzing website: {url} (max_depth={max_depth}, follow_links={follow_links})")
        
        # Validate URL
        if not self._is_valid_url(url):
            return {"error": f"Invalid URL: {url}"}
        
        # Check robots.txt if enabled
        if self.follow_robots and not self._check_robots_txt(url):
            return {"error": f"URL not allowed by robots.txt: {url}"}
        
        # Initialize results
        results = {
            "url": url,
            "pages_analyzed": 0,
            "structure": {
                "pages": {},
                "navigation": {},
                "depth": 0
            },
            "content": {
                "text_content": {},
                "media_content": {},
                "forms": {}
            },
            "seo": {
                "titles": {},
                "meta_descriptions": {},
                "headings": {},
                "images_without_alt": 0,
                "internal_links": 0,
                "external_links": 0
            },
            "performance": {
                "page_sizes": {},
                "response_times": {}
            },
            "accessibility": {
                "issues": []
            },
            "mobile_friendly": None,
            "technologies": [],
            "summary": {}
        }
        
        # Set maximum depth
        max_depth = min(max_depth, 3)  # Limit to 3 for demonstration
        
        # Crawl the URL and its linked pages
        visited = set()
        to_visit = [(url, 0)]  # (url, depth)
        
        while to_visit and results["pages_analyzed"] < self.max_pages:
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
                response = requests.get(current_url, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                
                # Record response time
                response_time = response.elapsed.total_seconds()
                results["performance"]["response_times"][current_url] = response_time
                
                # Record page size
                page_size = len(response.content) / 1024  # KB
                results["performance"]["page_sizes"][current_url] = page_size
                
                # Process HTML content
                if "text/html" in response.headers.get("Content-Type", "").lower():
                    soup = BeautifulSoup(response.text, "html.parser")
                    
                    # Extract page title
                    title = soup.title.string if soup.title else current_url
                    results["seo"]["titles"][current_url] = title
                    
                    # Extract meta description
                    meta_desc = soup.find("meta", attrs={"name": "description"})
                    if meta_desc and "content" in meta_desc.attrs:
                        results["seo"]["meta_descriptions"][current_url] = meta_desc["content"]
                    
                    # Extract headings
                    headings = {
                        "h1": [h.get_text(strip=True) for h in soup.find_all("h1")],
                        "h2": [h.get_text(strip=True) for h in soup.find_all("h2")],
                        "h3": [h.get_text(strip=True) for h in soup.find_all("h3")]
                    }
                    results["seo"]["headings"][current_url] = headings
                    
                    # Count images without alt text
                    images = soup.find_all("img")
                    images_without_alt = sum(1 for img in images if not img.get("alt"))
                    results["seo"]["images_without_alt"] += images_without_alt
                    
                    # Extract links
                    links = soup.find_all("a", href=True)
                    internal_links = 0
                    external_links = 0
                    
                    # Store page in structure
                    page_path = urlparse(current_url).path or "/"
                    results["structure"]["pages"][page_path] = {
                        "url": current_url,
                        "title": title,
                        "depth": depth
                    }
                    
                    # Update maximum depth
                    results["structure"]["depth"] = max(results["structure"]["depth"], depth)
                    
                    # If following links and not at max depth, add links to visit
                    if follow_links and depth < max_depth:
                        base_url = f"{urlparse(current_url).scheme}://{urlparse(current_url).netloc}"
                        
                        for link in links:
                            href = link["href"]
                            
                            # Skip empty, javascript, and anchor links
                            if not href or href.startswith(("javascript:", "#", "mailto:", "tel:")):
                                continue
                            
                            # Resolve relative URLs
                            if not href.startswith(("http://", "https://")):
                                href = urljoin(current_url, href)
                            
                            # Check if internal or external
                            if href.startswith(base_url):
                                internal_links += 1
                                
                                # Only follow internal links
                                if href not in visited:
                                    to_visit.append((href, depth + 1))
                            else:
                                external_links += 1
                    
                    # Update link counts
                    results["seo"]["internal_links"] += internal_links
                    results["seo"]["external_links"] += external_links
                    
                    # Extract forms
                    forms = soup.find_all("form")
                    if forms:
                        results["content"]["forms"][current_url] = len(forms)
                    
                    # Check for mobile viewport
                    viewport = soup.find("meta", attrs={"name": "viewport"})
                    if viewport and "content" in viewport.attrs:
                        if results["mobile_friendly"] is None:
                            results["mobile_friendly"] = True
                    else:
                        results["mobile_friendly"] = False
                    
                    # Detect technologies (simplified)
                    if soup.find("script", src=lambda s: s and "wp-content" in s):
                        if "WordPress" not in results["technologies"]:
                            results["technologies"].append("WordPress")
                    
                    if soup.find("link", href=lambda s: s and "bootstrap" in s):
                        if "Bootstrap" not in results["technologies"]:
                            results["technologies"].append("Bootstrap")
                    
                    if soup.find("script", src=lambda s: s and "jquery" in s):
                        if "jQuery" not in results["technologies"]:
                            results["technologies"].append("jQuery")
                
                # Increment pages analyzed
                results["pages_analyzed"] += 1
                
            except Exception as e:
                logger.error(f"Error analyzing {current_url}: {str(e)}")
        
        # Generate summary
        self._generate_summary(results)
        
        return results
    
    def _is_valid_url(self, url: str) -> bool:
        """
        Check if a URL is valid.
        
        Args:
            url: The URL to check
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _check_robots_txt(self, url: str) -> bool:
        """
        Check if a URL is allowed by robots.txt.
        
        Args:
            url: The URL to check
            
        Returns:
            bool: True if allowed, False otherwise
        """
        try:
            # Simplified implementation - in a real scenario, use a proper robots.txt parser
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            response = requests.get(robots_url, timeout=self.timeout)
            if response.status_code != 200:
                # If robots.txt doesn't exist or can't be fetched, assume allowed
                return True
            
            # Very simplified check - in reality, use a proper parser
            if "Disallow: /" in response.text:
                return False
            
            return True
        except:
            # If there's an error checking robots.txt, assume allowed
            return True
    
    def _generate_summary(self, results: Dict[str, Any]) -> None:
        """
        Generate a summary of the analysis results.
        
        Args:
            results: The analysis results to summarize
        """
        summary = {}
        
        # Calculate SEO score (simplified)
        seo_score = 100
        
        # Deduct for missing meta descriptions
        meta_desc_count = len(results["seo"]["meta_descriptions"])
        pages_count = results["pages_analyzed"]
        if pages_count > 0:
            meta_desc_percentage = meta_desc_count / pages_count
            if meta_desc_percentage < 0.8:
                seo_score -= 10
        
        # Deduct for missing alt text
        if results["seo"]["images_without_alt"] > 0:
            seo_score -= 5
        
        # Deduct for missing H1 headings
        h1_count = sum(1 for headings in results["seo"]["headings"].values() 
                       if headings.get("h1"))
        if pages_count > 0:
            h1_percentage = h1_count / pages_count
            if h1_percentage < 0.8:
                seo_score -= 10
        
        # Calculate performance score (simplified)
        perf_score = 100
        
        # Deduct for slow pages
        slow_pages = sum(1 for time in results["performance"]["response_times"].values() 
                         if time > 1.0)
        if pages_count > 0:
            slow_percentage = slow_pages / pages_count
            perf_score -= int(slow_percentage * 50)
        
        # Deduct for large pages
        large_pages = sum(1 for size in results["performance"]["page_sizes"].values() 
                          if size > 1000)  # > 1MB
        if pages_count > 0:
            large_percentage = large_pages / pages_count
            perf_score -= int(large_percentage * 30)
        
        # Mobile friendliness
        mobile_score = 100 if results["mobile_friendly"] else 0
        
        # Overall score
        overall_score = (seo_score + perf_score + mobile_score) / 3
        
        # Set summary
        summary["seo_score"] = max(0, min(100, seo_score))
        summary["performance_score"] = max(0, min(100, perf_score))
        summary["mobile_score"] = mobile_score
        summary["overall_score"] = max(0, min(100, overall_score))
        
        # Add recommendations
        recommendations = []
        
        if seo_score < 80:
            if meta_desc_percentage < 0.8:
                recommendations.append("Add meta descriptions to all pages")
            if results["seo"]["images_without_alt"] > 0:
                recommendations.append("Add alt text to all images")
            if h1_percentage < 0.8:
                recommendations.append("Ensure all pages have a single H1 heading")
        
        if perf_score < 80:
            if slow_percentage > 0.2:
                recommendations.append("Improve page load times")
            if large_percentage > 0.2:
                recommendations.append("Optimize page sizes")
        
        if not results["mobile_friendly"]:
            recommendations.append("Make the website mobile-friendly")
        
        summary["recommendations"] = recommendations
        
        # Update results
        results["summary"] = summary

# Create Tool instance
def get_website_analyzer_tool() -> Tool:
    """
    Get the website analyzer tool.
    
    Returns:
        Tool: The website analyzer tool
    """
    analyzer = WebsiteAnalyzer()
    
    return Tool(
        name="Website Analyzer",
        description="Analyzes a website's structure, content, SEO, and performance",
        func=analyzer.analyze_website
    )

if __name__ == "__main__":
    # Example usage
    analyzer = WebsiteAnalyzer()
    results = analyzer.analyze_website("https://example.com")
    print(json.dumps(results, indent=2))
