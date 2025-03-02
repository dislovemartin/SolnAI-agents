"""
WordPress Integration

This module provides tools for integrating with WordPress websites,
including content extraction, modification, and deployment.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional

import requests
from requests.auth import HTTPBasicAuth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WordPressIntegration:
    """
    Class for integrating with WordPress websites.
    """
    
    def __init__(self, site_url: str, username: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize the WordPress integration.
        
        Args:
            site_url: The WordPress site URL
            username: WordPress API username (optional)
            password: WordPress API password (optional)
        """
        self.site_url = site_url.rstrip("/")
        self.api_url = f"{self.site_url}/wp-json/wp/v2"
        self.username = username or os.getenv("WORDPRESS_API_USERNAME")
        self.password = password or os.getenv("WORDPRESS_API_PASSWORD")
        
        # Check if credentials are available
        self.has_auth = bool(self.username and self.password)
    
    def get_site_info(self) -> Dict[str, Any]:
        """
        Get basic information about the WordPress site.
        
        Returns:
            Dict with site information
        """
        try:
            response = requests.get(f"{self.site_url}/wp-json")
            response.raise_for_status()
            data = response.json()
            
            return {
                "name": data.get("name", "Unknown"),
                "description": data.get("description", ""),
                "url": data.get("url", self.site_url),
                "home": data.get("home", self.site_url),
                "gmt_offset": data.get("gmt_offset", 0),
                "timezone_string": data.get("timezone_string", "UTC"),
                "namespaces": data.get("namespaces", [])
            }
        except Exception as e:
            logger.error(f"Error getting site info: {str(e)}")
            return {"error": str(e)}
    
    def get_pages(self, per_page: int = 10, page: int = 1) -> List[Dict[str, Any]]:
        """
        Get pages from the WordPress site.
        
        Args:
            per_page: Number of pages per request
            page: Page number
            
        Returns:
            List of pages
        """
        try:
            params = {
                "per_page": per_page,
                "page": page
            }
            
            response = requests.get(f"{self.api_url}/pages", params=params)
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            logger.error(f"Error getting pages: {str(e)}")
            return []
    
    def get_posts(self, per_page: int = 10, page: int = 1) -> List[Dict[str, Any]]:
        """
        Get posts from the WordPress site.
        
        Args:
            per_page: Number of posts per request
            page: Page number
            
        Returns:
            List of posts
        """
        try:
            params = {
                "per_page": per_page,
                "page": page
            }
            
            response = requests.get(f"{self.api_url}/posts", params=params)
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            logger.error(f"Error getting posts: {str(e)}")
            return []
    
    def get_menus(self) -> List[Dict[str, Any]]:
        """
        Get menus from the WordPress site.
        
        Returns:
            List of menus
        """
        try:
            # This endpoint requires the WP API Menus plugin
            response = requests.get(f"{self.site_url}/wp-json/wp-api-menus/v2/menus")
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning("WP API Menus plugin may not be installed")
                return []
        except Exception as e:
            logger.error(f"Error getting menus: {str(e)}")
            return []
    
    def get_media(self, per_page: int = 10, page: int = 1) -> List[Dict[str, Any]]:
        """
        Get media from the WordPress site.
        
        Args:
            per_page: Number of media items per request
            page: Page number
            
        Returns:
            List of media items
        """
        try:
            params = {
                "per_page": per_page,
                "page": page
            }
            
            response = requests.get(f"{self.api_url}/media", params=params)
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            logger.error(f"Error getting media: {str(e)}")
            return []
    
    def update_page(self, page_id: int, title: str, content: str) -> Dict[str, Any]:
        """
        Update a page on the WordPress site.
        
        Args:
            page_id: The page ID
            title: The page title
            content: The page content
            
        Returns:
            Dict with update result
        """
        if not self.has_auth:
            return {"error": "Authentication credentials not provided"}
        
        try:
            auth = HTTPBasicAuth(self.username, self.password)
            data = {
                "title": title,
                "content": content
            }
            
            response = requests.post(
                f"{self.api_url}/pages/{page_id}",
                auth=auth,
                json=data
            )
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            logger.error(f"Error updating page: {str(e)}")
            return {"error": str(e)}
    
    def create_page(self, title: str, content: str, status: str = "draft") -> Dict[str, Any]:
        """
        Create a new page on the WordPress site.
        
        Args:
            title: The page title
            content: The page content
            status: The page status (draft, publish, etc.)
            
        Returns:
            Dict with creation result
        """
        if not self.has_auth:
            return {"error": "Authentication credentials not provided"}
        
        try:
            auth = HTTPBasicAuth(self.username, self.password)
            data = {
                "title": title,
                "content": content,
                "status": status
            }
            
            response = requests.post(
                f"{self.api_url}/pages",
                auth=auth,
                json=data
            )
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            logger.error(f"Error creating page: {str(e)}")
            return {"error": str(e)}
    
    def get_theme_info(self) -> Dict[str, Any]:
        """
        Get information about the active theme.
        
        Returns:
            Dict with theme information
        """
        try:
            response = requests.get(f"{self.site_url}/wp-json/wp/v2/themes")
            
            if response.status_code == 200:
                themes = response.json()
                active_theme = next((t for t in themes if t.get("active")), None)
                return active_theme or {}
            else:
                logger.warning("Could not get theme information")
                return {}
        except Exception as e:
            logger.error(f"Error getting theme info: {str(e)}")
            return {"error": str(e)}
    
    def export_content(self, output_dir: str) -> Dict[str, Any]:
        """
        Export content from the WordPress site to local files.
        
        Args:
            output_dir: Directory to save exported content
            
        Returns:
            Dict with export results
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Export pages
            pages = []
            page_num = 1
            while True:
                batch = self.get_pages(per_page=100, page=page_num)
                if not batch:
                    break
                pages.extend(batch)
                page_num += 1
            
            with open(os.path.join(output_dir, "pages.json"), "w") as f:
                json.dump(pages, f, indent=2)
            
            # Export posts
            posts = []
            post_num = 1
            while True:
                batch = self.get_posts(per_page=100, page=post_num)
                if not batch:
                    break
                posts.extend(batch)
                post_num += 1
            
            with open(os.path.join(output_dir, "posts.json"), "w") as f:
                json.dump(posts, f, indent=2)
            
            # Export menus
            menus = self.get_menus()
            with open(os.path.join(output_dir, "menus.json"), "w") as f:
                json.dump(menus, f, indent=2)
            
            # Export site info
            site_info = self.get_site_info()
            with open(os.path.join(output_dir, "site_info.json"), "w") as f:
                json.dump(site_info, f, indent=2)
            
            return {
                "success": True,
                "exported_items": {
                    "pages": len(pages),
                    "posts": len(posts),
                    "menus": len(menus)
                },
                "output_dir": output_dir
            }
        except Exception as e:
            logger.error(f"Error exporting content: {str(e)}")
            return {"error": str(e)}

if __name__ == "__main__":
    # Example usage
    wp = WordPressIntegration("https://example.com")
    site_info = wp.get_site_info()
    print(json.dumps(site_info, indent=2))
