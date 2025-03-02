"""
Demo script for N8N Integration Agent functionality.
This script demonstrates the caching and performance analysis capabilities.
"""

import os
import json
import asyncio

# Mock implementation for demonstration purposes
class MockN8NIntegrationAgent:
    def __init__(self, enable_caching=True, performance_tracking=True):
        self.enable_caching = enable_caching
        self.performance_tracking = performance_tracking
        self.cache = {}
        self.optimization_suggestions = set()
        self.performance_metrics = {}
        print(f"Initialized MockN8NIntegrationAgent with caching={enable_caching}, performance_tracking={performance_tracking}")
    
    def generate_cache_key(self, *args):
        """Generate a cache key from the arguments"""
        return "_".join(str(arg) for arg in args)
    
    def set_cached_result(self, key, value):
        """Store a result in the cache"""
        self.cache[key] = {
            "value": value,
            "timestamp": 123456789,
            "ttl": 3600
        }
        print(f"Cached result with key: {key}")
    
    def get_cached_result(self, key):
        """Get a result from the cache"""
        if key in self.cache:
            print(f"Cache hit for key: {key}")
            return self.cache[key]["value"]
        print(f"Cache miss for key: {key}")
        return None
    
    def clear_cache(self, pattern=None):
        """Clear the cache"""
        before_size = len(self.cache)
        if pattern:
            keys_to_remove = [k for k in self.cache.keys() if pattern in k]
            for k in keys_to_remove:
                del self.cache[k]
        else:
            self.cache.clear()
        print(f"Cleared cache: {before_size} items before, {len(self.cache)} items after")
        return {
            "status": "success",
            "cache_size_before": before_size,
            "cache_size_after": len(self.cache)
        }
    
    def get_cache_stats(self):
        """Get statistics about the cache"""
        print(f"Cache stats: {len(self.cache)} items in cache")
        return {
            "status": "success",
            "total_items": len(self.cache),
            "expired_items": 0,
            "active_items": len(self.cache)
        }
    
    def export_optimization_suggestions(self):
        """Export optimization suggestions"""
        # Add some example suggestions
        self.optimization_suggestions.add("Use Split In Batches nodes for parallel processing")
        self.optimization_suggestions.add("Cache results of expensive operations")
        self.optimization_suggestions.add("Implement retry logic for unreliable APIs")
        
        print(f"Exported {len(self.optimization_suggestions)} optimization suggestions")
        return {
            "status": "success",
            "total_suggestions": len(self.optimization_suggestions),
            "all_suggestions": list(self.optimization_suggestions)
        }
    
    def export_performance_analysis(self, workflow_json):
        """Export performance analysis for a workflow"""
        analysis = {
            "status": "success",
            "workflow_metrics": {
                "node_count": 10,
                "connection_count": 12,
                "api_node_count": 3
            },
            "complexity_analysis": {
                "complexity_score": 15.5,
                "complexity_level": "Medium"
            },
            "performance_risks": [
                "High number of API calls may cause performance issues"
            ]
        }
        print(f"Exported performance analysis for workflow")
        return analysis

async def demo():
    """Run a demonstration of the agent's capabilities"""
    print("Starting N8N Integration Agent Demo")
    
    # Create the agent
    agent = MockN8NIntegrationAgent()
    
    # Demonstrate caching
    print("\n=== Caching Demonstration ===")
    cache_key = agent.generate_cache_key("template", "data_processing", "customization1")
    
    # First access (cache miss)
    result = agent.get_cached_result(cache_key)
    if result is None:
        print("Generating new result...")
        result = {"data": "This is a generated result"}
        agent.set_cached_result(cache_key, result)
    
    # Second access (cache hit)
    cached_result = agent.get_cached_result(cache_key)
    print(f"Retrieved from cache: {cached_result}")
    
    # Get cache statistics
    cache_stats = agent.get_cache_stats()
    print(f"Cache statistics: {json.dumps(cache_stats, indent=2)}")
    
    # Demonstrate performance analysis
    print("\n=== Performance Analysis Demonstration ===")
    workflow_json = {
        "nodes": [{"id": "1", "type": "HttpRequest"}, {"id": "2", "type": "Set"}],
        "connections": {"main": [{"node": "1", "type": "main", "index": 0}]}
    }
    
    performance_analysis = agent.export_performance_analysis(workflow_json)
    print(f"Performance analysis: {json.dumps(performance_analysis, indent=2)}")
    
    # Demonstrate optimization suggestions
    print("\n=== Optimization Suggestions Demonstration ===")
    optimization_suggestions = agent.export_optimization_suggestions()
    print(f"Optimization suggestions: {json.dumps(optimization_suggestions, indent=2)}")
    
    # Demonstrate cache clearing
    print("\n=== Cache Clearing Demonstration ===")
    clear_result = agent.clear_cache()
    print(f"Cache clearing result: {json.dumps(clear_result, indent=2)}")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    asyncio.run(demo())
