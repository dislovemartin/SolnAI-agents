import React from 'react';

interface RedditResult {
  subreddit: string;
  description?: string;
  subscribers?: number;
  posts?: any[];
}

interface RedditResultsGridProps {
  results: string[] | RedditResult[];
  className?: string;
}

export function RedditResultsGrid({ results, className = '' }: RedditResultsGridProps) {
  // Handle simple string array (just subreddit names)
  const formattedResults = results.map(result => 
    typeof result === 'string' 
      ? { subreddit: result } 
      : result
  );

  return (
    <div className={`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 ${className}`}>
      {formattedResults.map((result, index) => (
        <div key={index} className="border rounded-md p-4 h-full">
          <div className="pb-2">
            <div className="text-lg flex items-center justify-between">
              <span>r/{result.subreddit}</span>
              {result.subscribers && (
                <span className="ml-2 text-sm bg-gray-100 px-2 py-1 rounded-full">
                  {new Intl.NumberFormat().format(result.subscribers)} members
                </span>
              )}
            </div>
          </div>
          <div>
            {result.description && (
              <p className="text-sm text-gray-600 line-clamp-3">{result.description}</p>
            )}
            {!result.description && (
              <p className="text-sm text-gray-500 italic">No description available</p>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
