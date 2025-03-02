import React, { useState } from 'react';

interface ToolResultsExpanderProps {
  toolResults: Record<string, any>;
  className?: string;
}

export function ToolResultsExpander({ toolResults, className = '' }: ToolResultsExpanderProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!toolResults || Object.keys(toolResults).length === 0) {
    return null;
  }

  const toggleExpand = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <div className={`w-full border rounded-md ${className}`}>
      <div className="p-4 pb-2">
        <div className="text-lg flex items-center justify-between">
          <div className="flex items-center">
            <span className="mr-2">🛠️</span>
            <span>Tool Results</span>
          </div>
          <button 
            className="p-1 rounded-md hover:bg-gray-100"
            onClick={toggleExpand}
            aria-label={isExpanded ? "Collapse tool results" : "Expand tool results"}
          >
            {isExpanded ? "▲" : "▼"}
          </button>
        </div>
      </div>
      {isExpanded && (
        <div className="px-4 pb-4">
          <pre className="bg-gray-100 p-4 rounded-md overflow-x-auto text-xs">
            {JSON.stringify(toolResults, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
