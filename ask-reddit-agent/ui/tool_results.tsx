"use client"

import { useState } from "react"
import { ChevronDown, ChevronUp } from 'lucide-react'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

// DEPENDS ON SHADCN CARD AND BUTTON, LUCIDE REACT

interface ToolResult {
  args: Record<string, any>
  result: Record<string, any>
  tool_name: string
}

interface ToolResultsProps {
  data: {
    tool_results: Record<string, ToolResult>
  }
}

function ToolResultExpander({ toolCall, result }: { toolCall: string; result: ToolResult }) {
  const [isExpanded, setIsExpanded] = useState(false)

  return (
    <Card className="mb-4">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg flex items-center justify-between">
          <div className="flex items-center">
            <span className="mr-2">🛠️</span>
            <span>{result.tool_name}</span>
          </div>
          <Button variant="ghost" size="sm" onClick={() => setIsExpanded(!isExpanded)} aria-label={isExpanded ? "Collapse tool result" : "Expand tool result"}>
            {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </Button>
        </CardTitle>
      </CardHeader>
      {isExpanded && (
        <CardContent>
          <div className="space-y-4">
            <div>
              <h4 className="text-sm font-medium mb-1">Arguments:</h4>
              <pre className="bg-gray-100 p-3 rounded-md overflow-x-auto text-xs">
                {JSON.stringify(result.args, null, 2)}
              </pre>
            </div>
            <div>
              <h4 className="text-sm font-medium mb-1">Result:</h4>
              <pre className="bg-gray-100 p-3 rounded-md overflow-x-auto text-xs">
                {JSON.stringify(result.result, null, 2)}
              </pre>
            </div>
          </div>
        </CardContent>
      )}
    </Card>
  )
}

export default function ToolResultsExpander({ data }: ToolResultsProps) {
  if (!data.tool_results || Object.keys(data.tool_results).length === 0) {
    return null
  }

  return (
    <div className="space-y-8">
      <h2 className="text-2xl font-bold mb-4">Detailed Tool Results</h2>
      {Object.entries(data.tool_results).map(([toolCall, result]) => (
        <ToolResultExpander key={toolCall} toolCall={toolCall} result={result} />
      ))}
    </div>
  )
}
