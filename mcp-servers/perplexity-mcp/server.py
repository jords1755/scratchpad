#!/usr/bin/env python3
"""
Perplexity MCP Server
Provides tools for querying Perplexity AI API with normal and deep research modes.
"""

import asyncio
import json
import os
from typing import Optional
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Perplexity API configuration
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")
if not PERPLEXITY_API_KEY:
    raise ValueError("PERPLEXITY_API_KEY environment variable is required")
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

# Available Perplexity models
MODELS = {
    "sonar": "sonar",  # Standard search model
    "sonar-pro": "sonar-pro",  # More capable search model
    "sonar-reasoning": "sonar-reasoning",  # Reasoning-focused model
    "sonar-reasoning-pro": "sonar-reasoning-pro",  # Advanced reasoning
    "sonar-deep-research": "sonar-deep-research",  # Deep research mode
}

# Default protocols that can be loaded
PROTOCOL_PATHS = {
    "deep_researcher": "/home/user/scratchpad/purpose-built/_Deep_Researcher_Protocol.txt",
    "tool_deep_researcher": "/home/user/scratchpad/assistant-workflows-tasks-personas/TOOL_Deep_Researcher_Protocol.txt",
}

server = Server("perplexity-mcp")


def load_protocol(protocol_name: str) -> Optional[str]:
    """Load a protocol file by name or path."""
    # Check if it's a known protocol name
    if protocol_name.lower() in PROTOCOL_PATHS:
        path = PROTOCOL_PATHS[protocol_name.lower()]
    else:
        path = protocol_name

    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return None
    except Exception as e:
        return f"Error loading protocol: {str(e)}"


async def query_perplexity(
    query: str,
    model: str = "sonar",
    system_prompt: Optional[str] = None,
    return_citations: bool = True,
    return_images: bool = False,
    search_recency_filter: Optional[str] = None,
) -> dict:
    """
    Query the Perplexity API.

    Args:
        query: The user query/question
        model: The Perplexity model to use
        system_prompt: Optional system prompt to guide the response
        return_citations: Whether to include source citations
        return_images: Whether to include related images
        search_recency_filter: Filter for recency (day, week, month, year)

    Returns:
        API response as dictionary
    """
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})

    payload = {
        "model": model,
        "messages": messages,
        "return_citations": return_citations,
        "return_images": return_images,
    }

    if search_recency_filter:
        payload["search_recency_filter"] = search_recency_filter

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            PERPLEXITY_API_URL,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        return response.json()


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available Perplexity tools."""
    return [
        Tool(
            name="perplexity_search",
            description="""Search the web using Perplexity AI. Use this for general queries,
            current events, and fact-finding. Returns web search results with citations.

            Models available:
            - sonar: Standard search (default, fast)
            - sonar-pro: More capable search
            - sonar-reasoning: Better for complex queries requiring reasoning
            - sonar-reasoning-pro: Advanced reasoning capabilities""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or question to ask Perplexity"
                    },
                    "model": {
                        "type": "string",
                        "enum": ["sonar", "sonar-pro", "sonar-reasoning", "sonar-reasoning-pro"],
                        "default": "sonar",
                        "description": "The Perplexity model to use"
                    },
                    "recency": {
                        "type": "string",
                        "enum": ["day", "week", "month", "year"],
                        "description": "Filter results by recency (optional)"
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "Optional system prompt to guide the response style/format"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="perplexity_deep_research",
            description="""Conduct deep research using Perplexity's sonar-deep-research model.
            This is designed for comprehensive, in-depth research on complex topics.

            You can optionally provide a protocol/instructions that will be passed verbatim
            to guide the research output format and methodology.

            Built-in protocols:
            - deep_researcher: Full Deep Researcher Protocol for 10000+ word academic reports
            - tool_deep_researcher: Tool version of the Deep Researcher Protocol

            Or provide custom instructions directly.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The research query or topic to investigate"
                    },
                    "protocol": {
                        "type": "string",
                        "description": "Name of built-in protocol (deep_researcher, tool_deep_researcher) OR custom instructions to pass verbatim"
                    },
                    "custom_instructions": {
                        "type": "string",
                        "description": "Additional custom instructions to append to the protocol"
                    },
                    "recency": {
                        "type": "string",
                        "enum": ["day", "week", "month", "year"],
                        "description": "Filter results by recency (optional)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="perplexity_with_protocol",
            description="""Query Perplexity with exact verbatim instructions from a protocol file.
            The protocol content is passed as the system prompt exactly as written.

            Use this when you need to execute a specific protocol/workflow using Perplexity.
            The protocol instructions will be followed exactly as specified.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to process according to the protocol"
                    },
                    "protocol_path": {
                        "type": "string",
                        "description": "Path to the protocol file OR name of built-in protocol"
                    },
                    "model": {
                        "type": "string",
                        "enum": ["sonar", "sonar-pro", "sonar-reasoning", "sonar-reasoning-pro", "sonar-deep-research"],
                        "default": "sonar-pro",
                        "description": "The Perplexity model to use"
                    },
                    "recency": {
                        "type": "string",
                        "enum": ["day", "week", "month", "year"],
                        "description": "Filter results by recency (optional)"
                    }
                },
                "required": ["query", "protocol_path"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""

    if name == "perplexity_search":
        query = arguments["query"]
        model = arguments.get("model", "sonar")
        recency = arguments.get("recency")
        system_prompt = arguments.get("system_prompt")

        try:
            result = await query_perplexity(
                query=query,
                model=model,
                system_prompt=system_prompt,
                search_recency_filter=recency,
            )

            # Extract the response content
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "No response")
            citations = result.get("citations", [])

            response_text = f"## Perplexity Search Results\n\n{content}"

            if citations:
                response_text += "\n\n### Sources\n"
                for i, citation in enumerate(citations, 1):
                    response_text += f"[{i}] {citation}\n"

            return [TextContent(type="text", text=response_text)]

        except httpx.HTTPStatusError as e:
            return [TextContent(type="text", text=f"API Error: {e.response.status_code} - {e.response.text}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    elif name == "perplexity_deep_research":
        query = arguments["query"]
        protocol = arguments.get("protocol")
        custom_instructions = arguments.get("custom_instructions", "")
        recency = arguments.get("recency")

        # Build system prompt from protocol
        system_prompt = None
        if protocol:
            # Check if it's a built-in protocol name
            loaded = load_protocol(protocol)
            if loaded and not loaded.startswith("Error"):
                system_prompt = loaded
            else:
                # Treat as custom instructions
                system_prompt = protocol

        if custom_instructions:
            if system_prompt:
                system_prompt += f"\n\n{custom_instructions}"
            else:
                system_prompt = custom_instructions

        try:
            result = await query_perplexity(
                query=query,
                model="sonar-deep-research",
                system_prompt=system_prompt,
                search_recency_filter=recency,
            )

            content = result.get("choices", [{}])[0].get("message", {}).get("content", "No response")
            citations = result.get("citations", [])

            response_text = f"## Deep Research Results\n\n{content}"

            if citations:
                response_text += "\n\n### Sources\n"
                for i, citation in enumerate(citations, 1):
                    response_text += f"[{i}] {citation}\n"

            return [TextContent(type="text", text=response_text)]

        except httpx.HTTPStatusError as e:
            return [TextContent(type="text", text=f"API Error: {e.response.status_code} - {e.response.text}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    elif name == "perplexity_with_protocol":
        query = arguments["query"]
        protocol_path = arguments["protocol_path"]
        model = arguments.get("model", "sonar-pro")
        recency = arguments.get("recency")

        # Load the protocol
        protocol_content = load_protocol(protocol_path)
        if not protocol_content:
            return [TextContent(type="text", text=f"Error: Could not load protocol from '{protocol_path}'")]
        if protocol_content.startswith("Error"):
            return [TextContent(type="text", text=protocol_content)]

        try:
            result = await query_perplexity(
                query=query,
                model=model,
                system_prompt=protocol_content,
                search_recency_filter=recency,
            )

            content = result.get("choices", [{}])[0].get("message", {}).get("content", "No response")
            citations = result.get("citations", [])

            response_text = f"## Protocol Execution Results\n\n**Protocol:** {protocol_path}\n**Model:** {model}\n\n{content}"

            if citations:
                response_text += "\n\n### Sources\n"
                for i, citation in enumerate(citations, 1):
                    response_text += f"[{i}] {citation}\n"

            return [TextContent(type="text", text=response_text)]

        except httpx.HTTPStatusError as e:
            return [TextContent(type="text", text=f"API Error: {e.response.status_code} - {e.response.text}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
