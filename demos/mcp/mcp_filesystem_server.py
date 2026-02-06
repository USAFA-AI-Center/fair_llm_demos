#!/usr/bin/env python3
# mcp_filesystem_server.py
"""
A simple Python-based MCP server that provides filesystem access tools.

This server exposes three tools:
- list_directory: List files and directories in a path
- read_file: Read the contents of a file
- get_file_info: Get metadata about a file

Usage:
    python mcp_filesystem_server.py /path/to/allowed/directory

The server runs via stdio and can be used with any MCP client.
"""
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


def create_filesystem_server(allowed_directory: str) -> Server:
    """
    Create an MCP server with filesystem tools.

    Args:
        allowed_directory: The root directory that tools can access.
                          All paths are restricted to this directory.

    Returns:
        Configured MCP Server instance.
    """
    server = Server("filesystem")
    allowed_path = Path(allowed_directory).resolve()

    def is_path_allowed(path: str) -> bool:
        """Check if a path is within the allowed directory."""
        try:
            resolved = Path(path).resolve()
            return str(resolved).startswith(str(allowed_path))
        except Exception:
            return False

    def resolve_path(path: str) -> Path:
        """Resolve a path relative to the allowed directory."""
        if os.path.isabs(path):
            resolved = Path(path).resolve()
        else:
            resolved = (allowed_path / path).resolve()

        if not str(resolved).startswith(str(allowed_path)):
            raise ValueError(f"Access denied: path outside allowed directory")

        return resolved

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """Return the list of available tools."""
        return [
            Tool(
                name="list_directory",
                description=(
                    "List files and directories in a specified path. "
                    "Returns names, types (file/directory), and sizes. "
                    f"All paths are relative to: {allowed_path}"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to list (relative or absolute within allowed area). Use '.' for current directory."
                        }
                    },
                    "required": ["path"]
                }
            ),
            Tool(
                name="read_file",
                description=(
                    "Read the contents of a text file. "
                    "Returns the file content as text. "
                    f"All paths are relative to: {allowed_path}"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to read"
                        },
                        "max_lines": {
                            "type": "integer",
                            "description": "Maximum number of lines to read (default: 100)"
                        }
                    },
                    "required": ["path"]
                }
            ),
            Tool(
                name="get_file_info",
                description=(
                    "Get metadata about a file or directory. "
                    "Returns size, modification time, and type."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file or directory"
                        }
                    },
                    "required": ["path"]
                }
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle tool calls."""
        try:
            if name == "list_directory":
                return await handle_list_directory(arguments, resolve_path)
            elif name == "read_file":
                return await handle_read_file(arguments, resolve_path)
            elif name == "get_file_info":
                return await handle_get_file_info(arguments, resolve_path)
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    return server


async def handle_list_directory(arguments: dict, resolve_path) -> list[TextContent]:
    """List contents of a directory."""
    path = arguments.get("path", ".")
    dir_path = resolve_path(path)

    if not dir_path.exists():
        return [TextContent(type="text", text=f"Directory not found: {path}")]

    if not dir_path.is_dir():
        return [TextContent(type="text", text=f"Not a directory: {path}")]

    entries = []
    for entry in sorted(dir_path.iterdir()):
        entry_type = "dir" if entry.is_dir() else "file"
        size = entry.stat().st_size if entry.is_file() else 0
        entries.append(f"  [{entry_type}] {entry.name}" + (f" ({size} bytes)" if entry.is_file() else ""))

    result = f"Contents of {dir_path}:\n" + "\n".join(entries) if entries else f"Directory {dir_path} is empty"
    return [TextContent(type="text", text=result)]


async def handle_read_file(arguments: dict, resolve_path) -> list[TextContent]:
    """Read contents of a file."""
    path = arguments.get("path")
    max_lines = arguments.get("max_lines", 100)

    if not path:
        return [TextContent(type="text", text="Error: path is required")]

    file_path = resolve_path(path)

    if not file_path.exists():
        return [TextContent(type="text", text=f"File not found: {path}")]

    if not file_path.is_file():
        return [TextContent(type="text", text=f"Not a file: {path}")]

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    lines.append(f"\n... (truncated at {max_lines} lines)")
                    break
                lines.append(line.rstrip())

        content = "\n".join(lines)
        return [TextContent(type="text", text=f"Contents of {file_path.name}:\n\n{content}")]
    except UnicodeDecodeError:
        return [TextContent(type="text", text=f"Error: {path} is not a text file")]


async def handle_get_file_info(arguments: dict, resolve_path) -> list[TextContent]:
    """Get metadata about a file or directory."""
    path = arguments.get("path")

    if not path:
        return [TextContent(type="text", text="Error: path is required")]

    file_path = resolve_path(path)

    if not file_path.exists():
        return [TextContent(type="text", text=f"Path not found: {path}")]

    stat = file_path.stat()
    file_type = "directory" if file_path.is_dir() else "file"
    mod_time = datetime.fromtimestamp(stat.st_mtime).isoformat()

    info = [
        f"Path: {file_path}",
        f"Type: {file_type}",
        f"Size: {stat.st_size} bytes",
        f"Modified: {mod_time}",
    ]

    return [TextContent(type="text", text="\n".join(info))]


async def main():
    """Main entry point for the MCP server."""
    if len(sys.argv) < 2:
        print("Usage: python mcp_filesystem_server.py <allowed_directory>", file=sys.stderr)
        sys.exit(1)

    allowed_directory = sys.argv[1]

    if not os.path.isdir(allowed_directory):
        print(f"Error: {allowed_directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    server = create_filesystem_server(allowed_directory)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
