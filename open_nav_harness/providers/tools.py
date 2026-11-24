"""Tool definitions and results exchanged between a model loop and its tools (MCP-shaped)."""

from typing import Literal

from pydantic import BaseModel


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    data: str
    mimeType: str


class ToolResult(BaseModel):
    content: list[TextContent | ImageContent]
    isError: bool = False


class ToolDefinition(BaseModel):
    name: str
    description: str
    inputSchema: dict
