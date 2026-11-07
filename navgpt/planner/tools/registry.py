"""Transport-neutral navigation tools and results. No MCP or model SDK dependency."""

from __future__ import annotations

import asyncio
import base64
import inspect
from dataclasses import dataclass
from typing import Any, Literal, get_type_hints

from pydantic import BaseModel, ConfigDict, create_model


@dataclass(frozen=True)
class Image:
    data: bytes
    format: str = "png"


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


class ToolRegistry:
    def __init__(self):
        self.functions: dict[str, Any] = {}
        self.arguments: dict[str, type[BaseModel]] = {}
        self.definitions: dict[str, ToolDefinition] = {}

    def register(self, name, description, fn):
        if name in self.functions:
            raise ValueError("Duplicate tool: " + name)
        hints = get_type_hints(fn)
        fields = {
            key: (hints.get(key, Any), ... if p.default is inspect.Parameter.empty else p.default)
            for key, p in inspect.signature(fn).parameters.items()
        }
        args = create_model(name + "Arguments", __config__=ConfigDict(extra="forbid"), **fields)
        self.arguments[name] = args
        self.functions[name] = fn
        self.definitions[name] = ToolDefinition(
            name=name, description=description or "", inputSchema=args.model_json_schema()
        )

    def invoke(self, name, arguments) -> ToolResult:
        """Execute once; the caller owns the journal and uncertain-effect handling."""
        if name not in self.functions:
            raise ValueError("Unknown navigation tool: " + name)
        args = self.arguments[name].model_validate(arguments, strict=True).model_dump()
        values = self.functions[name](**args)
        if isinstance(values, ToolResult):
            return values
        return ToolResult(
            content=[
                ImageContent(data=base64.b64encode(v.data).decode(), mimeType="image/" + v.format)
                if isinstance(v, Image)
                else TextContent(text=str(v))
                for v in values
            ]
        )

    async def call_tool(self, name, arguments):
        return await asyncio.to_thread(self.invoke, name, arguments)

    async def list_tools(self):
        return list(self.definitions.values())
