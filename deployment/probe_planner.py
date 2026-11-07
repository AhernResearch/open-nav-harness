"""Read-only protocol probe: multimodal token counting and tool-call validity."""

import argparse
import asyncio
import base64
import io
import json
from pathlib import Path
from PIL import Image
from nav_harness.core.artifacts import Journal, atomic_json
from nav_harness.core.config import PlannerProfile
from nav_harness.providers.compatible import CompatibleClient


async def probe(url, model, directory, image_count=2, image_size=256):
    client = CompatibleClient(PlannerProfile(model=model, base_url=url, output_tokens=128))
    buf = io.BytesIO()
    Image.new("RGB", (image_size, image_size), "red").save(buf, format="PNG")
    image = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()},
    }
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Call report_color with the shared color of all these pictures."},
                *[image] * image_count,
            ],
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "report_color",
                "description": "Report image color",
                "parameters": {
                    "type": "object",
                    "properties": {"color": {"type": "string", "enum": ["red", "blue"]}},
                    "required": ["color"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    journal = Journal(directory)
    try:
        counted = await client.count(messages, tools)
        response = await client.generate(messages, tools, journal)
        calls = response["choices"][0]["message"].get("tool_calls", [])
        assert len(calls) == 1 and calls[0]["function"]["name"] == "report_color"
        assert json.loads(calls[0]["function"]["arguments"]) == {"color": "red"}
        actual = response["usage"]["prompt_tokens"]
        assert counted == actual, (counted, actual)
        result = {
            "ok": True,
            "model": model,
            "counted_tokens": counted,
            "actual_prompt_tokens": actual,
            "image_count": image_count,
            "image_size": image_size,
            "schema_tool_call": True,
        }
        atomic_json(directory / "result.json", result)
        print(json.dumps(result))
    finally:
        await client.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--images", type=int, default=2)
    p.add_argument("--image-size", type=int, default=256)
    a = p.parse_args()
    asyncio.run(probe(a.url, a.model, a.output, a.images, a.image_size))
