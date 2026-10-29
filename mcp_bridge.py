# -*- coding: utf-8 -*-
"""Long-lived bridge: spawns the davedgd/colab-mcp server (stdio) and exposes its
MCP tools over a tiny JSON-line protocol on 127.0.0.1:8787.

Request:  {"tool": "get_cells", "args": {...}}\\n
Response: {"ok": true, "result": "..."}\\n

Run: .nb-venv/bin/python mcp_bridge.py   (background process)
"""
import asyncio, json, os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HOST, PORT = "127.0.0.1", 8787
REPO_DIR = "/tmp/colab-mcp-inspect"
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, ".colab-mcp-logs")

_lock = asyncio.Lock()


async def handle(reader, writer, session):
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            try:
                req = json.loads(line)
                tool = req.get("tool")
                args = req.get("args") or {}
                async with _lock:
                    result = await session.call_tool(tool, args)
                texts = []
                if hasattr(result, "content"):
                    texts = [c.text for c in result.content if hasattr(c, "text")]
                if not texts and hasattr(result, "structured_content") and result.structured_content:
                    texts = [json.dumps(result.structured_content)]
                payload = {"ok": True, "result": "\n".join(texts)}
            except Exception as e:
                payload = {"ok": False, "error": repr(e)}
            writer.write((json.dumps(payload) + "\n").encode())
            await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    params = StdioServerParameters(
        command="uv",
        args=["run", "--directory", REPO_DIR, "colab-mcp", "-l", LOG_DIR],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("MCP session initialized", flush=True)
            server = await asyncio.start_server(
                lambda r, w: handle(r, w, session), HOST, PORT
            )
            print(f"Bridge listening on {HOST}:{PORT}", flush=True)
            await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
