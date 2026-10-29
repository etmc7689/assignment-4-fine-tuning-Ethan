# -*- coding: utf-8 -*-
"""One-shot client for the mcp_bridge. Usage:
    .nb-venv/bin/python mcp_call.py '{"tool": "get_cells", "args": {}}'
    extra field "timeout" (seconds) raises the socket timeout, e.g. 5400 for long cell runs.
"""
import json, socket, sys

HOST, PORT = "127.0.0.1", 8787


def main():
    req = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    timeout = req.pop("timeout", 600)
    tool, args = req.get("tool"), req.get("args") or {}
    s = socket.create_connection((HOST, PORT), timeout=timeout)
    s.settimeout(timeout)
    s.sendall((json.dumps({"tool": tool, "args": args}) + "\n").encode())
    f = s.makefile("rb")
    line = f.readline()
    print(line.decode().strip())
    f.close()
    s.close()


if __name__ == "__main__":
    main()
