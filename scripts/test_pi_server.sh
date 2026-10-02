#!/bin/bash
# Use the ogbench venv python if present, otherwise fall back to isaac sim python
# (aliases aren't expanded in scripts, so use the path isaacpy points to).
OGBENCH_PY=/gscratch/weirdlab/will/venvs/ogbench/bin/python3
if [ -x "$OGBENCH_PY" ]; then
    PY="$OGBENCH_PY"
else
    PY=/isaac-sim/python.sh
fi

"$PY" -c "import socket;s=socket.create_connection(('localhost',8000),5);s.send(b'GET / HTTP/1.1\r\nHost: localhost:8000\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n');print(s.recv(200).split(b'\r\n')[0].decode())"
