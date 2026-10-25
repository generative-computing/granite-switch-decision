# SPDX-License-Identifier: Apache-2.0
"""granite-switch-decision: OpenAI Decisions (`POST /v1/decisions`) and jev system-one
(`POST /v1/systemone`) endpoints for granite-switch's vLLM backend, mounted on the same
server that already serves `/v1/chat/completions`: one engine, one KV cache, one container.

See `plugin.py`'s module docstring for the full design. `renderer.py` is the one
adapter-specific piece this package carries today (the decision adapter's trained
prompt convention); `plugin.py` itself has no knowledge of any specific adapter.
"""

__version__ = "0.2.0"
