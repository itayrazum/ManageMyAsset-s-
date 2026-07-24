"""LangChain tool wrappers around the pure functions in `data.py`.

`StructuredTool.from_function` derives each tool's name, argument schema, and
description automatically from the function's signature and docstring, so the
tool definitions stay in sync with the data layer with no duplication.
"""

from langchain_core.tools import StructuredTool

from . import data

# The tools the agent is allowed to call.
TOOLS = [
    StructuredTool.from_function(data.list_properties),
    StructuredTool.from_function(data.list_tenants),
    StructuredTool.from_function(data.calculate_pnl),
    StructuredTool.from_function(data.top_tenants),
    StructuredTool.from_function(data.breakdown_by),
]
