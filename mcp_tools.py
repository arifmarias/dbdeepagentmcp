from typing import Any, Callable, List, Generator, TypedDict, cast
from contextlib import asynccontextmanager
import asyncio
import nest_asyncio  # ✅ ADD THIS

from databricks_mcp import DatabricksOAuthClientProvider
from databricks.sdk import WorkspaceClient
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    Tool as MCPTool,
)
from langchain_core.tools import BaseTool, ToolException, StructuredTool, tool

# ✅ CRITICAL: Apply nest_asyncio to allow nested event loops in Flask/Dash
nest_asyncio.apply()

NonTextContent = ImageContent | EmbeddedResource
MAX_ITERATIONS = 1000

class DatabricksConnection(TypedDict):
    """Type definition for storing connection information to a Databricks MCP server.

    Attributes:
        server_url (str): Databricks MCP endpoint URL to connect to.
        workspace_client (WorkspaceClient): Databricks workspace client (for authentication).
    """
    server_url: str
    workspace_client: WorkspaceClient


@asynccontextmanager
async def _databricks_mcp_session(connection: DatabricksConnection):
    """Context manager for creating an asynchronous session to a Databricks MCP server.

    Args:
        connection (DatabricksConnection): Connection information to the MCP server.

    Yields:
        ClientSession: Initialized MCP client session.
    """
    async with streamablehttp_client(
        url=connection.get("server_url"),
        auth=DatabricksOAuthClientProvider(connection.get("workspace_client")),
        timeout=60,
    ) as (reader, writer, _):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            yield session


async def _list_all_tools(session: ClientSession) -> list[MCPTool]:
    """Retrieves all tool information from the MCP server using pagination.

    Args:
        session (ClientSession): MCP client session

    Returns:
        list[MCPTool]: List of all retrieved tools

    Raises:
        RuntimeError: If the page count exceeds the limit.
    """
    current_cursor: str | None = None
    all_tools: list[MCPTool] = []

    iterations = 0

    while True:
        iterations += 1
        if iterations > MAX_ITERATIONS:
            raise RuntimeError(
                f"Reached max of {MAX_ITERATIONS} iterations while listing tools."
            )

        list_tools_page_result = await session.list_tools(cursor=current_cursor)

        if list_tools_page_result.tools:
            all_tools.extend(list_tools_page_result.tools)

        if list_tools_page_result.nextCursor is None:
            break

        current_cursor = list_tools_page_result.nextCursor
    return all_tools


def _convert_call_tool_result(
    call_tool_result: CallToolResult,
) -> tuple[str | list[str], list[NonTextContent] | None]:
    """Returns the MCP tool call result, split into text and non-text content.

    Args:
        call_tool_result (CallToolResult): MCP tool call result.

    Returns:
        tuple[str | list[str], list[NonTextContent] | None]: Text content and non-text content.

    Raises:
        ToolException: If an error occurs.
    """
    text_contents: list[TextContent] = []
    non_text_contents = []
    for content in call_tool_result.content:
        if isinstance(content, TextContent):
            text_contents.append(content)
        else:
            non_text_contents.append(content)

    tool_content: str | list[str] = [content.text for content in text_contents]
    if not text_contents:
        tool_content = ""
    elif len(text_contents) == 1:
        tool_content = tool_content[0]

    if call_tool_result.isError:
        raise ToolException(tool_content)

    return tool_content, non_text_contents or None


def _convert_mcp_tool_to_langchain_tool(
    connection: DatabricksConnection,
    tool: MCPTool,
) -> BaseTool:
    """Converts MCP tool information to a LangChain StructuredTool.

    Args:
        connection (DatabricksConnection): MCP server connection information
        tool (MCPTool): MCP tool information

    Returns:
        BaseTool: LangChain-compatible tool
    """
    if connection is None:
        raise ValueError("a connection config must be provided")

    async def call_tool_async(
        **arguments: dict[str, Any],
    ) -> tuple[str | list[str], list[NonTextContent] | None]:
        async with _databricks_mcp_session(connection) as tool_session:
            call_tool_result = await cast(ClientSession, tool_session).call_tool(
                tool.name, arguments
            )
        return _convert_call_tool_result(call_tool_result)

    def call_tool_sync(
        **arguments: dict[str, Any]
    ) -> tuple[str | list[str], list[NonTextContent] | None]:
        # ✅ CHANGED: Use asyncio.run() directly (nest_asyncio allows this)
        return asyncio.run(call_tool_async(**arguments))

    return StructuredTool(
        name=tool.name,
        description=tool.description or "",
        args_schema=tool.inputSchema,
        coroutine=call_tool_async,
        func=call_tool_sync,
        response_format="content_and_artifact",
        metadata=tool.annotations.model_dump() if tool.annotations else None,
    )


def list_databricks_mcp_tools(
    connections: list[DatabricksConnection],
) -> list[BaseTool]:
    """Get all tools from multiple MCP servers and return them as a LangChain tool list.

    Args:
        connections (list[DatabricksConnection]): List of MCP server connection information.

    Returns:
        list[BaseTool]: List of all LangChain-compatible tools.
    """
    if type(connections) != list:
        raise ValueError("connections must be a list of DatabricksConnection")

    async def _load_databricks_mcp_tools(
        connection: DatabricksConnection,
    ) -> list[BaseTool]:
        if connection is None:
            raise ValueError("connection config must be provided")

        async with _databricks_mcp_session(connection) as tool_session:
            tools = await _list_all_tools(tool_session)

        converted_tools = [
            _convert_mcp_tool_to_langchain_tool(connection, tool) for tool in tools
        ]
        return converted_tools

    async def gather():
        tasks = [_load_databricks_mcp_tools(con) for con in connections]
        return await asyncio.gather(*tasks)

    # ✅ CHANGED: Use asyncio.run() directly (nest_asyncio allows this)
    all_tools = asyncio.run(gather())
    
    # Flatten the results and return them as a single list of tools
    return sum(all_tools, [])