from langchain_core.tools import tool
from datetime import datetime
import os
import logging
from flask import request
import pandas as pd
from genie_room import genie_query

logger = logging.getLogger(__name__)

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")

def get_response_from_genie(question: str) -> str:
    """
    Query Genie with a natural language question and return the response.
    
    This tool allows the agent to query data using natural language.
    Credentials are automatically extracted from the Flask request context.
    
    Args:
        question: Natural language question to ask Genie
        
    Returns:
        str: Response from Genie (text or formatted table data)
    """
    try:
        # Get user token from request headers (automatically available)
        headers = request.headers
        user_token = headers.get('X-Forwarded-Access-Token')
        
        if not user_token:
            return "Error: User authentication token not found."
        
        # Get Genie space ID from environment
        genie_space_id = os.environ.get("GENIE_SPACE")
        
        if not genie_space_id:
            return "Error: Genie space ID not configured."
        
        logger.info(f"Agent querying Genie: {question[:50]}...")
        
        # Call genie_query (same as in get_model_response)
        response, query_text = genie_query(question, user_token, genie_space_id)
        
        # Handle different response types
        if isinstance(response, pd.DataFrame):
            # Convert DataFrame to readable text for the agent
            result = f"Query Result ({len(response)} rows, {len(response.columns)} columns):\n\n"
            result += response.to_markdown(index=False)
            
            if query_text:
                result += f"\n\n[SQL Query used: {query_text}]"
            
            return result
        
        elif isinstance(response, str):
            # Text response - return as-is
            return response
        
        else:
            return f"Received response: {str(response)}"
        
    except Exception as e:
        logger.error(f"Error in get_response_from_genie: {str(e)}", exc_info=True)
        return f"Error querying Genie: {str(e)}"


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?
    - How complex is the question: Have I reached the number of search limits?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"