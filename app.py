import dash
from dash import html, dcc, Input, Output, State, callback, ALL, MATCH, callback_context, no_update, clientside_callback, dash_table
import dash_bootstrap_components as dbc
import json
from genie_room import genie_query
import pandas as pd
import os
from dotenv import load_dotenv
import sqlparse
from flask import request
import logging
from genie_room import GenieClient
from mcp_tools import DatabricksConnection, list_databricks_mcp_tools
import os
import uuid

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from databricks.sdk.config import Config
from langchain_databricks import ChatDatabricks

from deepagents import create_deep_agent
from tools import get_response_from_genie, think_tool
from subagents import critique_sub_agent, research_sub_agent
from prompts import RESEARCHER_INSTRUCTIONS

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the Genie space ID from environment variable
GENIE_SPACE_ID = os.environ.get("GENIE_SPACE")

# Create Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="CloudCFO",
    update_title=None,  # Disable automatic title updates
    suppress_callback_exceptions=True
)
app.server.config.update(
    TIMEOUT=300,  # 5 minutes
)
# Remove Dash favicon
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>CloudCFO - Cloud Cost AI Assistant</title>
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Add default welcome text that can be customized
DEFAULT_WELCOME_TITLE = "Welcome to CloudCFO - Cloud Cost Data Assistant"
DEFAULT_WELCOME_DESCRIPTION = "Explore and analyze Cloud Cost related data with AI-powered insights. Ask questions, discover trends, and make data-driven decisions."

# Define the layout
app.layout = html.Div([
    html.Div([
        dcc.Store(id="user-info", data={"initial": "Y", "username": "You"}),
        # Top navigation bar
        html.Div([
            # Left component containing both nav-left and sidebar
            html.Div([
                # Nav left
                html.Div([
                    html.Button([
                        html.Img(src="assets/menu_icon.svg", className="menu-icon")
                    ], id="sidebar-toggle", className="nav-button"),
                    html.Button([
                        html.Img(src="assets/plus_icon.svg", className="new-chat-icon")
                    ], id="new-chat-button", className="nav-button",disabled=False),
                    html.Button([
                        html.Img(src="assets/plus_icon.svg", className="new-chat-icon"),
                        html.Div("New chat", className="new-chat-text")
                    ], id="sidebar-new-chat-button", className="new-chat-button",disabled=False)
                ], id="nav-left", className="nav-left"),
                
                # Sidebar
                html.Div([
                    html.Div([
                        html.Div("Your conversations with Agent", className="sidebar-header-text"),
                    ], className="sidebar-header"),
                    html.Div([], className="chat-list", id="chat-list")
                ], id="sidebar", className="sidebar")
            ], id="left-component", className="left-component"),
            
            html.Div([
                html.Div("CloudCFO", id="logo-container", className="logo-container")
            ], className="nav-center"),
            html.Div([
                html.Div("Y", id="top-nav-avatar", className="user-avatar"),
                html.A(
                    html.Button(
                        "Logout",
                        id="logout-button",
                        className="logout-button"
                    ),
                    href=f"{os.getenv('DATABRICKS_APP_URL')}",
                    className="logout-link"
                )
            ], className="nav-right")
        ], className="top-nav"),
        
        # Main content area
        html.Div([
            html.Div([
                # Chat content
                html.Div([
                    # Welcome container
                    html.Div([
                        html.Div([html.Div([
                        html.Div(className="genie-logo")
                    ], className="genie-logo-container")],
                    className="genie-logo-container-header"),
                   
                        # Add settings button with tooltip
                        html.Div([
                            html.Div(id="welcome-title", className="welcome-message", children=DEFAULT_WELCOME_TITLE),
                        ], className="welcome-title-container"),
                        
                        html.Div(id="welcome-description", 
                                className="welcome-message-description",
                                children=DEFAULT_WELCOME_DESCRIPTION),
                        
                        # Suggestion buttons with IDs
                        html.Div([
                            html.Button([
                                html.Div(className="suggestion-icon"),
                                html.Div("What is the billing cost for last 6 months (month-wise) in GCP?", 
                                       className="suggestion-text", id="suggestion-1-text")
                            ], id="suggestion-1", className="suggestion-button"),
                            html.Button([
                                html.Div(className="suggestion-icon"),
                                html.Div("What is the cumulative cost saving for last 6 months (month-wise) in Azure?",
                                       className="suggestion-text", id="suggestion-2-text")
                            ], id="suggestion-2", className="suggestion-button"),
                            html.Button([
                                html.Div(className="suggestion-icon"),
                                html.Div("What is the cost saving for last 6 months (month-wise) in Azure?",
                                       className="suggestion-text", id="suggestion-3-text")
                            ], id="suggestion-3", className="suggestion-button"),
                            html.Button([
                                html.Div(className="suggestion-icon"),
                                html.Div("Show LBU, appref, apps, end date, source/target platforms for cloud migrations",
                                       className="suggestion-text", id="suggestion-4-text")
                            ], id="suggestion-4", className="suggestion-button")
                        ], className="suggestion-buttons")
                    ], id="welcome-container", className="welcome-container visible"),
                    
                    # Chat messages
                    html.Div([], id="chat-messages", className="chat-messages"),
                ], id="chat-content", className="chat-content"),
                
                # Input area
                html.Div([
                    html.Div([
                        dcc.Input(
                            id="chat-input-fixed",
                            placeholder="Ask your question...",
                            className="chat-input",
                            type="text",
                            disabled=False
                        ),
                        html.Div([
                            html.Button(
                                id="send-button-fixed", 
                                className="input-button send-button",
                                disabled=False
                            )
                        ], className="input-buttons-right"),
                        html.Div("You can only submit one query at a time", 
                                id="query-tooltip", 
                                className="query-tooltip")
                    ], id="fixed-input-container", className="fixed-input-container"),
                    html.Div("Always review the accuracy of responses.", className="disclaimer-fixed")
                ], id="fixed-input-wrapper", className="fixed-input-wrapper"),
            ], id="chat-container", className="chat-container"),
        ], id="main-content", className="main-content"),
        
        html.Div(id='dummy-output'),
        html.Div(id='dummy-insight-scroll'),
        dcc.Store(id="chat-trigger", data={"trigger": False, "message": ""}),
        dcc.Store(id="chat-history-store", data=[]),
        dcc.Store(id="query-running-store", data=False),
        dcc.Store(id="session-store", data={"current_session": None}),
    ], id="root-container", className="root-container")
], id="app-container", className="app-container")

# Store chat history
chat_history = []

def get_user_info_from_headers():
    """Extract user information from request headers"""
    try:
        username = request.headers.get("X-Forwarded-Preferred-Username", "").split("@")[0]
        username = username.split(".")
        username = [part[0].upper() + part[1:] for part in username]
        username = " ".join(username)

        if username:
            initial = username[0].upper()
            return {"initial": initial, "username": username}
        else:
            return {"initial": "Y", "username": "You"}
    except Exception as e:
        logger.warning(f"Failed to extract user info from headers: {str(e)}")
        return {"initial": "Y", "username": "You"}

def format_sql_query(sql_query):
    """Format SQL query using sqlparse library"""
    formatted_sql = sqlparse.format(
        sql_query,
        keyword_case='upper',  # Makes keywords uppercase
        identifier_case=None,  # Preserves identifier case
        reindent=True,         # Adds proper indentation
        indent_width=2,        # Indentation width
        strip_comments=False,  # Preserves comments
        comma_first=False      # Commas at the end of line, not beginning
    )
    return formatted_sql

def call_llm_for_insights(df, prompt=None):
    """
    Call an LLM to generate insights from a DataFrame.
    Args:
        df: pandas DataFrame
        prompt: Optional custom prompt
    Returns:
        str: Insights generated by the LLM
    """
    if prompt is None:
        prompt = (
            "You are a professional data analyst. Given the following table data, "
            "provide deep, actionable analysis for 1. Key insights and trends 2. Notable patterns and" 
            " anomalies 3. Business implications."
            "Be thorough, professional, and concise.\n\n"
        )
    csv_data = df.to_csv(index=False)
    full_prompt = f"{prompt}Table data:\n{csv_data}"
    # Call OpenAI (replace with your own LLM provider as needed)
    try:
        headers = request.headers
        user_token = headers.get('X-Forwarded-Access-Token')
        config = Config(
            host=f"https://{os.environ.get('DATABRICKS_HOST')}",
            token=user_token,
            auth_type="pat",  # Explicitly set authentication type to PAT
            retry_timeout_seconds=300,  # 5 minutes total retry timeout
            max_retries=5,              # Maximum number of retries
            retry_delay_seconds=2,      # Initial delay between retries
            retry_backoff_factor=2      # Exponential backoff factor
        )
        client = WorkspaceClient(config=config)
        response = client.serving_endpoints.query(
            os.getenv("SERVING_ENDPOINT_NAME"),
            messages=[ChatMessage(content=full_prompt, role=ChatMessageRole.USER)],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating insights: {str(e)}"
    
def deep_insight_function(user_question, agent):
    """
    Call an agent with the original user question to get deep insights.
    
    Args:
        user_question: The original question asked by the user
        agent: The agent instance to invoke
    
    Returns:
        str: Deep insights generated by the agent
    """
    try:
        logger.info("Invoking agent...")
        
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_question,
                    }
                ],
            }
        )
        
        logger.info("Agent invocation completed")
        
        # Extract the response from the agent
        messages = result.get("messages", [])
        
        # Get the last message (agent's response)
        if messages:
            # Handle different message formats
            last_message = messages[-1]
            if isinstance(last_message, dict):
                return last_message.get("content", "No response from agent")
            elif hasattr(last_message, "content"):
                return last_message.content
            else:
                return str(last_message)
        
        return "No response from agent"
        
    except Exception as e:
        logger.error(f"Error in deep_insight_function: {str(e)}", exc_info=True)
        # Re-raise with more context
        raise Exception(f"Agent invocation failed: {str(e)}") from e
    
# Callback to initialize user info from headers
@app.callback(
    Output("user-info", "data"),
    Input("user-info", "id"),
    prevent_initial_call=False
)
def initialize_user_info(_):
    """Initialize user info from request headers"""
    return get_user_info_from_headers()

# First callback: Handle inputs and show thinking indicator
@app.callback(
    [Output("chat-messages", "children", allow_duplicate=True),
     Output("chat-input-fixed", "value", allow_duplicate=True),
     Output("welcome-container", "className", allow_duplicate=True),
     Output("chat-trigger", "data", allow_duplicate=True),
     Output("query-running-store", "data", allow_duplicate=True),
     Output("chat-list", "children", allow_duplicate=True),
     Output("chat-history-store", "data", allow_duplicate=True),
     Output("session-store", "data", allow_duplicate=True)],
    [Input("suggestion-1", "n_clicks"),
     Input("suggestion-2", "n_clicks"),
     Input("suggestion-3", "n_clicks"),
     Input("suggestion-4", "n_clicks"),
     Input("send-button-fixed", "n_clicks"),
     Input("chat-input-fixed", "n_submit")],
    [State("suggestion-1-text", "children"),
     State("suggestion-2-text", "children"),
     State("suggestion-3-text", "children"),
     State("suggestion-4-text", "children"),
     State("chat-input-fixed", "value"),
     State("chat-messages", "children"),
     State("welcome-container", "className"),
     State("chat-list", "children"),
     State("chat-history-store", "data"),
     State("session-store", "data"),
     State("user-info", "data")],
    prevent_initial_call=True
)
def handle_all_inputs(s1_clicks, s2_clicks, s3_clicks, s4_clicks, send_clicks, submit_clicks,
                     s1_text, s2_text, s3_text, s4_text, input_value, current_messages,
                     welcome_class, current_chat_list, chat_history, session_data, user_info):
    ctx = callback_context
    if not ctx.triggered:
        return [no_update] * 8

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # Handle suggestion buttons
    suggestion_map = {
        "suggestion-1": s1_text,
        "suggestion-2": s2_text,
        "suggestion-3": s3_text,
        "suggestion-4": s4_text
    }
    
    # Get the user input based on what triggered the callback
    if trigger_id in suggestion_map:
        user_input = suggestion_map[trigger_id]
    else:
        user_input = input_value
    
    if not user_input:
        return [no_update] * 8
    
    # Create user message with user info
    user_initial = user_info.get("initial", "Y") if user_info else "Y"
    username = user_info.get("username", "You") if user_info else "You"
    
    user_message = html.Div([
        html.Div([
            html.Div(user_initial, className="user-avatar"),
            html.Span(username, className="model-name")
        ], className="user-info"),
        html.Div(user_input, className="message-text")
    ], className="user-message message")
    
    # Add the user message to the chat
    updated_messages = current_messages + [user_message] if current_messages else [user_message]
    
    # Add thinking indicator
    thinking_indicator = html.Div([
        html.Div([
            html.Span(className="spinner"),
            html.Span("Thinking...")
        ], className="thinking-indicator")
    ], className="bot-message message")
    
    updated_messages.append(thinking_indicator)
    
    # Handle session management
    if session_data["current_session"] is None:
        session_data = {"current_session": len(chat_history) if chat_history else 0}
    
    current_session = session_data["current_session"]
    
    # Update chat history
    if chat_history is None:
        chat_history = []
    
    if current_session < len(chat_history):
        chat_history[current_session]["messages"] = updated_messages
        chat_history[current_session]["queries"].append(user_input)
    else:
        chat_history.insert(0, {
            "session_id": current_session,
            "queries": [user_input],
            "messages": updated_messages
        })
    
    # Update chat list
    updated_chat_list = []
    for i, session in enumerate(chat_history):
        first_query = session["queries"][0]
        is_active = (i == current_session)
        updated_chat_list.append(
            html.Div(
                first_query,
                className=f"chat-item{'active' if is_active else ''}",
                id={"type": "chat-item", "index": i}
            )
        )
    
    return (updated_messages, "", "welcome-container hidden",
            {"trigger": True, "message": user_input}, True,
            updated_chat_list, chat_history, session_data)

# Second callback: Make API call and show response
@app.callback(
    [Output("chat-messages", "children", allow_duplicate=True),
     Output("chat-history-store", "data", allow_duplicate=True),
     Output("chat-trigger", "data", allow_duplicate=True),
     Output("query-running-store", "data", allow_duplicate=True)],
    [Input("chat-trigger", "data")],
    [State("chat-messages", "children"),
     State("chat-history-store", "data")],
    prevent_initial_call=True
)
def get_model_response(trigger_data, current_messages, chat_history):
    if not trigger_data or not trigger_data.get("trigger"):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    user_input = trigger_data.get("message", "")
    if not user_input:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    try:
        headers = request.headers
        user_token = headers.get('X-Forwarded-Access-Token')
        response, query_text = genie_query(user_input, user_token, GENIE_SPACE_ID)
        
        if isinstance(response, str):
            # Escape square brackets to prevent markdown auto-linking
            import re
            processed_response = response
            
            # Escape all square brackets to prevent markdown from interpreting them as links
            processed_response = processed_response.replace('[', '\\[').replace(']', '\\]')
            
            # Escape parentheses to prevent markdown from interpreting them as links
            processed_response = processed_response.replace('(', '\\(').replace(')', '\\)')
            
            # Escape angle brackets to prevent markdown from interpreting them as links
            processed_response = processed_response.replace('<', '\\<').replace('>', '\\>')
            
            content = dcc.Markdown(processed_response, className="message-text")
        else:
            # Data table response
            df = pd.DataFrame(response)
            
            # Store the DataFrame in chat_history for later retrieval by insight button
            if chat_history and len(chat_history) > 0:
                table_uuid = str(uuid.uuid4())
                chat_history[0].setdefault('dataframes', {})[table_uuid] = df.to_json(orient='split')
                chat_history[0].setdefault('user_questions', {})[table_uuid] = user_input
            else:
                table_uuid = str(uuid.uuid4())
                chat_history = [{
                    "dataframes": {table_uuid: df.to_json(orient='split')},
                    "user_questions": {table_uuid: user_input}
                    }]
            
            # Create the table with adjusted styles
            data_table = dash_table.DataTable(
                id=f"table-{len(chat_history)}",
                data=df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in df.columns],
                
                # Export configuration
                export_format="csv",
                export_headers="display",
                
                # Other table properties
                page_size=10,
                style_table={
                    'display': 'inline-block',
                    'overflowX': 'auto',
                    'width': '95%',
                    'marginRight': '20px'
                },
                style_cell={
                    'textAlign': 'left',
                    'fontSize': '12px',
                    'padding': '4px 10px',
                    'fontFamily': '-apple-system, BlinkMacSystemFont,Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif',
                    'backgroundColor': 'transparent',
                    'maxWidth': 'fit-content',
                    'minWidth': '100px'
                },
                style_header={
                    'backgroundColor': '#f8f9fa',
                    'fontWeight': '600',
                    'borderBottom': '1px solid #eaecef'
                },
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto'
                },
                fill_width=False,
                page_current=0,
                page_action='native'
            )

            # Format SQL query if available
            query_section = None
            if query_text is not None:
                formatted_sql = format_sql_query(query_text)
                query_index = f"{len(chat_history)}-{len(current_messages)}"
                
                query_section = html.Div([
                    html.Div([
                        html.Button([
                            html.Span("Show code", id={"type": "toggle-text", "index": query_index})
                        ], 
                        id={"type": "toggle-query", "index": query_index}, 
                        className="toggle-query-button",
                        n_clicks=0)
                    ], className="toggle-query-container"),
                    html.Div([
                        html.Pre([
                            html.Code(formatted_sql, className="sql-code")
                        ], className="sql-pre")
                    ], 
                    id={"type": "query-code", "index": query_index}, 
                    className="query-code-container hidden")
                ], id={"type": "query-section", "index": query_index}, className="query-section")
            
            insight_button = html.Button(
                "Generate Insights",
                id={"type": "insight-button", "index": table_uuid},
                className="insight-button",
                style={"border": "none", "background": "#f0f0f0", "padding": "8px 16px", "borderRadius": "4px", "cursor": "pointer"}
            )
            insight_output = dcc.Loading(
                id={"type": "insight-loading", "index": table_uuid},
                type="circle",
                color="#000000",
                children=html.Div(id={"type": "insight-output", "index": table_uuid})
            )
            # ADD THE NEW DEEP INSIGHT BUTTON:
            deep_insight_button = html.Button(
                "Create Deep Insight",
                id={"type": "deep-insight-button", "index": table_uuid},
                className="deep-insight-button",
                style={
                    "border": "none", 
                    "background": "#4A90E2",  # Different color to distinguish
                    "color": "white",
                    "padding": "8px 16px", 
                    "borderRadius": "4px", 
                    "cursor": "pointer",
                    "marginLeft": "10px"
                }
            )
            deep_insight_output = dcc.Loading(
                id={"type": "deep-insight-loading", "index": table_uuid},
                type="circle",
                color="#4A90E2",
                children=html.Div(id={"type": "deep-insight-output", "index": table_uuid})
            )



            # Create content with table and optional SQL section
            content = html.Div([
                html.Div([data_table], style={
                    'marginBottom': '20px',
                    'paddingRight': '5px'
                }),
                query_section if query_section else None,
                
                # Button container
                html.Div([
                    insight_button,
                    deep_insight_button  # ADD THIS
                ], style={"marginTop": "10px", "marginBottom": "10px"}),
                
                # Output containers
                insight_output,
                deep_insight_output,  # ADD THIS
            ])
            # content = html.Div([
            #     html.Div([data_table], style={
            #         'marginBottom': '20px',
            #         'paddingRight': '5px'
            #     }),
            #     query_section if query_section else None,
            #     insight_button,
            #     insight_output,
            # ])
        
        # Create bot response
        bot_response = html.Div([
            html.Div([
                html.Div(className="model-avatar"),
                html.Span("Agent", className="model-name")
            ], className="model-info"),
            html.Div([
                content,
            ], className="message-content")
        ], className="bot-message message")
        
        # Update chat history with both user message and bot response
        if chat_history and len(chat_history) > 0:
            chat_history[0]["messages"] = current_messages[:-1] + [bot_response]  
        return current_messages[:-1] + [bot_response], chat_history, {"trigger": False, "message": ""}, False
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}. Please try again later."
        error_response = html.Div([
            html.Div([
                html.Div(className="model-avatar"),
                html.Span("Agent", className="model-name")
            ], className="model-info"),
            html.Div([
                html.Div(error_msg, className="message-text")
            ], className="message-content")
        ], className="bot-message message")
        
        # Update chat history with both user message and error response
        if chat_history and len(chat_history) > 0:
            chat_history[0]["messages"] = current_messages[:-1] + [error_response]
        
        return current_messages[:-1] + [error_response], chat_history, {"trigger": False, "message": ""}, False

# Toggle sidebar and speech button
@app.callback(
    [Output("sidebar", "className"),
     Output("new-chat-button", "style"),
     Output("sidebar-new-chat-button", "style"),
     Output("logo-container", "className"),
     Output("nav-left", "className"),
     Output("left-component", "className"),
     Output("main-content", "className")],
    [Input("sidebar-toggle", "n_clicks")],
    [State("sidebar", "className"),
     State("left-component", "className"),
     State("main-content", "className")]
)
def toggle_sidebar(n_clicks, current_sidebar_class, current_left_component_class, current_main_content_class):
    if n_clicks:
        if "sidebar-open" in current_sidebar_class:
            # Sidebar is closing
            return "sidebar", {"display": "flex"}, {"display": "none"}, "logo-container", "nav-left", "left-component", "main-content"
        else:
            # Sidebar is opening
            return "sidebar sidebar-open", {"display": "none"}, {"display": "flex"}, "logo-container logo-container-open", "nav-left nav-left-open", "left-component left-component-open", "main-content main-content-shifted"
    # Initial state
    return current_sidebar_class, {"display": "flex"}, {"display": "none"}, "logo-container", "nav-left", "left-component", current_main_content_class

# Add callback for chat item selection
@app.callback(
    [Output("chat-messages", "children", allow_duplicate=True),
     Output("welcome-container", "className", allow_duplicate=True),
     Output("chat-list", "children", allow_duplicate=True),
     Output("session-store", "data", allow_duplicate=True)],
    [Input({"type": "chat-item", "index": ALL}, "n_clicks")],
    [State("chat-history-store", "data"),
     State("chat-list", "children"),
     State("session-store", "data")],
    prevent_initial_call=True
)
def show_chat_history(n_clicks, chat_history, current_chat_list, session_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Get the clicked item index
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    clicked_index = json.loads(triggered_id)["index"]
    
    if not chat_history or clicked_index >= len(chat_history):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Update session data to the clicked session
    new_session_data = {"current_session": clicked_index}
    
    # Update active state in chat list
    updated_chat_list = []
    for i, item in enumerate(current_chat_list):
        new_class = "chat-item active" if i == clicked_index else "chat-item"
        updated_chat_list.append(
            html.Div(
                item["props"]["children"],
                className=new_class,
                id={"type": "chat-item", "index": i}
            )
        )
    
    return (chat_history[clicked_index]["messages"], 
            "welcome-container hidden", 
            updated_chat_list,
            new_session_data)

# Modify the clientside callback to target the chat-container
app.clientside_callback(
    """
    function(children) {
        setTimeout(function() {
            var chatMessages = document.getElementById('chat-messages');
            if (chatMessages) {
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        }, 100);
        return '';
    }
    """,
    Output('dummy-output', 'children'),
    Input('chat-messages', 'children'),
    prevent_initial_call=True
)

# Modify the new chat button callback to reset session
@app.callback(
    [Output("welcome-container", "className", allow_duplicate=True),
     Output("chat-messages", "children", allow_duplicate=True),
     Output("chat-trigger", "data", allow_duplicate=True),
     Output("query-running-store", "data", allow_duplicate=True),
     Output("chat-history-store", "data", allow_duplicate=True),
     Output("session-store", "data", allow_duplicate=True)],
    [Input("new-chat-button", "n_clicks"),
     Input("sidebar-new-chat-button", "n_clicks")],
    [State("chat-messages", "children"),
     State("chat-trigger", "data"),
     State("chat-history-store", "data"),
     State("chat-list", "children"),
     State("query-running-store", "data"),
     State("session-store", "data")],
    prevent_initial_call=True
)
def reset_to_welcome(n_clicks1, n_clicks2, chat_messages, chat_trigger, chat_history_store, 
                    chat_list, query_running, session_data):
    # Reset session when starting a new chat
    new_session_data = {"current_session": None}
    return ("welcome-container visible", [], {"trigger": False, "message": ""}, 
            False, chat_history_store, new_session_data)

@app.callback(
    [Output("welcome-container", "className", allow_duplicate=True)],
    [Input("chat-messages", "children")],
    prevent_initial_call=True
)
def reset_query_running(chat_messages):
    # Return as a single-item list
    if chat_messages:
        return ["welcome-container hidden"]
    else:
        return ["welcome-container visible"]

# Add callback to disable input while query is running
@app.callback(
    [Output("chat-input-fixed", "disabled"),
     Output("send-button-fixed", "disabled"),
     Output("new-chat-button", "disabled"),
     Output("sidebar-new-chat-button", "disabled")],
    [Input("query-running-store", "data")],
    prevent_initial_call=True
)
def toggle_input_disabled(query_running):
    # Disable input and buttons when query is running
    return query_running, query_running, query_running, query_running

# Add callback for toggling SQL query visibility
@app.callback(
    [Output({"type": "query-code", "index": MATCH}, "className"),
     Output({"type": "toggle-text", "index": MATCH}, "children")],
    [Input({"type": "toggle-query", "index": MATCH}, "n_clicks")],
    prevent_initial_call=True
)
def toggle_query_visibility(n_clicks):
    if n_clicks % 2 == 1:
        return "query-code-container visible", "Hide code"
    return "query-code-container hidden", "Show code"

# Add callback for insight button
@app.callback(
    Output({"type": "insight-output", "index": dash.dependencies.MATCH}, "children"),
    Input({"type": "insight-button", "index": dash.dependencies.MATCH}, "n_clicks"),
    State({"type": "insight-button", "index": dash.dependencies.MATCH}, "id"),
    State("chat-history-store", "data"),
    prevent_initial_call=True
)
def generate_insights(n_clicks, btn_id, chat_history):
    if not n_clicks:
        return None
    table_id = btn_id["index"]
    df = None
    if chat_history and len(chat_history) > 0:
        df_json = chat_history[0].get('dataframes', {}).get(table_id)
        if df_json:
            df = pd.read_json(df_json, orient='split')
    if df is None:
        return html.Div("No data available for insights.", style={"color": "red"})
    insights = call_llm_for_insights(df)
    return html.Div([
        html.Div([
            dcc.Markdown(insights, className="insight-content")
        ], className="insight-body")
    ], className="insight-wrapper")

# In the generate_deep_insights callback:
@app.callback(
    Output({"type": "deep-insight-output", "index": dash.dependencies.MATCH}, "children"),
    Input({"type": "deep-insight-button", "index": dash.dependencies.MATCH}, "n_clicks"),
    State({"type": "deep-insight-button", "index": dash.dependencies.MATCH}, "id"),
    State("chat-history-store", "data"),
    prevent_initial_call=True
)
def generate_deep_insights(n_clicks, btn_id, chat_history):
    """
    Generate deep insights using the custom agent based on the original user question.
    """
    if not n_clicks:
        return None
    
    table_id = btn_id["index"]
    
    # Retrieve the original user question
    user_question = None
    if chat_history and len(chat_history) > 0:
        user_question = chat_history[0].get('user_questions', {}).get(table_id)
    
    if user_question is None:
        return html.Div(
            "No question available for deep insights.", 
            style={"color": "red", "marginTop": "10px"}
        )
    
    try:
        # Get user token
        headers = request.headers
        user_token = headers.get('X-Forwarded-Access-Token')
        
        if not user_token:
            logger.error("No user token found in headers")
            return html.Div(
                "Authentication error: No user token found.", 
                style={"color": "red", "marginTop": "10px"}
            )
        
        # Validate GENIE_SPACE_ID
        if not GENIE_SPACE_ID:
            logger.error("GENIE_SPACE environment variable is not set")
            return html.Div(
                "Configuration error: Genie space ID is not configured.", 
                style={"color": "red", "marginTop": "10px"}
            )
        
        logger.info(f"Starting deep insights for question: {user_question[:50]}...")
        logger.info(f"Using Genie Space ID: {GENIE_SPACE_ID}")
        
        # Import credentials provider
        from databricks_ai_bridge import ModelServingUserCredentials
        
        # ✅ CRITICAL FIX: Create WorkspaceClient with ONLY ModelServingUserCredentials
        # DO NOT mix with token/config when using MCP from Databricks Apps
        workspace_client = WorkspaceClient(
            credentials_strategy=ModelServingUserCredentials()
        )
        
        logger.info("WorkspaceClient created with ModelServingUserCredentials for OBO auth")
        
        # Validate Genie space access
        try:
            genie_client = GenieClient(
                host=os.environ.get("DATABRICKS_HOST"),
                space_id=GENIE_SPACE_ID,
                token=user_token
            )
            space_info = genie_client.get_space(GENIE_SPACE_ID)
            logger.info(f"Successfully validated access to Genie space: {space_info.get('title', 'Unknown')}")
        except Exception as space_error:
            logger.error(f"Failed to access Genie space: {str(space_error)}")
            return html.Div([
                html.Div("Error: Cannot access Genie space", style={
                    "color": "red", 
                    "fontWeight": "bold",
                    "marginTop": "10px",
                    "marginBottom": "10px"
                }),
                html.Div(f"Space ID: {GENIE_SPACE_ID}", style={
                    "fontSize": "12px",
                    "color": "#666",
                    "marginBottom": "5px"
                }),
                html.Div("Possible issues:", style={
                    "fontSize": "12px",
                    "marginTop": "10px",
                    "marginBottom": "5px"
                }),
                html.Ul([
                    html.Li("The Genie space ID may be incorrect"),
                    html.Li("You may not have permission to access this space"),
                    html.Li("Check GENIE_SPACE in app.yaml"),
                ], style={"fontSize": "12px", "color": "#666"})
            ])
        
        # ✅ Initialize the model with user token (for ChatDatabricks, still use token)
        model = ChatDatabricks(
            endpoint=os.getenv("SERVING_ENDPOINT_NAME"),
            token=user_token,
            temperature=0.0,  
            max_tokens=4000,  
        )
        
        # Build MCP URL
        databricks_host = workspace_client.config.host
        genie_mcp_url = f"{databricks_host}/api/2.0/mcp/genie/{GENIE_SPACE_ID}"
        
        logger.info(f"Connecting to Genie MCP at: {genie_mcp_url}")
        
        # ✅ Create MCP connection with OBO-authenticated workspace client
        mcp_connection_genie = DatabricksConnection(
            server_url=genie_mcp_url, 
            workspace_client=workspace_client  # This now has proper OBO credentials
        )
        
        # Get MCP tools
        try:
            logger.info("Loading MCP tools...")
            mcp_tools = list_databricks_mcp_tools(connections=[mcp_connection_genie])
            logger.info(f"Successfully loaded {len(mcp_tools)} MCP tools")
            
            # Log tool names
            for tool in mcp_tools:
                logger.info(f"  - Tool: {tool.name}")
                
        except Exception as e:
            logger.error(f"Error loading MCP tools: {str(e)}", exc_info=True)
            return html.Div([
                html.Div("Error loading MCP tools:", style={
                    "color": "red", 
                    "fontWeight": "bold",
                    "marginTop": "10px"
                }),
                html.Pre(str(e), style={
                    "background": "#f8f8f8",
                    "padding": "10px",
                    "borderRadius": "4px",
                    "fontSize": "12px",
                    "overflow": "auto",
                    "maxHeight": "200px"
                }),
                html.Div([
                    "If you see 403 errors, make sure:",
                    html.Ul([
                        html.Li("apps.apps scope is added to manifest.yaml"),
                        html.Li("The app has been redeployed after adding the scope"),
                        html.Li("You have CAN_RUN permission on the Genie space"),
                    ])
                ], style={"marginTop": "10px", "fontSize": "12px"})
            ])
        
        logger.info("Creating deep insight agent...")
        
        # Create agent with MCP tools
        agent = create_deep_agent(
            tools=mcp_tools,
            instructions=RESEARCHER_INSTRUCTIONS,
            model=model,
        ).with_config({"recursion_limit": 100})
        
        logger.info(f"Calling agent with question: {user_question[:50]}...")
        
        # Call the deep insight function
        deep_insights = deep_insight_function(user_question, agent)
        
        logger.info("Deep insights generated successfully")
        
        # Return formatted deep insights
        return html.Div([
            html.Div([
                html.H4("🔍 Deep Insights Analysis", style={
                    "color": "#4A90E2", 
                    "marginBottom": "10px",
                    "fontSize": "16px",
                    "fontWeight": "600"
                }),
                html.Div(f"Original Question: {user_question}", style={
                    "fontStyle": "italic",
                    "color": "#666",
                    "marginBottom": "15px",
                    "padding": "10px",
                    "background": "#f8f9fa",
                    "borderRadius": "4px"
                }),
                dcc.Markdown(deep_insights, className="deep-insight-content")
            ], className="deep-insight-body", style={
                "padding": "15px",
                "background": "#f0f8ff",
                "borderRadius": "8px",
                "marginTop": "15px",
                "border": "1px solid #4A90E2"
            })
        ], className="deep-insight-wrapper")
        
    except Exception as e:
        import traceback
        full_error = traceback.format_exc()
        logger.error(f"Error generating deep insights: {full_error}")
        
        return html.Div([
            html.Div("Error generating deep insights:", style={
                "color": "red", 
                "fontWeight": "bold",
                "marginTop": "10px",
                "marginBottom": "10px"
            }),
            html.Pre(str(e), style={
                "background": "#f8f8f8",
                "padding": "10px",
                "borderRadius": "4px",
                "fontSize": "12px",
                "overflow": "auto",
                "maxHeight": "200px"
            })
        ])
# Callback to fetch spaces on load
# Initialize welcome title and description from space info
@app.callback(
    [Output("welcome-title", "children"),
     Output("welcome-description", "children")],
    Input("user-info", "id"),
    prevent_initial_call=False
)
def initialize_welcome_info(_):
    if not GENIE_SPACE_ID:
        return DEFAULT_WELCOME_TITLE, DEFAULT_WELCOME_DESCRIPTION
    
    try:
        # Try to fetch space information to get title and description
        headers = request.headers
        token = headers.get('X-Forwarded-Access-Token')
        host = os.environ.get("DATABRICKS_HOST")
        client = GenieClient(host=host, space_id="", token=token)
        
        # Get the specific space details
        space_details = client.get_space(GENIE_SPACE_ID)
        
        title = space_details.get("title", DEFAULT_WELCOME_TITLE)
        description = space_details.get("description", DEFAULT_WELCOME_DESCRIPTION)
        return title, description
            
    except Exception as e:
        logger.warning(f"Could not fetch space information: {e}")
        return DEFAULT_WELCOME_TITLE, DEFAULT_WELCOME_DESCRIPTION

# Add clientside callback for scrolling to bottom of chat when insight is generated
app.clientside_callback(
    """
    function(children) {
        setTimeout(function() {
            var chatMessages = document.getElementById('chat-messages');
            if (chatMessages) {
                chatMessages.scrollTop = chatMessages.scrollHeight;
                if (chatMessages.lastElementChild) {
                    chatMessages.lastElementChild.scrollIntoView({behavior: 'auto', block: 'end'});
                }
            }
        }, 100);
        return '';
    }
    """,
    Output('dummy-insight-scroll', 'children'),
    Input({'type': 'insight-output', 'index': dash.dependencies.ALL}, 'children'),
    prevent_initial_call=True
)



# Add a callback to control the root-container style
@app.callback(
    Output("root-container", "style"),
    Input("user-info", "id"),
    prevent_initial_call=False
)
def set_root_style(_):
    if GENIE_SPACE_ID:
        return {"height": "auto", "overflow": "auto"}
    else:
        return {"height": "100vh", "overflow": "hidden"}

@app.callback(
    Output("query-tooltip", "className"),
    Input("query-running-store", "data"),
    prevent_initial_call=False
)
def update_query_tooltip_class(query_running):
    # Only show tooltip if query is running
    if query_running:
        return "query-tooltip query-tooltip-active"
    else:
        return "query-tooltip"

# Callback to update top navigation avatar
@app.callback(
    Output("top-nav-avatar", "children"),
    Input("user-info", "data"),
    prevent_initial_call=False
)
def update_top_nav_avatar(user_info):
    """Update the top navigation avatar with user initial"""
    if user_info and user_info.get("initial"):
        return user_info["initial"]
    return "Y"

if __name__ == "__main__":
    app.run_server(debug=True)
