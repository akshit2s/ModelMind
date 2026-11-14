from google.genai import types

from agent.llm_client import (
    create_gemini_client,
    MODEL_NAME
)

from agent.prompts import MODELMIND_SYSTEM_INSTRUCTION
from agent.tools import create_tool_registry


def create_agent_session(df, max_tool_calls=5):

    # Create Python tool registry
    tool_registry = create_tool_registry(df)

    # ----------------------------------
    # DEFINE AVAILABLE TOOLS
    # ----------------------------------

    def inspect_dataset() -> dict:
        """Inspect dataset structure and statistics."""

        print("\n[TOOL] Executing inspect_dataset...", flush=True)

        return tool_registry["inspect_dataset"]()

    def inspect_column(column_name: str) -> dict:
        """Inspect a specific dataset column."""

        print(
            f"\n[TOOL] Executing inspect_column: {column_name}",
            flush=True
        )

        return tool_registry["inspect_column"](
            column_name=column_name
        )

    available_tools = [
        inspect_dataset,
        inspect_column
    ]

    # ----------------------------------
    # SYSTEM INSTRUCTIONS
    # ----------------------------------

    system_instruction = (
        MODELMIND_SYSTEM_INSTRUCTION
        + """

        CONVERSATION RULES:

        1. Remember previous messages in the current conversation.

        2. Understand references such as "it",
           "that column" and "previous result".

        3. Use previous tool results when relevant.

        4. Never invent dataset statistics.

        5. Never expose actual dataset records.

        6. Available tools are read-only.

        7. Never claim to have trained or modified
           a model unless the operation was executed.

        """
    )

    # ----------------------------------
    # CREATE CLIENT
    # ----------------------------------

    client = create_gemini_client()

    # ----------------------------------
    # CONFIGURE CHAT
    # ----------------------------------

    config = types.GenerateContentConfig(

        system_instruction=system_instruction,

        tools=available_tools,

        automatic_function_calling=(
            types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=max_tool_calls
            )
        ),

        max_output_tokens=1024

    )

    chat = client.chats.create(
        model=MODEL_NAME,
        config=config
    )

    # IMPORTANT:
    # Return BOTH objects so the client remains alive.
    return {
        "client": client,
        "chat": chat
    }


def run_agent(session, user_prompt):

    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    # Retain a reference to the client during the request.
    client = session["client"]
    chat = session["chat"]

    print("\n[AGENT] Processing your request...", flush=True)

    try:

        response = chat.send_message(user_prompt)

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        print("[AGENT] Response received!", flush=True)

        return response.text.strip()

    except Exception as error:

        raise RuntimeError(
            f"Agent execution failed: {type(error).__name__}: {error}"
        ) from error


def close_agent_session(session):

    if session and "client" in session:

        session["client"].close()

        print("\n[AGENT] Session closed successfully.")