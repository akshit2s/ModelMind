from google.genai import types

from agent.llm_client import (
    create_gemini_client,
    MODEL_NAME
)

from agent.prompts import MODELMIND_SYSTEM_INSTRUCTION
from agent.tools import create_tool_registry


def run_agent(df, user_prompt, max_tool_calls=3):

    # ----------------------------------
    # 1. VALIDATE INPUTS
    # ----------------------------------

    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("User prompt cannot be empty.")

    if not isinstance(max_tool_calls, int) or max_tool_calls < 1:
        raise ValueError("Maximum tool calls must be at least 1.")

    # ----------------------------------
    # 2. CREATE TOOL REGISTRY
    # ----------------------------------

    tool_registry = create_tool_registry(df)

    # ----------------------------------
    # 3. DEFINE PERMITTED PYTHON TOOLS
    # ----------------------------------

    def inspect_dataset() -> dict:
        """
        Inspect the loaded dataset.

        Returns:
            Dataset dimensions, column names, data types,
            missing-value counts, unique-value counts
            and duplicate-row count.

        Does not return actual dataset records.
        """

        print(
            "[TOOL] Executing inspect_dataset...",
            flush=True
        )

        return tool_registry["inspect_dataset"]()

    def inspect_column(column_name: str) -> dict:
        """
        Inspect a specific dataset column.

        Args:
            column_name: Name of the column to inspect.

        Returns:
            Column data type, category, missing-value count,
            unique-value count and total records.

        Does not return actual column values.
        """

        print(
            f"[TOOL] Executing inspect_column: {column_name}",
            flush=True
        )

        return tool_registry["inspect_column"](
            column_name=column_name
        )

    # Only these two functions are exposed to Gemini.
    available_tools = [
        inspect_dataset,
        inspect_column
    ]

    # ----------------------------------
    # 4. CREATE GEMINI CLIENT
    # ----------------------------------

    client = create_gemini_client()

    # ----------------------------------
    # 5. EXTEND SYSTEM INSTRUCTIONS
    # ----------------------------------

    system_instruction = (
        MODELMIND_SYSTEM_INSTRUCTION
        + """

        TOOL USAGE RULES:

        1. Use inspect_dataset when the user asks
           about the overall dataset.

        2. Use inspect_column when the user asks
           about a specific column.

        3. Base dataset-related factual statements
           only on actual tool results.

        4. Never invent dataset statistics.

        5. Never expose actual dataset records.

        6. Never claim to have trained or modified
           a model unless the corresponding Python
           operation was actually executed.

        7. If a requested operation is unavailable,
           explain this limitation honestly.

        8. The available tools are read-only.

        """
    )

    # ----------------------------------
    # 6. CONFIGURE THE AGENT
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

    # ----------------------------------
    # 7. SEND REQUEST TO GEMINI
    # ----------------------------------

    print(
        "\n[AGENT] Sending request to Gemini...",
        flush=True
    )

    try:

        response = client.models.generate_content(

            model=MODEL_NAME,

            contents=user_prompt,

            config=config

        )

        # ----------------------------------
        # 8. RETURN FINAL RESPONSE
        # ----------------------------------

        if not response.text:

            raise RuntimeError(
                "Gemini returned an empty response."
            )

        print(
            "[AGENT] Response received successfully!",
            flush=True
        )

        return response.text.strip()

    except Exception as error:

        raise RuntimeError(
            f"Agent execution failed: {type(error).__name__}: {error}"
        ) from error