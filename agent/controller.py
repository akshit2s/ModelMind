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

    # ==================================
    # TOOL 1: DATASET INSPECTION
    # ==================================

    def inspect_dataset() -> dict:
        """Inspect the loaded dataset."""

        print("\n[TOOL] Executing inspect_dataset...", flush=True)

        return tool_registry["inspect_dataset"]()

    # ==================================
    # TOOL 2: COLUMN INSPECTION
    # ==================================

    def inspect_column(column_name: str) -> dict:
        """Inspect a specific dataset column."""

        print(
            f"\n[TOOL] Executing inspect_column: {column_name}",
            flush=True
        )

        return tool_registry["inspect_column"](
            column_name=column_name
        )

    # ==================================
    # TOOL 3: COMPLETE ML ANALYSIS
    # ==================================

    def run_model_analysis(
        target_column: str,
        problem_type: str
    ) -> dict:
        """
        Run model selection, training, evaluation,
        ranking and diagnostics.
        """

        print("\n[TOOL] Starting ML analysis...", flush=True)

        result = tool_registry["run_model_analysis"](
            target_column=target_column,
            problem_type=problem_type
        )

        print("[TOOL] ML analysis completed.", flush=True)

        return result

    # ==================================
    # TOOL 4: OPTIMIZATION PLAN
    # ==================================

    def prepare_optimization(model_name: str) -> dict:
        """
        Generate optimization recommendations.
        Does not modify or retrain any model.
        """

        print(
            f"\n[TOOL] Preparing optimization for {model_name}...",
            flush=True
        )

        return tool_registry["prepare_optimization"](
            model_name=model_name
        )

    # ==================================
    # SYSTEM INSTRUCTIONS
    # ==================================

    system_instruction = (
        MODELMIND_SYSTEM_INSTRUCTION
        + """

        CONVERSATION RULES:

        1. Remember previous messages.
        2. Understand references to previous results.
        3. Never invent dataset statistics or model metrics.
        4. Never expose actual dataset records.
        5. Report actual results returned by Python.

        ML ANALYSIS RULES:

        6. Use run_model_analysis when the user
           explicitly requests model training or evaluation.

        7. Determine classification or regression
           from the actual prediction objective.

        8. Ensure the target column and problem type
           are clear before running analysis.

        9. Explain when dataset size makes evaluation
           unreliable.

        OPTIMIZATION RULES:

        10. Use prepare_optimization when the user
            requests optimization recommendations.

        11. Use results from the latest ML analysis.

        12. If the user asks to optimize the best model,
            select the highest-ranked model from actual results.

        13. Explain optimization recommendations.

        14. Always ask for explicit user approval
            before applying changes.

        15. prepare_optimization only generates a plan.
            It does not modify or retrain models.

        16. If optimization is deferred because of
            insufficient data, explain why.

        17. Never claim optimization was applied
            unless the Python operation actually occurred.

        SECURITY RULES:

        18. Never execute arbitrary generated Python code.

        19. Only use registered Python tools.

        20. Never access secrets or API credentials.

        """
    )

    # ==================================
    # GEMINI TOOL CONFIGURATION
    # ==================================

    available_tools = [
        inspect_dataset,
        inspect_column,
        run_model_analysis,
        prepare_optimization
    ]

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

    # ==================================
    # CREATE CLIENT AND CHAT SESSION
    # ==================================

    client = create_gemini_client()

    chat = client.chats.create(
        model=MODEL_NAME,
        config=config
    )

    return {
    "client": client,
    "chat": chat,
    "tool_registry": tool_registry
}


def run_agent(session, user_prompt):

    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    # Keep client alive throughout the session
    client = session["client"]

    chat = session["chat"]

    print("\n[AGENT] Processing your request...", flush=True)

    try:

        response = chat.send_message(user_prompt)

        if not response.text:

            raise RuntimeError(
                "Gemini returned an empty response."
            )

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