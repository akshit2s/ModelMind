from google.genai import types

from agent.llm_client import (
    create_gemini_client,
    MODEL_NAME
)

from agent.prompts import MODELMIND_SYSTEM_INSTRUCTION
from agent.tools import create_tool_registry


def create_agent_session(df, max_tool_calls=5):

    # ==================================
    # CREATE TOOL REGISTRY
    # ==================================

    tool_registry = create_tool_registry(df)

    # ==================================
    # TOOL 1: DATASET INSPECTION
    # ==================================

    def inspect_dataset() -> dict:
        """Inspect the loaded dataset."""

        print("\n[TOOL] Inspecting dataset...", flush=True)

        return tool_registry["inspect_dataset"]()

    # ==================================
    # TOOL 2: COLUMN INSPECTION
    # ==================================

    def inspect_column(column_name: str) -> dict:
        """Inspect a specific dataset column."""

        print(
            f"\n[TOOL] Inspecting column: {column_name}",
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
        """Train, evaluate, rank and diagnose candidate models."""

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
        """Prepare optimization recommendations without executing them."""

        print(
            f"\n[TOOL] Preparing optimization for {model_name}...",
            flush=True
        )

        return tool_registry["prepare_optimization"](
            model_name=model_name
        )

    # ==================================
    # TOOL 5: EXPERIMENT HISTORY
    # ==================================

    def get_experiment_history(limit: int = 5) -> dict:
        """Retrieve recent saved ML experiments."""

        print("\n[TOOL] Retrieving experiment history...", flush=True)

        return tool_registry["get_experiment_history"](
            limit=limit
        )

    # ==================================
    # TOOL 6: EXPERIMENT COMPARISON
    # ==================================

    def compare_current_dataset_runs(
        problem_type: str,
        target_column: str,
        primary_metric: str = None
    ) -> dict:
        """Compare previous runs on the current dataset."""

        print("\n[TOOL] Comparing previous experiments...", flush=True)

        return tool_registry["compare_current_dataset_runs"](
            problem_type=problem_type,
            target_column=target_column,
            primary_metric=primary_metric
        )

    # ==================================
    # TOOL 7: OPTIMIZATION HISTORY
    # ==================================

    def get_saved_optimization_history(
        run_id: str = None
    ) -> dict:
        """Retrieve saved optimization attempts."""

        print("\n[TOOL] Retrieving optimization history...", flush=True)

        return tool_registry["get_saved_optimization_history"](
            run_id=run_id
        )

    # ==================================
    # SYSTEM INSTRUCTIONS
    # ==================================

    system_instruction = (
        MODELMIND_SYSTEM_INSTRUCTION
        + """

        CONVERSATION RULES:

        1. Remember previous messages in the current session.
        2. Understand references to previous results.
        3. Never invent dataset statistics or model metrics.
        4. Never expose actual dataset records.
        5. Report actual results returned by Python.

        ML ANALYSIS RULES:

        6. Use run_model_analysis when the user explicitly
           requests model training, evaluation or comparison.

        7. Determine classification or regression from
           the actual prediction objective.

        8. Ensure the target column and problem type
           are clear before running analysis.

        9. Explain limitations caused by insufficient data.

        OPTIMIZATION RULES:

        10. Use prepare_optimization when the user requests
            optimization recommendations.

        11. Use the latest actual ML analysis results.

        12. Never claim optimization has been executed
            when only a recommendation was generated.

        13. Never approve optimization on behalf of the user.

        EXPERIMENT TRACKING RULES:

        14. Use get_experiment_history to retrieve previous runs.

        15. Use compare_current_dataset_runs when the user
            requests comparison of previous experiments.

        16. Compare only matching dataset fingerprints,
            prediction targets and problem types.

        17. Use get_saved_optimization_history to retrieve
            previous optimization attempts.

        18. Report historical metrics exactly as retrieved.

        SECURITY RULES:

        19. Never execute arbitrary generated Python code.

        20. Only use registered Python tools.

        21. Never access API credentials or secrets.

        """
    )

    # ==================================
    # REGISTER AVAILABLE TOOLS
    # ==================================

    available_tools = [
        inspect_dataset,
        inspect_column,
        run_model_analysis,
        prepare_optimization,
        get_experiment_history,
        compare_current_dataset_runs,
        get_saved_optimization_history
    ]

    # ==================================
    # CONFIGURE GEMINI
    # ==================================

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
    # CREATE CLIENT AND CHAT
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


# ==================================
# RUN AGENT
# ==================================

def run_agent(session, user_prompt):

    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("Prompt cannot be empty.")

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


# ==================================
# CLOSE AGENT SESSION
# ==================================

def close_agent_session(session):

    if session and "client" in session:

        session["client"].close()

        print("\n[AGENT] Session closed successfully.")