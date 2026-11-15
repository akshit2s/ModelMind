import json
from pathlib import Path

from core.dataset_loader import load_dataset

from agent.controller import (
    create_agent_session,
    run_agent,
    close_agent_session
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = "datasets/students.csv"


def display_banner():
    print("\n" + "=" * 48)
    print("                    MODELMIND")
    print("       YOUR INTELLIGENT MACHINE LEARNING AGENT")
    print("=" * 48)


def resolve_dataset_path(user_path):
    """Resolve relative paths from the project root."""
    path = Path(user_path).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def load_dataset_interactively():
    """Ask the user which CSV dataset to load."""

    print("\n========== DATASET SELECTION ==========")
    print("Enter a CSV dataset path.")
    print(f"Press Enter to use the default: {DEFAULT_DATASET}")
    print("Type 'exit' to close ModelMind.")

    while True:
        user_path = input("\nDataset path: ").strip()

        if user_path.lower() in ("exit", "quit"):
            return None, None

        if not user_path:
            user_path = DEFAULT_DATASET

        # Handle paths copied from Windows Explorer.
        user_path = user_path.strip('"').strip("'")

        dataset_path = resolve_dataset_path(user_path)

        if not dataset_path.is_file():
            print(f"\n[ERROR] File not found: {dataset_path}")
            print("Please enter a valid CSV path.")
            continue

        try:
            df = load_dataset(str(dataset_path))

            print("\n[+] Dataset loaded successfully!")
            print(f"[+] File: {dataset_path.name}")

            if hasattr(df, "shape"):
                print(f"[+] Rows: {df.shape[0]}")
                print(f"[+] Columns: {df.shape[1]}")

            return df, dataset_path

        except Exception as error:
            print(f"\n[ERROR] Dataset loading failed: {error}")


def create_session(df):
    """Create a persistent Gemini agent session."""

    try:
        session = create_agent_session(df)
        print("[+] Agent session created successfully!")
        return session

    except Exception as error:
        print(f"\n[ERROR] Agent initialization failed: {error}")
        return None


def reset_conversation(agent_session, df):
    """Start a fresh conversation while retaining the loaded dataset."""

    new_session = create_session(df)

    if new_session is None:
        print("[ERROR] Could not reset the conversation.")
        return agent_session

    close_agent_session(agent_session)

    print("[+] Conversation history cleared.")
    return new_session


def handle_optimization_approval(agent_session):
    """Handle Python-side approval and execute optimization."""

    tool_registry = agent_session["tool_registry"]
    pending = tool_registry["_state"]["pending_optimization"]

    if pending is None:
        return

    print("\n========== OPTIMIZATION APPROVAL ==========")
    print(f"Model awaiting approval: {pending['model_name']}")

    while True:
        approval = input(
            "\nDo you approve this optimization? (yes/no): "
        ).strip().lower()

        if approval in ("yes", "y"):
            approved = True
            break

        if approval in ("no", "n"):
            approved = False
            break

        print("Please enter yes or no.")

    # Approval is enforced by Python, not Gemini.
    result = tool_registry["execute_approved_optimization"](
        approved=approved
    )

    print("\n========== OPTIMIZATION EXECUTION ==========")
    print(json.dumps(result, indent=4, default=str))

    # Ask Gemini to explain only verified Python results.
    if result.get("status") == "SUCCESS":

        explanation_prompt = (
            "The Python optimization engine returned the following "
            "verified results. Explain them accurately to the user. "
            "Do not invent metrics or claim improvements that are "
            "not present in the results.\n\n"
            + json.dumps(result, indent=4, default=str)
        )

        explanation = run_agent(
            agent_session,
            explanation_prompt
        )

        print("\n========== MODELMIND EXPLANATION ==========")
        print(explanation)


def run_application(agent_session, df):
    """Run the interactive ModelMind CLI."""

    print("\n========== MODELMIND READY ==========")
    print("Ask questions about your dataset or request ML analysis.")
    print("\nCommands:")
    print("  clear  - Reset conversation history")
    print("  exit   - Close ModelMind")

    while True:

        user_prompt = input("\nYou: ").strip()

        if user_prompt.lower() in ("exit", "quit"):
            print("\nModelMind: Goodbye!")
            break

        if not user_prompt:
            print("ModelMind: Please enter a question.")
            continue

        if user_prompt.lower() == "clear":
            agent_session = reset_conversation(agent_session, df)
            continue

        try:
            print("\n[AGENT] Processing your request...")

            response = run_agent(
                agent_session,
                user_prompt
            )

            print("\n========== MODELMIND ==========")
            print(response)

            handle_optimization_approval(agent_session)

        except Exception as error:
            print(f"\n[ERROR] Agent execution failed: {error}")
            print("You can try asking your question again.")

    return agent_session


def main():

    display_banner()

    agent_session = None

    try:
        df, dataset_path = load_dataset_interactively()

        if df is None:
            print("\nModelMind: Exiting.")
            return

        agent_session = create_session(df)

        if agent_session is None:
            return

        agent_session = run_application(agent_session, df)

    except KeyboardInterrupt:
        print("\n\nModelMind: Interrupted by user.")

    finally:
        if agent_session is not None:
            try:
                close_agent_session(agent_session)
                print("[+] Agent session closed.")
            except Exception:
                pass


if __name__ == "__main__":
    main()