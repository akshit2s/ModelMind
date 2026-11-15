from core.dataset_loader import load_dataset

from agent.controller import (
    create_agent_session,
    run_agent,
    close_agent_session
)


def main():

    print("\n======================================")
    print("              MODELMIND")
    print("       YOUR MACHINE LEARNING AGENT")
    print("======================================")

    try:
        df = load_dataset("datasets/students.csv")

        print("\n[+] Dataset loaded successfully!")

    except Exception as error:
        print(f"\n[ERROR] Dataset loading failed: {error}")
        return

    # Create one persistent agent session
    try:
        agent_session = create_agent_session(df)

        print("[+] Agent session created successfully!")

    except Exception as error:
        print(f"\n[ERROR] Agent initialization failed: {error}")
        return

    print("\nAsk ModelMind anything about your dataset.")
    print("Type 'exit' to close.")
    print("Type 'clear' to reset conversation history.")

    try:

        while True:

            user_prompt = input("\nYou: ").strip()

            if user_prompt.lower() in ["exit", "quit"]:
                print("\nModelMind: Goodbye!")
                break

            if not user_prompt:
                print("ModelMind: Please enter a question.")
                continue

            if user_prompt.lower() == "clear":

                close_agent_session(agent_session)

                agent_session = create_agent_session(df)

                print("[+] Conversation history cleared.")
                continue

            try:

                import json

                response = run_agent(
                    agent_session,
                    user_prompt
                )

                print("\n========== MODELMIND ==========")
                print(response)


                # ==================================
                # CHECK FOR PENDING OPTIMIZATION
                # ==================================

                tool_registry = agent_session["tool_registry"]

                pending = tool_registry["_state"]["pending_optimization"]

                if pending is not None:

                    print("\n========== OPTIMIZATION APPROVAL ==========")

                    print(
                        f"Model awaiting approval: {pending['model_name']}"
                    )

                    approval = input(
                        "\nDo you approve this optimization? (yes/no): "
                    ).strip().lower()

                    approved = approval in ["yes", "y"]

                    # Execute the internal Python approval function
                    result = tool_registry["execute_approved_optimization"](
                        approved=approved
                    )

                    print("\n========== OPTIMIZATION EXECUTION ==========")

                    print(json.dumps(result, indent=4, default=str))

                    # Ask Gemini to explain the actual execution result
                    if result["status"] == "SUCCESS":

                        explanation_prompt = (
                            "The Python optimization engine returned the following "
                            "verified results. Explain them to the user accurately. "
                            "Do not invent additional metrics or claim improvements "
                            "that are not present in the results.\n\n"
                            + json.dumps(result, indent=4, default=str)
                        )

                        explanation = run_agent(
                            agent_session,
                            explanation_prompt
                        )

                        print("\n========== MODELMIND EXPLANATION ==========")
                        print(explanation)

            except Exception as error:

                print(f"\n[ERROR] {error}")

    finally:

        close_agent_session(agent_session)


if __name__ == "__main__":
    main()