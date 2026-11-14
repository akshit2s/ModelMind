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

                response = run_agent(
                    agent_session,
                    user_prompt
                )

                print("\n========== MODELMIND ==========")
                print(response)

            except Exception as error:

                print(f"\n[ERROR] {error}")

    finally:

        close_agent_session(agent_session)


if __name__ == "__main__":
    main()