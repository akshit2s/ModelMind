from core.dataset_loader import load_dataset
from agent.controller import run_agent


def main():

    print("\n======================================")
    print("              MODELMIND")
    print("       YOUR MACHINE LEARNING AGENT")
    print("======================================")

    # Load dataset once when the program starts
    try:
        df = load_dataset("datasets/students.csv")
        print("\n[+] Dataset loaded successfully!")

    except Exception as error:
        print(f"\n[ERROR] Could not load dataset: {error}")
        return

    print("\nAsk ModelMind anything about your dataset.")
    print("Type 'exit' or 'quit' to close the program.")

    # Interactive conversation loop
    while True:

        user_prompt = input("\nYou: ").strip()

        # Exit condition
        if user_prompt.lower() in ["exit", "quit"]:
            print("\nModelMind: Goodbye! See you next time.")
            break

        # Handle empty input
        if not user_prompt:
            print("ModelMind: Please enter a question.")
            continue

        try:

            response = run_agent(
                df=df,
                user_prompt=user_prompt
            )

            print("\n========== MODELMIND ==========")
            print(response)

        except Exception as error:

            print(f"\n[ERROR] {error}")
            print("Please try again.")


if __name__ == "__main__":
    main()