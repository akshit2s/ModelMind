from agent.llm_client import generate_response
from agent.prompts import MODELMIND_SYSTEM_INSTRUCTION


def main():

    user_prompt = """
    I want to predict whether students will pass or fail
    based on their study hours, attendance and previous marks.

    Explain what type of ML problem this is and
    which evaluation metric you would prioritise.
    """

    try:

        response = generate_response(
            user_prompt=user_prompt,
            system_instruction=MODELMIND_SYSTEM_INSTRUCTION
        )

        print("\n========== MODELMIND AI RESPONSE ==========")
        print(response)

    except Exception as error:
        print(f"\n[ERROR] {error}")


if __name__ == "__main__":
    main()