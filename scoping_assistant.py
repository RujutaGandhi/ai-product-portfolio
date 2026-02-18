def generate_scope(client_input):
    prompt = f"""
    You are an AI scoping assistant.
    Extract:
    - Business objective
    - Required AI capabilities
    - Data sources
    - Risks
    - Estimated complexity

    Client description:
    {client_input}
    """

    return prompt

if __name__ == "__main__":
    example = "We want to build a churn prediction model for subscription customers."
    print(generate_scope(example))
