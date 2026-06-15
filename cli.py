from rag_engine import RAGEngine


def run_agent() -> None:
    engine = RAGEngine()
    history: list[dict] = []

    print("\n🚀 RAG Agent Started (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        answer = engine.chat(user_input, history)
        print("\nAssistant:", answer, "\n")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    run_agent()