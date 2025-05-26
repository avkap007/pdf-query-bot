from utils import PSAQABot

bot = PSAQABot()

while True:
    query = input("Ask a question: ")
    if query.lower() in {"exit", "quit"}:
        break

    result = bot.ask(query)
    print(f"\n🧠 Answer: {result['answer']}")
    print(f"\n📄 Sources: {result.get('sources', 'N/A')}")
