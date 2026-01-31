import asyncio
from langgraph_sdk import get_client

client = get_client(url="http://127.0.0.1:2024")

async def main():
    assistants = await client.assistants.search()
    for a in assistants:
        print(a["graph_id"], a["assistant_id"])

asyncio.run(main())