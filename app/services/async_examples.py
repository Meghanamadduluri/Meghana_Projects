import asyncio

async def fake_embedding(text: str) -> list:
    await asyncio.sleep(1)  # simulate API delay
    return [0.1, 0.2, 0.3]

async def fake_db_search(vector: list) -> list:
    await asyncio.sleep(1)  # simulate DB delay
    return ["chunk1", "chunk2"]