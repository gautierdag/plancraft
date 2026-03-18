"""
List available models via the GitHub Copilot SDK.

Usage:
    python scripts/list_copilot_models.py
"""

import asyncio
from copilot import CopilotClient


async def main():
    client = CopilotClient()
    await client.start()

    try:
        print("Available Copilot Models:")
        models = await client.list_models()
        for m in models:
            print(f"- {m.id}")
    except Exception as e:
        print(f"Error listing models: {e}")
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
