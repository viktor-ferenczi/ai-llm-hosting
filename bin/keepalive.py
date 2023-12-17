#!/usr/bin/python3
"""Keeps an LLM engine alive

- Periodically runs a short completion
- If the completion fails with retries, then invokes the restart script

Both the check and the restart have a timeout.

"""
import asyncio
import os
import sys
import traceback
from datetime import datetime
from subprocess import Popen
from time import time

from openai import AsyncOpenAI, APIError

AIDEV_OPENAI_BASE_URL = os.getenv('AIDEV_OPENAI_BASE_URL', 'http://127.0.0.1:8000/v1')
AIDEV_OPENAI_KEY = os.getenv('AIDEV_OPENAI_KEY', 'NO-KEY')

CHECK_TIMOUT = 60
CHECK_PERIOD = 120
RESTART_TIMEOUT = 60


async def check() -> bool:
    client = AsyncOpenAI(
        base_url=AIDEV_OPENAI_BASE_URL,
        api_key=AIDEV_OPENAI_KEY,
        max_retries=3,
    )

    create_completion_task = asyncio.create_task(
        client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant and give concise answers."},
                {"role": "user", "content": '1 + 1 = ?'}
            ],
            model='model',
            max_tokens=10,
            temperature=0.2,
        )
    )

    done, pending = await asyncio.wait([create_completion_task], timeout=CHECK_TIMOUT, return_when=asyncio.ALL_COMPLETED)

    if not done:
        return False

    try:
        completion = done.pop().result()
    except APIError as e:
        print(f'Check failed: [{e.__class__.__name__}] {e}')
        return False

    return bool(completion.choices) and bool(completion.choices[0].message.content)


def restart(script_path: str):
    print(f'{timestamp()}: Restarting')
    if not Popen([script_path], shell=True).wait(RESTART_TIMEOUT):
        print(f'{timestamp()}: Failed to restart')


def timestamp() -> str:
    return datetime.now().isoformat()


async def main():
    if len(sys.argv) < 2:
        print(r"Usage: {sys.argv[0]} [path-to-restart-script]")
        sys.exit(1)

    script_path = sys.argv[1]
    assert os.path.isfile(script_path)

    print(f'{timestamp()}: Keepalive loop started')

    next_check = 0.0
    while 1:

        if time() < next_check:
            await asyncio.sleep(1)
            continue

        next_check = time() + CHECK_PERIOD

        try:
            if not await check():
                restart(script_path)
        except KeyboardInterrupt:
            break
        except Exception:
            print(f'{timestamp()}: Unexpected error:')
            traceback.print_exc()
            print()

    print(f'{timestamp()}: Terminated')


if __name__ == '__main__':
    asyncio.run(main())
