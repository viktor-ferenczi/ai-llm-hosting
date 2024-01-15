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
from vllm_client import AsyncVllmClient, SamplingParams

AIDEV_OPENAI_BASE_URL = os.getenv('AIDEV_OPENAI_BASE_URL', 'http://127.0.0.1:8000/v1')
AIDEV_OPENAI_KEY = os.getenv('AIDEV_OPENAI_KEY', 'NO-KEY')

AIDEV_VLLM_BASE_URL = os.getenv('AIDEV_VLLM_BASE_URL', 'http://127.0.0.1:8000')

CHECK_TIMOUT = 60
CHECK_PERIOD = 120
RESTART_TIMEOUT = 60


async def check_openai() -> bool:
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


async def check_vllm() -> bool:
    client = AsyncVllmClient(AIDEV_VLLM_BASE_URL)
    create_completion_task = asyncio.create_task(
        client.generate(
            'You are a helpful AI assistant and give concise answers. 1 + 1 = ',
            SamplingParams(max_tokens=10, temperature=0.1)
        )
    )
    done, pending = await asyncio.wait([create_completion_task], timeout=CHECK_TIMOUT, return_when=asyncio.ALL_COMPLETED)

    if not done:
        return False

    try:
        completions = done.pop().result()
    except OSError as e:
        print(f'Check failed: [{e.__class__.__name__}] {e}')
        return False

    return bool(completions) and bool(completions[0])


def restart(script_path: str):
    print(f'{timestamp()}: Restarting')
    if Popen([script_path], shell=True).wait(RESTART_TIMEOUT):
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
            if not await check_vllm() and not await check_openai():
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
