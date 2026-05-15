"""
Chainlit web UI for 选品助手 — E-Commerce Product Selection Agent.
Run: chainlit run app.py
"""
import os
import json
from dotenv import load_dotenv

import chainlit as cl
from openai import OpenAI

from agents.chat import TOOLS, SYSTEM_PROMPT, TOOL_MAP, run_tool

load_dotenv()


@cl.on_chat_start
async def start():
    await cl.Message(
        content="👋 你好！我是**选品助手**，可以帮你：\n\n"
        "📊 **调研品类** — 市场规模、趋势、竞品、选品建议\n"
        "🔍 **发现趋势** — 当前热门品类 Top 5 推荐\n"
        "📈 **投资分析** — 产业链上市公司估值对比\n\n"
        "直接告诉我你想了解什么即可！例如：\n"
        '- "帮我调研AI硬件市场"\n'
        '- "最近有什么热门品类？"\n'
        '- "分析歌尔股份和立讯精密的投资价值"'
    ).send()

    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    cl.user_session.set("client", client)
    cl.user_session.set("messages", [{"role": "system", "content": SYSTEM_PROMPT}])


@cl.on_message
async def on_message(msg: cl.Message):
    client: OpenAI = cl.user_session.get("client")
    messages: list = cl.user_session.get("messages")

    messages.append({"role": "user", "content": msg.content})

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )
    assistant_msg = response.choices[0].message

    if assistant_msg.tool_calls:
        messages.append(assistant_msg.model_dump())

        for tc in assistant_msg.tool_calls:
            func_name = tc.function.name
            func_args = json.loads(tc.function.arguments)
            args_str = ", ".join(f"{k}={v}" for k, v in func_args.items())

            async with cl.Step(name=f"{func_name}", type="tool") as step:
                step.input = args_str or "(no arguments)"
                try:
                    result = run_tool(func_name, func_args)
                    step.output = result[:1000] + ("..." if len(result) > 1000 else "")
                except Exception as e:
                    result = f"Error: {e}"
                    step.output = result
                    step.is_error = True

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        stream = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=True,
        )

        final_msg = cl.Message(content="")
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                await final_msg.stream_token(delta.content)
        messages.append({"role": "assistant", "content": final_msg.content})
        await final_msg.send()

    elif assistant_msg.content:
        messages.append({"role": "assistant", "content": assistant_msg.content})

        stream = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=True,
        )

        final_msg = cl.Message(content="")
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                await final_msg.stream_token(delta.content)
        messages[-1]["content"] = final_msg.content
        await final_msg.send()

    cl.user_session.set("messages", messages)
