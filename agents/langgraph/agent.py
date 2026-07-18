"""
LangGraph ReAct 에이전트

ChatOpenAI (GPT-5-mini via OpenRouter) + DuckDuckGoSearchRun 기반.
langgraph.prebuilt.create_react_agent 로 ReAct 루프 구성.
"""
from __future__ import annotations

import os
import warnings
import httpx

from dotenv import load_dotenv

load_dotenv()


def create_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model="openrouter/openai/gpt-5-mini",
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
        http_client=httpx.Client(verify=False),
        temperature=0,
    )


def create_tools():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from langchain_community.tools import DuckDuckGoSearchRun
    return [DuckDuckGoSearchRun()]


def create_agent(llm=None, tools=None):
    """LangGraph ReAct 에이전트 그래프 반환."""
    from langgraph.prebuilt import create_react_agent
    llm   = llm   or create_llm()
    tools = tools or create_tools()
    return create_react_agent(llm, tools), llm, tools


def run_agent(graph, task: str) -> str:
    """에이전트 실행 → 최종 응답 텍스트 반환."""
    from langchain_core.messages import AIMessage
    result = graph.invoke({"messages": [("user", task)]})
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)
    return ""
