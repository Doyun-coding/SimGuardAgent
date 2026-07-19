FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 기본 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# SmolAgents 설치 (PyPI)
RUN pip install --no-cache-dir \
    smolagents \
    httpx \
    tavily-python \
    huggingface-hub

# SimGuardAgent 소스 복사
COPY simguard/           /app/simguard/
COPY agents/             /app/agents/
COPY attacks-bench/      /app/attacks-bench/
COPY run_benchmark.py    /app/
COPY eval_llamaguard.py  /app/
COPY main.py             /app/

# 결과 디렉토리 (호스트 볼륨으로 마운트)
RUN mkdir -p /app/results

# Docker-in-Docker: 내부 CodeSimulator가 컨테이너 실행할 수 있도록
# (docker.sock 마운트 방식 - docker-compose에서 설정)
RUN pip install --no-cache-dir docker

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "agents.smolagents.smolagents_adapter"]
