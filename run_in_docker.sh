#!/usr/bin/env bash
# SimGuardAgent를 MicroVM(Docker 컨테이너) 안에서 실행
# 사용법: ./run_in_docker.sh --benchmark GA_Jailbreak --mode simguard --limit 50

set -e

IMAGE="simguard-agent:latest"

# 이미지 없으면 빌드
if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    echo "[build] $IMAGE 빌드 중..."
    docker build -t "$IMAGE" .
fi

echo "[run] SimGuardAgent in Docker: $@"
docker run --rm \
    --env-file .env \
    -v "$(pwd)/results:/app/results" \
    -v /var/run/docker.sock:/var/run/docker.sock \
    "$IMAGE" "$@"
