"""
Risk Analyzer

SimulationTrace의 risk_score와 security_events를 검토한다.
현재는 백엔드가 이미 risk_score를 계산하므로 pass-through.
향후 여러 백엔드 결과를 종합하는 레이어로 확장 가능.
"""
from __future__ import annotations

from simguard.models import SimulationTrace


class RiskAnalyzer:
    async def analyze(self, trace: SimulationTrace) -> SimulationTrace:
        return trace
