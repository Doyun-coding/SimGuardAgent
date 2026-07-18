"""
Decision Controller

risk_score -> ALLOW / BLOCK

  BLOCK : risk >= RISK_THRESHOLD_BLOCK  (즉시 차단)
  ALLOW : risk <  RISK_THRESHOLD_BLOCK  (정상 실행)
"""
from __future__ import annotations

from simguard.config import RISK_THRESHOLD_BLOCK
from simguard.models import Decision


class DecisionController:
    def decide(self, risk_score: float) -> Decision:
        if risk_score >= RISK_THRESHOLD_BLOCK:
            return Decision.BLOCK
        return Decision.ALLOW
