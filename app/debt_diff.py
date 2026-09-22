"""可借贷桶情景差异试算：对 baseline / candidate 两组同构情景分别回放并对比。

两个情景的请求序列必须**同构**——请求数量相同，且按原始索引逐对的
请求 ID 与时间戳一致（由 DebtDiffRequest 的跨字段校验保证，非法时
返回 422、不进入回放、不写入任何状态）；桶集合（ID 与顺序）也必须
一致，但桶参数（capacity / refill_rate / initial / debt_limit）与
每个请求的 cost / chain 可以不同——参数差异正是候选情景要比较的内容。

对比内容：
- 按原始索引对齐，找首个 allowed / rejected_by / 拒绝原因分歧；
- 每一步给出两侧全部桶处理后的余额 / 债务，及 candidate - baseline
  的逐桶差值（链外桶因时间推进也会出现在快照中）；
- 终态给出两侧每桶快照（余额 / 债务 / 可立即支付总额）与差值。
"""
from __future__ import annotations

from app.debt import _Replay, _replay_debt
from app.models import (
    DebtBucketFinalState,
    DebtDiffBucketAfter,
    DebtDiffBucketDelta,
    DebtDiffFinalBucket,
    DebtDiffFirstDivergence,
    DebtDiffRequest,
    DebtDiffResponse,
    DebtDiffSideStep,
    DebtDiffStep,
    DebtDiffVerdict,
)
from app.simulation import _round


def _side_step(replay: _Replay, step) -> DebtDiffSideStep:
    """构造单边情景在某次请求后的结论与全部桶快照。"""
    verdict = DebtDiffVerdict(
        allowed=step.allowed,
        rejected_by=step.rejected_by,
        reason=step.reason,
    )
    buckets: dict[str, DebtDiffBucketAfter] = {}
    for bucket_id in replay.bucket_ids:
        balance, debt = step.after_all[bucket_id]
        debt_limit = replay.debt_limit[bucket_id]
        buckets[bucket_id] = DebtDiffBucketAfter(
            balance=_round(balance),
            debt=_round(debt),
            available=_round(balance + (debt_limit - debt)),
        )
    return DebtDiffSideStep(
        verdict=verdict,
        rejected_by=step.rejected_by,
        reason=step.reason,
        buckets=buckets,
    )


def _bucket_delta(
    base_balance: float,
    base_debt: float,
    cand_balance: float,
    cand_debt: float,
    replay_base: _Replay,
    replay_cand: _Replay,
    bucket_id: str,
) -> DebtDiffBucketDelta:
    """计算单个桶的 candidate - baseline 差值（统一取整后相减）。"""
    base_available = base_balance + (
        replay_base.debt_limit[bucket_id] - base_debt
    )
    cand_available = cand_balance + (
        replay_cand.debt_limit[bucket_id] - cand_debt
    )
    return DebtDiffBucketDelta(
        balance_delta=_round(_round(cand_balance) - _round(base_balance)),
        debt_delta=_round(_round(cand_debt) - _round(base_debt)),
        available_delta=_round(
            _round(cand_available) - _round(base_available)
        ),
    )


def _final_state_view(
    replay: _Replay, bucket_id: str
) -> DebtBucketFinalState:
    balance, debt = replay.final[bucket_id]
    return DebtBucketFinalState(
        id=bucket_id,
        capacity=_round(replay.capacity[bucket_id]),
        refill_rate=_round(replay.refill_rate[bucket_id]),
        debt_limit=_round(replay.debt_limit[bucket_id]),
        balance=_round(balance),
        debt=_round(debt),
        available=_round(balance + (replay.debt_limit[bucket_id] - debt)),
    )


def debt_diff(payload: DebtDiffRequest) -> DebtDiffResponse:
    # 两个情景完全独立回放，互不影响；本接口为纯计算，不保留任何状态。
    replay_base = _replay_debt(
        payload.baseline.buckets, payload.baseline.requests
    )
    replay_cand = _replay_debt(
        payload.candidate.buckets, payload.candidate.requests
    )

    steps_base = {step.index: step for step in replay_base.steps}
    steps_cand = {step.index: step for step in replay_cand.steps}

    results: list[DebtDiffStep] = []
    first_divergence: DebtDiffFirstDivergence | None = None
    divergence_count = 0

    # 按原始请求索引对齐（输入校验已保证两侧数量、ID、时间戳一致）
    for i in range(len(payload.baseline.requests)):
        step_b = steps_base[i]
        step_c = steps_cand[i]

        side_b = _side_step(replay_base, step_b)
        side_c = _side_step(replay_cand, step_c)

        diverged = (
            step_b.allowed != step_c.allowed
            or step_b.rejected_by != step_c.rejected_by
            or step_b.reason != step_c.reason
        )
        if diverged:
            divergence_count += 1
            if first_divergence is None:
                first_divergence = DebtDiffFirstDivergence(
                    request_id=step_b.request.id,
                    index=i,
                    timestamp=_round(step_b.timestamp),
                    baseline=side_b.verdict,
                    candidate=side_c.verdict,
                )

        bucket_deltas: dict[str, DebtDiffBucketDelta] = {}
        for bucket_id in replay_base.bucket_ids:
            bal_b, debt_b = step_b.after_all[bucket_id]
            bal_c, debt_c = step_c.after_all[bucket_id]
            bucket_deltas[bucket_id] = _bucket_delta(
                bal_b, debt_b, bal_c, debt_c,
                replay_base, replay_cand, bucket_id,
            )

        results.append(
            DebtDiffStep(
                request_id=step_b.request.id,
                index=i,
                timestamp=_round(step_b.timestamp),
                cost_baseline=_round(step_b.cost),
                cost_candidate=_round(step_c.cost),
                diverged=diverged,
                baseline=side_b,
                candidate=side_c,
                bucket_deltas=bucket_deltas,
            )
        )

    # 终态快照与差值（桶集合同构，按基线桶顺序即可）
    final_buckets: list[DebtDiffFinalBucket] = []
    for bucket_id in replay_base.bucket_ids:
        base_final = _final_state_view(replay_base, bucket_id)
        cand_final = _final_state_view(replay_cand, bucket_id)
        bal_b, debt_b = replay_base.final[bucket_id]
        bal_c, debt_c = replay_cand.final[bucket_id]
        final_buckets.append(
            DebtDiffFinalBucket(
                id=bucket_id,
                baseline=base_final,
                candidate=cand_final,
                delta=_bucket_delta(
                    bal_b, debt_b, bal_c, debt_c,
                    replay_base, replay_cand, bucket_id,
                ),
            )
        )

    return DebtDiffResponse(
        request_count=len(results),
        divergence_count=divergence_count,
        first_divergence=first_divergence,
        results=results,
        final_buckets=final_buckets,
    )
