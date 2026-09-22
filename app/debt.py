"""可借贷连续补充桶试算：允许借入缺口、债务先于余额得到偿还的链上原子扣减。

每个桶在普通令牌桶（capacity / refill_rate / initial）之外携带一个
累计债务上限 debt_limit：

- 时间推进时按 (now - last) · refill_rate 连续补充，**补充量先偿还债务**，
  偿还后剩余的部分才进入可用余额（余额仍封顶 capacity）；
- 判定一笔 cost 时，桶先把可用余额扣到 0，仍有缺口则允许借入，
  借入后累计债务不得超过 debt_limit，即可支付总额为
  balance + (debt_limit - debt)；
- cost > capacity + debt_limit 的请求对该桶永远不可达
  （cost_exceeds_limit）——即使满桶且零债务也无法支付；
- 其余当前无法在借贷上限内满足的情形以 debt_limit_exceeded 拒绝；
- 仅当链上每个桶都能在其借贷上限内满足 cost 时才原子扣减，否则链上
  所有桶保持不变（连续补充与债务偿还作为时间推进照常发生）。

模块同时对外暴露内部回放 _replay_debt：它与 debt_simulate 共享同一套
语义，但在每个事件时刻把**所有**桶（而不仅是链上桶）推进到当前时刻，
并记录每一步全部桶的扣减后 (balance, debt) 快照——/debt-diff 据此按
原始索引逐桶对比两个情景。逐步推进全部桶与惰性推进仅在取整时刻的
采样上等价（min 满足结合律的平移不变性），终态与链上视图完全一致。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.models import (
    EPS,
    BucketBalance,
    DebtBucketBalance,
    DebtBucketFinalState,
    DebtBucketSpec,
    DebtRejection,
    DebtSimResult,
    DebtSimulateRequest,
    DebtSimulateResponse,
    SimRequest,
)
from app.simulation import _round


class _BucketState:
    """桶运行时状态：可用余额 balance、累计债务 debt 与上次更新时刻。"""

    __slots__ = ("balance", "debt", "last_time")

    def __init__(self, balance: float, debt: float, last_time: float):
        self.balance = balance
        self.debt = debt
        self.last_time = last_time


@dataclass
class _StepRecord:
    """一次请求事件的回放记录（供 /debt-simulate 与 /debt-diff 共用）。"""

    index: int
    request: SimRequest
    timestamp: float
    cost: float
    chain: list[str]
    # 链上各桶补充还债之后、扣减之前的 (balance, debt)
    before: dict[str, tuple[float, float]]
    allowed: bool
    rejected_by: str | None
    reason: Literal["cost_exceeds_limit", "debt_limit_exceeded"] | None
    # 本事件处理之后**所有**桶的 (balance, debt)（链外桶只做时间推进）
    after_all: dict[str, tuple[float, float]]
    # 拒绝明细（raw 数值），放行为 None
    rejection_data: dict | None


@dataclass
class _Replay:
    """完整回放结果：逐步记录 + 终态与桶参数表。"""

    bucket_ids: list[str]
    capacity: dict[str, float]
    refill_rate: dict[str, float]
    debt_limit: dict[str, float]
    steps: list[_StepRecord]
    final: dict[str, tuple[float, float]]
    t_end: float


def _replay_debt(
    buckets: list[DebtBucketSpec], requests: list[SimRequest]
) -> _Replay:
    """按可借贷桶规则回放一条请求序列，返回逐步记录与终态。

    与 debt_simulate 的判定/取整语义完全一致，区别仅在于每个事件时刻
    把全部桶推进到 now（等价于惰性推进），以便逐桶对比不同情景。
    """
    capacity = {b.id: float(b.capacity) for b in buckets}
    refill_rate = {b.id: float(b.refill_rate) for b in buckets}
    debt_limit = {b.id: float(b.debt_limit) for b in buckets}
    bucket_ids = [b.id for b in buckets]

    # 与 /simulate 一致：模拟开始时刻 = 最早请求时间戳，所有桶在该时刻
    # 持有初始余额且无债务，时间戳因此可以使用任意一致的时间原点。
    t_start = min(float(request.timestamp) for request in requests)
    t_end = max(float(request.timestamp) for request in requests)

    states: dict[str, _BucketState] = {
        b.id: _BucketState(
            balance=float(b.capacity if b.initial is None else b.initial),
            debt=0.0,
            last_time=t_start,
        )
        for b in buckets
    }

    def advance(bucket_id: str, now: float) -> _BucketState:
        """把某个桶连续补充到时刻 now：补充量先偿还债务，余下进入余额。"""
        state = states[bucket_id]
        elapsed = now - state.last_time
        if elapsed > 0:
            refill = refill_rate[bucket_id] * elapsed
            # 先偿还债务；偿还后仍有剩余才增加可用余额，余额封顶容量
            repaid = min(state.debt, refill)
            state.debt -= repaid
            state.balance = min(
                capacity[bucket_id], state.balance + (refill - repaid)
            )
        state.last_time = now
        return state

    # 内部按时间顺序处理；同一时刻按原始输入顺序处理
    order = sorted(range(len(requests)), key=lambda i: (requests[i].timestamp, i))

    steps: list[_StepRecord] = []

    for i in order:
        request = requests[i]
        now = float(request.timestamp)
        cost = float(request.cost)

        # 先对**所有**桶做连续补充（无论是否在链上、无论最终是否放行，
        # 时间都会推进）；与只推进链上桶的惰性做法得到相同状态。
        for bucket_id in bucket_ids:
            advance(bucket_id, now)
        before: dict[str, tuple[float, float]] = {
            bucket_id: (states[bucket_id].balance, states[bucket_id].debt)
            for bucket_id in request.chain
        }

        # 链上第一个无法在借贷上限内满足成本的桶即拒绝桶：
        # cost > capacity + debt_limit 永远不可达；
        # 否则要求 可用余额 + 剩余借贷额度 >= cost。
        rejected_by: str | None = None
        reason: Literal["cost_exceeds_limit", "debt_limit_exceeded"] | None = None
        for bucket_id in request.chain:
            bal, debt = before[bucket_id]
            if cost > capacity[bucket_id] + debt_limit[bucket_id] + EPS:
                rejected_by, reason = bucket_id, "cost_exceeds_limit"
                break
            purchasable = bal + (debt_limit[bucket_id] - debt)
            if purchasable + EPS < cost:
                rejected_by, reason = bucket_id, "debt_limit_exceeded"
                break

        if rejected_by is None:
            # 全部通过：原子扣减。每桶先用可用余额支付，缺口计入债务。
            for bucket_id in request.chain:
                state = states[bucket_id]
                from_balance = min(state.balance, cost)
                state.balance -= from_balance
                state.debt += cost - from_balance

        rejection_data: dict | None = None
        if rejected_by is not None:
            bal_before, debt_before = before[rejected_by]
            remaining_credit = debt_limit[rejected_by] - debt_before
            available = bal_before + remaining_credit
            rejection_data = {
                "bucket_id": rejected_by,
                "reason": reason,
                "capacity": capacity[rejected_by],
                "debt_limit": debt_limit[rejected_by],
                "cost": cost,
                "balance_available": bal_before,
                "debt_before": debt_before,
                "available": available,
                "remaining_credit": remaining_credit,
                "shortage": cost - available,
            }

        steps.append(
            _StepRecord(
                index=i,
                request=request,
                timestamp=now,
                cost=cost,
                chain=list(request.chain),
                before=before,
                allowed=rejected_by is None,
                rejected_by=rejected_by,
                reason=reason,
                after_all={
                    bucket_id: (states[bucket_id].balance, states[bucket_id].debt)
                    for bucket_id in bucket_ids
                },
                rejection_data=rejection_data,
            )
        )

    # 终态：把所有桶补充到最后时刻（全部桶已逐步推进到最后事件时刻，
    # 此处再幂等推进一次以防 requests 时刻集合为空以外的边界情形）。
    for bucket_id in bucket_ids:
        advance(bucket_id, t_end)
    final = {
        bucket_id: (states[bucket_id].balance, states[bucket_id].debt)
        for bucket_id in bucket_ids
    }

    return _Replay(
        bucket_ids=bucket_ids,
        capacity=capacity,
        refill_rate=refill_rate,
        debt_limit=debt_limit,
        steps=steps,
        final=final,
        t_end=t_end,
    )


def _build_rejection(data: dict) -> DebtRejection:
    """把回放中的拒绝明细（raw 数值）转成输出模型（统一取整）。"""
    return DebtRejection(
        bucket_id=data["bucket_id"],
        reason=data["reason"],
        capacity=_round(data["capacity"]),
        debt_limit=_round(data["debt_limit"]),
        cost=_round(data["cost"]),
        balance_available=_round(data["balance_available"]),
        debt_before=_round(data["debt_before"]),
        available=_round(data["available"]),
        remaining_credit=_round(data["remaining_credit"]),
        shortage=_round(data["shortage"]),
    )


def debt_simulate(payload: DebtSimulateRequest) -> DebtSimulateResponse:
    replay = _replay_debt(payload.buckets, payload.requests)

    results_by_index: dict[int, DebtSimResult] = {}
    for step in replay.steps:
        bucket_views: dict[str, DebtBucketBalance] = {}
        for bucket_id in step.chain:
            bal_before, debt_before = step.before[bucket_id]
            if step.allowed:
                bal_after, debt_after = step.after_all[bucket_id]
            else:
                # 原子性：拒绝时链上所有桶的余额/债务保持不变
                bal_after, debt_after = bal_before, debt_before
            bucket_views[bucket_id] = DebtBucketBalance(
                balance=BucketBalance(
                    before=_round(bal_before), after=_round(bal_after)
                ),
                debt=BucketBalance(
                    before=_round(debt_before), after=_round(debt_after)
                ),
            )

        results_by_index[step.index] = DebtSimResult(
            request_id=step.request.id,
            index=step.index,
            timestamp=_round(step.timestamp),
            cost=_round(step.cost),
            allowed=step.allowed,
            buckets=bucket_views,
            rejected_by=step.rejected_by,
            rejection=(
                _build_rejection(step.rejection_data)
                if step.rejection_data is not None
                else None
            ),
        )

    results = [results_by_index[i] for i in range(len(payload.requests))]
    allowed_count = sum(1 for r in results if r.allowed)

    # 终态快照（所有桶推进到最后一个时间戳后）
    final_buckets: list[DebtBucketFinalState] = []
    for bucket_id in replay.bucket_ids:
        balance, debt = replay.final[bucket_id]
        final_buckets.append(
            DebtBucketFinalState(
                id=bucket_id,
                capacity=_round(replay.capacity[bucket_id]),
                refill_rate=_round(replay.refill_rate[bucket_id]),
                debt_limit=_round(replay.debt_limit[bucket_id]),
                balance=_round(balance),
                debt=_round(debt),
                available=_round(
                    balance + (replay.debt_limit[bucket_id] - debt)
                ),
            )
        )

    return DebtSimulateResponse(
        allowed_count=allowed_count,
        rejected_count=len(results) - allowed_count,
        results=results,
        final_buckets=final_buckets,
    )
