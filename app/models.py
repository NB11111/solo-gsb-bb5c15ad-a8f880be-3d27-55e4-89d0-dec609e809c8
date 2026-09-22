"""Pydantic models: request/response schemas and input validation."""
from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.errors import RequestValidationErrors

# Floating point tolerance for balance comparisons.
EPS = 1e-9


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class BucketSpec(BaseModel):
    """令牌桶定义。

    capacity: 桶容量（令牌上限）
    refill_rate: 每秒连续补充的令牌数
    initial: 初始余额，缺省为满桶
    """

    model_config = ConfigDict(extra="forbid")

    id: Annotated[str, Field(min_length=1, description="桶唯一标识")]
    capacity: Annotated[float, Field(description="桶容量，必须为正数")]
    refill_rate: Annotated[float, Field(description="令牌补充速率（令牌/秒），必须非负")]
    initial: Annotated[float | None, Field(default=None, description="初始余额，默认等于容量")] = None

    @field_validator("capacity", "refill_rate", mode="after")
    @classmethod
    def _validate_rate_fields(cls, value, info):
        if info.field_name == "capacity" and value <= 0:
            raise ValueError("capacity 必须为正数")
        if info.field_name == "refill_rate" and value < 0:
            raise ValueError("refill_rate 必须为非负数")
        return value

    @field_validator("initial", mode="before")
    @classmethod
    def _validate_initial(cls, value):
        if value is not None and (not _is_finite_number(value) or value < 0):
            raise ValueError("initial 必须是非负有限数值")
        return value

    @model_validator(mode="after")
    def _check_initial_within_capacity(self):
        if self.initial is not None and self.initial > self.capacity + EPS:
            raise ValueError("initial 不能大于 capacity")
        return self


class SimRequest(BaseModel):
    """单次试算请求。"""

    model_config = ConfigDict(extra="forbid")

    id: Annotated[str, Field(min_length=1, description="请求唯一标识")]
    timestamp: Annotated[float, Field(description="请求时间（秒，任意一致的时间原点）")]
    cost: Annotated[float, Field(gt=0, description="消耗量（从链上每个桶扣减的令牌数）")]
    chain: Annotated[list[str], Field(min_length=1, description="桶链：按顺序引用的桶 ID")]

    @field_validator("timestamp", mode="before")
    @classmethod
    def _validate_timestamp(cls, value):
        if not _is_finite_number(value):
            raise ValueError("timestamp 必须是有限数值")
        return value

    @field_validator("cost", mode="before")
    @classmethod
    def _validate_cost(cls, value):
        if not _is_finite_number(value):
            raise ValueError("cost 必须是有限数值")
        if value <= 0:
            raise ValueError("cost 必须为正数")
        return value

    @field_validator("chain")
    @classmethod
    def _validate_chain_shape(cls, value: list[str]):
        if not value:
            raise ValueError("chain 至少需要引用一个桶")
        seen: set[str] = set()
        for bucket_id in value:
            if not isinstance(bucket_id, str) or not bucket_id:
                raise ValueError("chain 中的桶 ID 必须是非空字符串")
            if bucket_id in seen:
                raise ValueError(f"chain 中存在重复桶引用: {bucket_id}")
            seen.add(bucket_id)
        return value


def _simulate_cross_validation_errors(
    buckets: list[BucketSpec], requests: list[SimRequest]
) -> list[dict]:
    """/simulate 系请求体的跨字段校验：重复桶、未知桶引用、重复请求 ID。

    /simulate、/plan-initial-balances、/usage-report 与 /fairness-report 共用。
    """
    errors: list[dict] = []

    # 重复桶 ID
    seen: set[str] = set()
    duplicates: set[str] = set()
    for index, bucket in enumerate(buckets):
        if bucket.id in seen:
            duplicates.add(bucket.id)
            errors.append(
                {
                    "type": "duplicate_bucket",
                    "loc": ("buckets", index, "id"),
                    "msg": f"重复的桶 ID: {bucket.id}",
                    "input": bucket.id,
                }
            )
        seen.add(bucket.id)

    if duplicates:
        # 已存在重复定义，引用校验意义不大，直接返回
        return errors

    # 链中未知引用
    for r_index, request in enumerate(requests):
        for c_index, bucket_id in enumerate(request.chain):
            if bucket_id not in seen:
                errors.append(
                    {
                        "type": "unknown_bucket_ref",
                        "loc": ("requests", r_index, "chain", c_index),
                        "msg": f"请求 {request.id!r} 引用了未定义的桶: {bucket_id}",
                        "input": bucket_id,
                    }
                )

    # 重复请求 ID（便于调用方定位模拟结果）
    seen_req: set[str] = set()
    for r_index, request in enumerate(requests):
        if request.id in seen_req:
            errors.append(
                {
                    "type": "duplicate_request",
                    "loc": ("requests", r_index, "id"),
                    "msg": f"重复的请求 ID: {request.id}",
                    "input": request.id,
                }
            )
        seen_req.add(request.id)

    return errors


class SimulateRequest(BaseModel):
    """POST /simulate 请求体。"""

    model_config = ConfigDict(extra="forbid")

    buckets: Annotated[list[BucketSpec], Field(min_length=1, description="令牌桶集合")]
    requests: Annotated[list[SimRequest], Field(min_length=1, description="按时间戳模拟的请求序列")]

    @model_validator(mode="after")
    def _cross_validate(self):
        errors = _simulate_cross_validation_errors(self.buckets, self.requests)
        if errors:
            raise RequestValidationErrors(errors)
        return self


class ForecastBucket(BaseModel):
    """预测时刻的桶状态快照。

    capacity: 桶容量（令牌上限）
    refill_rate: 每秒连续补充的令牌数
    balance: 当前余额（now 时刻持有）
    """

    model_config = ConfigDict(extra="forbid")

    id: Annotated[str, Field(min_length=1, description="桶唯一标识")]
    capacity: Annotated[float, Field(description="桶容量，必须为正数")]
    refill_rate: Annotated[float, Field(description="令牌补充速率（令牌/秒），必须非负")]
    balance: Annotated[float, Field(description="当前余额，要求 0 ≤ balance ≤ capacity")]

    @field_validator("capacity", "refill_rate", mode="after")
    @classmethod
    def _validate_rate_fields(cls, value, info):
        if info.field_name == "capacity" and value <= 0:
            raise ValueError("capacity 必须为正数")
        if info.field_name == "refill_rate" and value < 0:
            raise ValueError("refill_rate 必须为非负数")
        return value

    @field_validator("balance", mode="before")
    @classmethod
    def _validate_balance(cls, value):
        if not _is_finite_number(value) or value < 0:
            raise ValueError("balance 必须是非负有限数值")
        return value

    @model_validator(mode="after")
    def _check_balance_within_capacity(self):
        if self.balance > self.capacity + EPS:
            raise ValueError("balance 不能大于 capacity")
        return self


class ForecastRequest(BaseModel):
    """POST /forecast 请求体。"""

    model_config = ConfigDict(extra="forbid")

    now: Annotated[float, Field(description="当前时刻（秒，任意一致的时间原点）")]
    cost: Annotated[float, Field(gt=0, description="请求成本（链上每个桶都需持有的令牌数）")]
    chain: Annotated[list[str], Field(min_length=1, description="桶链：按顺序引用的桶 ID")]
    buckets: Annotated[list[ForecastBucket], Field(min_length=1, description="桶状态快照集合")]

    @field_validator("now", mode="before")
    @classmethod
    def _validate_now(cls, value):
        if not _is_finite_number(value):
            raise ValueError("now 必须是有限数值")
        return value

    @field_validator("cost", mode="before")
    @classmethod
    def _validate_cost(cls, value):
        if not _is_finite_number(value):
            raise ValueError("cost 必须是有限数值")
        if value <= 0:
            raise ValueError("cost 必须为正数")
        return value

    @field_validator("chain")
    @classmethod
    def _validate_chain_shape(cls, value: list[str]):
        if not value:
            raise ValueError("chain 至少需要引用一个桶")
        seen: set[str] = set()
        for bucket_id in value:
            if not isinstance(bucket_id, str) or not bucket_id:
                raise ValueError("chain 中的桶 ID 必须是非空字符串")
            if bucket_id in seen:
                raise ValueError(f"chain 中存在重复桶引用: {bucket_id}")
            seen.add(bucket_id)
        return value

    @model_validator(mode="after")
    def _cross_validate(self):
        errors: list[dict] = []

        # 重复桶 ID
        seen: set[str] = set()
        duplicates = False
        for index, bucket in enumerate(self.buckets):
            if bucket.id in seen:
                duplicates = True
                errors.append(
                    {
                        "type": "duplicate_bucket",
                        "loc": ("buckets", index, "id"),
                        "msg": f"重复的桶 ID: {bucket.id}",
                        "input": bucket.id,
                    }
                )
            seen.add(bucket.id)

        if duplicates:
            # 已存在重复定义，引用校验意义不大，直接返回
            raise RequestValidationErrors(errors)

        # 链中未知引用
        for c_index, bucket_id in enumerate(self.chain):
            if bucket_id not in seen:
                errors.append(
                    {
                        "type": "unknown_bucket_ref",
                        "loc": ("chain", c_index),
                        "msg": f"桶链引用了未定义的桶: {bucket_id}",
                        "input": bucket_id,
                    }
                )

        if errors:
            raise RequestValidationErrors(errors)
        return self


class ForecastBucketResult(BaseModel):
    """链上单个桶的可达性预测。"""

    id: str
    balance_now: Annotated[float, Field(description="now 时刻的余额")]
    reachable: Annotated[bool, Field(description="该桶单独是否可达成本")]
    reason: Annotated[
        str | None,
        Field(default=None, description="不可达原因：cost_exceeds_capacity / zero_refill_rate"),
    ] = None
    wait_seconds: Annotated[
        float | None, Field(default=None, description="该桶单独满足成本所需等待秒数；不可达为 null")
    ] = None
    projected_balance: Annotated[
        float | None,
        Field(default=None, description="预计可通过时刻的余额（不扣减）；链不可达时为 null"),
    ] = None


class ForecastResponse(BaseModel):
    reachable: Annotated[bool, Field(description="链上全部桶是否都能达到成本")]
    wait_seconds: Annotated[
        float | None, Field(description="全部桶首次同时满足成本所需等待秒数；不可达为 null")
    ]
    pass_at: Annotated[
        float | None, Field(description="预计可通过时刻 = now + wait_seconds；不可达为 null")
    ]
    limiting_bucket: Annotated[
        str | None,
        Field(description="等待最长的限制桶（并列取链上首个）；立即可通过或不可达时为 null"),
    ]
    buckets: Annotated[list[ForecastBucketResult], Field(description="链上各桶的预测，按链顺序")]


class BucketBalance(BaseModel):
    """链上单个桶在某次扣减判定时的余额视图。"""

    before: Annotated[float, Field(description="判定时刻（补充之后、扣减之前）的余额")]
    after: Annotated[float, Field(description="扣减之后的余额；被拒绝时与 before 相同")]


class SimResult(BaseModel):
    """单项请求的模拟结果。"""

    request_id: str
    index: Annotated[int, Field(description="该请求在原始 requests 数组中的下标")]
    timestamp: float
    allowed: bool
    rejected_by: Annotated[str | None, Field(default=None, description="导致拒绝的第一个余额不足的桶 ID")]
    balances: Annotated[dict[str, BucketBalance], Field(description="链上各桶扣减前后余额")]


class SimulateResponse(BaseModel):
    allowed_count: int
    rejected_count: int
    results: list[SimResult]
    final_balances: Annotated[dict[str, float], Field(description="模拟结束时所有桶的余额")]


class PlanBucketResult(BaseModel):
    """单个桶的初始余额规划结果。"""

    id: str
    required_initial: Annotated[
        float, Field(description="满足该桶全部链上消耗所需的最低初始余额（[0, capacity] 内）")
    ]
    requested_initial: Annotated[
        float, Field(description="请求中给出的 initial；缺省按 capacity 计")
    ]
    delta: Annotated[
        float,
        Field(description="required_initial - requested_initial；为正表示请求初始余额不足"),
    ]


class PlanFailure(BaseModel):
    """满桶起步仍不可行时，按处理顺序首个无法满足的请求与桶。"""

    request_id: str
    index: Annotated[int, Field(description="该请求在原始 requests 数组中的下标")]
    timestamp: float
    bucket_id: Annotated[str, Field(description="链上首个余额不足的桶 ID")]
    cost: Annotated[float, Field(description="该请求的消耗量")]
    available: Annotated[
        float, Field(description="满桶起步回放中，该桶在判定时刻（补充之后、扣减之前）的余额")
    ]


class PlanInitialBalancesResponse(BaseModel):
    reachable: Annotated[bool, Field(description="满桶起步是否能让全部请求放行")]
    buckets: Annotated[
        list[PlanBucketResult] | None,
        Field(description="各桶规划结果，按 buckets 输入顺序；不可行为 null"),
    ]
    final_balances: Annotated[
        dict[str, float] | None,
        Field(description="用规划值完整回放得到的终态余额；不可行为 null"),
    ]
    first_failure: Annotated[
        PlanFailure | None,
        Field(description="不可行时按处理顺序首个无法满足的请求与桶；可行时为 null"),
    ]


class UsageReportRequest(SimulateRequest):
    """POST /usage-report 请求体：/simulate 请求体 + 统计窗口宽度。

    桶与请求的字段、跨字段校验（重复桶、未知引用、重复请求 ID）全部沿用
    SimulateRequest。
    """

    window_seconds: Annotated[float, Field(description="统计窗口宽度（秒），必须为正的有限数值")]

    @field_validator("window_seconds", mode="before")
    @classmethod
    def _validate_window_seconds(cls, value):
        if not _is_finite_number(value):
            raise ValueError("window_seconds 必须是有限数值")
        if value <= 0:
            raise ValueError("window_seconds 必须为正数")
        return value


class WindowRange(BaseModel):
    """半开窗口区间 [start, end)。"""

    index: Annotated[int, Field(description="窗口序号（从 0 开始）")]
    start: Annotated[float, Field(description="窗口起点（含）")]
    end: Annotated[float, Field(description="窗口终点（不含）")]


class UsageWindowStats(BaseModel):
    """单个桶在单个窗口内的使用统计。"""

    index: Annotated[int, Field(description="窗口序号")]
    allowed_consumption: Annotated[
        float, Field(description="窗口内放行请求对该桶的消耗合计")
    ]
    rejected_count: Annotated[
        int, Field(description="窗口内引用该桶且被原子拒绝的请求数（无论限制桶是否为该桶）")
    ]
    peak_occupancy: Annotated[
        float, Field(description="窗口内峰值占用 = capacity - 窗口内最低余额")
    ]


class UsageBucketReport(BaseModel):
    """单个桶的逐窗统计与峰值信息。"""

    id: str
    allowed_consumption: Annotated[float, Field(description="全部窗口放行消耗合计")]
    rejected_count: Annotated[int, Field(description="全部窗口拒绝次数合计")]
    peak_occupancy: Annotated[float, Field(description="全部窗口中的最高峰值占用")]
    peak_window: Annotated[
        WindowRange, Field(description="达到峰值占用的窗口起止时间（并列取最早窗口）")
    ]
    windows: Annotated[
        list[UsageWindowStats], Field(description="逐窗口统计，按窗口序号升序")
    ]


class BusiestWindow(WindowRange):
    """全局最忙窗口：跨桶放行消耗合计最大的窗口（并列取最早）。"""

    allowed_consumption: Annotated[float, Field(description="该窗口全部桶放行消耗之和")]
    allowed_count: Annotated[int, Field(description="该窗口放行的请求数")]
    rejected_count: Annotated[int, Field(description="该窗口被拒的请求数")]


class UsageReportResponse(BaseModel):
    window_seconds: Annotated[float, Field(description="统计窗口宽度（秒）")]
    window_count: Annotated[int, Field(description="覆盖请求时间跨度的窗口总数")]
    windows: Annotated[
        list[WindowRange], Field(description="全部窗口的起止时间，按窗口序号升序")
    ]
    buckets: Annotated[
        list[UsageBucketReport], Field(description="各桶统计，按 buckets 输入顺序")
    ]
    busiest_window: Annotated[BusiestWindow, Field(description="全局最忙窗口")]


class FairnessSimRequest(BaseModel):
    """公平性试算请求：在 /simulate 请求字段上增加 principal 与 weight。"""

    model_config = ConfigDict(extra="forbid")

    id: Annotated[str, Field(min_length=1, description="请求唯一标识")]
    timestamp: Annotated[float, Field(description="请求时间（秒，任意一致的时间原点）")]
    cost: Annotated[float, Field(gt=0, description="消耗量（从链上每个桶扣减的令牌数）")]
    chain: Annotated[list[str], Field(min_length=1, description="桶链：按顺序引用的桶 ID")]
    principal: Annotated[
        str, Field(min_length=1, description="请求归属主体（租户/用户/调用方 ID）")
    ]
    weight: Annotated[
        float,
        Field(default=1.0, description="主体权重，必须为正的有限数值；缺省为 1.0"),
    ] = 1.0

    @field_validator("timestamp", mode="before")
    @classmethod
    def _validate_timestamp(cls, value):
        if not _is_finite_number(value):
            raise ValueError("timestamp 必须是有限数值")
        return value

    @field_validator("cost", mode="before")
    @classmethod
    def _validate_cost(cls, value):
        if not _is_finite_number(value):
            raise ValueError("cost 必须是有限数值")
        if value <= 0:
            raise ValueError("cost 必须为正数")
        return value

    @field_validator("principal", mode="before")
    @classmethod
    def _validate_principal(cls, value):
        if not isinstance(value, str) or not value:
            raise ValueError("principal 必须是非空字符串")
        return value

    @field_validator("weight", mode="before")
    @classmethod
    def _validate_weight(cls, value):
        if not _is_finite_number(value):
            raise ValueError("weight 必须是有限数值")
        if value <= 0:
            raise ValueError("weight 必须为正数")
        return value

    @field_validator("chain")
    @classmethod
    def _validate_chain_shape(cls, value: list[str]):
        if not value:
            raise ValueError("chain 至少需要引用一个桶")
        seen: set[str] = set()
        for bucket_id in value:
            if not isinstance(bucket_id, str) or not bucket_id:
                raise ValueError("chain 中的桶 ID 必须是非空字符串")
            if bucket_id in seen:
                raise ValueError(f"chain 中存在重复桶引用: {bucket_id}")
            seen.add(bucket_id)
        return value


class FairnessReportRequest(BaseModel):
    """POST /fairness-report 请求体：带 principal/weight 的请求序列 + 统计窗口宽度。

    桶与请求的字段、跨字段校验（重复桶、未知引用、重复请求 ID）沿用
    SimulateRequest 的同一套规则。
    """

    model_config = ConfigDict(extra="forbid")

    buckets: Annotated[list[BucketSpec], Field(min_length=1, description="令牌桶集合")]
    requests: Annotated[
        list[FairnessSimRequest], Field(min_length=1, description="按时间戳模拟的请求序列")
    ]
    window_seconds: Annotated[float, Field(description="统计窗口宽度（秒），必须为正的有限数值")]

    @field_validator("window_seconds", mode="before")
    @classmethod
    def _validate_window_seconds(cls, value):
        if not _is_finite_number(value):
            raise ValueError("window_seconds 必须是有限数值")
        if value <= 0:
            raise ValueError("window_seconds 必须为正数")
        return value

    @model_validator(mode="after")
    def _cross_validate(self):
        errors = _simulate_cross_validation_errors(self.buckets, self.requests)

        # 同一 principal 的权重必须一致：加权 Jain 指数以主体为单位计算
        principal_weight: dict[str, float] = {}
        for r_index, request in enumerate(self.requests):
            previous = principal_weight.get(request.principal)
            if previous is None:
                principal_weight[request.principal] = float(request.weight)
            elif abs(previous - float(request.weight)) > EPS:
                errors.append(
                    {
                        "type": "inconsistent_principal_weight",
                        "loc": ("requests", r_index, "weight"),
                        "msg": (
                            f"principal {request.principal!r} 的权重不一致："
                            f"{previous} 与 {float(request.weight)}"
                        ),
                        "input": request.weight,
                    }
                )

        if errors:
            raise RequestValidationErrors(errors)
        return self


class FairnessPrincipalRow(BaseModel):
    """单个主体在某个统计范围（全局或单个窗口）内的公平性数据。"""

    principal: Annotated[str, Field(description="主体标识")]
    weight: Annotated[float, Field(description="主体权重")]
    request_count: Annotated[int, Field(description="该范围内该主体的请求总数")]
    allowed_count: Annotated[int, Field(description="放行请求数")]
    rejected_count: Annotated[int, Field(description="被原子拒绝的请求数")]
    allowed_consumption: Annotated[
        float, Field(description="放行请求的 cost 合计（该主体的实际获得量）")
    ]
    theoretical_share: Annotated[
        float,
        Field(description="理论权重份额 = 该范围全部放行消耗 × weight / 权重合计"),
    ]
    deficit: Annotated[
        float, Field(description="短缺量 = max(0, theoretical_share - allowed_consumption)")
    ]
    deficit_ratio: Annotated[
        float,
        Field(description="短缺比例 = deficit / theoretical_share；理论份额为 0 时为 0"),
    ]


class FirstShortfall(BaseModel):
    """最受限主体的短缺证据：按处理顺序首个造成其短缺的拒绝。"""

    request_id: Annotated[str, Field(description="被拒绝的请求 ID")]
    index: Annotated[int, Field(description="该请求在原始 requests 数组中的下标")]
    timestamp: Annotated[float, Field(description="该请求的时间戳")]
    bucket_id: Annotated[str, Field(description="导致原子拒绝的第一个余额不足的桶 ID")]
    cost: Annotated[float, Field(description="该请求的消耗量")]
    available: Annotated[
        float, Field(description="判定时刻（补充之后、扣减之前）限制桶的余额")
    ]


class FairnessScopeReport(BaseModel):
    """一个统计范围（全局或单个窗口）的公平性报告。"""

    principal_count: Annotated[int, Field(description="参与该范围统计的主体数")]
    request_count: Annotated[int, Field(description="该范围内的请求总数")]
    allowed_count: Annotated[int, Field(description="放行请求总数")]
    rejected_count: Annotated[int, Field(description="拒绝请求总数")]
    allowed_consumption: Annotated[
        float, Field(description="全部主体放行消耗（实际获得量）合计")
    ]
    weighted_jain_index: Annotated[
        float,
        Field(
            description=(
                "加权 Jain 公平指数 = (Σ x)² / (Σ w · Σ x²/w)，"
                "x 为主体实际获得量、w 为权重；取值 (0, 1]，越大越公平"
            )
        ),
    ]
    most_constrained_principal: Annotated[
        str | None,
        Field(
            description=(
                "短缺比例最高且有被拒请求的主体（并列取首次拒绝最早者）；"
                "不存在有短缺的被拒主体时为 null"
            )
        ),
    ]
    first_shortfall: Annotated[
        FirstShortfall | None,
        Field(
            description="最受限主体按处理顺序的首个短缺拒绝；无最受限主体时为 null"
        ),
    ]
    principals: Annotated[
        list[FairnessPrincipalRow],
        Field(description="各主体统计，按主体首次请求的处理顺序"),
    ]


class FairnessWindowReport(FairnessScopeReport):
    """单个窗口的公平性报告：范围数据 + 窗口起止。"""

    index: Annotated[int, Field(description="窗口序号")]
    start: Annotated[float, Field(description="窗口起点（含）")]
    end: Annotated[float, Field(description="窗口终点（不含）")]


class FairnessReportResponse(BaseModel):
    window_seconds: Annotated[float, Field(description="统计窗口宽度（秒）")]
    window_count: Annotated[int, Field(description="覆盖请求时间跨度的窗口总数")]
    windows: Annotated[
        list[WindowRange], Field(description="全部窗口的起止时间，按窗口序号升序")
    ]
    global_report: Annotated[FairnessScopeReport, Field(description="跨全部窗口的全局报告")]
    window_reports: Annotated[
        list[FairnessWindowReport], Field(description="逐窗口报告，按窗口序号升序")
    ]


class ReservationBucket(BaseModel):
    """预留试算桶定义：只有容量上限，没有补充概念。

    capacity: 桶容量，即任意时刻允许的并发占用上限
    """

    model_config = ConfigDict(extra="forbid")

    id: Annotated[str, Field(min_length=1, description="桶唯一标识")]
    capacity: Annotated[float, Field(description="桶容量（并发占用上限），必须为正数")]

    @field_validator("capacity", mode="before")
    @classmethod
    def _validate_capacity(cls, value):
        if not _is_finite_number(value):
            raise ValueError("capacity 必须是有限数值")
        if value <= 0:
            raise ValueError("capacity 必须为正数")
        return value


class ReservationRequest(BaseModel):
    """单次预留请求：在半开区间 [start, end) 内占用链上每个桶的 cost 容量。"""

    model_config = ConfigDict(extra="forbid")

    start: Annotated[float, Field(description="预留起始时刻（含），秒，任意一致的时间原点")]
    end: Annotated[float, Field(description="预留结束时刻（不含），必须严格大于 start")]
    cost: Annotated[float, Field(gt=0, description="预留期间对链上每个桶的并发占用量")]
    chain: Annotated[list[str], Field(min_length=1, description="桶链：按顺序引用的桶 ID")]

    @field_validator("start", "end", mode="before")
    @classmethod
    def _validate_time(cls, value, info):
        if not _is_finite_number(value):
            raise ValueError(f"{info.field_name} 必须是有限数值")
        return value

    @field_validator("cost", mode="before")
    @classmethod
    def _validate_cost(cls, value):
        if not _is_finite_number(value):
            raise ValueError("cost 必须是有限数值")
        if value <= 0:
            raise ValueError("cost 必须为正数")
        return value

    @field_validator("chain")
    @classmethod
    def _validate_chain_shape(cls, value: list[str]):
        if not value:
            raise ValueError("chain 至少需要引用一个桶")
        seen: set[str] = set()
        for bucket_id in value:
            if not isinstance(bucket_id, str) or not bucket_id:
                raise ValueError("chain 中的桶 ID 必须是非空字符串")
            if bucket_id in seen:
                raise ValueError(f"chain 中存在重复桶引用: {bucket_id}")
            seen.add(bucket_id)
        return value

    @model_validator(mode="after")
    def _check_interval(self):
        if self.end <= self.start:
            raise ValueError("end 必须严格大于 start")
        return self


class ReservationSimulateRequest(BaseModel):
    """POST /reservation-simulate 请求体。"""

    model_config = ConfigDict(extra="forbid")

    buckets: Annotated[list[ReservationBucket], Field(min_length=1, description="预留桶集合")]
    reservations: Annotated[
        list[ReservationRequest], Field(min_length=1, description="按 start 升序试算的预留序列")
    ]

    @model_validator(mode="after")
    def _cross_validate(self):
        errors: list[dict] = []

        # 重复桶 ID
        seen: set[str] = set()
        duplicates: set[str] = set()
        for index, bucket in enumerate(self.buckets):
            if bucket.id in seen:
                duplicates.add(bucket.id)
                errors.append(
                    {
                        "type": "duplicate_bucket",
                        "loc": ("buckets", index, "id"),
                        "msg": f"重复的桶 ID: {bucket.id}",
                        "input": bucket.id,
                    }
                )
            seen.add(bucket.id)

        if duplicates:
            # 已存在重复定义，引用校验意义不大，直接返回
            raise RequestValidationErrors(errors)

        # 链中未知引用
        for r_index, reservation in enumerate(self.reservations):
            for c_index, bucket_id in enumerate(reservation.chain):
                if bucket_id not in seen:
                    errors.append(
                        {
                            "type": "unknown_bucket_ref",
                            "loc": ("reservations", r_index, "chain", c_index),
                            "msg": f"预留 {r_index} 引用了未定义的桶: {bucket_id}",
                            "input": bucket_id,
                        }
                    )

        if errors:
            raise RequestValidationErrors(errors)
        return self


class OccupancyInterval(BaseModel):
    """单个桶上并发占用量恒定的合并区间 [start, end)。"""

    start: Annotated[float, Field(description="区间起点（含）")]
    end: Annotated[float, Field(description="区间终点（不含）")]
    occupancy: Annotated[float, Field(description="该区间内的并发占用总量")]


class ReservationResult(BaseModel):
    """单项预留的试算结果。"""

    index: Annotated[int, Field(description="该预留在原始 reservations 数组中的下标")]
    start: float
    end: float
    cost: float
    allowed: Annotated[bool, Field(description="是否原子接受")]
    rejected_by: Annotated[
        str | None,
        Field(default=None, description="导致拒绝的第一个放不下的桶 ID；接受时为 null"),
    ] = None


class ReservationBucketOccupancy(BaseModel):
    """单个桶在全部接受预留叠加后的占用视图。"""

    id: str
    capacity: Annotated[float, Field(description="桶容量（并发占用上限）")]
    peak_occupancy: Annotated[float, Field(description="整个时间轴上的峰值并发占用")]
    intervals: Annotated[
        list[OccupancyInterval],
        Field(description="合并后的非零占用区间，按起点升序、相邻同占用相接区间已合并"),
    ]


class ReservationSimulateResponse(BaseModel):
    allowed_count: int
    rejected_count: int
    results: Annotated[
        list[ReservationResult], Field(description="逐项预留结果，按原始输入索引升序")
    ]
    buckets: Annotated[
        list[ReservationBucketOccupancy],
        Field(description="各桶合并后的占用区间与峰值并发，按 buckets 输入顺序"),
    ]


class RouteSimRequest(BaseModel):
    """单次路由试算请求：携带多条候选桶链，由服务挑选一条放行。"""

    model_config = ConfigDict(extra="forbid")

    id: Annotated[str, Field(min_length=1, description="请求唯一标识")]
    timestamp: Annotated[float, Field(description="请求时间（秒，任意一致的时间原点）")]
    cost: Annotated[float, Field(gt=0, description="消耗量（从选中链上每个桶扣减的令牌数）")]
    chains: Annotated[
        list[list[str]],
        Field(min_length=1, description="候选桶链列表：至少一条，每条链按顺序引用桶 ID"),
    ]

    @field_validator("timestamp", mode="before")
    @classmethod
    def _validate_timestamp(cls, value):
        if not _is_finite_number(value):
            raise ValueError("timestamp 必须是有限数值")
        return value

    @field_validator("cost", mode="before")
    @classmethod
    def _validate_cost(cls, value):
        if not _is_finite_number(value):
            raise ValueError("cost 必须是有限数值")
        if value <= 0:
            raise ValueError("cost 必须为正数")
        return value

    @field_validator("chains")
    @classmethod
    def _validate_chains_shape(cls, value: list[list[str]]):
        if not value:
            raise ValueError("chains 至少需要一条候选链")
        for chain in value:
            if not chain:
                raise ValueError("候选链至少需要引用一个桶")
            seen: set[str] = set()
            for bucket_id in chain:
                if not isinstance(bucket_id, str) or not bucket_id:
                    raise ValueError("候选链中的桶 ID 必须是非空字符串")
                if bucket_id in seen:
                    raise ValueError(f"候选链中存在重复桶引用: {bucket_id}")
                seen.add(bucket_id)
        return value


class RouteSimulateRequest(BaseModel):
    """POST /route-simulate 请求体。"""

    model_config = ConfigDict(extra="forbid")

    buckets: Annotated[list[BucketSpec], Field(min_length=1, description="令牌桶集合")]
    requests: Annotated[
        list[RouteSimRequest], Field(min_length=1, description="按时间戳路由试算的请求序列")
    ]

    @model_validator(mode="after")
    def _cross_validate(self):
        errors: list[dict] = []

        # 重复桶 ID
        seen: set[str] = set()
        duplicates: set[str] = set()
        for index, bucket in enumerate(self.buckets):
            if bucket.id in seen:
                duplicates.add(bucket.id)
                errors.append(
                    {
                        "type": "duplicate_bucket",
                        "loc": ("buckets", index, "id"),
                        "msg": f"重复的桶 ID: {bucket.id}",
                        "input": bucket.id,
                    }
                )
            seen.add(bucket.id)

        if duplicates:
            # 已存在重复定义，引用校验意义不大，直接返回
            raise RequestValidationErrors(errors)

        # 候选链中未知引用
        for r_index, request in enumerate(self.requests):
            for c_index, chain in enumerate(request.chains):
                for b_index, bucket_id in enumerate(chain):
                    if bucket_id not in seen:
                        errors.append(
                            {
                                "type": "unknown_bucket_ref",
                                "loc": ("requests", r_index, "chains", c_index, b_index),
                                "msg": (
                                    f"请求 {request.id!r} 的候选链 {c_index} "
                                    f"引用了未定义的桶: {bucket_id}"
                                ),
                                "input": bucket_id,
                            }
                        )

        # 重复请求 ID（便于调用方定位模拟结果）
        seen_req: set[str] = set()
        for r_index, request in enumerate(self.requests):
            if request.id in seen_req:
                errors.append(
                    {
                        "type": "duplicate_request",
                        "loc": ("requests", r_index, "id"),
                        "msg": f"重复的请求 ID: {request.id}",
                        "input": request.id,
                    }
                )
            seen_req.add(request.id)

        if errors:
            raise RequestValidationErrors(errors)
        return self


class RouteCandidateRejection(BaseModel):
    """候选链的拒绝原因：该链上第一个余额不足的桶。"""

    chain_index: Annotated[int, Field(description="候选链在 chains 中的下标")]
    rejected_by: Annotated[str, Field(description="该候选链上第一个余额不足的桶 ID")]
    available: Annotated[
        float, Field(description="判定时刻（补充之后、扣减之前）该桶的余额")
    ]


class RouteSimResult(BaseModel):
    """单项路由请求的试算结果。"""

    request_id: str
    index: Annotated[int, Field(description="该请求在原始 requests 数组中的下标")]
    timestamp: float
    allowed: bool
    selected_chain_index: Annotated[
        int | None,
        Field(default=None, description="选中候选链在 chains 中的下标；被拒绝时为 null"),
    ] = None
    selected_chain: Annotated[
        list[str] | None,
        Field(default=None, description="选中的桶链；被拒绝时为 null"),
    ] = None
    balances: Annotated[
        dict[str, BucketBalance],
        Field(description="本请求触及的所有桶（全部候选链的并集）扣减前后余额"),
    ]
    rejections: Annotated[
        list[RouteCandidateRejection] | None,
        Field(default=None, description="被拒绝时逐候选链的首个不足桶；放行时为 null"),
    ] = None


class RouteSimulateResponse(BaseModel):
    allowed_count: int
    rejected_count: int
    results: Annotated[
        list[RouteSimResult], Field(description="逐项路由结果，按原始输入索引升序")
    ]
    final_balances: Annotated[dict[str, float], Field(description="模拟结束时所有桶的余额")]


class QueueSimRequest(SimRequest):
    """单次排队试算请求：/simulate 请求字段 + 不早于 timestamp 的 deadline。

    deadline: 最晚可执行时刻（秒）；队首在 deadline 前（含）余额充足即原子扣减，
    否则等待至 deadline 标记 expired 出队。
    """

    deadline: Annotated[
        float, Field(description="最晚可执行时刻（秒），必须为有限数值且不早于 timestamp")
    ]

    @field_validator("deadline", mode="before")
    @classmethod
    def _validate_deadline(cls, value):
        if not _is_finite_number(value):
            raise ValueError("deadline 必须是有限数值")
        return value

    @model_validator(mode="after")
    def _check_deadline_not_before_timestamp(self):
        if self.deadline < self.timestamp:
            raise ValueError("deadline 必须不早于 timestamp")
        return self


class QueueSimulateRequest(BaseModel):
    """POST /queue-simulate 请求体。"""

    model_config = ConfigDict(extra="forbid")

    buckets: Annotated[list[BucketSpec], Field(min_length=1, description="令牌桶集合")]
    requests: Annotated[
        list[QueueSimRequest],
        Field(min_length=1, description="按（到达时间, 输入顺序）排入严格 FIFO 队列的请求序列"),
    ]

    @model_validator(mode="after")
    def _cross_validate(self):
        errors = _simulate_cross_validation_errors(self.buckets, self.requests)
        if errors:
            raise RequestValidationErrors(errors)
        return self


class QueueSimResult(BaseModel):
    """单项排队请求的结果。"""

    request_id: str
    index: Annotated[int, Field(description="该请求在原始 requests 数组中的下标")]
    timestamp: float
    deadline: float
    status: Annotated[
        Literal["executed", "expired"],
        Field(
            description="executed=在 deadline 前（含）原子扣减放行；expired=等待至 deadline 仍不可执行"
        ),
    ]
    executed_at: Annotated[
        float | None, Field(default=None, description="实际执行时刻；expired 时为 null")
    ] = None
    expired_at: Annotated[
        float | None, Field(default=None, description="过期出队时刻；executed 时为 null")
    ] = None
    wait_seconds: Annotated[float, Field(description="从到达到执行/过期的等待秒数")]
    limiting_bucket: Annotated[
        str | None,
        Field(
            default=None,
            description="就绪最晚的限制桶（并列取链上首个）；成为队首即可执行时为 null",
        ),
    ] = None
    balances: Annotated[
        dict[str, BucketBalance],
        Field(description="链上各桶在判定时刻（补充之后、扣减之前/之后）的余额"),
    ]


class QueueSimulateResponse(BaseModel):
    executed_count: int
    expired_count: int
    results: Annotated[
        list[QueueSimResult], Field(description="逐项排队结果，按原始输入索引升序")
    ]
    final_balances: Annotated[
        dict[str, float], Field(description="最后一个请求出队时刻所有桶的余额")
    ]


class WindowPolicySpec(BaseModel):
    """半开滑动窗口配额策略。

    window_seconds: 滑动窗口宽度（秒）
    limit: 窗口内按归属主体（principal）统计的累计消耗上限
    """

    model_config = ConfigDict(extra="forbid")

    id: Annotated[str, Field(min_length=1, description="窗口策略唯一标识")]
    window_seconds: Annotated[float, Field(description="滑动窗口宽度（秒），必须为正数")]
    limit: Annotated[float, Field(description="窗口内累计消耗上限，必须为正数")]

    @field_validator("window_seconds", "limit", mode="before")
    @classmethod
    def _validate_positive_fields(cls, value, info):
        if not _is_finite_number(value):
            raise ValueError(f"{info.field_name} 必须是有限数值")
        if value <= 0:
            raise ValueError(f"{info.field_name} 必须为正数")
        return value


class WindowSimRequest(BaseModel):
    """单次滑动窗口试算请求。"""

    model_config = ConfigDict(extra="forbid")

    id: Annotated[str, Field(min_length=1, description="请求唯一标识")]
    timestamp: Annotated[float, Field(description="请求时间（秒，任意一致的时间原点）")]
    principal: Annotated[
        str, Field(min_length=1, description="请求归属主体（租户/用户/调用方 ID）")
    ]
    cost: Annotated[float, Field(gt=0, description="消耗量（对每个引用策略各计此数量）")]
    policy_refs: Annotated[
        list[str], Field(min_length=1, description="引用的窗口策略 ID 列表，按顺序判定")
    ]

    @field_validator("timestamp", mode="before")
    @classmethod
    def _validate_timestamp(cls, value):
        if not _is_finite_number(value):
            raise ValueError("timestamp 必须是有限数值")
        return value

    @field_validator("cost", mode="before")
    @classmethod
    def _validate_cost(cls, value):
        if not _is_finite_number(value):
            raise ValueError("cost 必须是有限数值")
        if value <= 0:
            raise ValueError("cost 必须为正数")
        return value

    @field_validator("principal", mode="before")
    @classmethod
    def _validate_principal(cls, value):
        if not isinstance(value, str) or not value:
            raise ValueError("principal 必须是非空字符串")
        return value

    @field_validator("policy_refs")
    @classmethod
    def _validate_refs_shape(cls, value: list[str]):
        if not value:
            raise ValueError("policy_refs 至少需要引用一个窗口策略")
        seen: set[str] = set()
        for policy_id in value:
            if not isinstance(policy_id, str) or not policy_id:
                raise ValueError("policy_refs 中的策略 ID 必须是非空字符串")
            if policy_id in seen:
                raise ValueError(f"policy_refs 中存在重复策略引用: {policy_id}")
            seen.add(policy_id)
        return value


class WindowSimulateRequest(BaseModel):
    """POST /window-simulate 请求体。"""

    model_config = ConfigDict(extra="forbid")

    policies: Annotated[
        list[WindowPolicySpec], Field(min_length=1, description="滑动窗口策略集合")
    ]
    requests: Annotated[
        list[WindowSimRequest],
        Field(min_length=1, description="按时间戳试算的请求序列（可乱序输入）"),
    ]

    @model_validator(mode="after")
    def _cross_validate(self):
        errors: list[dict] = []

        # 重复策略 ID
        seen: set[str] = set()
        duplicates = False
        for index, policy in enumerate(self.policies):
            if policy.id in seen:
                duplicates = True
                errors.append(
                    {
                        "type": "duplicate_policy",
                        "loc": ("policies", index, "id"),
                        "msg": f"重复的窗口策略 ID: {policy.id}",
                        "input": policy.id,
                    }
                )
            seen.add(policy.id)

        if duplicates:
            # 已存在重复定义，引用校验意义不大，直接返回
            raise RequestValidationErrors(errors)

        # 引用中未知策略
        for r_index, request in enumerate(self.requests):
            for p_index, policy_id in enumerate(request.policy_refs):
                if policy_id not in seen:
                    errors.append(
                        {
                            "type": "unknown_policy_ref",
                            "loc": ("requests", r_index, "policy_refs", p_index),
                            "msg": f"请求 {request.id!r} 引用了未定义的窗口策略: {policy_id}",
                            "input": policy_id,
                        }
                    )

        # 重复请求 ID（便于调用方定位模拟结果）
        seen_req: set[str] = set()
        for r_index, request in enumerate(self.requests):
            if request.id in seen_req:
                errors.append(
                    {
                        "type": "duplicate_request",
                        "loc": ("requests", r_index, "id"),
                        "msg": f"重复的请求 ID: {request.id}",
                        "input": request.id,
                    }
                )
            seen_req.add(request.id)

        if errors:
            raise RequestValidationErrors(errors)
        return self


class SlidingWindowRange(BaseModel):
    """请求判定时刻的半开滑动窗口 (start, end]：左端不含、右端含。"""

    start: Annotated[float, Field(description="窗口左端（不含）= end - window_seconds")]
    end: Annotated[float, Field(description="窗口右端（含）= 请求时间戳")]


class WindowPolicyView(BaseModel):
    """单个请求对某个引用策略的判定视图。"""

    id: str
    window_seconds: Annotated[float, Field(description="该策略的窗口宽度（秒）")]
    limit: Annotated[float, Field(description="该策略的窗口累计消耗上限")]
    window: Annotated[SlidingWindowRange, Field(description="判定时刻的半开窗口边界")]
    used_before: Annotated[
        float, Field(description="移除左端过期事件后、本请求记账前的窗口内累计消耗")
    ]
    used_after: Annotated[
        float, Field(description="本请求记账后的窗口内累计消耗；被拒绝时与 used_before 相同")
    ]
    remaining_before: Annotated[
        float, Field(description="记账前剩余额度 = limit - used_before")
    ]
    remaining_after: Annotated[
        float,
        Field(description="记账后剩余额度 = limit - used_after；被拒绝时与 remaining_before 相同"),
    ]


class WindowRejection(BaseModel):
    """拒绝原因：首个累计消耗超过上限的引用策略。"""

    policy_id: Annotated[str, Field(description="首个超限的窗口策略 ID（按 policy_refs 顺序）")]
    reason: Annotated[
        Literal["limit_exceeded"],
        Field(description="拒绝原因：窗口内累计消耗加本次 cost 超过 limit"),
    ]
    window_seconds: Annotated[float, Field(description="该策略的窗口宽度（秒）")]
    limit: Annotated[float, Field(description="该策略的窗口累计消耗上限")]
    used: Annotated[float, Field(description="判定时窗口内已占用的累计消耗")]
    cost: Annotated[float, Field(description="本次请求的消耗量")]
    required: Annotated[float, Field(description="放行所需额度 = used + cost")]
    available: Annotated[float, Field(description="判定时剩余额度 = limit - used")]
    window: Annotated[SlidingWindowRange, Field(description="判定时刻的半开窗口边界")]


class WindowSimResult(BaseModel):
    """单项请求的滑动窗口试算结果。"""

    request_id: str
    index: Annotated[int, Field(description="该请求在原始 requests 数组中的下标")]
    timestamp: float
    principal: str
    cost: float
    allowed: bool
    policies: Annotated[
        dict[str, WindowPolicyView],
        Field(description="各引用策略的判定视图，按 policy_refs 顺序"),
    ]
    rejected_by: Annotated[
        str | None, Field(default=None, description="首个超限的策略 ID；放行时为 null")
    ] = None
    rejection: Annotated[
        WindowRejection | None, Field(default=None, description="拒绝原因明细；放行时为 null")
    ] = None


class WindowPrincipalOccupancy(BaseModel):
    """终态时刻某个主体在单个策略窗口内的占用。"""

    principal: Annotated[str, Field(description="主体标识")]
    event_count: Annotated[int, Field(description="终态窗口内仍有效（未过期）的记账事件数")]
    used: Annotated[float, Field(description="终态窗口内该主体的累计消耗")]
    remaining: Annotated[float, Field(description="终态剩余额度 = limit - used")]
    window: Annotated[SlidingWindowRange, Field(description="终态时刻的半开窗口边界")]


class WindowPolicyFinalState(BaseModel):
    """单个策略在模拟结束时刻（最后一个请求时间戳）的终态占用。"""

    id: str
    window_seconds: Annotated[float, Field(description="该策略的窗口宽度（秒）")]
    limit: Annotated[float, Field(description="该策略的窗口累计消耗上限")]
    total_used: Annotated[
        float, Field(description="终态窗口内全部主体的累计消耗合计")
    ]
    window: Annotated[SlidingWindowRange, Field(description="终态时刻的半开窗口边界")]
    principals: Annotated[
        list[WindowPrincipalOccupancy],
        Field(description="各主体终态占用，按该策略上首次成功记账的顺序"),
    ]


class WindowSimulateResponse(BaseModel):
    allowed_count: int
    rejected_count: int
    results: Annotated[
        list[WindowSimResult], Field(description="逐项请求结果，按原始输入索引升序")
    ]
    policies: Annotated[
        list[WindowPolicyFinalState],
        Field(description="各策略的终态占用，按 policies 输入顺序"),
    ]


# ---------- 周期结转额度试算 (/rollover-simulate) ----------


class RolloverPolicySpec(BaseModel):
    """周期结转额度策略。

    anchor: 周期对齐锚点时刻（秒，任意一致的时间原点）
    period_seconds: 周期长度（秒）
    allowance: 每个周期发放的基础额度
    carry_cap: 上一周期未用基础额度结转到下一周期的上限（旧结转额跨期一律失效）
    """

    model_config = ConfigDict(extra="forbid")

    id: Annotated[str, Field(min_length=1, description="结转策略唯一标识")]
    anchor: Annotated[float, Field(description="周期对齐锚点时刻（秒），可为任意有限数值（含负数）")]
    period_seconds: Annotated[float, Field(description="周期长度（秒），必须为正数")]
    allowance: Annotated[float, Field(description="每个周期的基础额度，必须为正数")]
    carry_cap: Annotated[
        float, Field(description="上期未用基础额度结转到下期的上限，必须为非负数；0 表示不结转")
    ]

    @field_validator("anchor", "period_seconds", "allowance", "carry_cap", mode="before")
    @classmethod
    def _validate_numeric_fields(cls, value, info):
        if not _is_finite_number(value):
            raise ValueError(f"{info.field_name} 必须是有限数值")
        if info.field_name in ("period_seconds", "allowance") and value <= 0:
            raise ValueError(f"{info.field_name} 必须为正数")
        if info.field_name == "carry_cap" and value < 0:
            raise ValueError("carry_cap 必须为非负数")
        return value


class RolloverSimRequest(BaseModel):
    """单次周期结转试算请求。"""

    model_config = ConfigDict(extra="forbid")

    id: Annotated[str, Field(min_length=1, description="请求唯一标识")]
    timestamp: Annotated[float, Field(description="请求时间（秒，任意一致的时间原点）")]
    cost: Annotated[float, Field(gt=0, description="消耗量（对每个引用策略各计此数量）")]
    policy_refs: Annotated[
        list[str], Field(min_length=1, description="引用的结转策略 ID 列表，按顺序判定")
    ]

    @field_validator("timestamp", mode="before")
    @classmethod
    def _validate_timestamp(cls, value):
        if not _is_finite_number(value):
            raise ValueError("timestamp 必须是有限数值")
        return value

    @field_validator("cost", mode="before")
    @classmethod
    def _validate_cost(cls, value):
        if not _is_finite_number(value):
            raise ValueError("cost 必须是有限数值")
        if value <= 0:
            raise ValueError("cost 必须为正数")
        return value

    @field_validator("policy_refs")
    @classmethod
    def _validate_refs_shape(cls, value: list[str]):
        if not value:
            raise ValueError("policy_refs 至少需要引用一个结转策略")
        seen: set[str] = set()
        for policy_id in value:
            if not isinstance(policy_id, str) or not policy_id:
                raise ValueError("policy_refs 中的策略 ID 必须是非空字符串")
            if policy_id in seen:
                raise ValueError(f"policy_refs 中存在重复策略引用: {policy_id}")
            seen.add(policy_id)
        return value


class RolloverSimulateRequest(BaseModel):
    """POST /rollover-simulate 请求体。"""

    model_config = ConfigDict(extra="forbid")

    policies: Annotated[
        list[RolloverPolicySpec], Field(min_length=1, description="周期结转策略集合")
    ]
    requests: Annotated[
        list[RolloverSimRequest],
        Field(min_length=1, description="按时间戳试算的请求序列（可乱序输入）"),
    ]

    @model_validator(mode="after")
    def _cross_validate(self):
        errors: list[dict] = []

        # 重复策略 ID
        seen: set[str] = set()
        duplicates = False
        for index, policy in enumerate(self.policies):
            if policy.id in seen:
                duplicates = True
                errors.append(
                    {
                        "type": "duplicate_policy",
                        "loc": ("policies", index, "id"),
                        "msg": f"重复的结转策略 ID: {policy.id}",
                        "input": policy.id,
                    }
                )
            seen.add(policy.id)

        if duplicates:
            # 已存在重复定义，引用校验意义不大，直接返回
            raise RequestValidationErrors(errors)

        # 引用中未知策略
        for r_index, request in enumerate(self.requests):
            for p_index, policy_id in enumerate(request.policy_refs):
                if policy_id not in seen:
                    errors.append(
                        {
                            "type": "unknown_policy_ref",
                            "loc": ("requests", r_index, "policy_refs", p_index),
                            "msg": f"请求 {request.id!r} 引用了未定义的结转策略: {policy_id}",
                            "input": policy_id,
                        }
                    )

        # 重复请求 ID（便于调用方定位模拟结果）
        seen_req: set[str] = set()
        for r_index, request in enumerate(self.requests):
            if request.id in seen_req:
                errors.append(
                    {
                        "type": "duplicate_request",
                        "loc": ("requests", r_index, "id"),
                        "msg": f"重复的请求 ID: {request.id}",
                        "input": request.id,
                    }
                )
            seen_req.add(request.id)

        if errors:
            raise RequestValidationErrors(errors)
        return self


class PeriodRange(BaseModel):
    """周期区间：编号 index 与半开时间区间 [start, end)。"""

    index: Annotated[int, Field(description="周期编号 = floor((timestamp - anchor) / period_seconds)）")]
    start: Annotated[float, Field(description="周期起点（含）= anchor + index · period_seconds")]
    end: Annotated[float, Field(description="周期终点（不含）= start + period_seconds")]


class RolloverBalances(BaseModel):
    """某策略在一次请求判定时两类余额的扣减前后视图。"""

    base: Annotated[
        BucketBalance, Field(description="本期基础额度余额（每周期初重置为 allowance）")
    ]
    carry: Annotated[
        BucketBalance, Field(description="结转余额（上期未用基础额度结转而来，跨期即失效）")
    ]


class RolloverPolicyView(BaseModel):
    """单个请求对某个引用策略的判定视图。"""

    id: str
    period: Annotated[int, Field(description="判定时刻所在周期的编号")]
    period_window: Annotated[PeriodRange, Field(description="所在周期的编号与起止时间")]
    allowance: Annotated[float, Field(description="该策略每周期的基础额度")]
    carry_cap: Annotated[float, Field(description="该策略的结转上限")]
    balances: Annotated[RolloverBalances, Field(description="两类余额扣减前后快照")]


class RolloverRejection(BaseModel):
    """拒绝原因：两类余额合计不足的首个引用策略。"""

    policy_id: Annotated[str, Field(description="首个余额不足的策略 ID（按 policy_refs 顺序）")]
    reason: Annotated[
        Literal["insufficient_balance"],
        Field(description="拒绝原因：本期基础额度与结转余额合计不足以支付 cost"),
    ]
    period: Annotated[int, Field(description="判定时刻所在周期的编号")]
    period_window: Annotated[PeriodRange, Field(description="所在周期的编号与起止时间")]
    allowance: Annotated[float, Field(description="该策略每周期的基础额度")]
    carry_cap: Annotated[float, Field(description="该策略的结转上限")]
    cost: Annotated[float, Field(description="本次请求的消耗量")]
    base_available: Annotated[float, Field(description="判定时本期基础额度余额")]
    carry_available: Annotated[float, Field(description="判定时结转余额")]
    available: Annotated[float, Field(description="判定时两类余额合计 = base + carry")]
    shortage: Annotated[float, Field(description="缺口 = cost - available")]


class RolloverSimResult(BaseModel):
    """单项请求的周期结转试算结果。"""

    request_id: str
    index: Annotated[int, Field(description="该请求在原始 requests 数组中的下标")]
    timestamp: float
    cost: float
    allowed: bool
    policies: Annotated[
        dict[str, RolloverPolicyView],
        Field(description="各引用策略的判定视图，按 policy_refs 顺序"),
    ]
    rejected_by: Annotated[
        str | None, Field(default=None, description="首个余额不足的策略 ID；放行时为 null")
    ] = None
    rejection: Annotated[
        RolloverRejection | None, Field(default=None, description="拒绝原因明细；放行时为 null")
    ] = None


class RolloverPolicyFinalState(BaseModel):
    """单个策略在模拟结束时刻（最后一个请求时间戳）的终态。"""

    id: str
    anchor: Annotated[float, Field(description="周期对齐锚点时刻（秒）")]
    period_seconds: Annotated[float, Field(description="周期长度（秒）")]
    allowance: Annotated[float, Field(description="每周期基础额度")]
    carry_cap: Annotated[float, Field(description="结转上限")]
    period: Annotated[int, Field(description="终态观察时刻所在周期的编号")]
    period_window: Annotated[PeriodRange, Field(description="所在周期的编号与起止时间")]
    base: Annotated[float, Field(description="本期基础额度终态余额")]
    carry: Annotated[float, Field(description="结转余额终态")]
    available: Annotated[float, Field(description="两类余额终态合计 = base + carry")]


class RolloverSimulateResponse(BaseModel):
    allowed_count: int
    rejected_count: int
    results: Annotated[
        list[RolloverSimResult], Field(description="逐项请求结果，按原始输入索引升序")
    ]
    policies: Annotated[
        list[RolloverPolicyFinalState],
        Field(description="各策略在最后请求时刻的终态，按 policies 输入顺序"),
    ]


# ---------- 可借贷连续补充桶试算 (/debt-simulate) ----------


class DebtBucketSpec(BucketSpec):
    """可借贷令牌桶定义：在 /simulate 桶字段上增加 debt_limit。

    debt_limit: 允许累计借入（债务余额）的上限，必须为非负有限数值；
    0 表示完全不允许借债，此时语义与 /simulate 相同。
    """

    debt_limit: Annotated[
        float,
        Field(description="累计债务上限，必须为非负有限数值；0 表示不允许借债"),
    ]

    @field_validator("debt_limit", mode="before")
    @classmethod
    def _validate_debt_limit(cls, value):
        if not _is_finite_number(value):
            raise ValueError("debt_limit 必须是有限数值")
        if value < 0:
            raise ValueError("debt_limit 必须为非负数")
        return value


class DebtSimulateRequest(BaseModel):
    """POST /debt-simulate 请求体。"""

    model_config = ConfigDict(extra="forbid")

    buckets: Annotated[list[DebtBucketSpec], Field(min_length=1, description="可借贷令牌桶集合")]
    requests: Annotated[
        list[SimRequest], Field(min_length=1, description="按时间戳模拟的请求序列（可乱序输入）")
    ]

    @model_validator(mode="after")
    def _cross_validate(self):
        errors = _simulate_cross_validation_errors(self.buckets, self.requests)
        if errors:
            raise RequestValidationErrors(errors)
        return self


class DebtBucketBalance(BaseModel):
    """链上单个桶在某次请求判定时的余额/债务视图。"""

    balance: Annotated[BucketBalance, Field(description="可用余额（补充偿还债务后的余额）扣减前后")]
    debt: Annotated[BucketBalance, Field(description="累计债务在偿还/新增借贷前后的数额")]


class DebtRejection(BaseModel):
    """拒绝原因：链上首个无法在借贷上限内满足成本的桶。"""

    bucket_id: Annotated[str, Field(description="首个无法满足成本的桶 ID（按 chain 顺序）")]
    reason: Annotated[
        Literal["cost_exceeds_limit", "debt_limit_exceeded"],
        Field(
            description=(
                "拒绝原因：cost_exceeds_limit=成本超过 capacity + debt_limit，"
                "任何时刻都不可达；debt_limit_exceeded=成本未超总上限，"
                "但当前可用余额加剩余借贷额度不足"
            )
        ),
    ]
    capacity: Annotated[float, Field(description="该桶容量")]
    debt_limit: Annotated[float, Field(description="该桶累计债务上限")]
    cost: Annotated[float, Field(description="本次请求的消耗量")]
    balance_available: Annotated[
        float, Field(description="判定时刻（补充偿还债务之后）的可用余额")
    ]
    debt_before: Annotated[float, Field(description="判定时刻偿还债务之后的累计债务")]
    available: Annotated[
        float, Field(description="判定时刻可支付总额 = 可用余额 + 剩余借贷额度")
    ]
    remaining_credit: Annotated[
        float, Field(description="剩余可借额度 = debt_limit - debt_before")
    ]
    shortage: Annotated[float, Field(description="缺口 = cost - available")]


class DebtSimResult(BaseModel):
    """单项请求的可借贷试算结果。"""

    request_id: str
    index: Annotated[int, Field(description="该请求在原始 requests 数组中的下标")]
    timestamp: float
    cost: float
    allowed: bool
    buckets: Annotated[
        dict[str, DebtBucketBalance],
        Field(description="链上各桶判定时刻可用余额与累计债务的扣减前后快照，按 chain 顺序"),
    ]
    rejected_by: Annotated[
        str | None, Field(default=None, description="首个无法满足成本的桶 ID；放行时为 null")
    ] = None
    rejection: Annotated[
        DebtRejection | None, Field(default=None, description="拒绝原因明细；放行时为 null")
    ] = None


class DebtBucketFinalState(BaseModel):
    """单个可借贷桶在模拟结束时刻（最后一个请求时间戳）的终态快照。"""

    id: str
    capacity: Annotated[float, Field(description="桶容量（令牌上限）")]
    refill_rate: Annotated[float, Field(description="令牌补充速率（令牌/秒）")]
    debt_limit: Annotated[float, Field(description="累计债务上限")]
    balance: Annotated[float, Field(description="终态可用余额")]
    debt: Annotated[float, Field(description="终态累计债务")]
    available: Annotated[
        float, Field(description="终态可立即支付总额 = balance + (debt_limit - debt)")
    ]


class DebtSimulateResponse(BaseModel):
    allowed_count: int
    rejected_count: int
    results: Annotated[
        list[DebtSimResult], Field(description="逐项请求结果，按原始输入索引升序")
    ]
    final_buckets: Annotated[
        list[DebtBucketFinalState],
        Field(description="各桶在最后请求时刻的终态快照，按 buckets 输入顺序"),
    ]


# ---------- 可借贷桶情景差异试算 (/debt-diff) ----------


class DebtDiffScenario(BaseModel):
    """情景差异试算中的单边输入（baseline / candidate），格式与 /debt-simulate 相同。

    本模型**不**自带跨字段校验器：嵌套在 DebtDiffRequest 中时由父模型的
    校验器统一收集错误（保留 baseline./candidate. 定位前缀），避免
    Pydantic 把嵌套校验器抛出的异常提升到父级根节点。
    """

    model_config = ConfigDict(extra="forbid")

    buckets: Annotated[list[DebtBucketSpec], Field(min_length=1, description="可借贷令牌桶集合")]
    requests: Annotated[
        list[SimRequest], Field(min_length=1, description="按时间戳模拟的请求序列（可乱序输入）")
    ]


class DebtDiffRequest(BaseModel):
    """POST /debt-diff 请求体：baseline 与 candidate 两组同构的可借贷桶情景。"""

    model_config = ConfigDict(extra="forbid")

    baseline: Annotated[DebtDiffScenario, Field(description="基线情景（差值以 candidate - baseline 计算）")]
    candidate: Annotated[DebtDiffScenario, Field(description="候选情景")]

    @model_validator(mode="after")
    def _cross_validate(self):
        errors: list[dict] = []
        baseline = self.baseline
        candidate = self.candidate

        # 结构性对齐校验先于引用解析：桶集合不一致时，单边的未知引用
        # 多半只是缺桶的连锁噪声，应优先报告对齐错误本身。
        # 先检查各情景内部是否存在重复桶定义（这会让集合比较失去意义）。
        set_differs = False
        for side_name, scenario in (
            ("baseline", baseline),
            ("candidate", candidate),
        ):
            seen: set[str] = set()
            for b_index, bucket in enumerate(scenario.buckets):
                if bucket.id in seen:
                    errors.append(
                        {
                            "type": "duplicate_bucket",
                            "loc": (side_name, "buckets", b_index, "id"),
                            "msg": f"重复的桶 ID: {bucket.id}",
                            "input": bucket.id,
                        }
                    )
                seen.add(bucket.id)
        if errors:
            raise RequestValidationErrors(errors)

        # 请求数量必须一致，否则无法按原始索引对齐
        if len(baseline.requests) != len(candidate.requests):
            errors.append(
                {
                    "type": "debt_diff_request_count_mismatch",
                    "loc": ("candidate", "requests"),
                    "msg": (
                        "baseline 与 candidate 的请求数量不一致："
                        f"{len(baseline.requests)} != {len(candidate.requests)}"
                    ),
                    "input": {
                        "baseline_count": len(baseline.requests),
                        "candidate_count": len(candidate.requests),
                    },
                }
            )
            raise RequestValidationErrors(errors)

        # 按原始索引逐对检查请求 ID 与时间戳（不依赖内部时间排序）
        for i, (req_b, req_c) in enumerate(
            zip(baseline.requests, candidate.requests)
        ):
            if req_b.id != req_c.id:
                errors.append(
                    {
                        "type": "debt_diff_request_id_mismatch",
                        "loc": ("candidate", "requests", i, "id"),
                        "msg": (
                            f"请求索引 {i} 的 ID 不一致："
                            f"baseline={req_b.id!r}，candidate={req_c.id!r}"
                        ),
                        "input": {
                            "index": i,
                            "baseline_id": req_b.id,
                            "candidate_id": req_c.id,
                        },
                    }
                )
            if abs(float(req_b.timestamp) - float(req_c.timestamp)) > EPS:
                errors.append(
                    {
                        "type": "debt_diff_timestamp_mismatch",
                        "loc": ("candidate", "requests", i, "timestamp"),
                        "msg": (
                            f"请求 {req_b.id!r}（索引 {i}）的时间戳不一致："
                            f"baseline={float(req_b.timestamp)}，"
                            f"candidate={float(req_c.timestamp)}"
                        ),
                        "input": {
                            "index": i,
                            "request_id": req_b.id,
                            "baseline_timestamp": float(req_b.timestamp),
                            "candidate_timestamp": float(req_c.timestamp),
                        },
                    }
                )

        # 桶集合必须同构（ID 与顺序一致），桶参数（capacity/refill_rate/
        # initial/debt_limit）允许不同——参数差异正是候选情景的比较内容。
        bucket_ids_b = [b.id for b in baseline.buckets]
        bucket_ids_c = [b.id for b in candidate.buckets]
        if bucket_ids_b != bucket_ids_c:
            set_differs = True
            errors.append(
                {
                    "type": "debt_diff_bucket_mismatch",
                    "loc": ("candidate", "buckets"),
                    "msg": (
                        "baseline 与 candidate 的桶集合（ID 及顺序）不一致："
                        f"{bucket_ids_b} != {bucket_ids_c}"
                    ),
                    "input": {
                        "baseline_bucket_ids": bucket_ids_b,
                        "candidate_bucket_ids": bucket_ids_c,
                    },
                }
            )

        if set_differs:
            # 桶集合不同构时，链引用的未知桶多为缺桶的连锁结果，不再追加
            raise RequestValidationErrors(errors)

        # 桶集合同构：对两个情景各自执行与 /debt-simulate 相同的跨字段
        # 校验（未知引用、重复请求 ID 等），错误 loc 带情景前缀以便定位。
        for side_name in ("baseline", "candidate"):
            scenario = getattr(self, side_name)
            for error in _simulate_cross_validation_errors(
                scenario.buckets, scenario.requests
            ):
                errors.append({**error, "loc": (side_name, *error["loc"])})

        if errors:
            raise RequestValidationErrors(errors)
        return self


class DebtDiffVerdict(BaseModel):
    """单边情景对单个请求的放行结论。"""

    allowed: Annotated[bool, Field(description="该情景下此请求是否放行")]
    rejected_by: Annotated[
        str | None, Field(description="该情景下导致拒绝的第一个桶 ID；放行时为 null")
    ] = None
    reason: Annotated[
        str | None,
        Field(
            description=(
                "该情景下的拒绝原因（cost_exceeds_limit / "
                "debt_limit_exceeded）；放行时为 null"
            )
        ),
    ] = None


class DebtDiffBucketAfter(BaseModel):
    """单边情景在某次请求处理之后单个桶的余额/债务。"""

    balance: Annotated[float, Field(description="该请求处理之后的可用余额（被拒绝时保持判定前状态）")]
    debt: Annotated[float, Field(description="该请求处理之后的累计债务（被拒绝时保持判定前状态）")]
    available: Annotated[
        float, Field(description="该请求处理之后可立即支付总额 = balance + (debt_limit - debt)")
    ]


class DebtDiffSideStep(BaseModel):
    """单个请求在某一情景下的结论与全部桶的处理后快照。"""

    verdict: Annotated[DebtDiffVerdict, Field(description="该情景下的放行/拒绝结论")]
    rejected_by: Annotated[
        str | None, Field(description="= verdict.rejected_by（便于直接读取）")
    ] = None
    reason: Annotated[str | None, Field(description="= verdict.reason（便于直接读取）")] = None
    buckets: Annotated[
        dict[str, DebtDiffBucketAfter],
        Field(description="该请求处理之后全部桶的余额/债务/可支付总额，按桶 ID"),
    ]


class DebtDiffBucketDelta(BaseModel):
    """候选相对基线在单个桶上的差值（candidate - baseline）。"""

    balance_delta: Annotated[float, Field(description="可用余额差值 = candidate - baseline")]
    debt_delta: Annotated[float, Field(description="累计债务差值 = candidate - baseline")]
    available_delta: Annotated[
        float, Field(description="可立即支付总额差值 = candidate - baseline")
    ]


class DebtDiffStep(BaseModel):
    """按原始索引对齐后的单请求逐项对比。"""

    request_id: str
    index: Annotated[int, Field(description="该请求在原始 requests 数组中的下标")]
    timestamp: float
    cost_baseline: Annotated[float, Field(description="基线情景中该请求的 cost")]
    cost_candidate: Annotated[float, Field(description="候选情景中该请求的 cost")]
    diverged: Annotated[
        bool,
        Field(
            description=(
                "allowed/rejected_by/reason 是否存在分歧；两侧均拒绝但"
                "rejected_by 或原因不同同样为 true"
            )
        ),
    ]
    baseline: Annotated[DebtDiffSideStep, Field(description="基线情景回放结论与桶快照")]
    candidate: Annotated[DebtDiffSideStep, Field(description="候选情景回放结论与桶快照")]
    bucket_deltas: Annotated[
        dict[str, DebtDiffBucketDelta],
        Field(description="该请求处理之后各桶 candidate - baseline 的余额/债务差值"),
    ]


class DebtDiffFirstDivergence(BaseModel):
    """首个 allowed/拒绝原因分歧的定位与两侧结论。"""

    request_id: Annotated[str, Field(description="首个分歧请求的 ID")]
    index: Annotated[int, Field(description="首个分歧请求在原始 requests 数组中的下标")]
    timestamp: Annotated[float, Field(description="首个分歧请求的时间戳")]
    baseline: Annotated[DebtDiffVerdict, Field(description="基线情景下该请求的结论")]
    candidate: Annotated[DebtDiffVerdict, Field(description="候选情景下该请求的结论")]


class DebtDiffFinalBucket(BaseModel):
    """单个桶在两个情景终态（最后请求时刻）的快照与差值。"""

    id: str
    baseline: Annotated[
        DebtBucketFinalState, Field(description="基线情景终态快照（余额/债务/可支付总额）")
    ]
    candidate: Annotated[
        DebtBucketFinalState, Field(description="候选情景终态快照（余额/债务/可支付总额）")
    ]
    delta: Annotated[
        DebtDiffBucketDelta, Field(description="终态 candidate - baseline 的余额/债务差值")
    ]


class DebtDiffResponse(BaseModel):
    request_count: Annotated[int, Field(description="对齐后的请求数量（两情景相等）")]
    divergence_count: Annotated[
        int, Field(description="存在 allowed/拒绝原因分歧的请求数量")
    ]
    first_divergence: Annotated[
        DebtDiffFirstDivergence | None,
        Field(description="按原始索引首个分歧；两侧结论完全一致时为 null"),
    ]
    results: Annotated[
        list[DebtDiffStep],
        Field(description="逐请求对比，按原始输入索引升序（含各桶逐步差值）"),
    ]
    final_buckets: Annotated[
        list[DebtDiffFinalBucket],
        Field(description="各桶在最后请求时刻的双情景终态快照与差值，按桶输入顺序"),
    ]
