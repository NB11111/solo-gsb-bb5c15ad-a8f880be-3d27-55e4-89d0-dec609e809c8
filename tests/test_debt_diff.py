"""可借贷桶情景差异试算（/debt-diff）引擎与 HTTP 层测试：pytest 运行。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.errors import RequestValidationErrors
from app.main import app
from app.models import DebtDiffRequest
from app.debt import debt_simulate
from app.debt_diff import debt_diff
from app.models import DebtSimulateRequest

client = TestClient(app)

# 手工回放的标准情景（见 test_first_divergence_and_deltas）：
# A：capacity=10, refill_rate=1, initial=2（baseline debt_limit=5 / candidate=2）
# B：capacity=4,  refill_rate=0, initial=0（两侧 debt_limit=4）
#
# r1 t=0 cost=4 chain [A,B]
# r2 t=0 cost=3 chain [A]      baseline 放行（借到债务 5）；candidate 借贷额度不足
# r3 t=3 cost=2 chain [A]
# r4 t=3 cost=1 chain [A,B]    两侧均因 B 拒绝

_BUCKETS_BASE = [
    {"id": "A", "capacity": 10, "refill_rate": 1, "initial": 2, "debt_limit": 5},
    {"id": "B", "capacity": 4, "refill_rate": 0, "initial": 0, "debt_limit": 4},
]
_BUCKETS_CAND = [
    {"id": "A", "capacity": 10, "refill_rate": 1, "initial": 2, "debt_limit": 2},
    {"id": "B", "capacity": 4, "refill_rate": 0, "initial": 0, "debt_limit": 4},
]
_REQUESTS = [
    {"id": "r1", "timestamp": 0, "cost": 4, "chain": ["A", "B"]},
    {"id": "r2", "timestamp": 0, "cost": 3, "chain": ["A"]},
    {"id": "r3", "timestamp": 3, "cost": 2, "chain": ["A"]},
    {"id": "r4", "timestamp": 3, "cost": 1, "chain": ["A", "B"]},
]


def _run(baseline, candidate):
    return debt_diff(
        DebtDiffRequest.model_validate(
            {"baseline": baseline, "candidate": candidate}
        )
    )


def _payload():
    return {
        "baseline": {"buckets": _BUCKETS_BASE, "requests": _REQUESTS},
        "candidate": {"buckets": _BUCKETS_CAND, "requests": _REQUESTS},
    }


# ---------- 分歧定位与差值 ----------

def test_first_divergence_and_deltas():
    resp = _run(
        {"buckets": _BUCKETS_BASE, "requests": _REQUESTS},
        {"buckets": _BUCKETS_CAND, "requests": _REQUESTS},
    )

    assert resp.request_count == 4
    assert resp.divergence_count == 1

    allowed_b = [r.baseline.verdict.allowed for r in resp.results]
    allowed_c = [r.candidate.verdict.allowed for r in resp.results]
    assert allowed_b == [True, True, True, False]
    assert allowed_c == [True, False, True, False]
    assert [r.diverged for r in resp.results] == [False, True, False, False]

    first = resp.first_divergence
    assert first is not None
    assert (first.request_id, first.index, first.timestamp) == ("r2", 1, 0.0)
    assert first.baseline.allowed is True
    assert first.baseline.rejected_by is None and first.baseline.reason is None
    assert first.candidate.allowed is False
    assert first.candidate.rejected_by == "A"
    assert first.candidate.reason == "debt_limit_exceeded"


def test_step_bucket_snapshots_include_all_buckets_every_step():
    resp = _run(
        {"buckets": _BUCKETS_BASE, "requests": _REQUESTS},
        {"buckets": _BUCKETS_CAND, "requests": _REQUESTS},
    )
    for step in resp.results:
        # 即使请求链只引用 A，B（以及任何链外桶）也应出现在两侧快照中
        assert set(step.baseline.buckets) == {"A", "B"}
        assert set(step.candidate.buckets) == {"A", "B"}
        assert set(step.bucket_deltas) == {"A", "B"}

    # r1 两侧一致：A 余额 2→0 借债 2；B 全额借债 4
    r1 = resp.results[0]
    assert (r1.baseline.buckets["A"].balance, r1.baseline.buckets["A"].debt) == (0, 2)
    assert (r1.candidate.buckets["B"].balance, r1.candidate.buckets["B"].debt) == (0, 4)
    assert r1.bucket_deltas["A"].debt_delta == 0
    assert r1.bucket_deltas["B"].balance_delta == 0

    # r2：baseline 放行后 A 债务=5；candidate 拒绝，A 债务保持 2
    r2 = resp.results[1]
    assert (r2.baseline.buckets["A"].balance, r2.baseline.buckets["A"].debt) == (0, 5)
    assert (r2.candidate.buckets["A"].balance, r2.candidate.buckets["A"].debt) == (0, 2)
    assert r2.bucket_deltas["A"].debt_delta == -3
    assert r2.bucket_deltas["A"].balance_delta == 0
    # 两侧 A 此刻可立即支付均为 0（额度各自借满）
    assert r2.baseline.buckets["A"].available == 0
    assert r2.candidate.buckets["A"].available == 0
    assert r2.bucket_deltas["A"].available_delta == 0

    # r3：baseline 还债 3 后借 2 → 债务 4；candidate 还债 2 后借 1 → 债务 1
    r3 = resp.results[2]
    assert (r3.baseline.buckets["A"].balance, r3.baseline.buckets["A"].debt) == (0, 4)
    assert (r3.candidate.buckets["A"].balance, r3.candidate.buckets["A"].debt) == (0, 1)
    assert r3.bucket_deltas["A"].debt_delta == -3

    # r4 两侧均因 B 原子拒绝 → 状态与 r3 完全一致
    r4 = resp.results[3]
    assert r4.diverged is False
    assert r4.baseline.rejected_by == "B" and r4.candidate.rejected_by == "B"
    assert r4.baseline.reason == r4.candidate.reason == "debt_limit_exceeded"
    assert r4.bucket_deltas["A"].debt_delta == -3
    # 被拒绝桶两侧 after == before（原子性）
    assert (
        r4.candidate.buckets["A"].balance,
        r4.candidate.buckets["A"].debt,
    ) == (r3.candidate.buckets["A"].balance, r3.candidate.buckets["A"].debt)


def test_final_snapshots_and_deltas():
    resp = _run(
        {"buckets": _BUCKETS_BASE, "requests": _REQUESTS},
        {"buckets": _BUCKETS_CAND, "requests": _REQUESTS},
    )
    finals = {b.id: b for b in resp.final_buckets}

    fb, fc, fd = finals["A"].baseline, finals["A"].candidate, finals["A"].delta
    assert (fb.balance, fb.debt, fb.available) == (0, 4, 1)
    assert (fc.balance, fc.debt, fc.available) == (0, 1, 1)
    # 终态快照保留两侧各自的桶参数
    assert fb.debt_limit == 5 and fc.debt_limit == 2
    assert (fd.balance_delta, fd.debt_delta, fd.available_delta) == (0, -3, 0)

    bb, bc, bd = finals["B"].baseline, finals["B"].candidate, finals["B"].delta
    assert (bb.balance, bb.debt, bb.available) == (0, 4, 0)
    assert (bc.balance, bc.debt, bc.available) == (0, 4, 0)
    assert (bd.balance_delta, bd.debt_delta, bd.available_delta) == (0, 0, 0)


def test_unreferenced_bucket_is_advanced_and_compared_every_step():
    extra_b = _BUCKETS_BASE + [
        {"id": "IDLE", "capacity": 10, "refill_rate": 2, "initial": 0, "debt_limit": 3}
    ]
    extra_c = _BUCKETS_CAND + [
        {"id": "IDLE", "capacity": 10, "refill_rate": 1, "initial": 0, "debt_limit": 3}
    ]
    resp = _run(
        {"buckets": extra_b, "requests": _REQUESTS},
        {"buckets": extra_c, "requests": _REQUESTS},
    )
    # t=3 时链外桶按各自速率连续补充：baseline 6、candidate 3（均未触顶）
    idle_final = {b.id: b for b in resp.final_buckets}["IDLE"]
    assert idle_final.baseline.balance == 6
    assert idle_final.candidate.balance == 3
    assert idle_final.delta.balance_delta == -3
    # 逐步快照中 IDLE 同样按事件时刻推进
    assert resp.results[0].baseline.buckets["IDLE"].balance == 0
    assert resp.results[2].baseline.buckets["IDLE"].balance == 6
    assert resp.results[2].candidate.buckets["IDLE"].balance == 3


# ---------- 同情景一致性 ----------

def test_identical_scenarios_have_no_divergence_and_zero_deltas():
    payload = {"buckets": _BUCKETS_BASE, "requests": _REQUESTS}
    resp = _run(payload, payload)
    assert resp.divergence_count == 0 and resp.first_divergence is None
    assert all(not step.diverged for step in resp.results)
    for step in resp.results:
        for delta in step.bucket_deltas.values():
            assert (
                delta.balance_delta == 0
                and delta.debt_delta == 0
                and delta.available_delta == 0
            )
    for final in resp.final_buckets:
        assert final.delta.balance_delta == 0
        assert final.delta.debt_delta == 0
        assert final.delta.available_delta == 0


def test_diff_side_verdicts_match_debt_simulate_outputs():
    # 两侧各自的结论/快照必须与直接调用 /debt-simulate 引擎完全一致
    base_req = DebtSimulateRequest.model_validate(
        {"buckets": _BUCKETS_BASE, "requests": _REQUESTS}
    )
    cand_req = DebtSimulateRequest.model_validate(
        {"buckets": _BUCKETS_CAND, "requests": _REQUESTS}
    )
    sim_b = debt_simulate(base_req)
    sim_c = debt_simulate(cand_req)
    diff = _run(
        {"buckets": _BUCKETS_BASE, "requests": _REQUESTS},
        {"buckets": _BUCKETS_CAND, "requests": _REQUESTS},
    )
    for sb, sc, d in zip(sim_b.results, sim_c.results, diff.results):
        assert d.baseline.verdict.allowed == sb.allowed
        assert d.baseline.verdict.rejected_by == sb.rejected_by
        assert d.candidate.verdict.allowed == sc.allowed
        assert d.candidate.verdict.rejected_by == sc.rejected_by
        # diff 的处理后快照与 /debt-simulate 链上视图的 after 一致
        for bucket_id, view in sb.buckets.items():
            assert d.baseline.buckets[bucket_id].balance == view.balance.after
            assert d.baseline.buckets[bucket_id].debt == view.debt.after


# ---------- 分歧类型 ----------

def test_reason_divergence_cost_exceeds_limit_vs_debt_limit_exceeded():
    # cost=13：baseline（cap 10 + limit 5 = 15）非永久不可达但当前付不起
    # → debt_limit_exceeded；candidate（cap 10 + limit 2 = 12 < 13）
    # → cost_exceeds_limit。rejected_by 相同但原因不同，仍计为分歧。
    buckets_b = [
        {"id": "B", "capacity": 10, "refill_rate": 0, "initial": 0, "debt_limit": 5}
    ]
    buckets_c = [
        {"id": "B", "capacity": 10, "refill_rate": 0, "initial": 0, "debt_limit": 2}
    ]
    requests = [{"id": "r", "timestamp": 0, "cost": 13, "chain": ["B"]}]
    resp = _run(
        {"buckets": buckets_b, "requests": requests},
        {"buckets": buckets_c, "requests": requests},
    )
    assert resp.divergence_count == 1
    first = resp.first_divergence
    assert first.baseline.reason == "debt_limit_exceeded"
    assert first.candidate.reason == "cost_exceeds_limit"
    assert first.baseline.rejected_by == first.candidate.rejected_by == "B"


def test_both_rejected_with_different_rejected_bucket_is_divergence():
    # 同一请求两侧都拒绝，但限制桶不同
    requests = [{"id": "r", "timestamp": 0, "cost": 5, "chain": ["A", "B"]}]
    base = [
        {"id": "A", "capacity": 10, "refill_rate": 0, "initial": 10, "debt_limit": 0},
        {"id": "B", "capacity": 4, "refill_rate": 0, "initial": 0, "debt_limit": 0},
    ]
    cand = [
        {"id": "A", "capacity": 4, "refill_rate": 0, "initial": 0, "debt_limit": 0},
        {"id": "B", "capacity": 4, "refill_rate": 0, "initial": 4, "debt_limit": 0},
    ]
    resp = _run(
        {"buckets": base, "requests": requests},
        {"buckets": cand, "requests": requests},
    )
    assert resp.results[0].diverged is True
    assert resp.first_divergence.baseline.rejected_by == "B"
    assert resp.first_divergence.candidate.rejected_by == "A"


def test_different_cost_per_side_is_allowed_and_aligned_by_index():
    # 仅 ID/时间戳要求一致；cost 允许不同（链也可不同）
    buckets = [
        {"id": "B", "capacity": 10, "refill_rate": 0, "initial": 5, "debt_limit": 0}
    ]
    req_b = [{"id": "r", "timestamp": 0, "cost": 5, "chain": ["B"]}]
    req_c = [{"id": "r", "timestamp": 0, "cost": 6, "chain": ["B"]}]
    resp = _run(
        {"buckets": buckets, "requests": req_b},
        {"buckets": buckets, "requests": req_c},
    )
    step = resp.results[0]
    assert step.cost_baseline == 5 and step.cost_candidate == 6
    assert step.baseline.verdict.allowed is True
    assert step.candidate.verdict.allowed is False
    assert step.diverged is True


def test_first_divergence_uses_original_index_not_processing_order():
    # 乱序输入：首个分歧在原始索引 0（t=5），而非先处理的 t=0 请求
    buckets_b = [
        {"id": "B", "capacity": 10, "refill_rate": 0, "initial": 10, "debt_limit": 0}
    ]
    buckets_c = [
        {"id": "B", "capacity": 10, "refill_rate": 0, "initial": 3, "debt_limit": 0}
    ]
    requests = [
        {"id": "late", "timestamp": 5, "cost": 4, "chain": ["B"]},
        {"id": "early", "timestamp": 0, "cost": 4, "chain": ["B"]},
    ]
    resp = _run(
        {"buckets": buckets_b, "requests": requests},
        {"buckets": buckets_c, "requests": requests},
    )
    # baseline：两笔都放行；candidate：early（t=0）拒绝，late 补不到（rate 0）也拒绝
    assert resp.first_divergence.index == 0
    assert resp.first_divergence.request_id == "late"
    assert [r.index for r in resp.results] == [0, 1]


# ---------- 对齐校验（422，不回放） ----------

def _validate_error(payload):
    with pytest.raises(RequestValidationErrors) as exc:
        DebtDiffRequest.model_validate(payload)
    return exc.value.errors


def test_request_count_mismatch_rejected():
    errors = _validate_error(
        {
            "baseline": {"buckets": _BUCKETS_BASE, "requests": _REQUESTS},
            "candidate": {
                "buckets": _BUCKETS_CAND,
                "requests": _REQUESTS[:3],
            },
        }
    )
    assert any(
        e["type"] == "debt_diff_request_count_mismatch" for e in errors
    )


def test_request_id_mismatch_rejected_with_index_location():
    cand_requests = [{**r} for r in _REQUESTS]
    cand_requests[2] = {**cand_requests[2], "id": "RX"}
    errors = _validate_error(
        {
            "baseline": {"buckets": _BUCKETS_BASE, "requests": _REQUESTS},
            "candidate": {"buckets": _BUCKETS_CAND, "requests": cand_requests},
        }
    )
    match = [e for e in errors if e["type"] == "debt_diff_request_id_mismatch"]
    assert len(match) == 1
    assert match[0]["loc"] == ("candidate", "requests", 2, "id")
    assert match[0]["input"] == {
        "index": 2,
        "baseline_id": "r3",
        "candidate_id": "RX",
    }


def test_timestamp_mismatch_rejected_with_index_location():
    cand_requests = [{**r} for r in _REQUESTS]
    cand_requests[1] = {**cand_requests[1], "timestamp": 0.5}
    errors = _validate_error(
        {
            "baseline": {"buckets": _BUCKETS_BASE, "requests": _REQUESTS},
            "candidate": {"buckets": _BUCKETS_CAND, "requests": cand_requests},
        }
    )
    match = [e for e in errors if e["type"] == "debt_diff_timestamp_mismatch"]
    assert len(match) == 1
    assert match[0]["loc"] == ("candidate", "requests", 1, "timestamp")


def test_timestamp_equal_within_eps_is_accepted():
    cand_requests = [{**r} for r in _REQUESTS]
    cand_requests[1] = {**cand_requests[1], "timestamp": 1e-10}
    DebtDiffRequest.model_validate(
        {
            "baseline": {"buckets": _BUCKETS_BASE, "requests": _REQUESTS},
            "candidate": {"buckets": _BUCKETS_CAND, "requests": cand_requests},
        }
    )


def test_timestamp_beyond_eps_is_rejected():
    cand_requests = [{**r} for r in _REQUESTS]
    cand_requests[1] = {**cand_requests[1], "timestamp": 1e-8}
    errors = _validate_error(
        {
            "baseline": {"buckets": _BUCKETS_BASE, "requests": _REQUESTS},
            "candidate": {"buckets": _BUCKETS_CAND, "requests": cand_requests},
        }
    )
    assert any(e["type"] == "debt_diff_timestamp_mismatch" for e in errors)


def test_bucket_set_mismatch_rejected():
    cand_buckets = [_BUCKETS_CAND[0]]  # 缺少 B
    errors = _validate_error(
        {
            "baseline": {"buckets": _BUCKETS_BASE, "requests": _REQUESTS},
            "candidate": {"buckets": cand_buckets, "requests": _REQUESTS},
        }
    )
    match = [e for e in errors if e["type"] == "debt_diff_bucket_mismatch"]
    assert len(match) == 1
    assert match[0]["loc"] == ("candidate", "buckets")
    assert match[0]["input"]["baseline_bucket_ids"] == ["A", "B"]
    assert match[0]["input"]["candidate_bucket_ids"] == ["A"]


def test_bucket_order_mismatch_rejected_even_with_same_ids():
    cand_buckets = list(reversed(_BUCKETS_CAND))
    errors = _validate_error(
        {
            "baseline": {"buckets": _BUCKETS_BASE, "requests": _REQUESTS},
            "candidate": {"buckets": cand_buckets, "requests": _REQUESTS},
        }
    )
    assert any(e["type"] == "debt_diff_bucket_mismatch" for e in errors)


def test_bucket_parameter_difference_is_allowed():
    # 同 ID 同顺序，capacity/debt_limit 不同：合法
    DebtDiffRequest.model_validate(_payload())


def test_single_side_inherits_debt_simulate_cross_validation_with_prefix():
    # baseline 内部未知桶引用：错误 loc 必须带 baseline 前缀；
    # candidate 用同 ID/时间戳的合法请求以排除对齐噪声
    bad_requests = [
        {"id": "r1", "timestamp": 0, "cost": 4, "chain": ["A", "GHOST"]}
    ]
    good_requests = [
        {"id": "r1", "timestamp": 0, "cost": 4, "chain": ["A"]}
    ]
    errors = _validate_error(
        {
            "baseline": {"buckets": _BUCKETS_BASE, "requests": bad_requests},
            "candidate": {"buckets": _BUCKETS_CAND, "requests": good_requests},
        }
    )
    unknown = [e for e in errors if e["type"] == "unknown_bucket_ref"]
    assert len(unknown) == 1
    assert unknown[0]["loc"] == ("baseline", "requests", 0, "chain", 1)


def test_single_side_duplicate_request_id_rejected():
    dup = [_REQUESTS[0], _REQUESTS[0]]
    errors = _validate_error(
        {
            "baseline": {"buckets": _BUCKETS_BASE, "requests": dup},
            "candidate": {"buckets": _BUCKETS_CAND, "requests": dup},
        }
    )
    assert any(e["type"] == "duplicate_request" for e in errors)


def test_single_side_bad_debt_limit_rejected_by_pydantic():
    from pydantic import ValidationError

    bad_buckets = [
        {**_BUCKETS_BASE[0], "debt_limit": -1},
        _BUCKETS_BASE[1],
    ]
    with pytest.raises(ValidationError):
        DebtDiffRequest.model_validate(
            {
                "baseline": {"buckets": bad_buckets, "requests": _REQUESTS},
                "candidate": {"buckets": _BUCKETS_CAND, "requests": _REQUESTS},
            }
        )


def test_extra_field_rejected_by_pydantic():
    from pydantic import ValidationError

    payload = _payload()
    payload["unexpected"] = 1
    with pytest.raises(ValidationError):
        DebtDiffRequest.model_validate(payload)


def test_alignment_error_aborts_before_replay_and_keeps_stateless():
    # 时间戳不一致直接 422；接口本身无状态，重复调用结果一致
    good = client.post("/debt-diff", json=_payload())
    assert good.status_code == 200
    again = client.post("/debt-diff", json=_payload())
    assert again.json() == good.json()

    cand_requests = [{**r} for r in _REQUESTS]
    cand_requests[0] = {**cand_requests[0], "timestamp": 99}
    bad = client.post(
        "/debt-diff",
        json={
            "baseline": {"buckets": _BUCKETS_BASE, "requests": _REQUESTS},
            "candidate": {"buckets": _BUCKETS_CAND, "requests": cand_requests},
        },
    )
    assert bad.status_code == 422
    assert any(
        e["type"] == "debt_diff_timestamp_mismatch"
        for e in bad.json()["detail"]
    )
    # 非法请求后再发同一合法请求，结果不变（不写入状态）
    after = client.post("/debt-diff", json=_payload())
    assert after.json() == good.json()


# ---------- HTTP ----------

def test_http_debt_diff_ok():
    response = client.post("/debt-diff", json=_payload())
    assert response.status_code == 200
    data = response.json()
    assert data["request_count"] == 4
    assert data["divergence_count"] == 1
    first = data["first_divergence"]
    assert first["request_id"] == "r2" and first["index"] == 1
    assert first["baseline"]["allowed"] is True
    assert first["candidate"] == {
        "allowed": False,
        "rejected_by": "A",
        "reason": "debt_limit_exceeded",
    }
    r2 = data["results"][1]
    assert r2["diverged"] is True
    assert r2["bucket_deltas"]["A"] == {
        "balance_delta": 0.0,
        "debt_delta": -3.0,
        "available_delta": 0.0,
    }
    # 终态结构完整：两侧快照 + delta
    final_a = {b["id"]: b for b in data["final_buckets"]}["A"]
    assert final_a["baseline"] == {
        "id": "A",
        "capacity": 10.0,
        "refill_rate": 1.0,
        "debt_limit": 5.0,
        "balance": 0.0,
        "debt": 4.0,
        "available": 1.0,
    }
    assert final_a["candidate"]["debt"] == 1.0
    assert final_a["delta"] == {
        "balance_delta": 0.0,
        "debt_delta": -3.0,
        "available_delta": 0.0,
    }


def test_http_identical_scenarios():
    payload = {
        "baseline": {"buckets": _BUCKETS_BASE, "requests": _REQUESTS},
        "candidate": {"buckets": _BUCKETS_BASE, "requests": _REQUESTS},
    }
    data = client.post("/debt-diff", json=payload).json()
    assert data["divergence_count"] == 0
    assert data["first_divergence"] is None


def test_http_count_mismatch_is_422():
    response = client.post(
        "/debt-diff",
        json={
            "baseline": {"buckets": _BUCKETS_BASE, "requests": _REQUESTS},
            "candidate": {
                "buckets": _BUCKETS_CAND,
                "requests": _REQUESTS[:2],
            },
        },
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(
        e["type"] == "debt_diff_request_count_mismatch" for e in detail
    )


def test_http_request_id_mismatch_is_422():
    cand_requests = [{**r} for r in _REQUESTS]
    cand_requests[0]["id"] = "other"
    response = client.post(
        "/debt-diff",
        json={
            "baseline": {"buckets": _BUCKETS_BASE, "requests": _REQUESTS},
            "candidate": {
                "buckets": _BUCKETS_CAND,
                "requests": cand_requests,
            },
        },
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    match = [e for e in detail if e["type"] == "debt_diff_request_id_mismatch"]
    assert match and match[0]["loc"] == ["candidate", "requests", 0, "id"]


def test_http_bucket_mismatch_is_422():
    response = client.post(
        "/debt-diff",
        json={
            "baseline": {"buckets": _BUCKETS_BASE, "requests": _REQUESTS},
            "candidate": {
                "buckets": [{**_BUCKETS_CAND[0], "id": "Z"}],
                "requests": _REQUESTS,
            },
        },
    )
    assert response.status_code == 422
    assert any(
        e["type"] == "debt_diff_bucket_mismatch"
        for e in response.json()["detail"]
    )


def test_http_root_lists_debt_diff():
    root = client.get("/").json()
    assert root["debt_diff"] == "/debt-diff"


def test_http_openapi_documents_debt_diff():
    paths = client.get("/openapi.json").json()["paths"]
    assert "/debt-diff" in paths
