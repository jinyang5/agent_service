"""R28:eval_rag_prompt 判分加固的回归测试(不调真实 LLM)。"""
import importlib.util
import pathlib


def _load_script():
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "eval_rag_prompt.py"
    spec = importlib.util.spec_from_file_location("eval_rag_prompt_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


script = _load_script()


def _delivery_sample() -> dict:
    return {
        "id": "delivery-timing",
        "evidence": ["发货后配送时效以平台实际处理为准,具体以物流信息为准。"],
        "expect_citation": [1],
        "expect_any": ["以物流信息为准"],
        "expect_all": [],
        "forbid_patterns": [r"\d+\s*天(内)?(到|送达)", r"保证.{0,6}送达"],
    }


def test_negated_commitment_is_not_miss():
    """「无法保证 3 天内送达」是在否定承诺,不能判成违规。"""
    passed, misses = script._check(
        _delivery_sample(), "无法保证3天内送达,具体以物流信息为准[1]。")
    assert passed, misses


def test_positive_commitment_is_still_miss():
    passed, misses = script._check(
        _delivery_sample(), "一般3天内送达,具体以物流信息为准[1]。")
    assert not passed
    assert any("违规承诺" in m for m in misses)


def test_bounded_number_is_not_treated_as_negation():
    """「不超过 3 天」是限定而非否定,仍属于实质时效承诺。"""
    passed, misses = script._check(
        _delivery_sample(), "不超过3天送达,具体以物流信息为准[1]。")
    assert not passed
    assert any("违规承诺" in m for m in misses)


def test_future_is_not_treated_as_negation():
    """「未来 3 天内送达」里的「未」不是否定,不能豁免违规承诺。"""
    passed, misses = script._check(
        _delivery_sample(), "未来3天内送达,具体以物流信息为准[1]。")
    assert not passed
    assert any("违规承诺" in m for m in misses)


def test_expected_citation_out_of_range_is_miss():
    sample = {"id": "cite-bad", "evidence": ["唯一证据"],
              "expect_citation": [2], "expect_any": [], "expect_all": [],
              "forbid_patterns": []}
    passed, misses = script._check(sample, "[2] 唯一证据")
    assert not passed
    assert any("越界" in m for m in misses)


def test_answer_citation_out_of_range_is_miss():
    sample = {"id": "cite-ok", "evidence": ["e1", "e2"],
              "expect_citation": [1], "expect_any": [], "expect_all": [],
              "forbid_patterns": []}
    passed, misses = script._check(sample, "[1] e1 [3] 编造证据")
    assert not passed
    assert any("不存在的证据编号" in m and "[3]" in m for m in misses)


def test_refusal_with_substantive_number_is_miss():
    sample = {"id": "refusal", "evidence": ["e1", "e2"],
              "expect_citation": [], "expect_any": ["没有查到"],
              "expect_all": ["人工客服"], "forbid_patterns": []}
    passed, misses = script._check(
        sample, "没有查到相关信息,建议联系人工客服,通常3天到账。")
    assert not passed
    assert any("实质数值" in m for m in misses)


def test_refusal_with_chinese_number_is_miss():
    """中文数字同样是实质数值,不能绕过拒答检查。"""
    sample = {"id": "refusal", "evidence": ["e1", "e2"],
              "expect_citation": [], "expect_any": ["没有查到"],
              "expect_all": ["人工客服"], "forbid_patterns": []}
    passed, misses = script._check(
        sample, "没有查到相关信息,建议联系人工客服,通常三天到账。")
    assert not passed
    assert any("实质数值" in m for m in misses)


def test_plain_refusal_passes():
    sample = {"id": "refusal", "evidence": ["e1", "e2"],
              "expect_citation": [], "expect_any": ["没有查到"],
              "expect_all": ["人工客服"], "forbid_patterns": []}
    passed, misses = script._check(
        sample, "没有查到相关信息,建议联系人工客服进一步确认。")
    assert passed, misses
