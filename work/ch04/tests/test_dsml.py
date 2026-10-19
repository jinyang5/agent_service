# tests/test_dsml.py
"""DSML 泄漏标记的识别与拦截。

背景:上游(DeepSeek V3.2/V4)在长上下文下偶发把工具调用写成正文,
标记有多种变体(全角双竖线 / 单竖线 / ASCII 管道;calls / function_calls / tool_calls;
有时缺开头的 wrapper)。生产库实测 16 条,收敛步泄漏率约 5.9%。
"""
import pytest

from app.core.dsml import DSML_FALLBACK, StreamGate, clean_answer, has_dsml, strip_dsml

# 生产库里真实出现过的形态(带前言)
REAL_LEAK = (
    "亲，帮您查一下退货政策的具体内容哈～ 🐾\n\n"
    '<｜｜DSML｜｜ calls>\n'
    '<｜｜DSML｜｜ invoke name="query_faq">\n'
    '<｜｜DSML｜｜ parameter name="keyword" string="true">退货</｜｜DSML｜｜ parameter>\n'
    "</｜｜DSML｜｜ invoke>\n"
    "</｜｜DSML｜｜ calls>"
)


@pytest.mark.parametrize("text", [
    REAL_LEAK,                                                     # 全角双竖线 + calls
    '<｜DSML｜ invoke name="query_faq">',                           # 单竖线
    '<|DSML| tool_calls>',                                         # ASCII 管道
    '<|DSML| function_calls>',                                     # V3.2 变体
    '<DSML｜ calls>',                                              # 缺前导竖线
    '｜｜DSML｜｜ calls>',                                          # 缺开头的 <
])
def test_has_dsml_covers_variants(text):
    assert has_dsml(text)


@pytest.mark.parametrize("text", [
    "您的订单正在运输中，预计明天送达。",
    "| 地区 | 首重(1kg) | 续重(每 kg) |\n|---|---|---|",          # markdown 表格
    "满 99 元包邮，偏远地区另计 <下单页面为准>",                    # 普通尖括号
    "运费 10 元 | 会员 9 折",                                      # 普通竖线
    "",
])
def test_has_dsml_no_false_positive(text):
    assert not has_dsml(text)


def test_strip_dsml_keeps_human_preamble():
    assert strip_dsml(REAL_LEAK) == "亲，帮您查一下退货政策的具体内容哈～ 🐾"


def test_strip_dsml_pure_markup_returns_empty():
    assert strip_dsml('<｜｜DSML｜｜ calls>\n<｜｜DSML｜｜ invoke name="query_faq">') == ""


def test_strip_dsml_leaves_normal_text_untouched():
    text = "您的订单已发货。\n\n| 地区 | 运费 |\n|---|---|\n| 江浙沪 | 免费 |"
    assert strip_dsml(text) == text


def test_clean_answer_falls_back_when_only_markup():
    assert clean_answer(REAL_LEAK) == "亲，帮您查一下退货政策的具体内容哈～ 🐾"
    # 纯标记(没有半句人话)→ 整句换成兜底话术,绝不把标记给用户
    assert clean_answer('<｜｜DSML｜｜ calls>\n</｜｜DSML｜｜ calls>') == DSML_FALLBACK


def test_clean_answer_passthrough_when_clean():
    assert clean_answer("满 99 元包邮。") == "满 99 元包邮。"


# ---------------- 流式闸门:一个字都不许漏给用户 ----------------

def _drain(gate, tokens):
    out = []
    for t in tokens:
        out += gate.feed(t)
    tail, leaked = gate.finish()
    return "".join(out + tail), leaked


def test_stream_gate_withholds_leak_inside_holdback():
    tokens = ["亲，帮您查一下～\n\n", "<｜｜DSML", "｜｜ calls>\n",
              '<｜｜DSML｜｜ invoke name="query_faq">']
    text, leaked = _drain(StreamGate(holdback=40), tokens)
    assert leaked is True
    assert text == ""                       # 一个字都没吐给用户


def test_stream_gate_streams_clean_answer_unchanged():
    tokens = list("您的订单正在运输中，预计明天送达。")
    text, leaked = _drain(StreamGate(holdback=5), tokens)
    assert leaked is False
    assert text == "您的订单正在运输中，预计明天送达。"


def test_stream_gate_late_marker_keeps_only_clean_prefix():
    tokens = ["满 99 元包邮，偏远地区另计。", "这是官方规则。",
              '<｜｜DSML｜｜ calls>\n', "</｜｜DSML｜｜ calls>"]
    text, leaked = _drain(StreamGate(holdback=8), tokens)
    assert leaked is True
    assert "DSML" not in text
    assert "满 99 元包邮" in text            # 已吐出的干净前缀不收回
