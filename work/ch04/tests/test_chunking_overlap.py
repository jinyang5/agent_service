from app.kb import chunking


def test_overlap_is_whole_trailing_sentence():
    chunks = ["前面很多内容。中间一句话。最后收尾句。", "下一块正文。"]
    out = chunking.apply_sentence_overlap(chunks, overlap=6)
    assert out[0] == chunks[0]
    # 末6字符恰为「最后收尾句。」,是完整句 → 整句作为重叠前缀
    assert out[1] == "最后收尾句。下一块正文。"


def test_overlap_never_starts_mid_sentence():
    chunks = ["这是一个非常非常非常长的句子没有中间标点结尾才有。", "新块。"]
    out = chunking.apply_sentence_overlap(chunks, overlap=5)
    # overlap=5 放不下整句,但不能留半句 → 宁可整句(优先不留半截话)
    assert out[1].startswith("这是") or out[1] == "新块。"
