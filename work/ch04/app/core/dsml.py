"""DSML 泄漏标记:识别与拦截。

上游(DeepSeek V3.2/V4)在长上下文下偶发把「工具调用」写成正文 -- 完整格式良好,
但 `tool_calls` 为空,上游解析器没进工具模式(缺开头的 wrapper token)。标记有多个变体:
全角双竖线 `｜｜DSML｜｜`、单竖线 `｜DSML｜`、ASCII 管道 `|DSML|`;种类 `calls` /
`function_calls` / `tool_calls`;有时连开头的 `<` 都没有。

生产库实测:16 条泄漏消息 / 13 个会话;收敛步(工具查空后模型想重试)泄漏率约 5.9%。

本模块只做「认出来 + 剥干净 + 一个字都不许漏给用户」。**不**负责把泄漏的调用解析回来
重新执行 -- 那是下一步;在那之前,至少不能把机器标记当成回答喂给用户。
"""
import re

DSML_FALLBACK = "抱歉,刚才没答利索。麻烦再问一次,或者把问题说得更具体一点,我马上帮您查。"

# 只要出现 DSML 字样,且旁边有竖线(全角/半角)或前导尖括号,就认定是标记。
# 普通中文里几乎不会出现 "DSML" 这个词,所以误报面很窄;markdown 表格的 |---| 不含 DSML,不受影响。
_DSML_RE = re.compile(r"(?:<\s*)?(?:[|｜]{1,3}\s*DSML|DSML\s*[|｜]{1,3})", re.IGNORECASE)


def has_dsml(text: str) -> bool:
    """文本里是否含 DSML 工具调用标记(任意变体)。"""
    return bool(text) and _DSML_RE.search(text) is not None


def strip_dsml(text: str) -> str:
    """剥掉含标记的整行,保留模型说的人话。只剩标记时返回空串。

    按行剔除而不是正则替换整段:泄漏块每行都带标记,而且可能缺 wrapper、被截断,
    按行处理对残缺形态最稳;代价是「同一行里标记后面的人话」会一起丢掉(实测没有这种形态)。
    """
    if not text or not has_dsml(text):
        return text or ""
    kept = [line for line in text.splitlines() if not has_dsml(line)]
    return "\n".join(kept).strip()


def clean_answer(text: str) -> str:
    """给用户看的最终答案:剥掉标记;剥完没实质内容就换兜底话术。"""
    cleaned = strip_dsml(text)
    return cleaned if cleaned.strip() else DSML_FALLBACK


class StreamGate:
    """流式闸门:滚动留一小段尾巴,确认这段里没有标记再吐出去。

    留尾的作用是**拦住还没收全的标记**。一旦在缓冲区里看到标记,就停止输出并置 leaked,
    后续内容一律不吐 —— 调用方据此补一句兜底话术。
    已经吐出去的干净前缀不收回(用户看到的和落库的内容保持一致)。
    """

    def __init__(self, holdback: int = 40):
        if holdback < 0:
            raise ValueError("holdback 不能为负")
        self._holdback = holdback
        self._buf = ""
        self._leaked = False

    def feed(self, text: str) -> list[str]:
        """喂一段新文本,返回现在可以安全吐出的片段(可能为空)。"""
        if self._leaked or not text:
            return []
        self._buf += text
        if has_dsml(self._buf):
            self._leaked = True
            self._buf = ""
            return []
        if len(self._buf) <= self._holdback:
            return []
        cut = len(self._buf) - self._holdback
        out, self._buf = self._buf[:cut], self._buf[cut:]
        return [out]

    def finish(self) -> tuple[list[str], bool]:
        """收尾:返回 (剩余可吐片段, 是否发生过泄漏)。"""
        if self._leaked:
            return [], True
        tail, self._buf = self._buf, ""
        return ([tail] if tail else []), False
