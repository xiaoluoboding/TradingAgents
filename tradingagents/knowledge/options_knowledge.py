"""Lightweight retrieval for an Obsidian options knowledge graph."""

from pathlib import Path
import re


DEFAULT_OPTIONS_KNOWLEDGE_PATH = "/Users/robertshaw/GitHub/private/obsidian/期权知识图谱"
_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text) if len(token) > 1}


def retrieve_options_knowledge(symbol: str, root: str | Path | None = None, max_documents: int = 6, max_chars: int = 12000) -> str:
    """Retrieve relevant Markdown notes from an Obsidian vault, read-only."""
    vault = Path(root or DEFAULT_OPTIONS_KNOWLEDGE_PATH).expanduser()
    if not vault.is_dir():
        return f"Options knowledge graph unavailable: directory does not exist: {vault}"
    symbol_upper = symbol.strip().upper()
    query = _tokens(f"{symbol_upper} Delta Gamma Theta Vega Rho IV implied volatility expiration moneyness strategy risk")
    candidates = []
    for path in vault.rglob("*.md"):
        if any(part.startswith(".") for part in path.parts):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        tokens = _tokens(f"{path.stem} {path} {content}")
        score = len(query & tokens)
        if path.stem.upper() == symbol_upper:
            score += 100
        if path.parent.name in {"术语", "期权策略", "高级期权策略", "中级期权策略"}:
            score += 4
        if score:
            candidates.append((score, str(path.relative_to(vault)), content))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if not candidates:
        return f"No matching options knowledge notes found for {symbol_upper} in {vault}."
    sections = []
    used = 0
    for _, relative_path, content in candidates[:max_documents]:
        remaining = max_chars - used
        if remaining <= 0:
            break
        excerpt = content[:remaining].strip()
        sections.append(f"### Knowledge note: {relative_path}\n{excerpt}")
        used += len(excerpt)
    return "\n\n".join(sections)
