from sb.llm.models import PreprocessingConfig


def remove_solidity_comments(source: str) -> str:
    """Remove Solidity comments without changing character or newline positions."""
    result = list(source)
    state = "code"
    quote = ""
    index = 0
    while index < len(source):
        char = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            if char in {'"', "'"}:
                state, quote = "string", char
            elif char == "/" and following == "/":
                result[index] = result[index + 1] = " "
                state = "line_comment"
                index += 1
            elif char == "/" and following == "*":
                result[index] = result[index + 1] = " "
                state = "block_comment"
                index += 1
        elif state == "string":
            if char == "\\":
                state = "escape"
            elif char == quote:
                state = "code"
        elif state == "escape":
            state = "string"
        elif state == "line_comment":
            if char in "\r\n":
                state = "code"
            else:
                result[index] = " "
        elif state == "block_comment":
            if char == "*" and following == "/":
                result[index] = result[index + 1] = " "
                state = "code"
                index += 1
            elif char not in "\r\n":
                result[index] = " "
        index += 1
    return "".join(result)


def add_line_numbers(source: str) -> str:
    lines = source.splitlines(keepends=True)
    if not lines:
        return ""
    width = max(3, len(str(len(lines))))
    return "".join(f"{number:0{width}d} | {line}" for number, line in enumerate(lines, 1))


def preprocess(source: str, config: PreprocessingConfig) -> str:
    processed = remove_solidity_comments(source) if config.remove_comments else source
    return add_line_numbers(processed) if config.add_line_numbers else processed
