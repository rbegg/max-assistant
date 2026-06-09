
def log_banner(msgs: list[str]) -> str:
    max_length = max(len(msg) for msg in msgs)

    banner = "*" * (max_length + 4)
    lines = [banner]
    for msg in msgs:
        lines.append(f"* {msg:<{max_length}} *")
    lines.append(banner)

    return "\n".join([""] + lines)