import re, html

LOG_REGEX = re.compile(
    r"^nova-compute\.log\.\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2} "
    r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{3})? "
    r"\d+ (INFO|ERROR|DEBUG|WARNING|WARN|CRITICAL) "
    r"[a-zA-Z0-9_.]+ (?:\[[^\]]*\] )?.+"
)

def validate_line(line: str) -> str:
    stripped = line.strip()
    if not LOG_REGEX.match(stripped):
        raise ValueError(f"Invalid OpenStack log format: {stripped}")
    return html.escape(stripped)
