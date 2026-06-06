from collections import deque
from datetime import datetime

MAX_LOG_LINES = 2000

_logs = deque(maxlen=MAX_LOG_LINES)


def log(
    component,
    *parts,
    level="INFO"
):

    message = " ".join(
        str(p)
        for p in parts
        if p is not None
    ).strip()

    if not message:
        message = "-"

    timestamp = datetime.now().strftime(
        "%H:%M:%S"
    )

    entry = {
        "time": timestamp,
        "timestamp": datetime.now().isoformat(),
        "component": component,
        "level": level,
        "message": message
    }

    _logs.append(entry)

    print(
        f"[{timestamp}] "
        f"[{level}] "
        f"[{component}] "
        f"{message}"
    )


def get_logs():

    return list(_logs)


def clear_logs():

    _logs.clear()