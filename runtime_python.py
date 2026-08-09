"""Python runtime policy shared by all supervised entrypoints."""

import platform
import sys


REQUIRED_MAJOR_MINOR = (3, 12)


def runtime_report(version_info=None):
    info = version_info or sys.version_info
    current = (int(info.major), int(info.minor))
    supported = current == REQUIRED_MAJOR_MINOR
    return {
        "supported": supported,
        "required": ".".join(map(str, REQUIRED_MAJOR_MINOR)),
        "current": f"{info.major}.{info.minor}.{info.micro}",
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
    }


def require_supported_python(service="ETH-bot"):
    report = runtime_report()
    if not report["supported"]:
        raise RuntimeError(
            f"{service} requires Python {report['required']}.x; "
            f"current={report['current']} executable={report['executable']}"
        )
    return report
