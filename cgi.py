"""
Compatibility shim for the deprecated stdlib `cgi` module.

Python 3.13 removed `cgi` (PEP 594), but some third-party dependencies
such as `httpx` used by `googletrans` still import it for `parse_header`.
This lightweight module re-implements `parse_header` using the `email`
package, which continues to be available.
"""

from __future__ import annotations

from email.message import Message
from typing import Dict, Tuple

__all__ = ["parse_header"]


def parse_header(line: str) -> Tuple[str, Dict[str, str]]:
    """
    Parse a Content-Type style header into a (value, params) tuple.

    Mirrors the public API of the removed stdlib helper so libraries that
    still import `cgi.parse_header` keep working on Python 3.13+.
    """

    if not isinstance(line, str):
        line = line.decode("utf-8", "surrogateescape")

    msg = Message()
    msg["content-type"] = line
    params = msg.get_params()

    if not params:
        return line.strip(), {}

    value = params[0][0] or line.strip()
    param_pairs = params[1:]

    parsed_params: Dict[str, str] = {}
    for key, val in param_pairs:
        if key is None:
            continue
        parsed_params[key.lower()] = "" if val is None else val.strip()

    return value, parsed_params

