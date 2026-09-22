#!/usr/bin/env python3
"""WCAG AA contrast check for every palette in exec-shell.css (same rules as contrast-check.mjs).
Usage: python3 contrast_check.py path/to/exec-shell.css -> exit 1 on any failure."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def say(msg: str) -> None:
    sys.stdout.write(msg + "\n")


RULES: list[tuple[list[str], list[str], float]] = [
    (["text", "muted", "primary", "secondary"], ["bg", "panel", "panel-2"], 4.5),
    (["ok", "warn", "danger"], ["bg", "panel", "panel-2"], 4.5),
    ([f"chart-{i}" for i in range(1, 9)], ["bg", "panel"], 3.0),
    (["on-accent"], ["primary", "secondary"], 4.5),
    (["border"], ["bg"], 1.2),
]


def lum(hex_colour: str) -> float:
    def chan(c: int) -> float:
        v = c / 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (chan(int(hex_colour[i : i + 2], 16)) for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ratio(a: str, b: str) -> float:
    x, y = lum(a), lum(b)
    return (max(x, y) + 0.05) / (min(x, y) + 0.05)


def themes_from(css: str) -> dict[str, dict[str, dict[str, str]]]:
    start = css.index("@media (prefers-color-scheme: light)")
    end = css.index("\n}\n", start) + 3
    parts = {"dark": css[:start] + css[end:], "light": css[start:end]}
    out: dict[str, dict[str, dict[str, str]]] = {}
    for scheme, text in parts.items():
        for m in re.finditer(r'\[data-theme="([a-z]+)"\]\s*\{([^}]*)\}', text):
            tokens = {
                t.group(1): t.group(2).lower()
                for t in re.finditer(r"--([a-z0-9-]+):\s*(#[0-9a-fA-F]{6})", m.group(2))
            }
            if tokens:
                out.setdefault(m.group(1), {})[scheme] = tokens
    return out


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("exec-shell.css")
    themes = themes_from(path.read_text(encoding="utf-8"))
    failures = checked = 0
    for name, schemes in themes.items():
        for scheme, t in schemes.items():
            for fgs, bgs, minimum in RULES:
                for fg in fgs:
                    for bg in bgs:
                        if fg not in t or bg not in t:
                            say(f"{name}/{scheme}: missing token --{fg} or --{bg}")
                            failures += 1
                            continue
                        r = ratio(t[fg], t[bg])
                        checked += 1
                        if r < minimum:
                            say(
                                f"FAIL {name}/{scheme} --{fg} {t[fg]} on --{bg} {t[bg]}: "
                                f"{r:.2f} < {minimum}"
                            )
                            failures += 1
    say(f"contrast-check: {len(themes)} themes x 2 schemes, {checked} pairs, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
