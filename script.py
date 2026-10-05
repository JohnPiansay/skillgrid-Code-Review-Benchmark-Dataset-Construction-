from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple
from urllib.parse import urlparse


@dataclass(frozen=True)
class BenchmarkProblem:
    problem_id: str
    title: str
    prompt: str
    reference: Callable[..., Any]
    tests: Tuple[Tuple[Tuple[Any, ...], Any], ...]


# ---------------------------- Reference implementations ----------------------------

def p001_parse_jsonl_counts(text: str) -> Dict[str, int]:
    """Count valid records by their string-valued `type`; ignore blank lines."""
    counts: Dict[str, int] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        kind = record.get("type")
        if isinstance(kind, str) and kind:
            counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


def p002_merge_intervals(intervals: Sequence[Sequence[int]]) -> List[List[int]]:
    """Merge overlapping or directly touching closed integer intervals."""
    if not intervals:
        return []
    normalized = sorted(
        (int(a), int(b)) if a <= b else (int(b), int(a))
        for a, b in intervals
    )
    merged: List[List[int]] = [[normalized[0][0], normalized[0][1]]]
    for start, end in normalized[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


_ALLOWED_HOSTS = {"example.com", "api.example.com"}


def p003_safe_https_url(url: str) -> bool:
    """Allow only HTTPS URLs to an exact approved host, with no userinfo."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme.lower() != "https":
        return False
    if parsed.username is not None or parsed.password is not None:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    return host in _ALLOWED_HOSTS


def p004_token_bucket_allow(
    requests: Sequence[Tuple[float, int]],
    capacity: int = 5,
    refill_per_second: float = 2.0,
) -> List[bool]:
    """Pure token-bucket simulation. Each tuple is (timestamp, cost)."""
    if capacity <= 0 or refill_per_second < 0:
        raise ValueError("invalid bucket configuration")

    tokens = float(capacity)
    previous = requests[0][0] if requests else 0.0
    result: List[bool] = []

    for timestamp, cost in requests:
        if cost <= 0:
            raise ValueError("cost must be positive")
        if timestamp < previous:
            raise ValueError("timestamps must be nondecreasing")

        tokens = min(
            float(capacity),
            tokens + (timestamp - previous) * refill_per_second,
        )
        previous = timestamp

        if tokens >= cost:
            tokens -= cost
            result.append(True)
        else:
            result.append(False)

    return result


def p005_csv_safe_export(
    rows: Sequence[Dict[str, str]], fields: Sequence[str]
) -> str:
    """Export CSV while neutralizing spreadsheet formula injection."""
    out = io.StringIO(newline="")
    writer = csv.DictWriter(
        out,
        fieldnames=list(fields),
        extrasaction="ignore",
    )
    writer.writeheader()

    for row in rows:
        safe: Dict[str, str] = {}
        for field in fields:
            value = "" if row.get(field) is None else str(row.get(field))
            if value.startswith(("=", "+", "-", "@")):
                value = "'" + value
            safe[field] = value
        writer.writerow(safe)

    return out.getvalue()


# ---------------------------- Five new benchmark problems ----------------------------

PROBLEMS = (
    BenchmarkProblem(
        "P001",
        "JSONL event counter",
        (
            "Count valid JSON Lines records by non-empty string `type`. Ignore blank lines, "
            "malformed JSON, non-object JSON, and records whose type is missing, non-string, "
            "or empty."
        ),
        p001_parse_jsonl_counts,
        (
            (('{"type":"click"}\n{"type":"view"}\n{"type":"click"}\n',),
             {"click": 2, "view": 1}),
            (('\nnot-json\n42\n{"type":""}\n{"type":7}\n{"type":"ok"}\n',),
             {"ok": 1}),
            (('{"type":"a"}\n{"type":"a"}\n{"x":1}\n',), {"a": 2}),
        ),
    ),
    BenchmarkProblem(
        "P002",
        "Merge integer intervals",
        (
            "Normalize reversed endpoints, then merge overlapping or directly touching "
            "closed intervals. Return intervals sorted by start."
        ),
        p002_merge_intervals,
        (
            (([[1, 3], [2, 5], [7, 8]],), [[1, 5], [7, 8]]),
            (([[5, 2], [8, 9], [10, 10]],), [[2, 5], [8, 10]]),
            (([],), []),
        ),
    ),
    BenchmarkProblem(
        "P003",
        "HTTPS host allowlist",
        (
            "Return true only for HTTPS URLs whose exact hostname is example.com or "
            "api.example.com. Reject userinfo, subdomains outside the allowlist, HTTP, "
            "malformed URLs, and lookalike hosts."
        ),
        p003_safe_https_url,
        (
            (("https://example.com/path",), True),
            (("https://api.example.com/v1",), True),
            (("http://example.com",), False),
            (("https://evil-example.com",), False),
            (("https://example.com.evil.test",), False),
            (("https://user:pass@example.com/",), False),
        ),
    ),
    BenchmarkProblem(
        "P004",
        "Token-bucket request limiter",
        (
            "Simulate a token bucket with configurable capacity and refill rate. Each request "
            "is (timestamp, cost). Refill before each request; approve iff enough tokens remain. "
            "Reject invalid configuration, non-positive costs, and decreasing timestamps."
        ),
        p004_token_bucket_allow,
        (
            (([(0.0, 3), (0.0, 3), (1.0, 2)],), [True, False, True]),
            (([(0.0, 5), (2.0, 4), (2.0, 2)],), [True, True, False]),
            (([],), []),
        ),
    ),
    BenchmarkProblem(
        "P005",
        "Formula-safe CSV export",
        (
            "Export selected fields as CSV. Prefix values beginning with =, +, -, or @ with "
            "an apostrophe to prevent spreadsheet formula injection. Preserve commas, quotes, "
            "and newlines using standard CSV quoting."
        ),
        p005_csv_safe_export,
        (
            (([{"name": "Alice", "note": "hello"}], ["name", "note"]),
             'name,note\r\nAlice,hello\r\n'),
            (([{"name": "=1+1", "note": "a,b"}], ["name", "note"]),
             "name,note\r\n'=1+1,\"a,b\"\r\n"),
            (([{"name": "Bob", "note": 'say "hi"\nnext'}], ["name", "note"]),
             'name,note\r\nBob,"say ""hi""\nnext"\r\n'),
        ),
    ),
)


def grade(
    candidate: Callable[..., Any],
    problem: BenchmarkProblem,
) -> Tuple[bool, List[str]]:
    """Run a candidate against every public benchmark case."""
    failures: List[str] = []

    for index, (args, expected) in enumerate(problem.tests, 1):
        try:
            actual = candidate(*args)
        except Exception as exc:
            failures.append(
                f"case {index}: exception {type(exc).__name__}: {exc}"
            )
            continue

        if actual != expected:
            failures.append(
                f"case {index}: expected {expected!r}, got {actual!r}"
            )

    return not failures, failures


def self_verify() -> Dict[str, Any]:
    """Verify reference implementations before they are accepted as benchmark gold."""
    report: Dict[str, Any] = {}

    for problem in PROBLEMS:
        passed, failures = grade(problem.reference, problem)
        report[problem.problem_id] = {
            "title": problem.title,
            "passed": passed,
            "failures": failures,
        }

    return report


if __name__ == "__main__":
    verification = self_verify()
    print(json.dumps(verification, indent=2))

    if not all(item["passed"] for item in verification.values()):
        raise SystemExit("Reference verification failed.")
