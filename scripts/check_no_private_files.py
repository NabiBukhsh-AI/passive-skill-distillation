"""Block private specification files from entering this public repository.

Build playbook Section 2 gives this check as a shell one-liner and suggests making it a
pre-commit hook rather than a thing you remember to run. This is that hook.

The failure mode it prevents is unrecoverable: a specification pushed to a public remote
cannot be un-pushed in any meaningful sense.

No third-party dependencies, so it runs even when the project environment is broken.
"""

from __future__ import annotations

import re
import subprocess
import sys

# Patterns matched against staged paths, case-insensitively.
BLOCKED = [
    (re.compile(r"(^|/)\.spec(/|$)"), "the .spec link into the private files repo"),
    (re.compile(r"PASSIVE_SKILL_DISTILLATION_SPEC\.md$", re.I), "the engineering specification"),
    (re.compile(r"BUILD_PLAYBOOK\.md$", re.I), "the build playbook"),
    (re.compile(r"(^|/)docs/spec(/|$)"), "the docs/spec private directory"),
    (re.compile(r"\.private\.md$", re.I), "a file marked private"),
]


def staged_paths() -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    # Check the staged set, not the filenames pre-commit passes, so the hook is correct
    # even when invoked with --all-files or with no arguments.
    paths = staged_paths()
    findings: list[str] = []
    for path in paths:
        for pattern, description in BLOCKED:
            if pattern.search(path):
                findings.append(f"  {path}  ({description})")
                break

    if findings:
        print("STOP: private file staged for a public repository:", file=sys.stderr)
        print("\n".join(findings), file=sys.stderr)
        print(
            "\nUnstage it. The specification and playbook live in the separate private "
            "repository and reach this one through the gitignored .spec link only.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
