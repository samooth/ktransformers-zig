#!/usr/bin/env python3
"""verify_workflow.py — CI workflow gates that PyYAML will not give you for free.

WHY THIS EXISTS. `yaml.safe_load` silently accepts a DUPLICATE MAPPING KEY and
keeps the last one. A workflow step that accidentally carries two `run:` keys
therefore parses cleanly, prints as valid, and behaves differently in CI from how
it reads on screen. That is not hypothetical: the orphan-test step in wheels.yml
had two `run:` keys, so the detection block was silently discarded AND the
"Test suites" step above it lost its command entirely — while `yaml.safe_load`
reported the file as valid, twice.

So this checks the two things a parser cannot:

  1. no duplicate keys anywhere in the file
  2. every `- name:` step actually has a `run:` or `uses:`

EXIT: 0 clean, 1 a gate failed.

USAGE: python3 tools/verify_workflow.py [.github/workflows/*.yml]
"""
import sys
import glob
import yaml


class Strict(yaml.SafeLoader):
    """SafeLoader that raises on a duplicate mapping key instead of keeping the last."""


def _no_dupes(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.YAMLError(f"duplicate key {key!r} at line {key_node.start_mark.line + 1}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


Strict.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_dupes)


def check(path):
    problems = []
    with open(path) as fh:
        try:
            doc = yaml.load(fh, Loader=Strict)
        except yaml.YAMLError as exc:
            return [f"{path}: {exc}"]

    for job_name, job in (doc.get("jobs") or {}).items():
        for index, step in enumerate(job.get("steps") or []):
            name = step.get("name") or step.get("uses") or f"#{index}"
            if "run" not in step and "uses" not in step:
                problems.append(
                    f"{path}: job {job_name!r} step {name!r} has neither `run:` nor `uses:` "
                    f"— CI would fail or skip it"
                )
    return problems


def main():
    paths = sys.argv[1:] or sorted(glob.glob(".github/workflows/*.yml") + glob.glob(".github/workflows/*.yaml"))
    if not paths:
        print("verify_workflow: no workflow files found")
        return 1
    problems = []
    for path in paths:
        found = check(path)
        print(f"{path}: {'OK' if not found else str(len(found)) + ' problem(s)'}")
        problems.extend(found)
    for problem in problems:
        print(f"  FAIL {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
