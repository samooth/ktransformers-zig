#!/usr/bin/env bash
# check_staged_claims.sh — fail a commit that sweeps another session's work.
#
# WHY THIS EXISTS. The protocol in AGENTS.md says "stage by filename". That advice
# is wrong and has now been demonstrated three times on one file:
#
#   - `bench/cpp_vs_zig/measure.zig` was staged by one session while another had
#     half-finished edits in it. Twice. Once the committed tool did not COMPILE,
#     because the swept-in work referenced a parser field that did not exist yet.
#   - `src/runtime/worker_pool.zig` was truncated to zero lines by a write on a
#     full disk, and the same file was corrupted a second time by a partial write.
#   - A `TODO.md` write-up and a test suite each landed inside another session's
#     commit.
#
# A `// CLAIM:` marker cannot prevent any of that. The mechanism is
# `git add <file>`, which stages the file's ENTIRE working state — so whoever
# commits first sweeps in whatever the other session had in flight, including work
# its author knows does not compile yet. "Stage by filename" was the wrong
# instinct; per-stream files are the real fix, and this is the enforcement that
# makes the rule stick in the meantime.
#
# WHAT IT DOES. For every staged file, collect the claim handles it carries. Every
# one of them must be declared in the commit message as `Handles: <handle>`. A
# commit that sweeps another session's claim therefore cannot land silently — the
# message has to say whose work went in with it.
#
# USAGE
#   tools/check_staged_claims.sh              # check the staged index
#   git commit -m "... Handles: bench-harness, fp8-e5m2"
#
# EXIT STATUS
#   0  every staged claim is declared in the message
#   1  an undeclared claim is staged
#   2  nothing staged (nothing to check)

set -uo pipefail

staged="$(git diff --cached --name-only 2>/dev/null)"
[ -z "$staged" ] && { echo "check_staged_claims: nothing staged, nothing to check"; exit 2; }

# The message being COMMITTED is in .git/COMMIT_EDITMSG (git writes it for both
# -m and the editor). `git log -1` is only the PREVIOUS commit's message, so
# reaching for it first means this script never sees the message under test and
# refuses every commit. Read COMMIT_EDITMSG first; fall back to HEAD's message
# when running against an already-made commit.
if [ -r .git/COMMIT_EDITMSG ]; then
    msg="$(cat .git/COMMIT_EDITMSG)"
else
    msg="$(git log -1 --format=%B 2>/dev/null || true)"
fi
msg_lc="$(printf '%s' "$msg" | tr '[:upper:]' '[:lower:]')"

undeclared=""
declared_any=0

while read -r f; do
    [ -z "$f" ] && continue
    [ -f "$f" ] || continue
    # Claim markers live at the top of a source file; grep the staged blob, not
    # the working tree, because the working tree may already have moved on.
    handles="$(git show ":$f" 2>/dev/null | grep -oE 'CLAIM:[[:space:]]*[a-zA-Z0-9._-]+' | awk '{print $2}' | sort -u)"
    [ -z "$handles" ] && continue
    for h in $handles; do
        declared_any=1
        if printf '%s' "$msg_lc" | grep -q "handles:.*$h"; then
            continue
        fi
        undeclared="$undeclared $f($h)"
    done
done <<EOF
$staged
EOF

if [ -z "$undeclared" ]; then
    if [ "$declared_any" -eq 1 ]; then
        echo "check_staged_claims: OK — every staged claim is declared in the message."
    else
        echo "check_staged_claims: OK — no claim markers in the staged files."
    fi
    exit 0
fi

cat >&2 <<EOF

check_staged_claims: FAIL — this commit would sweep another session's work.

  $undeclared

Those files carry someone else's CLAIM marker, and \`git add <file>\` stages their
ENTIRE working state — including edits their author has not finished and may know
do not compile. Three times on one file that has left HEAD broken.

Fix one of these, not the marker:
  1. Unstage, and stage only your own hunks (git add -p / git diff | apply).
  2. Unstage entirely and wait for the other session to land.
  3. If their work genuinely belongs in this commit, say so in the message:
       Handles: <handle>
     which is what this script requires. Example: "Handles: bench-harness".

A claim marker documents intent. It cannot enforce it; this script can.

EOF
exit 1