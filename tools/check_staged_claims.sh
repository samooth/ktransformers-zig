#!/usr/bin/env bash
# check_staged_claims.sh — report (and optionally annotate) a commit that sweeps
# another session's work.
#
# WHY THIS EXISTS. The protocol in AGENTS.md used to say "stage by filename".
# That advice is wrong and this repo paid for it three times on one file:
#
#   - `bench/cpp_vs_zig/measure.zig` was staged by one session while another had
#     half-finished edits in it. Twice. Once the committed tool did not COMPILE,
#     because the swept-in work referenced a parser field that did not exist yet.
#   - `src/runtime/worker_pool.zig` was truncated to zero lines by a write on a
#     full disk, and corrupted a second time by a partial write.
#   - A `TODO.md` write-up and a test suite each landed inside another session's
#     commit.
#
# A `// CLAIM:` marker cannot prevent any of that. The mechanism is
# `git add <file>`, which stages the file's ENTIRE working state — so whoever
# commits first sweeps in whatever the other session had in flight, including work
# its author knows does not compile. The marker documents intent; this enforces it.
#
# WHAT IT DOES. For every staged file, collect the claim handles it carries. Each
# one must be declared in the commit message as `Handles: <handle>`. A commit that
# sweeps another session's claim therefore cannot land silently.
#
# USAGE
#   tools/check_staged_claims.sh                       # check the staged index
#   git commit -m "... Handles: bench-harness, fp8-e5m2"
#
#   --message-file PATH   read the message from PATH. A prepare-commit-msg hook
#                         receives the message file as its argument; git writes
#                         .git/COMMIT_EDITMSG for `-m` and for the editor, but
#                         reading that alone is wrong under a hook.
#   --annotate            print the warning to stdout and ALWAYS exit 0, so a hook
#                         can append it to the message instead of blocking. A
#                         blocking hook would break every session on the clone,
#                         since .git/hooks/ is per-repository, not per-user.
#
# EXIT STATUS (without --annotate)
#   0  every staged claim is declared in the message
#   1  an undeclared claim is staged
#   2  nothing staged

set -uo pipefail

msg_file=""
annotate=0

while [ $# -gt 0 ]; do
    case "$1" in
        --message-file) msg_file="${2:-}"; shift 2 ;;
        --annotate) annotate=1; shift ;;
        -h|--help) sed -n '2,38p' "$0"; exit 0 ;;
        *) printf 'check_staged_claims: unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

staged="$(git diff --cached --name-only 2>/dev/null)"
if [ -z "$staged" ]; then
    [ "$annotate" -eq 1 ] && exit 0
    echo "check_staged_claims: nothing staged, nothing to check"
    exit 2
fi

# The message being committed: an explicit path first (the hook's argument),
# then COMMIT_EDITMSG. `git log -1` is only the PREVIOUS commit's message, so
# reaching for it first means the check never sees the message under test and
# refuses every commit, including correct ones.
if [ -n "$msg_file" ] && [ -r "$msg_file" ]; then
    msg="$(cat "$msg_file")"
elif [ -r .git/COMMIT_EDITMSG ]; then
    msg="$(cat .git/COMMIT_EDITMSG)"
else
    msg="$(git log -1 --format=%B 2>/dev/null || true)"
fi
msg_lc="$(printf '%s' "$msg" | tr '[:upper:]' '[:lower:]')"

undeclared=""
declared_any=0

while read -r f; do
    [ -z "$f" ] && continue
    # ANCHORED, and that matters: this script's own prose contains the literal
    # "// CLAIM:" and so does AGENTS.md, so an unanchored match reports a phantom
    # sweep on every commit that merely *documents* the convention. A real marker
    # is `// CLAIM: <handle>` at the start of a line.
    handles="$(git show ":$f" 2>/dev/null \
        | grep -oE '^[[:space:]]*//+[[:space:]]*CLAIM:[[:space:]]*[a-zA-Z0-9._-]+' \
        | grep -oE 'CLAIM:[[:space:]]*[a-zA-Z0-9._-]+' \
        | awk '{print $2}' | sort -u)"
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
    [ "$annotate" -eq 1 ] && exit 0
    if [ "$declared_any" -eq 1 ]; then
        echo "check_staged_claims: OK — every staged claim is declared in the message."
    else
        echo "check_staged_claims: OK — no claim markers in the staged files."
    fi
    exit 0
fi

if [ "$annotate" -eq 1 ]; then
    cat <<EOF
--------------------------------------------------------------------
NOTE FROM check_staged_claims.sh (advisory — this did not block your commit):
it appears to sweep another session's work:
$undeclared
\`git add <file>\` stages a file's ENTIRE working state, including edits their
author has not finished. If that work belongs in this commit, add
    Handles: <handle>
to this message and the note will not reappear. If it does not, stage only
your own hunks (\`git add -p\`) — \`bench/cpp_vs_zig/measure.zig\` has been swept
this way three times, twice leaving HEAD not compiling.
--------------------------------------------------------------------
EOF
    exit 0
fi

cat >&2 <<EOF

check_staged_claims: FAIL — this commit would sweep another session's work.

  $undeclared

Those files carry someone else's CLAIM marker, and \`git add <file>\` stages their
ENTIRE working state — including edits their author has not finished and may know
do not compile.

Fix one of these, not the marker:
  1. Unstage, and stage only your own hunks (git add -p / git diff | apply).
  2. Unstage entirely and wait for the other session to land.
  3. If their work genuinely belongs in this commit, say so in the message:
       Handles: <handle>
     which is what this script requires. Example: "Handles: bench-harness".

A claim marker documents intent. It cannot enforce it; this script can.

EOF
exit 1