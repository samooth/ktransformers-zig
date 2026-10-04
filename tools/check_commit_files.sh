#!/usr/bin/env bash
# check_commit_files.sh — require every staged file to be NAMED in the commit
# message. BLOCKING.
#
# WHY THIS IS BLOCKING AND THE CLAIM CHECK IS NOT. Those two failures are not the
# same kind, and they should not get the same response:
#
#   - A staged file whose name is absent from the message is unambiguously a
#     mistake. There is no legitimate reading of "I am committing this file" that
#     does not mention the file. Block it.
#   - A staged file carrying another session's claim may be perfectly correct:
#     sometimes their work genuinely belongs in your commit. Warning about it four
#     times and being ignored four times is worth exactly as much as not warning,
#     which is what happened — the measurement session read the advisory note four
#     times in ten minutes and committed anyway. So the claim check stays advisory
#     (tools/check_staged_claims.sh) and only this one blocks.
#
# WHY IT EXISTS ANYWAY. Four sweeps in ten minutes took another session's staged
# work, split their attribution, and broke HEAD twice. The root cause is a commit
# taking what happened to be staged rather than what it wrote, and the fix is to
# make the commit say what it contains. Naming the files is also the only remedy
# for attribution that survives: a commit which lists its files records who was
# holding what.
#
# USAGE
#   tools/check_commit_files.sh [--message-file PATH]
#
#   Wired into .git/hooks/prepare-commit-msg, which receives the message file as
#   its first argument. Copy from tools/prepare-commit-msg after editing.
#
# EXIT STATUS
#   0  every staged file is named in the message
#   1  a staged file is not named — the commit is refused
#   2  nothing staged

set -uo pipefail

msg_file=""
while [ $# -gt 0 ]; do
    case "$1" in
        --message-file) msg_file="${2:-}"; shift 2 ;;
        -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
        *) printf 'check_commit_files: unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

staged="$(git diff --cached --name-only 2>/dev/null)"
[ -z "$staged" ] && { echo "check_commit_files: nothing staged, nothing to check"; exit 2; }

if [ -n "$msg_file" ] && [ -r "$msg_file" ]; then
    msg="$(cat "$msg_file")"
elif [ -r .git/COMMIT_EDITMSG ]; then
    msg="$(cat .git/COMMIT_EDITMSG)"
else
    msg="$(git log -1 --format=%B 2>/dev/null || true)"
fi
msg_lc="$(printf '%s' "$msg" | tr '[:upper:]' '[:lower:]')"

missing=""
while read -r f; do
    [ -z "$f" ] && continue
    base="${f##*/}"
    # Accept the basename or the full path — the message should be readable, not
    # a checklist of repository-relative paths.
    #
    # BOTH sides must be lowercased. The message is lowercased above (so the
    # check is case-insensitive by design), but the needle was NOT, which made any
    # staged file containing an uppercase letter — `TODO.md`, `AGENTS.md`,
    # `README.md` — impossible to satisfy: the lowercased message can never match
    # an uppercase needle. Every shared document in this repo was therefore
    # unnameable, and the check would refuse a commit that named all of them.
    # Cost a real commit here: the message did name TODO.md and was still rejected.
    base_lc="$(printf '%s' "$base" | tr '[:upper:]' '[:lower:]')"
    f_lc="$(printf '%s' "$f" | tr '[:upper:]' '[:lower:]')"
    if printf '%s' "$msg_lc" | grep -qF -- "$base_lc" || printf '%s' "$msg_lc" | grep -qF -- "$f_lc"; then
        continue
    fi
    missing="$missing $f"
done <<EOF
$staged
EOF

if [ -z "$missing" ]; then
    echo "check_commit_files: OK — all $(printf '%s\n' "$staged" | wc -l) staged file(s) named in the message."
    exit 0
fi

cat >&2 <<EOF

check_commit_files: REFUSING the commit — staged file(s) not named in the message:
$missing

Every staged file must appear in the commit message (basename is enough). Two
reasons, and the second is the one that keeps biting:

  1. It catches the accident — a habitually staged file riding along in a commit
     that is about something else.
  2. It records ATTRIBUTION. Four sweeps in ten minutes took another session's
     work and split their provenance, because \`git add <file>\` stages a file's
     entire working state and the message said nothing about which files were
     really in there.

Add the name(s) to the message, or unstage the file:

    git restore --staged <path>

EOF
exit 1