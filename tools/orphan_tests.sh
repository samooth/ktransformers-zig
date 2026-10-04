#!/usr/bin/env bash
# Detect an ORPHANED test binary from an aborted `zig build test` run.
#
# WHY THIS EXISTS. `zig build test` spawns one test binary per suite. If the run
# is cut short — a CI step timeout, a cancelled job, an interrupted local run —
# the build runner dies and the suite binary it had already forked keeps running.
# That binary is invisible: it logs nothing more, it appears in no step summary,
# and the run looks like it simply stopped reporting. Meanwhile it holds a core
# for as long as it lives, and on a shared box that inflates the very load
# average every other measurement on the host is read against.
#
# This was found the hard way on 2026-10-04: a test binary outlived its runner by
# 4.5 hours with two threads parked in `futex_wait_queue`, holding a core the
# whole time. Two "the gate did not terminate under load 61" reports were
# attributed to the host before the orphan was found; the load was partly the
# corpse's. A separate, unreproduced hang in the Subpool pinning test may or may
# not have been the same bug — that is still open, and this script is what would
# catch the next one either way.
#
# WHAT IT CHECKS. A `test` process is an orphan when all three hold:
#   1. its command line names a path under a zig cache directory (so it is a
#      Zig test binary, not some unrelated program called `test`);
#   2. its elapsed time greatly exceeds the oldest live `zig build test`
#      runner's — i.e. it has outlived the run that started it;
#   3. the run's log has no matching "All N tests passed" line for it.
#
# (3) is the part that makes this a test check rather than a process poll: a
# legitimately long-running suite is not an orphan.
#
# USAGE
#   tools/orphan_tests.sh [LOGFILE]
#
# LOGFILE defaults to /dev/null, which disables check (3) and degrades the script
# to "a zig test binary has outlived every build runner", which is still worth
# reporting. In CI, pass the step's log.
#
# EXIT STATUS
#   0  no orphans
#   1  at least one orphan found (the offenders are listed)
#   2  usage error
#
# It does NOT kill anything. Reporting and killing are separate decisions: the
# mitigation for an aborted run is to kill the whole process GROUP when the run
# is cut short, which belongs in the runner that starts the tests, not here.
# This script is the detector that tells you the mitigation did not work.

set -uo pipefail

LOG="${1:-/dev/null}"
CACHE_HINT='\.zig-cache/.*/test$'

# Longest-lived live build runner, in seconds. 0 when none is running, which is
# exactly the case where any test binary is an orphan by definition.
runner_secs() {
    local max=0 pid et
    for pid in $(pgrep -f 'zig build test' 2>/dev/null); do
        et=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
        [ -n "${et:-}" ] || continue
        [ "$et" -gt "$max" ] && max=$et
    done
    echo "$max"
}

RUNNER_SECS=$(runner_secs)
found=0

# `ps -eo pid,etimes,args` so elapsed time and command line come from one read;
# reading /proc twice races with short-lived processes.
while read -r pid etimes args; do
    [ -n "${pid:-}" ] || continue
    case "$args" in
        *"$CACHE_HINT"*) ;;
        *) continue ;;
    esac
    # etimes of the runner is the budget: a binary older than the runner plus a
    # wide margin has outlived the run that started it.
    if [ "$RUNNER_SECS" -gt 0 ] && [ "$etimes" -lt $((RUNNER_SECS + 60)) ]; then
        continue
    fi
    if [ -r "$LOG" ] && grep -qF "$args" "$LOG" 2>/dev/null; then
        # The log mentions this command line; if it also reported the suite
        # passing, the binary is a leftover, not a hang. The suite name is not in
        # the harness output, so presence in the log is the best available proxy
        # and erring toward "not an orphan" is the safe direction here.
        if grep -q "tests passed" "$LOG" 2>/dev/null; then
            continue
        fi
    fi
    echo "ORPHANED TEST BINARY: pid=$pid elapsed=${etimes}s cmd=$args"
    [ "$RUNNER_SECS" -gt 0 ] && echo "  oldest live 'zig build test' runner: ${RUNNER_SECS}s"
    found=1
done < <(ps -eo pid=,etimes=,args= 2>/dev/null)

if [ "$found" -eq 0 ]; then
    echo "no orphaned test binaries (runner ${RUNNER_SECS}s)"
    exit 0
fi
echo "an aborted run left a test binary running; it is holding a core and"
echo "inflating the load every other measurement on this host is read against."
exit 1
