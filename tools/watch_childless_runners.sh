#!/usr/bin/env bash
# Report build runners that stay CHILDLESS across several observations.
#
# WHAT IT OBSERVES, AND WHAT IT DOES NOT. The name is the contract: this reports
# a runner with no children, sampled repeatedly. It does NOT report a runner
# that is STALLED, and the two are not the same claim:
#
#   * a runner between suites is briefly childless while it compiles the next
#     test binary — normal, and observed in real builds;
#   * a runner whose compile is slow is childless for the whole compile;
#   * a runner whose build or test has wedged is childless indefinitely.
#
# One sample cannot tell those apart, which is why this takes several spaced
# observations rather than one snapshot. Even so the output is deliberately NOT
# called "stalled": what it establishes is "childless for at least N seconds
# across K observations", which bounds a compile and does not exclude one. A
# check that fires on healthy runs gets ignored within a week, so the honest
# claim is the narrow one. An earlier single-shot version of this idea was
# written and NOT shipped for exactly that reason: it fired on a healthy build.
#
# WHY IT EXISTS. `zig build test` forks one test binary per suite. If the run is
# cut short the runner dies and the binary it forked keeps running, holding a
# core invisibly and inflating the load average every other measurement on a
# shared host is read against. That case is `tools/orphan_tests.sh`, which is a
# gate. This one is the complement: the runner is alive but going nowhere. It
# reports and NEVER signals — killing a process group from inside a CI step
# takes the runner and every later step with it.
#
# USAGE
#   tools/watch_childless_runners.sh [--runner PATTERN] [--samples K] [--spacing SECONDS]
#
#   --runner PATTERN   argv substring identifying a runner. Default: "zig build"
#                      — deliberately broad, so it covers `zig build test`,
#                      `zig build-exe` and a bare `zig build` alike.
#   --samples K        consecutive childless observations required. Default 3.
#                      Must be >= 2: one observation is a snapshot, not evidence.
#   --spacing SECONDS  seconds between observations. Default 5.
#
# EXIT STATUS
#   0  no runner met the condition
#   1  at least one runner was childless across every observation (reported)
#   2  usage error
#
# It observes runners that are alive AT THE FIRST OBSERVATION. A runner that
# starts later is not picked up; this is not a supervisor and does not pretend
# to be. In CI, run it in a loop or as a background step for the duration of the
# run you want covered.

set -uo pipefail

RUNNER='zig build'
SAMPLES=3
SPACING=5

while [ $# -gt 0 ]; do
    case "$1" in
        --runner) RUNNER="${2:-}"; shift 2 ;;
        --samples) SAMPLES="${2:-}"; shift 2 ;;
        --spacing) SPACING="${2:-}"; shift 2 ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

case "$SAMPLES" in
    ''|*[!0-9]*) echo "--samples must be a positive integer" >&2; exit 2 ;;
esac
[ "$SAMPLES" -ge 2 ] || { echo "--samples must be >= 2 (one observation is a snapshot, not evidence)" >&2; exit 2; }
case "$SPACING" in
    ''|*[!0-9]*) echo "--spacing must be a positive integer" >&2; exit 2 ;;
esac

# pids whose argv contains the runner pattern, as "pid<TAB>etimes<TAB>args".
live_runners() {
    ps -eo pid=,etimes=,args= 2>/dev/null |
        grep -F -- "$RUNNER" |
        grep -vE 'grep -F|watch_childless_runners' |
        awk '{print $1"\t"$2"\t"substr($0, index($0,$3))}'
}

# A runner is childless when `pgrep -P` finds nothing. Note this counts ANY
# child, not just a test binary: a runner that forked a compiler is doing work.
childless_pids() {
    local out='' pid kids
    while IFS=$'\t' read -r pid _ _; do
        [ -n "$pid" ] || continue
        kids=$(pgrep -P "$pid" 2>/dev/null | tr -d '[:space:]')
        [ -z "$kids" ] && out="${out}${pid}"$'\n'
    done <<< "$LIVE"
    printf '%s' "$out"
}

LIVE="$(live_runners)"
if [ -z "$LIVE" ]; then
    echo "no live runner matching '$RUNNER' at the first observation; nothing to watch"
    exit 0
fi

# Candidates are runners alive now; a runner must stay alive AND childless
# across every observation, so a runner that exits mid-watch is not reported.
CANDIDATES="$(printf '%s\n' "$LIVE" | cut -f1 | tr -d '[:space:]')"
SURVIVORS="$CANDIDATES"

i=1
while [ "$i" -lt "$SAMPLES" ]; do
    sleep "$SPACING"
    LIVE="$(live_runners)"
    still=''
    for pid in $SURVIVORS; do
        if printf '%s\n' "$LIVE" | grep -q "^${pid}"$'\t'; then
            kids=$(pgrep -P "$pid" 2>/dev/null | tr -d '[:space:]')
            [ -z "$kids" ] && still="${still}${pid}"$'\n'
        fi
    done
    SURVIVORS="$still"
    i=$((i + 1))
done

if [ -z "$SURVIVORS" ]; then
    echo "no runner was childless across ${SAMPLES} observations ${SPACING}s apart"
    exit 0
fi

rc=1
while read -r pid; do
    [ -n "$pid" ] || continue
    line=$(ps -o etimes=,args= -p "$pid" 2>/dev/null | sed 's/^ *//')
    echo "RUNNER CHILDLESS across ${SAMPLES} observations ${SPACING}s apart:"
    echo "  pid=${pid} elapsed/args: ${line}"
    echo "  This is the OBSERVATION, not a diagnosis. A long compile is childless too."
    echo "  Correlate with the build's own log before concluding anything, and note that"
    echo "  the orphan case (runner dead, test binary alive) is tools/orphan_tests.sh."
done <<< "$SURVIVORS"
exit $rc
