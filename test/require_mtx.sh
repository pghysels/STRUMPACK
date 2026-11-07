#!/bin/sh
#
# Usage: require_mtx.sh <matrix file> <command> [args...]
#
# Runs <command> only if <matrix file> exists and is non-empty.
# Otherwise exits with 77, which ctest reports as "skipped" (the test
# has SKIP_RETURN_CODE 77). This is used for tests that need one of the
# matrices downloaded by download_mtx.sh, so that a failed download
# skips those tests instead of failing them.
#
# The exit code of <command> is passed through unchanged, so real test
# failures are still reported as failures.

if [ $# -lt 2 ]; then
    echo "usage: $0 <matrix file> <command> [args...]" >&2
    exit 2
fi

mtx=$1
shift
if [ ! -s "$mtx" ]; then
    echo "SKIPPED: test matrix $mtx is not available (download failed?)"
    exit 77
fi
exec "$@"
