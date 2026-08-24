#!/usr/bin/env sh
# POSIX entry point named by spec Section 18. The driver itself is reproduce.py so that
# the same code path runs on every platform, including the Windows dev machines.
set -e
exec python "$(dirname "$0")/reproduce.py" "$@"
