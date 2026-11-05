#!/bin/sh
set -e
if [ "${AUTO_MIGRATE:-true}" = "true" ]; then
    alembic upgrade head
fi
exec "$@"
