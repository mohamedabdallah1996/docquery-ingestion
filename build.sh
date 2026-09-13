#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

SKIP_DEPS=""

for arg in "$@"; do
  case "$arg" in
    --skip-deps)
    SKIP_DEPS=true
    ;;

    --headless)
    :
    ;; # no-op, see docquery-core/build.sh for why this flag exists

    *)
    warn "build.sh: ignoring unknown argument: $arg"
    ;;
  esac
done

cd "$(dirname "${BASH_SOURCE[0]}")"

info "Building docquery-ingestion package ..."

if [[ "$SKIP_DEPS" != true ]]; then
    info "Syncing dependencies"
    uv sync --locked
else
    info "Skipping dependency sync"
fi

info "Cleaning previous build artifacts"

rm -rf dist
# --out-dir is required: this package's own uv.lock/env can still be
# resolved from a workspace-adjacent context, and uv build otherwise
# writes to a workspace root's dist/, not this package's own directory.
uv build --out-dir dist

success "docquery-ingestion built successfully -> $(ls dist)"
