#!/usr/bin/env bash
# Slicer Validation Test Runner
#
# This script runs the integration tests that validate the Rust slicer
# against BambuStudio-generated reference G-code.
#
# Usage:
#   ./scripts/run_validation.sh              # Run all validation tests
#   ./scripts/run_validation.sh --verbose    # Run with verbose output
#   ./scripts/run_validation.sh --quick      # Run only fast tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Parse arguments
VERBOSE=""
QUICK=""

for arg in "$@"; do
    case $arg in
        --verbose|-v)
            VERBOSE="-- --nocapture"
            ;;
        --quick|-q)
            QUICK="--skip test_slicing_performance"
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose    Show test output (println! statements)"
            echo "  -q, --quick      Skip performance tests"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
    esac
done

echo "========================================"
echo "  Slicer Validation Test Suite"
echo "========================================"
echo ""

# Check that test data exists
echo "Checking test data..."

if [ ! -f "data/test_stls/3DBenchy.stl" ]; then
    echo "ERROR: data/test_stls/3DBenchy.stl not found"
    echo "Please add the 3DBenchy STL file to run validation tests."
    exit 1
fi

if [ ! -f "data/reference_gcodes/3DBenchy.gcode" ]; then
    echo "ERROR: data/reference_gcodes/3DBenchy.gcode not found"
    echo "Please add the reference G-code file to run validation tests."
    exit 1
fi

echo "  ✓ Test data found"
echo ""

# Build the project first
echo "Building project..."
cargo build --release 2>&1 | grep -E "(Compiling|Finished)" || true
echo "  ✓ Build complete"
echo ""

# Run integration tests
echo "Running validation tests..."
echo "----------------------------------------"

if [ -n "$VERBOSE" ]; then
    cargo test --test benchy_integration $QUICK $VERBOSE
else
    cargo test --test benchy_integration $QUICK 2>&1 | tail -20
fi

echo "----------------------------------------"
echo ""

# Summary
PASSED=$(cargo test --test benchy_integration 2>&1 | grep -oE '[0-9]+ passed' | head -1 || echo "0 passed")
IGNORED=$(cargo test --test benchy_integration 2>&1 | grep -oE '[0-9]+ ignored' | head -1 || echo "0 ignored")

echo "========================================"
echo "  Results: $PASSED, $IGNORED"
echo "========================================"
echo ""
echo "To see detailed output, run:"
echo "  cargo test --test benchy_integration -- --nocapture"
