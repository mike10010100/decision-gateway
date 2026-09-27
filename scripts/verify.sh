#!/usr/bin/env bash
# ==============================================================================
# Decision Gateway Local Verification Pipeline
# ==============================================================================
# Stages:
#   1. Code formatting check (black --check app tests)
#   2. Linting check (flake8 app tests)
#   3. Static type check (mypy app)
#   4. Unit & E2E tests (pytest -v)
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# 1. Project Directory & PATH Setup
# ------------------------------------------------------------------------------
# Resolve project root regardless of where script is invoked from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Add user-local binary path if present
if [ -d "${HOME}/.local/bin" ] && [[ ":${PATH}:" != *":${HOME}/.local/bin:"* ]]; then
    export PATH="${HOME}/.local/bin:${PATH}"
fi

# ------------------------------------------------------------------------------
# 2. Terminal Styling & Color Detection
# ------------------------------------------------------------------------------
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    BOLD="\033[1m"
    DIM="\033[2m"
    RESET="\033[0m"
    GREEN="\033[32m"
    RED="\033[31m"
    YELLOW="\033[33m"
    CYAN="\033[36m"
    BLUE="\033[34m"
else
    BOLD=""
    DIM=""
    RESET=""
    GREEN=""
    RED=""
    YELLOW=""
    CYAN=""
    BLUE=""
fi

# ------------------------------------------------------------------------------
# 3. Stage Metadata & Timing State
# ------------------------------------------------------------------------------
TOTAL_START_TIME=$SECONDS

STAGE_1_NAME="Code Formatting Check (black)"
STAGE_2_NAME="Linting Check (flake8)"
STAGE_3_NAME="Static Type Check (mypy)"
STAGE_4_NAME="Unit & E2E Tests (pytest)"

STAGE_1_DURATION=0
STAGE_2_DURATION=0
STAGE_3_DURATION=0
STAGE_4_DURATION=0

# ------------------------------------------------------------------------------
# 4. Banner Functions
# ------------------------------------------------------------------------------
print_header() {
    echo -e "${BLUE}${BOLD}================================================================================${RESET}"
    echo -e "${BLUE}${BOLD}            DECISION-GATEWAY LOCAL VERIFICATION PIPELINE                        ${RESET}"
    echo -e "${BLUE}${BOLD}================================================================================${RESET}"
    echo -e "${DIM}Project Root : ${PROJECT_ROOT}${RESET}"
    echo -e "${DIM}Python Path  : $(command -v python3 || echo 'unknown')${RESET}"
    echo -e "${DIM}Date / Time  : $(date -u '+%Y-%m-%d %H:%M:%SZ')${RESET}"
    echo -e "${BLUE}--------------------------------------------------------------------------------${RESET}"
    echo -e "Verification Stages to Execute (Fail-Fast):"
    echo -e "  [Stage 1/4] black --check app tests"
    echo -e "  [Stage 2/4] flake8 app tests"
    echo -e "  [Stage 3/4] mypy app"
    echo -e "  [Stage 4/4] pytest -v"
    echo -e "${BLUE}================================================================================${RESET}"
    echo ""
}

print_summary_success() {
    local total_elapsed=$((SECONDS - TOTAL_START_TIME))
    echo ""
    echo -e "${GREEN}${BOLD}================================================================================${RESET}"
    echo -e "${GREEN}${BOLD}              VERIFICATION SUCCESSFUL (100% PASS)                              ${RESET}"
    echo -e "${GREEN}${BOLD}================================================================================${RESET}"
    printf "  %-35s %-12s %s\n" "Stage" "Status" "Duration"
    echo -e "${DIM}  ----------------------------------------------------------------------------${RESET}"
    printf "  %-35s ${GREEN}%-12s${RESET} %ds\n" "${STAGE_1_NAME}" "[PASSED]" "${STAGE_1_DURATION}"
    printf "  %-35s ${GREEN}%-12s${RESET} %ds\n" "${STAGE_2_NAME}" "[PASSED]" "${STAGE_2_DURATION}"
    printf "  %-35s ${GREEN}%-12s${RESET} %ds\n" "${STAGE_3_NAME}" "[PASSED]" "${STAGE_3_DURATION}"
    printf "  %-35s ${GREEN}%-12s${RESET} %ds\n" "${STAGE_4_NAME}" "[PASSED]" "${STAGE_4_DURATION}"
    echo -e "${DIM}  ----------------------------------------------------------------------------${RESET}"
    echo -e "  Total Verification Time : ${BOLD}${total_elapsed}s${RESET}"
    echo -e "  Repository Status       : ${GREEN}${BOLD}READY FOR COMMIT / PR${RESET}"
    echo -e "${GREEN}${BOLD}================================================================================${RESET}"
    echo ""
}

print_stage_failure() {
    local stage_idx="$1"
    local stage_desc="$2"
    local cmd_str="$3"
    local exit_code="$4"
    local elapsed="$5"

    echo ""
    echo -e "${RED}${BOLD}================================================================================${RESET}"
    echo -e "${RED}${BOLD}  VERIFICATION FAILED AT STAGE ${stage_idx}/4: ${stage_desc}${RESET}"
    echo -e "${RED}${BOLD}================================================================================${RESET}"
    echo -e "  Command   : ${BOLD}${cmd_str}${RESET}"
    echo -e "  Exit Code : ${RED}${BOLD}${exit_code}${RESET}"
    echo -e "  Duration  : ${elapsed}s"
    echo -e "${RED}--------------------------------------------------------------------------------${RESET}"
    echo -e "${YELLOW}${BOLD}Remediation Suggestions:${RESET}"
    case "$stage_idx" in
        1)
            echo -e "  - Run ${BOLD}black app tests${RESET} to automatically reformat files."
            echo -e "  - Inspect file diffs with ${BOLD}git diff${RESET}."
            ;;
        2)
            echo -e "  - Run ${BOLD}flake8 app tests${RESET} to inspect all lint violations."
            echo -e "  - Ensure lines do not exceed 100 characters and remove unused imports."
            ;;
        3)
            echo -e "  - Run ${BOLD}mypy app${RESET} to view static typing errors."
            echo -e "  - Correct type annotations in reported source files."
            ;;
        4)
            echo -e "  - Run ${BOLD}pytest -v tests/<test_file>.py${RESET} to isolate failing tests."
            echo -e "  - Review assertion logs and mock fixtures."
            ;;
    esac
    echo -e "${RED}${BOLD}================================================================================${RESET}"
    echo ""
}

# ------------------------------------------------------------------------------
# 5. Stage Execution Runner
# ------------------------------------------------------------------------------
run_stage() {
    local stage_idx="$1"
    local stage_desc="$2"
    shift 2
    local stage_cmd=("$@")
    local cmd_str="${stage_cmd[*]}"

    echo -e "${CYAN}${BOLD}>>> [Stage ${stage_idx}/4] ${stage_desc}${RESET}"
    echo -e "${DIM}    Command: ${cmd_str}${RESET}"
    echo -e "${CYAN}--------------------------------------------------------------------------------${RESET}"

    local stage_start=$SECONDS

    # Execute command, capturing exit code without triggering immediate shell abort
    set +e
    "${stage_cmd[@]}"
    local exit_code=$?
    set -e

    local stage_elapsed=$((SECONDS - stage_start))

    case "$stage_idx" in
        1) STAGE_1_DURATION=$stage_elapsed ;;
        2) STAGE_2_DURATION=$stage_elapsed ;;
        3) STAGE_3_DURATION=$stage_elapsed ;;
        4) STAGE_4_DURATION=$stage_elapsed ;;
    esac

    if [ $exit_code -ne 0 ]; then
        print_stage_failure "$stage_idx" "$stage_desc" "$cmd_str" "$exit_code" "$stage_elapsed"
        exit "$exit_code"
    fi

    echo -e "${CYAN}--------------------------------------------------------------------------------${RESET}"
    echo -e "${GREEN}${BOLD}[PASS] Stage ${stage_idx}/4 passed in ${stage_elapsed}s.${RESET}"
    echo ""
}

# ------------------------------------------------------------------------------
# 6. Pre-flight Dependency Verification
# ------------------------------------------------------------------------------
preflight_check() {
    local missing_tools=()
    for tool in black flake8 mypy pytest; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            missing_tools+=("$tool")
        fi
    done

    if [ ${#missing_tools[@]} -gt 0 ]; then
        echo -e "${RED}${BOLD}[ERROR] Missing required development tools: ${missing_tools[*]}${RESET}"
        echo -e "Please install development dependencies via:"
        echo -e "    pip install -r requirements-dev.txt"
        exit 127
    fi
}

# ------------------------------------------------------------------------------
# 7. Main Execution Flow
# ------------------------------------------------------------------------------
main() {
    preflight_check
    print_header

    # Stage 1: Code formatting check
    run_stage 1 "${STAGE_1_NAME}" black --check app tests

    # Stage 2: Linting check
    run_stage 2 "${STAGE_2_NAME}" flake8 app tests

    # Stage 3: Static type check
    run_stage 3 "${STAGE_3_NAME}" mypy app

    # Stage 4: Unit & E2E tests
    run_stage 4 "${STAGE_4_NAME}" pytest -v

    print_summary_success
}

main "$@"
