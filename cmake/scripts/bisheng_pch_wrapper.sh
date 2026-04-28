#!/bin/bash
# bisheng wrapper that injects -include-pch for FIAS arch32 kernel compiles.
#
# Goal: amortize the ~36K LOC of macro-invariant header parsing (attn_infra/* +
# AscendC runtime) across the ~1k per-tiling-key bisheng invocations of FIAS.
#
# Strategy:
#   1. On the first FIA-source compile per build, treat the captured cmdline
#      as a template: replace the source with our prelude header, drop the
#      object output and per-binary -DORIG_DTYPE_*/-DDTYPE_*/-DFORMAT_* flags,
#      run bisheng once with -x c++-header to produce the PCH.
#   2. On every FIA-source compile (including the first, after PCH is built),
#      prepend `-include-pch <pch>` to the cmdline.
#   3. Non-FIA compiles pass through unchanged.
#
# Activated only when env var FIAS_PCH=1 is set; otherwise behaves exactly
# like the upstream wrapper (real bisheng invocation, optionally via ccache).
#
# Required env vars:
#   BISHENG_REAL_PATH   absolute path to the real bisheng binary
#   FIAS_PCH_DIR        directory where prelude.h and prelude.pch live
#   FIAS_PRELUDE_HEADER absolute path to fias_kernel_prelude.h
# Optional:
#   CCACHE_PROGRAM      ccache binary (when set, prefixes the bisheng call)
#   FIAS_PCH_DEBUG=1    log each invocation decision to FIAS_PCH_DIR/wrapper.log

set -e

REAL=${BISHENG_REAL_PATH:?BISHENG_REAL_PATH unset}
CCACHE=${CCACHE_PROGRAM:-}

# Pass-through path: if FIAS_PCH not enabled, behave like the original wrapper.
if [ "${FIAS_PCH:-0}" != "1" ]; then
    if [ -n "$CCACHE" ]; then
        exec "$CCACHE" "$REAL" "$@"
    else
        exec "$REAL" "$@"
    fi
fi

PCH_DIR=${FIAS_PCH_DIR:?FIAS_PCH_DIR unset when FIAS_PCH=1}
PRELUDE=${FIAS_PRELUDE_HEADER:?FIAS_PRELUDE_HEADER unset when FIAS_PCH=1}
PCH_FILE="$PCH_DIR/fias_prelude.pch"
PCH_BUILD_LOG="$PCH_DIR/pch_build.log"
LOCK="$PCH_DIR/pch.lock"
WRAPPER_LOG="$PCH_DIR/wrapper.log"
mkdir -p "$PCH_DIR"

dbg() { if [ "${FIAS_PCH_DEBUG:-0}" = "1" ]; then printf '%s\n' "$*" >> "$WRAPPER_LOG"; fi; }

# Decide if this invocation is a FIA arch32 kernel compile we should PCH-ify.
# We match by source-file basename in argv. The list mirrors the kernel TUs
# that include the prelude content (see audit in plan Step 1).
is_fia_kernel_compile=0
src_arg=""
for arg in "$@"; do
    case "$arg" in
        */fused_infer_attention_score.cpp \
        |*/fused_infer_attention_score_v3.cpp \
        |*/flash_attention_interface.cpp \
        |*/fused_infer_attention_score_apt.cpp)
            is_fia_kernel_compile=1
            src_arg=$arg
            ;;
    esac
done
dbg "argv: $*"
dbg "is_fia=$is_fia_kernel_compile src=$src_arg"

run_real() {
    if [ -n "$CCACHE" ]; then
        "$CCACHE" "$REAL" "$@"
    else
        "$REAL" "$@"
    fi
}

if [ $is_fia_kernel_compile -eq 0 ]; then
    if [ -n "$CCACHE" ]; then
        exec "$CCACHE" "$REAL" "$@"
    else
        exec "$REAL" "$@"
    fi
fi

# Build the PCH once (serialized via flock).
if [ ! -f "$PCH_FILE" ]; then
    (
        # Wait for the first concurrent compile that gets here; others will
        # block, then see PCH_FILE present and skip the build.
        flock 9
        if [ ! -f "$PCH_FILE" ]; then
            dbg "building PCH from template cmd"
            # Reconstruct a PCH-build cmdline from this invocation:
            #   - keep -I, -D (excluding per-binary), --cce-*, -std=*
            #   - drop the source file argument, -c, -o <obj>
            #   - append `-x c++-header <prelude> -o <pch>`
            pch_argv=()
            skip_next=0
            for arg in "$@"; do
                if [ $skip_next -eq 1 ]; then skip_next=0; continue; fi
                case "$arg" in
                    -DORIG_DTYPE_*|-DDTYPE_*|-DFORMAT_*) continue ;;
                    -c) continue ;;
                    -o) skip_next=1; continue ;;
                    "$src_arg") continue ;;
                esac
                pch_argv+=("$arg")
            done
            pch_argv+=(-x c++-header "$PRELUDE" -o "$PCH_FILE")
            dbg "pch cmd: ${pch_argv[*]}"
            if "$REAL" "${pch_argv[@]}" >"$PCH_BUILD_LOG" 2>&1; then
                dbg "PCH built: $(stat -c%s "$PCH_FILE" 2>/dev/null) bytes"
            else
                dbg "PCH build failed; see $PCH_BUILD_LOG. Falling back to no-PCH path."
                rm -f "$PCH_FILE"
            fi
        fi
    ) 9>"$LOCK"
fi

# Inject -include-pch if the PCH exists; else fall back to the original cmd.
if [ -f "$PCH_FILE" ]; then
    exec "$REAL" -include-pch "$PCH_FILE" "$@"
else
    exec "$REAL" "$@"
fi
