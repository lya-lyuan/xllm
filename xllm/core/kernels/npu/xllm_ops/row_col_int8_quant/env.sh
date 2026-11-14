#!/usr/bin/env bash
# Source this file; keep all custom packages in the user-owned prefix.
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
_rowcol_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -x "$_rowcol_root/.venv/bin/python3.12" ]]; then
    export PATH="$_rowcol_root/.venv/bin:$PATH"
fi
unset _rowcol_root
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export ROWCOL_PREFIX=${ROWCOL_PREFIX:-$HOME/.local/rowcol-int8}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
if [[ -f "$ROWCOL_PREFIX/vendors/rowcol_int8/bin/set_env.bash" ]]; then
    source "$ROWCOL_PREFIX/vendors/rowcol_int8/bin/set_env.bash"
fi
