# scratch-env.sh — redirect all install locations and caches off $HOME onto scratch.
#
# For users whose home directory has little/no free space. Source this BEFORE
# installing or running LidarStudio, in every shell where you use the project:
#
#     source scratch-env.sh          # uses a sensible default scratch path
#     SCRATCH=/scratch/$USER source scratch-env.sh   # or set your own
#
# Nothing here needs root. Note: scratch filesystems are usually auto-purged,
# so caches/venv may vanish — just re-run `uv sync` to rebuild, and keep any
# generated clouds/splats somewhere persistent.

# --- Pick a scratch root ---------------------------------------------------
# Honour an already-set $SCRATCH; otherwise fall back to common locations.
if [ -z "${SCRATCH:-}" ]; then
    for _cand in "/scratch/$USER" "/scratch/$(id -un)" "/scratch"; do
        if [ -d "$_cand" ] && [ -w "$_cand" ]; then
            SCRATCH="$_cand"
            break
        fi
    done
    unset _cand
fi

if [ -z "${SCRATCH:-}" ]; then
    echo "scratch-env.sh: no writable scratch dir found." >&2
    echo "  Set one explicitly, e.g.:  SCRATCH=/scratch/\$USER source scratch-env.sh" >&2
    return 1 2>/dev/null || exit 1
fi

export SCRATCH
echo "scratch-env.sh: using SCRATCH=$SCRATCH"

# --- uv: binary, caches, downloaded Pythons, and the project venv ----------
export UV_INSTALL_DIR="$SCRATCH/uv/bin"             # where the uv binary lands
export UV_CACHE_DIR="$SCRATCH/uv/cache"             # wheel/download cache (large)
export UV_PYTHON_INSTALL_DIR="$SCRATCH/uv/python"   # standalone CPython 3.11+
export UV_PROJECT_ENVIRONMENT="$SCRATCH/lidarstudio-venv"  # the .venv, off-home

# --- Node / npm ------------------------------------------------------------
export NVM_DIR="$SCRATCH/nvm"
export npm_config_cache="$SCRATCH/npm-cache"

# --- pip fallback cache, XDG caches, temp (open3d/torch build+unpack temp) --
export PIP_CACHE_DIR="$SCRATCH/pip-cache"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export TMPDIR="$SCRATCH/tmp"

# --- Create the dirs and put uv on PATH ------------------------------------
mkdir -p "$UV_INSTALL_DIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" \
         "$NVM_DIR" "$npm_config_cache" "$PIP_CACHE_DIR" \
         "$XDG_CACHE_HOME" "$TMPDIR"

case ":$PATH:" in
    *":$UV_INSTALL_DIR:"*) ;;                       # already on PATH
    *) export PATH="$UV_INSTALL_DIR:$PATH" ;;
esac

echo "scratch-env.sh: redirected uv/npm/pip caches and venv onto scratch."
echo "  next steps:  npm install  &&  uv sync  &&  uv run lidarstudio"
