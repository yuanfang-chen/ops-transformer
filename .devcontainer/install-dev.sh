#!/usr/bin/env bash
set -euo pipefail

install_bubblewrap() {
    if command -v bwrap >/dev/null 2>&1; then
        return
    fi

    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y bubblewrap
    apt-get clean
    rm -rf /var/lib/apt/lists/*
}

check_bubblewrap() {
    bwrap --version
    if ! bwrap --unshare-user --unshare-pid --ro-bind / / /bin/true; then
        cat >&2 <<'EOF'
bubblewrap is installed, but it cannot create the namespaces Codex needs.
Make sure the selected devcontainer config includes:
  --security-opt=apparmor=unconfined
  --security-opt=seccomp=unconfined
EOF
        return 1
    fi
}

install_codex_cli() {
    if command -v codex >/dev/null 2>&1; then
        return
    fi

    install_bubblewrap
    check_bubblewrap
    npm install -g @openai/codex@latest
}

install_claude_code() {
    if command -v claude >/dev/null 2>&1; then
        return
    fi

    curl -fsSL https://claude.ai/install.sh | bash
}

configure_claude_code_env() {
    cat > /etc/profile.d/claude-code.sh <<'EOF'
export PATH="/root/.local/bin:${PATH}"
EOF
}

install_codex_cli
install_claude_code
configure_claude_code_env
