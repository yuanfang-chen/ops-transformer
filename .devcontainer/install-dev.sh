#!/usr/bin/env bash
set -euo pipefail

clean_apt_lists() {
    apt-get clean
    rm -rf /var/lib/apt/lists/*
}

install_apt_packages() {
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        bubblewrap \
        ripgrep
    clean_apt_lists
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

    check_bubblewrap
    npm install -g @openai/codex@latest
}

install_opencode_cli() {
    if command -v opencode >/dev/null 2>&1; then
        return
    fi

    npm install -g opencode-ai@latest
}

configure_codex_zsh_completion() {
    if ! command -v codex >/dev/null 2>&1; then
        return
    fi

    install -d /usr/local/share/zsh/site-functions
    codex completion zsh > /usr/local/share/zsh/site-functions/_codex
}

install_claude_code() {
    if command -v claude >/dev/null 2>&1; then
        return
    fi

    curl -fsSL https://claude.ai/install.sh | bash
}

configure_claude_code_env() {
    export PATH="/root/.local/bin:${PATH}"

    cat > /etc/profile.d/claude-code.sh <<'EOF'
export PATH="/root/.local/bin:${PATH}"
EOF
}

install_apt_packages
install_codex_cli
install_opencode_cli
configure_codex_zsh_completion
install_claude_code
configure_claude_code_env
