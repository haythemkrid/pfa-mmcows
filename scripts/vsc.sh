#!/bin/bash

# Configuration
INSTALL_DIR="$HOME/.local/bin"
mkdir -p "$INSTALL_DIR"
# Define here for use in the setup logic
MY_TUNNEL_NAME="pfa-$(whoami)"

# 1. Connectivity Check
echo "Checking internet connectivity..."
if ! ping -c 1 8.8.8.8 &>/dev/null; then
    echo "❌ ERROR: No internet connection. DNS and Tunnels will fail."
    echo "Please check the ZTM mesh status or server gateway."
fi

# 2. Download VS Code CLI if missing
if [ ! -f "$INSTALL_DIR/code" ]; then
    echo "Downloading VS Code CLI..."
    wget -O "$HOME/vscode_cli.tar.gz" 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64'
    tar -xf "$HOME/vscode_cli.tar.gz" -C "$INSTALL_DIR"
    chmod +x "$INSTALL_DIR/code"
    rm "$HOME/vscode_cli.tar.gz"
fi

# 3. Create the robust start script
# We use single quotes around 'EOF' to prevent variable expansion during creation
cat << 'EOF' > "$INSTALL_DIR/start_dev_env"
#!/bin/bash
# Re-detect user to ensure name is correct
TUNNEL_NAME="pfa-$(whoami)"

echo "Starting VS Code Tunnel: $TUNNEL_NAME"

# Start tunnel in background
$HOME/.local/bin/code tunnel --name "$TUNNEL_NAME" --accept-server-license-terms &
TUNNEL_PID=$!

# Improved trap: kills the specific PID and ignores errors if already dead
trap "echo 'Closing Tunnel...'; kill $TUNNEL_PID 2>/dev/null" EXIT

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Tunnel is ACTIVE: https://vscode.dev/tunnel/$TUNNEL_NAME"
echo "Keep this ZTM terminal open to keep the instance alive."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Wait for the tunnel process
wait $TUNNEL_PID
EOF

chmod +x "$INSTALL_DIR/start_dev_env"

# Ensure ~/.local/bin is in PATH for this session
export PATH="$PATH:$INSTALL_DIR"

echo "Setup complete. You can now start your instance by running: start_dev_env"