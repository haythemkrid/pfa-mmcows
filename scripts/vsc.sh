#!/bin/bash

# Configuration
INSTALL_DIR="$HOME/.local/bin"
mkdir -p "$INSTALL_DIR"
TUNNEL_NAME="pfa-$(whoami)" # e.g., pfa-haythem, pfa-iheb, pfa-oussema

# 1. Download VS Code CLI if missing
if [ ! -f "$INSTALL_DIR/code" ]; then
    echo "Downloading VS Code CLI..."
    # Added _$USER to the filename to prevent permission conflicts between accounts
    wget -O /tmp/vscode_cli_$USER.tar.gz 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64'
    tar -xf /tmp/vscode_cli_$USER.tar.gz -C "$INSTALL_DIR"
    chmod +x "$INSTALL_DIR/code"
    
    # Clean up the downloaded tar file to save space
    rm /tmp/vscode_cli_$USER.tar.gz
fi

# 2. Setup a "Session-Locked" Tunnel script
# We don't use systemd here because ZTM sessions can be wonky with user-services.
# Instead, we create a script that runs the tunnel and kills it on exit.
cat <<EOF > "$INSTALL_DIR/start_dev_env"
#!/bin/bash
echo "Starting VS Code Tunnel: \$TUNNEL_NAME"
# The tunnel runs in the background. 
# 'trap' ensures that when you close the terminal or exit, the tunnel dies.
$INSTALL_DIR/code tunnel --name $TUNNEL_NAME --accept-server-license-terms &
TUNNEL_PID=\$!

trap "echo 'Closing Tunnel...'; kill \$TUNNEL_PID" EXIT

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Tunnel is ACTIVE: https://vscode.dev/tunnel/$TUNNEL_NAME"
echo "Keep this terminal open to keep the instance alive."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Wait for the tunnel process
wait \$TUNNEL_PID
EOF

chmod +x "$INSTALL_DIR/start_dev_env"
echo "Setup complete. You can now start your instance by running: start_dev_env"