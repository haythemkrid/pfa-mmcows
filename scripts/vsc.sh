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
cat << 'EOF' > $HOME/.local/bin/start_dev_env
#!/bin/bash
USER_NAME=$(whoami)
TUNNEL_NAME="pfa-$USER_NAME"

# Nettoyage plus propre
pkill -u $USER_NAME -f "code tunnel" 2>/dev/null
sleep 2 # Laisse le temps au système de libérer les sockets

# Bypass SSL
export NODE_TLS_REJECT_UNAUTHORIZED=0

# On définit le répertoire de données
DATA_DIR="$HOME/.vscode-server-data"
mkdir -p "$DATA_DIR"

echo "🚀 Tentative de lancement du tunnel pour $USER_NAME..."

# Ajout d'une boucle de retry en cas de DNS failure
for i in {1..3}; do
    $HOME/.local/bin/code tunnel \
        --name "$TUNNEL_NAME" \
        --accept-server-license-terms \
        --server-data-dir "$DATA_DIR" && break || \
        echo "⚠️ Échec de connexion (tentative $i/3), nouvel essai dans 5s..."
    sleep 5
done
EOF

chmod +x "$INSTALL_DIR/start_dev_env"

# Ensure ~/.local/bin is in PATH for this session
export PATH="$PATH:$INSTALL_DIR"

echo "Setup complete. You can now start your instance by running: start_dev_env"