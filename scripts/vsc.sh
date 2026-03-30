%%writefile setup_vscode.sh
#!/bin/bash

# 1. Download the CLI
echo "Downloading VS Code CLI..."
wget -O vscode_cli.tar.gz 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64'

# 2. Extract
echo "Extracting..."
tar -xf vscode_cli.tar.gz
chmod +x code

# 3. Start Tunnel
echo "Starting tunnel... Follow the GitHub link and enter the code provided below."
./code tunnel --accept-server-license-terms