#!/bin/bash

set -euo pipefail  # Exit on error, undefined vars, and pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Function to setup virtual environment and install requirements
# Function to setup virtual environment and install requirements
setup_python_environment() {
    log_info "Environment setup requested. Checking Python environment..."

    # 1. Detect if our shared Conda environment is active
    if [[ "${CONDA_DEFAULT_ENV:-}" == "pfa-mmcows-env" ]]; then
        log_info "Shared Conda environment 'pfa-mmcows-env' detected. Skipping venv creation ✓"
        return 0
    fi

    # 2. Check if already active
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        log_info "Already running inside a virtual environment: $VIRTUAL_ENV"
    else
        # 3. Handle .venv creation/repair
        if [[ -d ".venv" && -f ".venv/bin/activate" ]]; then
            log_info "Found valid .venv directory. Activating..."
            source .venv/bin/activate
        else
            if [[ -d ".venv" ]]; then
                log_warn "Existing .venv is broken. Repairing..."
                rm -rf .venv
            fi
            log_info "Creating a new virtual environment..."
            python3 -m venv .venv
            source .venv/bin/activate
        fi
    fi

    # Install requirements if they exist
    if [[ -f "requirements.txt" ]]; then
        log_info "Installing dependencies from requirements.txt..."
        python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt
    fi
}

# Function to load .env file
load_env() {
    local env_file="${1:-.env}"
    
    if [[ ! -f "$env_file" ]]; then
        log_error ".env file not found: $env_file"
        exit 1
    fi
    
    log_info "Loading environment from: $env_file"
    
    # Read the file line by line, skip comments and empty lines
    while IFS='=' read -r key value || [[ -n "$key" ]]; do
        # Skip comments
        if [[ "$key" =~ ^#.* ]] || [[ -z "$key" ]]; then
            continue
        fi
        
        # Clean the key and value (remove whitespace/quotes/carriage returns)
        key=$(echo "$key" | tr -d '[:space:]')
        value=$(echo "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^["'\'']//' -e 's/["'\'']$//' -e 's/\r$//')

        if [[ -n "$key" ]]; then
            export "$key=$value"
            log_debug "Exported: $key"
        fi
    done < "$env_file"
    
    log_info "Environment variables loaded successfully ✓"
}

# Function to check if DVC is installed
check_dvc_installed() {
    if ! command -v dvc &> /dev/null; then
        log_error "DVC is not installed."
        log_error ""
        log_error "Installing DVC..."
        
        if command -v pip &> /dev/null || command -v pip3 &> /dev/null; then
            pip install 'dvc[s3]' || pip3 install 'dvc[s3]' || {
                log_error "Failed to install DVC"
                log_error "Please install manually: pip install 'dvc[s3]'"
                exit 1
            }
            log_info "DVC installed successfully ✓"
        else
            log_error "pip is not available. Please install DVC manually:"
            log_error "  pip install 'dvc[s3]'"
            exit 1
        fi
    fi
    log_info "DVC found: $(dvc version | head -1)"
}

# Function to validate credentials
validate_credentials() {
    local errors=0
    
    if [[ -z "${DVC_ACCESS_KEY_ID:-}" ]]; then
        log_error "DVC_ACCESS_KEY_ID is not set in .env"
        errors=$((errors + 1))
    elif [[ "${DVC_ACCESS_KEY_ID}" == "your_access_key_here" ]]; then
        log_error "DVC_ACCESS_KEY_ID still has the default placeholder value"
        errors=$((errors + 1))
    fi
    
    if [[ -z "${DVC_SECRET_ACCESS_KEY:-}" ]]; then
        log_error "DVC_SECRET_ACCESS_KEY is not set in .env"
        errors=$((errors + 1))
    elif [[ "${DVC_SECRET_ACCESS_KEY}" == "your_secret_key_here" ]]; then
        log_error "DVC_SECRET_ACCESS_KEY still has the default placeholder value"
        errors=$((errors + 1))
    fi
    
    if [[ $errors -gt 0 ]]; then
        log_error ""
        log_error "Please update your .env file with valid credentials"
        exit 1
    fi
    
    log_info "Credentials validated ✓"
}

# Function to validate configuration
validate_config() {
    if [[ -z "${DVC_REMOTE_NAME:-}" ]]; then
        log_error "DVC_REMOTE_NAME is not set in .env"
        exit 1
    fi
    
    if [[ -z "${DVC_S3_BUCKET:-}" ]]; then
        log_error "DVC_S3_BUCKET is not set in .env"
        exit 1
    fi
    
    if [[ -z "${DVC_ENDPOINT_URL:-}" ]]; then
        log_error "DVC_ENDPOINT_URL is not set in .env"
        exit 1
    fi
    
    log_info "Configuration validated ✓"
}

# Function to initialize DVC
init_dvc() {
    if [[ -d .dvc ]]; then
        log_info "DVC already initialized. Reinitializing with --force..."
    fi
    
    log_info "Initializing DVC..."
    # Using --force ensures it runs without asking for confirmation
    if dvc init --force; then
        log_info "DVC initialized successfully ✓"
    else
        log_error "Failed to initialize DVC"
        exit 1
    fi
}

# Function to configure remote
configure_remote() {
    log_info "Configuring DVC remote: $DVC_REMOTE_NAME"
    
    # 1. Add (or overwrite) the remote and set as default (-d)
    # Using -f (force) handles existing remotes gracefully
    if dvc remote add -d -f "$DVC_REMOTE_NAME" "$DVC_S3_BUCKET" -q; then
        log_info "Remote added: $DVC_REMOTE_NAME -> $DVC_S3_BUCKET"
    else
        log_error "Failed to add remote"
        exit 1
    fi
    
    # 2. Configure endpoint URL (Critical for MinIO/S3-compatible storage)
    if dvc remote modify "$DVC_REMOTE_NAME" endpointurl "$DVC_ENDPOINT_URL" -q; then
        log_info "Endpoint URL configured: $DVC_ENDPOINT_URL"
    else
        log_error "Failed to configure endpoint URL"
        exit 1
    fi
    
    # 3. Configure credentials (Local-only, secure)
    # We wrap these in a subshell or silence them to avoid leaking keys in logs
    dvc remote modify "$DVC_REMOTE_NAME" --local access_key_id "$DVC_ACCESS_KEY_ID" -q
    dvc remote modify "$DVC_REMOTE_NAME" --local secret_access_key "$DVC_SECRET_ACCESS_KEY" -q
    
    log_info "Credentials configured in .dvc/config.local ✓"
}

# Function to verify configuration
verify_configuration() {
    log_info "Verifying DVC configuration..."
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "Remote list:"
    dvc remote list
    
    echo ""
    log_info "Remote details (global config):"
    dvc config remote."$DVC_REMOTE_NAME".url 2>/dev/null || echo "  (not set in global config)"
    dvc config remote."$DVC_REMOTE_NAME".endpointurl 2>/dev/null || echo "  (not set in global config)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# Function to update .gitignore
update_gitignore() {
    local gitignore_entries=(
        "/.dvc/config.local"
        "/.env"
        "/.dvc/tmp"
        "/.dvc/cache"
        "/.venv"
    )
    
    if [[ ! -f .gitignore ]]; then
        log_info "Creating .gitignore..."
        cat > .gitignore << 'EOF'
# DVC
/.dvc/config.local
/.dvc/tmp
/.dvc/cache

# Environment variables
/.env

# Python Virtual Environment
/.venv

EOF
        log_info ".gitignore created ✓"
        return
    fi
    
    local added=0
    for entry in "${gitignore_entries[@]}"; do
        if ! grep -qF "$entry" .gitignore; then
            if [[ $added -eq 0 ]]; then
                echo "" >> .gitignore
                echo "# DVC, credentials, and Python env (auto-added by setup script)" >> .gitignore
            fi
            echo "$entry" >> .gitignore
            log_info "Added to .gitignore: $entry"
            added=$((added + 1))
        fi
    done
    
    if [[ $added -eq 0 ]]; then
        log_info ".gitignore already up to date ✓"
    else
        log_info "Updated .gitignore with $added entries ✓"
    fi
}

# Function to test connection
test_connection() {
    log_info "Testing DVC remote connection..."
    
    if dvc status --cloud 2>&1 | grep -q "ERROR\|error"; then
        log_warn "Could not verify remote connection (this is normal if no data is tracked yet)"
    else
        log_info "Remote connection test passed ✓"
    fi
}

# Function to display summary
display_summary() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${GREEN}✅ DVC setup completed successfully!${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Configuration:"
    echo "  Remote name: $DVC_REMOTE_NAME"
    echo "  S3 bucket:   $DVC_S3_BUCKET"
    echo "  Endpoint:    $DVC_ENDPOINT_URL"
    if [[ "${VIRTUAL_ENV:-}" != "" ]]; then
        echo "  Python venv: $VIRTUAL_ENV"
    fi
    echo ""
    echo "Next steps:"
    echo "  1. Activate virtual environment (if not already):"
    echo "     source .venv/bin/activate"
    echo ""
    echo "  2. Track data with DVC:"
    echo "     dvc add data/"
    echo ""
    echo "  3. Push data to remote:"
    echo "     dvc push"
    echo ""
    echo "  4. Commit DVC files to git:"
    echo "     git add .dvc/config .dvc/.gitignore data.dvc .gitignore"
    echo "     git commit -m 'Initialize DVC tracking'"
    echo ""
    echo "  5. Pull data on another machine:"
    echo "     git clone <repo>"
    echo "     cp .env.example .env  # and fill credentials"
    echo "     ./setup_dvc.sh"
    echo "     dvc pull"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Main execution
# Main execution
main() {
    local setup_env=${1:-false}

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}DVC Setup Script for DagHub${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # STEP 1: Setup Python environment (Only if flag is true)
    if [ "$setup_env" = true ]; then
        setup_python_environment
    else
        log_info "Skipping environment setup (use --setup-env to enable)."
    fi
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "Setting up DVC..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    load_env
    check_dvc_installed
    validate_config
    validate_credentials
    init_dvc
    configure_remote
    update_gitignore
    verify_configuration
    test_connection
    display_summary
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --setup-env    Create and configure the Python virtual environment"
        echo "  --help, -h     Show this help message"
        exit 0
        ;;
    --setup-env)
        main true
        ;;
    *)
        main false
        ;;
esac