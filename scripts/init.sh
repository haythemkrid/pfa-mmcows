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
setup_python_enviroment() {
    log_info "Checking Python environment..."

    # 1. Check if already active or exists
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        log_info "Already running inside a virtual environment: $VIRTUAL_ENV"
    elif [[ -d ".venv" ]]; then
        log_info "Found existing .venv directory. Activating..."
        source .venv/bin/activate
    else
        log_warn "No virtual environment found. Creating one..."
        
        # Try standard creation first
        if python3 -m venv .venv; then
            log_info "Virtual environment created successfully ✓"
            source .venv/bin/activate
        else
            log_warn "Standard venv creation failed (ensurepip error). Attempting bootstrap..."
            # Fallback: Create without pip to avoid the exit status 1 error
            if python3 -m venv --without-pip .venv; then
                source .venv/bin/activate
                log_info "Venv created without pip. Bootstrapping pip now..."
                # Download and install pip manually inside the venv
                if curl -sS https://bootstrap.pypa.io/get-pip.py | python3; then
                    log_info "Pip bootstrapped successfully ✓"
                else
                    log_error "Failed to bootstrap pip. Please install python3-venv on your system."
                    exit 1
                fi
            else
                log_error "Critical: Could not create virtual environment directory."
                exit 1
            fi
        fi
    fi

    # 2. Install requirements.txt
    if [[ -f "requirements.txt" ]]; then
        log_info "Installing dependencies from requirements.txt..."
        # Use python3 -m pip to ensure we stay within the venv context
        if python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt; then
            log_info "Dependencies installed successfully ✓"
        else
            log_error "Failed to install dependencies from requirements.txt"
            exit 1
        fi
    else
        log_warn "requirements.txt not found. Skipping dependency installation."
        # Even if no requirements, we still need DVC for the rest of your script
        if ! command -v dvc &> /dev/null; then
            log_info "DVC not found in venv, installing dvc[s3]..."
            python3 -m pip install "dvc[s3]"
        fi
    fi
}

# Function to load .env file
load_env() {
    local env_file="${1:-.env}"
    
    if [[ ! -f "$env_file" ]]; then
        log_error ".env file not found: $env_file"
        log_error ""
        log_error "Please create a .env file with your credentials:"
        log_error "  cp .env.example .env"
        log_error "  # Edit .env and add your actual credentials"
        exit 1
    fi
    
    log_info "Loading environment from: $env_file"
    
    # Load .env file, ignoring comments and empty lines
    set -a  # Automatically export all variables
    source <(grep -v '^#' "$env_file" | grep -v '^$' | sed 's/\r$//')
    set +a
    
    log_debug "Environment variables loaded"
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
        log_warn "DVC already initialized in this directory"
        read -p "Do you want to reinitialize? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping DVC initialization"
            return 0
        fi
    fi
    
    log_info "Initializing DVC..."
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
    
    # Check if remote already exists
    if dvc remote list 2>/dev/null | grep -q "^$DVC_REMOTE_NAME"; then
        log_warn "Remote '$DVC_REMOTE_NAME' already exists - removing and recreating"
        dvc remote remove "$DVC_REMOTE_NAME" 2>/dev/null || true
    fi
    
    # Add remote
    if dvc remote add "$DVC_REMOTE_NAME" "$DVC_S3_BUCKET"; then
        log_info "Remote added: $DVC_REMOTE_NAME -> $DVC_S3_BUCKET"
    else
        log_error "Failed to add remote"
        exit 1
    fi
    
    # Configure endpoint URL
    if dvc remote modify "$DVC_REMOTE_NAME" endpointurl "$DVC_ENDPOINT_URL"; then
        log_info "Endpoint URL configured: $DVC_ENDPOINT_URL"
    else
        log_error "Failed to configure endpoint URL"
        exit 1
    fi
    
    # Configure credentials (local only, not committed to git)
    if dvc remote modify "$DVC_REMOTE_NAME" --local access_key_id "$DVC_ACCESS_KEY_ID"; then
        log_info "Access key ID configured (local) ✓"
    else
        log_error "Failed to configure access key ID"
        exit 1
    fi
    
    if dvc remote modify "$DVC_REMOTE_NAME" --local secret_access_key "$DVC_SECRET_ACCESS_KEY"; then
        log_info "Secret access key configured (local) ✓"
    else
        log_error "Failed to configure secret access key"
        exit 1
    fi
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
main() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}DVC Setup Script for DagHub${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # STEP 1: Setup Python environment (NEW!)
    setup_python_environment
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "Setting up DVC..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Load environment variables from .env
    load_env
    
    # Validate everything
    check_dvc_installed
    validate_config
    validate_credentials
    
    # Setup DVC
    init_dvc
    configure_remote
    update_gitignore
    
    # Verify and test
    verify_configuration
    test_connection
    
    # Show summary
    display_summary
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo ""
        echo "This script:"
        echo "  1. Sets up Python virtual environment (.venv)"
        echo "  2. Installs requirements from requirements.txt"
        echo "  3. Initializes DVC with credentials from .env file"
        echo ""
        echo "Make sure to create .env from .env.example first"
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac