#!/usr/bin/env bash
#
# Deploy VariationSampler to Google Cloud for training.
#
# Usage:
#   ./scripts/deploy_cloud.sh            # Create VM (if needed) + upload code + install deps
#   ./scripts/deploy_cloud.sh --start    # Start a stopped VM + upload code
#   ./scripts/deploy_cloud.sh --stop     # Stop the VM (preserves disk, stops billing for GPU)
#   ./scripts/deploy_cloud.sh --upload   # Upload code only (VM already running)
#   ./scripts/deploy_cloud.sh --ssh      # SSH into the VM
#
set -euo pipefail

INSTANCE="variation-sampler"
ZONES=("europe-west4-a" "europe-west4-b" "europe-west4-c")
MACHINE_TYPE="g2-standard-8"
ACCELERATOR="type=nvidia-l4,count=1"
BOOT_DISK_SIZE="100GB"
# NOTE: Image families change over time. To list available families, run:
#   gcloud compute images list --project=deeplearning-platform-release --no-standard-images | grep pytorch
IMAGE_FAMILY="pytorch-2-7-cu128-ubuntu-2204-nvidia-570"
IMAGE_PROJECT="deeplearning-platform-release"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="~/variation-sampler"
ZONE_FILE="$PROJECT_ROOT/.cloud_zone"

# ---------- helpers ----------

log() { echo "[deploy] $*"; }

save_zone() {
    echo "$1" > "$ZONE_FILE"
    log "Saved zone $1 to .cloud_zone"
}

load_zone() {
    if [ -f "$ZONE_FILE" ]; then
        cat "$ZONE_FILE"
        return 0
    fi
    return 1
}

vm_exists() {
    gcloud compute instances describe "$INSTANCE" --zone="$1" &>/dev/null 2>&1
}

vm_status() {
    gcloud compute instances describe "$INSTANCE" --zone="$1" \
        --format="get(status)" 2>/dev/null || echo "NOT_FOUND"
}

find_vm_zone() {
    # Check saved zone first
    if saved_zone=$(load_zone) && vm_exists "$saved_zone"; then
        echo "$saved_zone"
        return 0
    fi
    # Fall back to scanning all zones
    for z in "${ZONES[@]}"; do
        if vm_exists "$z"; then
            save_zone "$z"
            echo "$z"
            return 0
        fi
    done
    return 1
}

create_vm() {
    local zone="$1"
    log "Creating VM '$INSTANCE' in $zone..."
    if ! gcloud compute instances create "$INSTANCE" \
        --zone="$zone" \
        --machine-type="$MACHINE_TYPE" \
        --accelerator="$ACCELERATOR" \
        --boot-disk-size="$BOOT_DISK_SIZE" \
        --image-family="$IMAGE_FAMILY" \
        --image-project="$IMAGE_PROJECT" \
        --maintenance-policy=TERMINATE \
        --scopes=default 2>&1; then

        log "Zone $zone unavailable, trying alternatives..."
        return 1
    fi
    save_zone "$zone"
    log "VM created in $zone"
    return 0
}

wait_for_ssh() {
    local zone="$1"
    log "Waiting for SSH to become available..."
    local retries=0
    while ! gcloud compute ssh "$INSTANCE" --zone="$zone" --tunnel-through-iap \
            --command="echo ready" &>/dev/null 2>&1; do
        retries=$((retries + 1))
        if [ $retries -ge 30 ]; then
            log "ERROR: SSH not available after 30 attempts"
            return 1
        fi
        sleep 5
    done
    log "SSH ready"
}

upload_code() {
    local zone="$1"
    log "Uploading project files to $INSTANCE..."

    # Create remote directory structure
    gcloud compute ssh "$INSTANCE" --zone="$zone" --tunnel-through-iap --command="
        mkdir -p $REMOTE_DIR/{src,scripts,configs,data/codegrams,data/splits,data/baselines,checkpoints}
    "

    # Upload code, configs, scripts (small — use scp directly)
    gcloud compute scp --recurse --zone="$zone" --tunnel-through-iap --compress \
        "$PROJECT_ROOT/src/" "$INSTANCE:$REMOTE_DIR/src/"
    gcloud compute scp --recurse --zone="$zone" --tunnel-through-iap --compress \
        "$PROJECT_ROOT/scripts/" "$INSTANCE:$REMOTE_DIR/scripts/"
    gcloud compute scp --recurse --zone="$zone" --tunnel-through-iap --compress \
        "$PROJECT_ROOT/configs/" "$INSTANCE:$REMOTE_DIR/configs/"
    gcloud compute scp --zone="$zone" --tunnel-through-iap \
        "$PROJECT_ROOT/requirements.txt" "$INSTANCE:$REMOTE_DIR/"

    # Upload splits and baselines (small files, scp is fine)
    if [ -d "$PROJECT_ROOT/data/splits" ]; then
        gcloud compute scp --recurse --zone="$zone" --tunnel-through-iap --compress \
            "$PROJECT_ROOT/data/splits/" "$INSTANCE:$REMOTE_DIR/data/splits/"
    fi
    if [ -d "$PROJECT_ROOT/data/baselines" ]; then
        gcloud compute scp --recurse --zone="$zone" --tunnel-through-iap --compress \
            "$PROJECT_ROOT/data/baselines/" "$INSTANCE:$REMOTE_DIR/data/baselines/"
    fi

    # Upload codegrams via tar (thousands of small .npy files — scp is too slow)
    if [ -d "$PROJECT_ROOT/data/codegrams" ]; then
        log "Packing codegrams into tar archive..."
        local tar_file="/tmp/vs-codegrams.tar.gz"
        tar czf "$tar_file" -C "$PROJECT_ROOT" data/codegrams/
        local size_mb
        size_mb=$(du -m "$tar_file" | cut -f1)
        log "Uploading codegrams archive (${size_mb} MB)..."
        gcloud compute scp --zone="$zone" --tunnel-through-iap --compress \
            "$tar_file" "$INSTANCE:~/vs-codegrams.tar.gz"
        gcloud compute ssh "$INSTANCE" --zone="$zone" --tunnel-through-iap --command="
            tar xzf ~/vs-codegrams.tar.gz -C $REMOTE_DIR && rm ~/vs-codegrams.tar.gz
        "
        rm "$tar_file"
        log "Codegrams uploaded and extracted"
    fi

    log "Upload complete"
}

install_deps() {
    local zone="$1"
    log "Installing dependencies..."
    gcloud compute ssh "$INSTANCE" --zone="$zone" --tunnel-through-iap --command="
        cd $REMOTE_DIR && pip3 install -q -r requirements.txt
    "
    log "Dependencies installed"
}

print_instructions() {
    local zone="$1"
    echo ""
    echo "============================================================"
    echo "  VM '$INSTANCE' is ready in $zone"
    echo "============================================================"
    echo ""
    echo "  SSH in:"
    echo "    gcloud compute ssh $INSTANCE --zone=$zone --tunnel-through-iap"
    echo ""
    echo "  Start training (note: use python3 on the VM):"
    echo "    cd ~/variation-sampler"
    echo "    python3 scripts/train.py --config configs/default.yaml"
    echo ""
    echo "  With W&B:"
    echo "    python3 scripts/train.py --config configs/default.yaml --wandb"
    echo ""
    echo "  Stop VM (saves disk, stops GPU billing):"
    echo "    ./scripts/deploy_cloud.sh --stop"
    echo ""
    echo "  Download checkpoints:"
    echo "    gcloud compute scp --recurse --tunnel-through-iap $INSTANCE:~/variation-sampler/checkpoints/ ./checkpoints/ --zone=$zone"
    echo ""
    echo "============================================================"
}

# ---------- main ----------

ACTION="${1:-deploy}"

case "$ACTION" in
    --stop)
        if active_zone=$(find_vm_zone); then
            log "Stopping VM in $active_zone..."
            gcloud compute instances stop "$INSTANCE" --zone="$active_zone"
            log "VM stopped. Disk preserved, GPU billing stopped."
        else
            log "VM '$INSTANCE' not found in any zone."
            exit 1
        fi
        ;;

    --start)
        if active_zone=$(find_vm_zone); then
            status=$(vm_status "$active_zone")
            if [ "$status" = "TERMINATED" ] || [ "$status" = "STOPPED" ]; then
                log "Starting VM in $active_zone..."
                gcloud compute instances start "$INSTANCE" --zone="$active_zone"
                wait_for_ssh "$active_zone"
                upload_code "$active_zone"
                print_instructions "$active_zone"
            elif [ "$status" = "RUNNING" ]; then
                log "VM already running in $active_zone"
                upload_code "$active_zone"
                print_instructions "$active_zone"
            else
                log "VM in unexpected state: $status"
                exit 1
            fi
        else
            log "VM '$INSTANCE' not found. Run without flags to create it."
            exit 1
        fi
        ;;

    --upload)
        if active_zone=$(find_vm_zone); then
            upload_code "$active_zone"
            log "Code uploaded."
        else
            log "VM '$INSTANCE' not found."
            exit 1
        fi
        ;;

    --ssh)
        if active_zone=$(find_vm_zone); then
            exec gcloud compute ssh "$INSTANCE" --zone="$active_zone" --tunnel-through-iap
        else
            log "VM '$INSTANCE' not found."
            exit 1
        fi
        ;;

    deploy|--deploy)
        # Check if VM already exists
        if active_zone=$(find_vm_zone); then
            status=$(vm_status "$active_zone")
            log "VM exists in $active_zone (status: $status)"

            if [ "$status" = "TERMINATED" ] || [ "$status" = "STOPPED" ]; then
                log "Starting stopped VM..."
                gcloud compute instances start "$INSTANCE" --zone="$active_zone"
            fi
            wait_for_ssh "$active_zone"
            upload_code "$active_zone"
            install_deps "$active_zone"
            print_instructions "$active_zone"
        else
            # Try creating in each zone
            created=false
            for zone in "${ZONES[@]}"; do
                if create_vm "$zone"; then
                    created=true
                    active_zone="$zone"
                    break
                fi
            done

            if [ "$created" = false ]; then
                log "ERROR: Could not create VM in any zone."
                exit 1
            fi

            wait_for_ssh "$active_zone"
            upload_code "$active_zone"
            install_deps "$active_zone"
            print_instructions "$active_zone"
        fi
        ;;

    *)
        echo "Usage: $0 [--deploy|--start|--stop|--upload|--ssh]"
        exit 1
        ;;
esac
