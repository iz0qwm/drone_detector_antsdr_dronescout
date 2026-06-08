#!/bin/bash

set -e

UPDATER_DIR="/home/pi/tracker-mini-updater"
TRACKER_DIR="/home/pi/tracker-mini"

REQUEST_FILE="${UPDATER_DIR}/install-request.json"
CURRENT_FILE="${UPDATER_DIR}/current.json"

log() {

    echo "[INSTALLER] $(date '+%F %T') $1"

}

while true
do

    if [ ! -f "$REQUEST_FILE" ]; then

        sleep 5
        continue

    fi

    log "Install request detected"

    PACKAGE=$(jq -r '.package' "$REQUEST_FILE")
    BACKUP=$(jq -r '.backup' "$REQUEST_FILE")

    log "Package: $PACKAGE"
    log "Backup: $BACKUP"

    BACKUP_DIR="${UPDATER_DIR}/backups/${BACKUP}"

    TEST_INSTALL="${UPDATER_DIR}/test-install"

    #
    # STOP SERVICE
    #

    log "Stopping tracker-mini"

    systemctl stop tracker-mini

    sleep 3

    #
    # DEPLOY
    #

    log "Deploying backend"

    rm -rf \
        "${TRACKER_DIR}/backend"

    cp -a \
        "${TEST_INSTALL}/backend" \
        "${TRACKER_DIR}/"

    log "Deploying frontend"

    rm -rf \
        "${TRACKER_DIR}/frontend"

    cp -a \
        "${TEST_INSTALL}/frontend" \
        "${TRACKER_DIR}/"

    #
    # START SERVICE
    #

    log "Starting tracker-mini"

    systemctl start tracker-mini

    log "Waiting 30 seconds"

    sleep 30

    #
    # HEALTH CHECK
    #

    if curl \
        -s \
        --max-time 10 \
        http://127.0.0.1:5000/api/update/status \
        >/dev/null
    then

        log "Health check OK"

        cat > "$CURRENT_FILE" <<EOF
{
    "installed_package": "${PACKAGE}",
    "installed_at": "$(date --iso-8601=seconds)",
    "backup": "${BACKUP}",
    "status": "installed"
}
EOF

        rm -f "$REQUEST_FILE"

        log "Installation completed"

        continue

    fi

    #
    # ROLLBACK
    #

    log "Health check FAILED"

    log "Starting rollback"

    systemctl stop tracker-mini

    rm -rf \
        "${TRACKER_DIR}/backend"

    rm -rf \
        "${TRACKER_DIR}/frontend"

    tar -xzf \
        "${BACKUP_DIR}/backend.tar.gz" \
        -C "${TRACKER_DIR}"

    tar -xzf \
        "${BACKUP_DIR}/frontend.tar.gz" \
        -C "${TRACKER_DIR}"

    log "Backup restored"

    systemctl start tracker-mini

    sleep 30

    if curl \
        -s \
        --max-time 10 \
        http://127.0.0.1:5000/api/update/status \
        >/dev/null
    then

        log "Rollback successful"

        cat > "$CURRENT_FILE" <<EOF
{
    "installed_package": "${PACKAGE}",
    "installed_at": "$(date --iso-8601=seconds)",
    "backup": "${BACKUP}",
    "status": "rollback",
    "rollback_reason": "health_check_failed"
}
EOF

    else

        log "CRITICAL: rollback failed"

        cat > "$CURRENT_FILE" <<EOF
{
    "installed_package": "${PACKAGE}",
    "installed_at": "$(date --iso-8601=seconds)",
    "backup": "${BACKUP}",
    "status": "failed",
    "rollback_reason": "rollback_failed"
}
EOF

    fi

    rm -f "$REQUEST_FILE"

done