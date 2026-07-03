#!/bin/bash

set -euo pipefail
trap 'trap - INT TERM; kill 0; exit 130' INT TERM

SERVER="192.168.200.105"
LOCAL_USER="jaoschmidt"
REMOTE_USER="admin"
PROJECT_DIR="Flappybird"


LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PASSWORD=$LOCAL_DIR/PASSWORD

REMOTE_SRC="/home/${REMOTE_USER}/projects/${PROJECT_DIR}"
SSH_OPTS="-o StrictHostKeyChecking=no
        -o ControlMaster=auto
        -o ControlPersist=3600
        -o ControlPath=/tmp/ssh_mux_%h_%p_%r"
REMOTE_BUILD="${REMOTE_SRC}/build"

echo "Syncing source with rsync"

sshpass -f${PASSWORD} rsync -az --delete --mkpath \
    --exclude=".git" \
    --exclude="build/" \
    --exclude=".cache" \
    -e "ssh $SSH_OPTS" \
    "$LOCAL_DIR/" \
    "$REMOTE_USER@$SERVER:$REMOTE_SRC/"

echo "Cmake build"

if sshpass -f${PASSWORD} ssh -t \
    "$REMOTE_USER@$SERVER" << EOF
docker run --rm \
    -v "$REMOTE_SRC:${LOCAL_DIR}" \
    -v "$REMOTE_BUILD:${LOCAL_DIR}/build" \
    build-container \
    bash -c '
        cd ${LOCAL_DIR}
        if [ ! -f build/CMakeCache.txt ]; then
            echo "Configuring project..."
            rm -rf build
            cmake --preset default || {
                rm -rf build
                exit 1
            }
        fi

        echo "Building..."
        cmake --build ./build -j$(nproc)
    '
EOF
then
    :   
else
    DBUS="unix:path=/run/user/$(id -u)/bus"
    DBUS_SESSION_BUS_ADDRESS=$DBUS notify-send "Build Failed" "Command failed" &
    DBUS_SESSION_BUS_ADDRESS=$DBUS paplay --volume 45536 /usr/share/sounds/freedesktop/stereo/suspend-error.oga &
    exit 1
fi

echo "Downloading artifacts..."

mkdir -p "$LOCAL_DIR/build"

sshpass -f${PASSWORD} rsync -az \
    -e "ssh $SSH_OPTS" \
    "$REMOTE_USER@$SERVER:$REMOTE_BUILD/" \
    "$LOCAL_DIR/build/"

echo "Build successful."

DBUS="unix:path=/run/user/$(id -u)/bus"
DBUS_SESSION_BUS_ADDRESS=$DBUS notify-send "Build Complete" "Command succeeded" &
DBUS_SESSION_BUS_ADDRESS=$DBUS paplay /usr/share/sounds/freedesktop/stereo/complete.oga &
