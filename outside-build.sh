#!/bin/bash

set -euo pipefail
trap 'trap - INT TERM; kill 0; exit 130' INT TERM

SKIP_BUILD=0
CLEAN=0
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip) SKIP_BUILD=1; shift ;;
        --clean) CLEAN=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done


SOURCE_HASH=$(
    find "$LOCAL_DIR" \
        -type d \( \
            -name .git -o \
            -name build -o \
            -name .cache -o \
            -name external \
        \) -prune -o \
        \( -name '*.cpp' -o -name '*.h' \) \
        -type f -print0 |
    sort -z |
    xargs -0 sha256sum |
    sha256sum |
    awk '{print $1}'
)
BUILD_STAMP="$LOCAL_DIR/.build-source-hash"

if [[ -f "$BUILD_STAMP" ]] && [[ -d "build" ]] && [[ "$(cat "$BUILD_STAMP")" == "$SOURCE_HASH" ]]; then
    echo "No .cpp/.h changes since last build. Using rsync locally"
    rsync -av  "${LOCAL_DIR}/Pain/resources/" "${LOCAL_DIR}/resources/"
    # rsync -av "${LOCAL_DIR}/Example/PainlessEditor/resources/" "${LOCAL_DIR}/resources/"
    rsync -av --chmod=F444,D775 "${LOCAL_DIR}/resources/" "${LOCAL_DIR}/build/resources/"
    exit 0
fi


VOLUME=65536
SERVER="192.168.200.105"
LOCAL_USER="jaoschmidt"
REMOTE_USER="admin"
PROJECT_DIR="Flappybird"

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
        cmake --version
        if [ "${CLEAN}" == "1" ]; then
            echo "Clean build requested, removing remote build..."
            rm -rf build
        fi
        if [ ! -f build/CMakeCache.txt ]; then
            echo "Configuring project..."
            rm -rf build
            cmake --preset default || {
                rm -rf build
                exit 1
            }
        fi

        echo "Building..."
        export CLICOLOR_FORCE=1
        cmake --build ./build -j$(nproc)
    '
EOF
then
    :   
else
    DBUS="unix:path=/run/user/$(id -u)/bus"
    DBUS_SESSION_BUS_ADDRESS=$DBUS notify-send "Build Failed" "Command failed" &
    DBUS_SESSION_BUS_ADDRESS=$DBUS paplay --volume=$VOLUME /usr/share/sounds/freedesktop/stereo/suspend-error.oga &
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
DBUS_SESSION_BUS_ADDRESS=$DBUS paplay --volume=$VOLUME /usr/share/sounds/freedesktop/stereo/complete.oga &

printf '%s\n' "$SOURCE_HASH" > "$BUILD_STAMP"
