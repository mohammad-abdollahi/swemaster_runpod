#!/usr/bin/env bash
# Wait for the parallel SWE-World training to finish (or the GPU to free up),
# then launch a clean SWE-Master scaled run.
#
# Watches for the parallel run's process to exit; once it does, launches
# run_scaled.sh.
set -e

ROGUE_PID=${ROGUE_PID:-161602}

echo "Waiting for parallel rogue training (pid ${ROGUE_PID}) to exit..."
while kill -0 ${ROGUE_PID} 2>/dev/null; do
    sleep 60
    # Re-check the gcs_server is still alive (Ray cluster healthy)
    if ! pgrep -f "gcs_server" >/dev/null 2>&1; then
        echo "$(date): Ray gcs_server gone — rogue probably crashed; proceeding"
        break
    fi
done
echo "$(date): rogue exited or Ray cluster down; cleaning up..."

# Wait an extra 30s for Ray/Docker cleanup
sleep 30
sg docker -c "docker ps -a -q --filter status=exited | xargs -r docker rm -f" 2>/dev/null | wc -l

echo "$(date): launching clean SWE-Master scaled-3..."
exec bash /home/temp/swemaster_runpod/run_scaled.sh
