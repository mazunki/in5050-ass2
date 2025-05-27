#!/bin/sh
set -eu

PROJECT_USER=in5050-g01
PROJECT_ROOT="/home/${PROJECT_USER}/in5050-ass3/project"
SRC_DIR="${PROJECT_ROOT}/src"
BUILD_DIR="${PROJECT_ROOT}/build"
WORKDIR="${PROJECT_ROOT}/workdir"

ASSETS_DIR="/home/${PROJECT_USER}/assets"
# ASSETS_DIR="/mnt/sdcard/cipr"
REPORT_FILE="report.nsys-rep"

# BUILDER="${BUILDER:-${PROJECT_USER}@in5050}"
# RUNNER="${RUNNER:-${PROJECT_USER}@in5050-2016-10}"
BUILDER="${BUILDER:-${PROJECT_USER}@tegra-3}"
COMPRUNNER="${RUNNER:-${PROJECT_USER}@tegra-3}"
IORUNNER="${RUNNER:-${PROJECT_USER}@in5050-2016-10}"
BUILD_MODE="${BUILD_MODE:-Debug}"
TRACE_LEVEL="${TRACE_LEVEL:-4}"

VID_HEIGHT="${VID_HEIGHT:-288}"
VID_WIDTH="${VID_WIDTH:-352}"
VID_OUTPUT_ENC="output.c63"
VID_OUTPUT_DEC="output"
REPORT_FILE_ENC="encoding.nsys-rep"
REPORT_FILE_DEC="decoding.nsys-rep"
VID_INPUT="${ASSETS_DIR}/${VIDEO:-foreman}.yuv"
VID_FLAGS="$@"

cd "$(dirname "$0")"

builder() {
	echo "[BUILDER] $*" >&2
	ssh "$BUILDER" "$*"
}

iorunner() {
	echo "[IO] $*" >&2
	ssh "$IORUNNER" "$*"
}

comprunner() {
	echo "[SERVER] $*" >&2
	ssh "$COMPRUNNER" "$*"
}

TEGRA_NODE=$(builder /opt/DIS/sbin/disinfo get-nodeid -hostname "${COMPRUNNER#*@}")
PC_NODE=$(builder /opt/DIS/sbin/disinfo get-nodeid -hostname "${IORUNNER#*@}")

pipeline() {
	echo "[PIPELINE] Updating build server..."
	iorunner "mkdir -p '${WORKDIR}'"
	(set -x; rsync -av --progress . "${BUILDER}:${PROJECT_ROOT}/")

	echo "[PIPELINE] updating cmake..."
	builder "cd '${PROJECT_ROOT}' && rm -rf build && cmake -B build -DCMAKE_BUILD_TYPE='${BUILD_MODE}' -DTRACE_LEVEL='${TRACE_LEVEL}'"

	echo "[PIPELINE] building project..."
	builder "cd '${BUILD_DIR}' && make"

	iorunner "mkdir -p '${WORKDIR}'"
	if [ ! "${IORUNNER}" = "${BUILDER}" ]; then
	  echo "[PIPELINE] syncing build machine with gpu machine..."
	  (set -x; ssh "${BUILDER}" "rsync -av --progress '${BUILD_DIR}/' '${IORUNNER}:${BUILD_DIR}/'")
	  (set -x; ssh "${BUILDER}" "rsync -av --progress '${SRC_DIR}/' '${IORUNNER}:${SRC_DIR}/'")
	fi

	echo "[PIPELINE] running profiling on gpu machine..."
	echo "[PIPELINE] wiping workdir..."
	iorunner "rm -rf '${WORKDIR}'"
	iorunner "mkdir -p '${WORKDIR}'"


	cmd_srv="${BUILD_DIR}/c63server -r '${PC_NODE}'"
	cmd_enc="${BUILD_DIR}/c63client -r '${TEGRA_NODE}' -h '${VID_HEIGHT}' -w '${VID_WIDTH}' ${VID_FLAGS} -o '${VID_OUTPUT_ENC}' '${VID_INPUT}'"
	cmd_dec="${BUILD_DIR}/c63dec '${VID_OUTPUT_ENC}' '${VID_OUTPUT_DEC}'"
	echo ${cmd_dec}

	echo "[PIPELINE] server..."
	comprunner "cd '${WORKDIR}' && ${cmd_srv}" &

	echo "[PIPELINE] encoding..."
	iorunner "cd '${WORKDIR}' && nsys profile --trace=cuda,nvtx --output '${REPORT_FILE_ENC}' ${cmd_enc}" || { echo "runner encoder failed with errno $?"; exit 1; }
	# runner "cd '${WORKDIR}' && ${cmd_enc}" || { echo "runner encoder failed with errno $?"; exit 1; }

	echo "[PIPELINE] decoding..."
	# runner "cd '${WORKDIR}' && nsys profile --trace=cuda,nvtx --output '${REPORT_FILE_DEC}' ${cmd_dec}" || { echo "runner decoder failed with errno $?"; true; }
	iorunner "cd '${WORKDIR}' && ${cmd_dec}" || { echo "runner decoder failed with errno $?"; exit 2; }

	echo "[PIPELINE] fetching profiling report..."
  (set -x; rm -r ../workdir || true)
	(set -x; rsync -av --progress "$IORUNNER:$WORKDIR/" "../workdir/")
}


pipeline
