#!/usr/bin/env bash
set -o pipefail
set -o errexit
set -o nounset

PACKAGE_ROOT=$(cd "$(dirname "${0}")"; echo $(pwd))
GLOBAL_PYTHON_PATH=$(which python3)
DEPENDENCIES=("pandas" "openpyxl" "streamlit" "langchain" "cohere" "openpyxl" "openai" "tiktoken" "faiss-cpu" "nemoguardrails" "Flask" "langchain-openai" "pypdf")

echo "[Info] PACKAGE_ROOT: ${PACKAGE_ROOT}"
echo "[Info] GLOBAL_PYTHON_PATH: ${GLOBAL_PYTHON_PATH}"

# check python path using "which python"
echo "[Info] Setting up Pipenv in ${PACKAGE_ROOT}"
cd "${PACKAGE_ROOT}"
pipenv --python "${GLOBAL_PYTHON_PATH}"

# From now on we are in the virtual python environment created by pipenv
echo "[Info] Update pip in Pipenv"
pipenv run pip install --upgrade pip

# installing dependencies
for i in ${!DEPENDENCIES[@]};
do
  echo "[Info] Installing ${DEPENDENCIES[$i]}"
  pipenv install "${DEPENDENCIES[$i]}"
done

echo "[Done]"
echo ""
echo "[Info] Configuration direct in start.sh, execute to start server"
exit 0