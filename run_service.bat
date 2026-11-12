@echo off
setlocal
echo ========================================================
echo Starting NCM Sovereign Summarizer & Inspector Service
echo ========================================================

set NCM_HOME=%~dp0
set NCM_ROOT=%~dp0ncm
set NCM_DEVICE=cpu
set HF_HOME=%~dp0hf-home
set HF_HUB_CACHE=%~dp0hf-home\hub
set TRANSFORMERS_CACHE=%~dp0hf-home\hub
set HF_HUB_DISABLE_XET=1
set HF_HUB_OFFLINE=1

if exist "%~dp0.venv-cpu\Scripts\python.exe" (
    set PY="%~dp0.venv-cpu\Scripts\python.exe"
) else (
    set PY=python
)

echo Using Python: %PY%
%PY% service\summarize_server.py --port 8031 --host 127.0.0.1 --quantize int8
endlocal

