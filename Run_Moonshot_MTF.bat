@echo off
chcp 65001 >NUL
title GG Moonshot v1d — MTF (LIVE)
setlocal
set "PY=python"
where %PY% >NUL 2>&1 || set "PY=py -3"

echo -- Upgrading pip (user) --
%PY% -m pip install --upgrade --user pip

echo -- Installing requirements --
%PY% -m pip install --user ccxt rich pandas numpy tzdata

echo -- Launching (LIVE) --
set GG_PAPER=0
%PY% gg_moonshot_mtf.py
echo -- Bot exited --
pause
