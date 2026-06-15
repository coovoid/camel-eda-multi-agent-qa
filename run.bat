@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1
title EDA多智能体问答系统

cd /d "%~dp0"

echo ==================================================
echo   EDA 多智能体整合问答系统
echo ==================================================
echo.

set "PROJECT_DIR=%~dp0"
if not exist "%PROJECT_DIR%agent.py" (
    echo [错误] 未找到 agent.py
    echo 请在本脚本所在目录运行，或确认项目已完整解压
    goto :fail
)

echo 项目目录: %PROJECT_DIR%
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python
    echo 请安装 Python 3.10 / 3.11 / 3.12，并勾选 Add Python to PATH
    echo 下载: https://www.python.org/downloads/
    goto :fail
)

python --version
python -c "import sys; print('Python', sys.version); sys.exit(1 if sys.version_info >= (3, 13) else 0)"
if errorlevel 1 (
    echo [警告] Python 3.13+ 可能与 camel-ai 0.2.38 不兼容，建议使用 3.10-3.12
    echo.
)

set "VENV_DIR=%PROJECT_DIR%.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [1/5] 创建虚拟环境 .venv ...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败
        goto :fail
    )
) else (
    echo [1/5] 虚拟环境已存在
)

if not exist "%VENV_PY%" (
    echo [错误] 未找到虚拟环境 Python: %VENV_PY%
    goto :fail
)

echo [2/5] 安装依赖（首次约 3-8 分钟，请耐心等待）...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail
if exist "%PROJECT_DIR%requirements.txt" (
    "%VENV_PY%" -m pip install -r "%PROJECT_DIR%requirements.txt"
    if errorlevel 1 goto :fail
) else (
    "%VENV_PY%" -m pip install "camel-ai==0.2.38" "requests_oauthlib>=2.0.0" "unstructured>=0.16.0"
    if errorlevel 1 goto :fail
    "%VENV_PY%" -m pip install "pydantic>=2.9,<2.10" "psutil>=5.9.8,<6"
    if errorlevel 1 goto :fail
    "%VENV_PY%" -m pip install "streamlit>=1.28.0" "requests>=2.28.0" "python-dotenv>=1.0.0" "langchain-text-splitters>=0.2.0" "PyPDF2>=3.0.0" "python-docx>=1.0.0" "pandas>=2.0.0" "openpyxl>=3.1.0"
    if errorlevel 1 goto :fail
)

echo       验证模块导入 ...
"%VENV_PY%" -c "from multi_agent_backend import initialize_system; print('模块检查通过')"
if errorlevel 1 (
    echo [错误] 模块导入失败，可删除 .venv 后重新运行 run.bat
    goto :fail
)

echo [3/5] 检查配置 ...
if not exist "%PROJECT_DIR%api_key.env" (
    if exist "%PROJECT_DIR%api_key.env.example" (
        copy /Y "%PROJECT_DIR%api_key.env.example" "%PROJECT_DIR%api_key.env" >nul
        echo [提示] 已生成 api_key.env，请填入魔搭令牌后重新运行
        echo 获取地址: https://modelscope.cn/my/overview
        start "" notepad "%PROJECT_DIR%api_key.env"
        goto :fail
    ) else (
        echo [提示] 未找到 api_key.env，可在网页侧栏填写 API 密钥
        echo.
    )
)

echo [4/5] 启动服务 ...
echo 地址: http://localhost:8501
echo 按 Ctrl+C 可停止服务
echo.

"%VENV_PY%" -m streamlit run agent.py --server.port 8501 --browser.gatherUsageStats false
if errorlevel 1 (
    echo.
    echo [错误] Streamlit 启动失败，请查看上方输出
    goto :fail
)

goto :end

:fail
echo.
echo 按任意键关闭窗口 ...
pause
exit /b 1

:end
endlocal
exit /b 0
