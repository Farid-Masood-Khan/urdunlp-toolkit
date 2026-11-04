# UrduNLP Execution Automation Script
param(
    [string]$Command = "demo",
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$VenvStreamlit = Join-Path $PSScriptRoot ".venv\Scripts\streamlit.exe"
$VenvPytest = Join-Path $PSScriptRoot ".venv\Scripts\pytest.exe"
$VenvRuff = Join-Path $PSScriptRoot ".venv\Scripts\ruff.exe"
$VenvMypy = Join-Path $PSScriptRoot ".venv\Scripts\mypy.exe"

switch ($Command) {
    "demo" {
        Write-Host "Launching UrduNLP Streamlit Demo App on port $Port..." -ForegroundColor Green
        & $VenvStreamlit run "demo/app.py" --server.port $Port --server.headless true
    }
    "eval" {
        Write-Host "Running evaluation benchmark and error slice analysis..." -ForegroundColor Yellow
        & $VenvPython -m src.urdu_nlp.evaluate
    }
    "data" {
        Write-Host "Generating stratified 80/10/10 dataset splits..." -ForegroundColor Yellow
        & $VenvPython -m src.urdu_nlp.dataset
    }
    "test" {
        Write-Host "Running UrduNLP test suite..." -ForegroundColor Cyan
        & $VenvPytest -v tests/
    }
    "lint" {
        Write-Host "Running ruff and mypy code quality checks..." -ForegroundColor Magenta
        & $VenvRuff check src tests demo
        & $VenvMypy src
    }
    default {
        Write-Host "Usage: .\run.ps1 [demo | eval | data | test | lint]" -ForegroundColor Red
    }
}

