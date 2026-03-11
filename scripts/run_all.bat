@echo off
REM ============================================================
REM SimCLR Full Pipeline - Windows Batch Script
REM ============================================================
REM Usage: scripts\run_all.bat [DATASET]
REM ============================================================

setlocal

set DATASET=%1
if "%DATASET%"=="" set DATASET=stl10

echo ============================================
echo  SimCLR Full Pipeline
echo  Dataset: %DATASET%
echo ============================================

cd /d "%~dp0\.."

echo.
echo ========== STEP 1: SimCLR Pretraining ==========
python -m training.train_simclr --dataset %DATASET% --epochs 200 --batch_size 256
if errorlevel 1 goto :error

echo.
echo ========== STEP 2: Linear Evaluation ==========
python -m training.linear_probe --dataset %DATASET% --epochs 100
if errorlevel 1 goto :error

echo.
echo ========== STEP 3: kNN Evaluation ==========
python -m evaluation.knn_eval --dataset %DATASET% --k 200
if errorlevel 1 goto :error

echo.
echo ========== STEP 4: Semi-Supervised Fine-Tuning ==========
python -m training.fine_tune --dataset %DATASET% --fractions 0.01 0.10 1.0 --epochs 100
if errorlevel 1 goto :error

echo.
echo ========== STEP 5: Supervised Baseline ==========
python -m training.train_supervised --dataset %DATASET% --fractions 0.01 0.10 1.0 --epochs 100
if errorlevel 1 goto :error

echo.
echo ========== STEP 6: Visualization ==========
python -m evaluation.tsne_visualization --dataset %DATASET% --label_curve
if errorlevel 1 goto :error

echo.
echo ============================================
echo  Full pipeline completed successfully!
echo  Check results/ for outputs.
echo ============================================
goto :end

:error
echo.
echo ERROR: Pipeline failed at the above step.
exit /b 1

:end
endlocal
