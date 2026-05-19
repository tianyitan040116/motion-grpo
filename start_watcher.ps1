# Start watch_and_eval.py in background
$pythonPath = "C:\Users\tianyi\motion-agent\venv_grpo\Scripts\python.exe"
$scriptPath = "C:\Users\tianyi\motion-agent\watch_and_eval.py"
$logPath = "C:\Users\tianyi\motion-agent\experiments_grpo\grpo_kinematic\watcher.log"

Start-Process -FilePath $pythonPath `
    -ArgumentList "$scriptPath --exp-dir experiments_grpo/grpo_kinematic --every 100 --interval 60 --device cuda:0" `
    -WorkingDirectory "C:\Users\tianyi\motion-agent" `
    -RedirectStandardOutput $logPath `
    -RedirectStandardError "$logPath.err" `
    -WindowStyle Hidden

Write-Host "Watcher started in background. Logs: $logPath"
Write-Host "To stop: Get-Process python | Where-Object {`$_.CommandLine -like '*watch_and_eval*'} | Stop-Process"
