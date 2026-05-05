@echo off
for %%a in ("*.mp4") do ffmpeg -i "%%~a" "%%~na.wav"