# Rebuild the search index after adding/changing PDFs in data\pdfs\.
$venv = "C:\Users\mounika_c\venvs\pdfrag"
Set-Location $PSScriptRoot
& "$venv\Scripts\python.exe" -m src.ingest
