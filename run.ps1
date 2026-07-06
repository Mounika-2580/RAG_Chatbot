# Launch the PDF RAG chatbot.
# The virtual environment lives OUTSIDE OneDrive (OneDrive locks/dehydrates venv
# files and breaks it). Code stays here in OneDrive; packages live in $venv.
$venv = "C:\Users\mounika_c\venvs\pdfrag"
Set-Location $PSScriptRoot
& "$venv\Scripts\python.exe" -m streamlit run app.py
