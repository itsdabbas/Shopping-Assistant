# Environment Configuration

When executing Python commands, ALWAYS use the interpreter located at:
.\.venv\Scripts\python.exe

Do not run `python` directly from the global path.
Do not use the system Python.
Before running project scripts or installing packages, use the project virtual environment at `.\.venv\Scripts\python.exe`.
For package installation, use:
`.\.venv\Scripts\python.exe -m pip install ...`
For script execution, use:
`.\.venv\Scripts\python.exe <script_name>.py`