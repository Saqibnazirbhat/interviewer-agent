"""Launch script for the Interviewer Agent — checks deps, starts server, opens browser."""

import sys
import time
import webbrowser


REQUIRED_PACKAGES = [
    "fastapi",
    "uvicorn",
    "python-multipart",
    "PyPDF2",
    "python-docx",
    "openai",
    "PyGithub",
    "python-dotenv",
    "rich",
    "reportlab",
]

HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"


def check_deps():
    """Check all required packages are installed. Exit with instructions if any are missing."""
    missing = []
    for pkg in REQUIRED_PACKAGES:
        import_name = pkg.replace("-", "_").lower()
        # Map package names to import names where they differ
        import_map = {
            "python_multipart": "multipart",
            "pypdf2": "PyPDF2",
            "python_docx": "docx",
            "pygithub": "github",
            "python_dotenv": "dotenv",
        }
        check_name = import_map.get(import_name, import_name)

        try:
            __import__(check_name)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"  Missing packages: {', '.join(missing)}")
        print()
        print("  Install all dependencies with:")
        print(f"    {sys.executable} -m pip install -r requirements.txt")
        print()
        sys.exit(1)
    else:
        print("  All dependencies are installed.")


def verify_api():
    """Verify NVIDIA API key and model connectivity on startup."""
    from dotenv import load_dotenv
    load_dotenv()

    from src.llm_client import verify_nvidia_connection
    ok, msg = verify_nvidia_connection()
    if ok:
        print(f"  [OK] {msg}")
    else:
        print(f"  [FAIL] {msg}")
        print()
        print("  The server will start, but AI features will fail.")
        print("  Add a valid NVIDIA_API_KEY to your .env file.")
        print("  Get one free at https://build.nvidia.com")
    print()
    return ok


def main():
    print()
    print("  +------------------------------------------+")
    print("  |        Interviewer Agent  v2.0            |")
    print("  |        Powered by NVIDIA NIM              |")
    print("  +------------------------------------------+")
    print()

    print("  Checking dependencies...")
    check_deps()
    print()

    print("  Verifying AI connection...")
    verify_api()

    print(f"  Interviewer Agent is running -> {URL}")
    print()
    print("  Press Ctrl+C to stop the server.")
    print()

    # Open browser after a short delay
    def open_browser():
        time.sleep(1.5)
        webbrowser.open(URL)

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Start FastAPI server
    import uvicorn
    uvicorn.run(
        "src.web.app:app",
        host=HOST,
        port=PORT,
        log_level="warning",
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        sys.exit(0)
