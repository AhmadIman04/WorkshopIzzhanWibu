import subprocess
from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def home():
    # This command runs the Linux 'figlet' tool
    # It attempts to turn text into ASCII art
    try:
        result = subprocess.check_output(['figlet', 'Hello Workshop!'])
        return f"<pre>{result.decode('utf-8')}</pre>"
    except FileNotFoundError:
        return "Error: 'figlet' is not installed on this system!"


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000)
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)