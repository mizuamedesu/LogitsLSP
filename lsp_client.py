import subprocess
import json
import threading
import queue
import time
from typing import Optional, Dict, Any, List


class LSPClient:
    def __init__(self, server_command: List[str], workspace_path: str):
        self.server_command = server_command
        self.workspace_path = workspace_path
        self.process = None
        self.request_id = 0
        self.response_queue = queue.Queue()
        self.initialized = False
        self.document_uri = None
        self.document_version = 0

    def start(self):
        self.process = subprocess.Popen(
            self.server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False
        )

        reader_thread = threading.Thread(target=self._read_responses, daemon=True)
        reader_thread.start()

        self._initialize()

    def _initialize(self):
        init_params = {
            "processId": None,
            "rootUri": f"file://{self.workspace_path}",
            "capabilities": {
                "textDocument": {
                    "completion": {
                        "dynamicRegistration": True,
                        "completionItem": {
                            "snippetSupport": True
                        }
                    }
                }
            }
        }

        response = self._send_request("initialize", init_params)
        if response:
            self.initialized = True
            self._send_notification("initialized", {})

    def open_document(self, file_path: str, content: str):
        self.document_uri = f"file://{file_path}"
        self.document_version = 1

        params = {
            "textDocument": {
                "uri": self.document_uri,
                "languageId": self._get_language_id(file_path),
                "version": self.document_version,
                "text": content
            }
        }

        self._send_notification("textDocument/didOpen", params)

    def update_document(self, content: str):
        if not self.document_uri:
            return

        self.document_version += 1

        params = {
            "textDocument": {
                "uri": self.document_uri,
                "version": self.document_version
            },
            "contentChanges": [{
                "text": content
            }]
        }

        self._send_notification("textDocument/didChange", params)

    def get_completions(self, line: int, character: int) -> List[Dict[str, Any]]:
        if not self.initialized or not self.document_uri:
            return []

        params = {
            "textDocument": {
                "uri": self.document_uri
            },
            "position": {
                "line": line,
                "character": character
            }
        }

        response = self._send_request("textDocument/completion", params)

        if response and "result" in response:
            result = response["result"]
            if isinstance(result, dict) and "items" in result:
                return result["items"]
            elif isinstance(result, list):
                return result

        return []

    def _send_request(self, method: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params
        }

        self._send_message(request)

        timeout = 5
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self.response_queue.get(timeout=0.1)
                if response.get("id") == self.request_id:
                    return response
                else:
                    self.response_queue.put(response)
            except queue.Empty:
                continue

        return None

    def _send_notification(self, method: str, params: Dict[str, Any]):
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }

        self._send_message(notification)

    def _send_message(self, message: Dict[str, Any]):
        if not self.process or self.process.poll() is not None:
            return

        content = json.dumps(message)
        content_bytes = content.encode('utf-8')
        header = f"Content-Length: {len(content_bytes)}\r\n\r\n"

        self.process.stdin.write(header.encode('utf-8'))
        self.process.stdin.write(content_bytes)
        self.process.stdin.flush()

    def _read_responses(self):
        while True:
            if not self.process or self.process.poll() is not None:
                break

            try:
                headers = {}
                while True:
                    line = self.process.stdout.readline()
                    if not line:
                        return

                    line = line.decode('utf-8').strip()
                    if not line:
                        break

                    key, value = line.split(": ", 1)
                    headers[key] = value

                if "Content-Length" in headers:
                    content_length = int(headers["Content-Length"])
                    content = self.process.stdout.read(content_length)

                    try:
                        response = json.loads(content.decode('utf-8'))
                        if "id" in response:
                            self.response_queue.put(response)
                    except json.JSONDecodeError:
                        pass

            except Exception:
                continue

    def _get_language_id(self, file_path: str) -> str:
        extensions = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".rs": "rust",
            ".go": "go"
        }

        for ext, lang_id in extensions.items():
            if file_path.endswith(ext):
                return lang_id

        return "plaintext"

    def shutdown(self):
        if self.initialized:
            self._send_request("shutdown", {})
            self._send_notification("exit", {})

        if self.process:
            self.process.terminate()
            self.process.wait()