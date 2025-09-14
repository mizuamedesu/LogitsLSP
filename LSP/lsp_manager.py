import os
import subprocess
import shutil
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from lsp_client import LSPClient


class LSPManager:
    LSP_CONFIGS = {
        "python": {
            "servers": [
                {
                    "name": "pylsp",
                    "command": ["pylsp"],
                    "install_check": "pylsp",
                    "install_cmd": "pip install python-lsp-server"
                },
                {
                    "name": "pyright",
                    "command": ["pyright-langserver", "--stdio"],
                    "install_check": "pyright-langserver",
                    "install_cmd": "npm install -g pyright"
                }
            ],
            "extensions": [".py", ".pyw"],
            "language_id": "python"
        },
        "typescript": {
            "servers": [
                {
                    "name": "typescript-language-server",
                    "command": ["typescript-language-server", "--stdio"],
                    "install_check": "typescript-language-server",
                    "install_cmd": "npm install -g typescript typescript-language-server"
                }
            ],
            "extensions": [".ts", ".tsx"],
            "language_id": "typescript"
        },
        "javascript": {
            "servers": [
                {
                    "name": "typescript-language-server",
                    "command": ["typescript-language-server", "--stdio"],
                    "install_check": "typescript-language-server",
                    "install_cmd": "npm install -g typescript typescript-language-server"
                }
            ],
            "extensions": [".js", ".jsx", ".mjs"],
            "language_id": "javascript"
        },
        "rust": {
            "servers": [
                {
                    "name": "rust-analyzer",
                    "command": ["rust-analyzer"],
                    "install_check": "rust-analyzer",
                    "install_cmd": "rustup component add rust-analyzer"
                }
            ],
            "extensions": [".rs"],
            "language_id": "rust"
        },
        "cpp": {
            "servers": [
                {
                    "name": "clangd",
                    "command": ["clangd"],
                    "install_check": "clangd",
                    "install_cmd": "apt-get install clangd or brew install llvm"
                }
            ],
            "extensions": [".cpp", ".cc", ".cxx", ".hpp", ".h"],
            "language_id": "cpp"
        },
        "c": {
            "servers": [
                {
                    "name": "clangd",
                    "command": ["clangd"],
                    "install_check": "clangd",
                    "install_cmd": "apt-get install clangd or brew install llvm"
                }
            ],
            "extensions": [".c", ".h"],
            "language_id": "c"
        },
        "go": {
            "servers": [
                {
                    "name": "gopls",
                    "command": ["gopls"],
                    "install_check": "gopls",
                    "install_cmd": "go install golang.org/x/tools/gopls@latest"
                }
            ],
            "extensions": [".go"],
            "language_id": "go"
        },
        "java": {
            "servers": [
                {
                    "name": "jdtls",
                    "command": ["jdtls"],
                    "install_check": "jdtls",
                    "install_cmd": "Download Eclipse JDT Language Server"
                }
            ],
            "extensions": [".java"],
            "language_id": "java"
        }
    }

    def __init__(self, workspace_path: str = None, auto_install: bool = False):
        self.workspace_path = workspace_path or os.getcwd()
        self.auto_install = auto_install
        self.active_clients: Dict[str, LSPClient] = {}
        self.server_cache: Dict[str, Dict[str, Any]] = {}
        self._load_server_cache()

    def _load_server_cache(self):
        cache_file = Path.home() / ".cache" / "logitslsp" / "servers.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self.server_cache = json.load(f)
            except:
                self.server_cache = {}

    def _save_server_cache(self):
        cache_dir = Path.home() / ".cache" / "logitslsp"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "servers.json"

        with open(cache_file, 'w') as f:
            json.dump(self.server_cache, f, indent=2)

    def detect_language(self, code: str, file_path: Optional[str] = None) -> Optional[str]:

        if file_path:
            ext = Path(file_path).suffix.lower()
            for lang, config in self.LSP_CONFIGS.items():
                if ext in config["extensions"]:
                    return lang

        code_lower = code.lower()

        if "def " in code or "import " in code or "class " in code and ":" in code:
            return "python"

        if "function " in code or "const " in code or "let " in code or "interface " in code:
            if "interface " in code or ": " in code and "{" in code:
                return "typescript"
            return "javascript"

        if "fn " in code or "impl " in code or "trait " in code or "pub " in code:
            return "rust"

        if "#include" in code or "int main" in code:
            if "class " in code or "namespace " in code or "template" in code:
                return "cpp"
            return "c"

        if "package " in code or "func " in code and "{" in code:
            return "go"

        if "public class" in code or "private " in code or "protected " in code:
            return "java"

        return None

    def _check_server_installed(self, command: str) -> bool:
        return shutil.which(command) is not None

    def _find_available_server(self, language: str) -> Optional[Dict[str, Any]]:
        if language not in self.LSP_CONFIGS:
            return None

        config = self.LSP_CONFIGS[language]

        for server in config["servers"]:
            if self._check_server_installed(server["install_check"]):
                return server

        return None

    def _install_server(self, language: str) -> bool:
        if language not in self.LSP_CONFIGS:
            return False

        config = self.LSP_CONFIGS[language]
        server = config["servers"][0]


        if self.auto_install:
            try:
                if "pip " in server['install_cmd']:
                    subprocess.run(server['install_cmd'].split(), check=True)
                elif "npm " in server['install_cmd']:
                    subprocess.run(server['install_cmd'].split(), check=True)
                else:
                    return False

                return self._check_server_installed(server["install_check"])
            except:
                return False
        else:
            return False

    def get_client(self, language: str = None, code: str = None,
                   file_path: str = None) -> Optional[LSPClient]:

        if not language:
            language = self.detect_language(code, file_path)
            if not language:
                return None

        if language in self.active_clients:
            return self.active_clients[language]

        server = self._find_available_server(language)

        if not server:

            if self.auto_install:
                if self._install_server(language):
                    server = self._find_available_server(language)

            if not server:
                config = self.LSP_CONFIGS.get(language, {})
                if config and config.get("servers"):
                    return None

        client = LSPClient(server["command"], self.workspace_path)

        try:
            client.start()
            self.active_clients[language] = client

            self.server_cache[language] = {
                "server": server["name"],
                "command": server["command"]
            }
            self._save_server_cache()

            return client

        except Exception as e:
            return None

    def open_document(self, client: LSPClient, file_path: str, content: str):
        if client:
            client.open_document(file_path, content)

    def update_document(self, client: LSPClient, content: str):
        if client:
            client.update_document(content)

    def get_completions(self, language: str, code: str,
                        line: int, character: int) -> List[Dict[str, Any]]:
        client = self.get_client(language=language, code=code)

        if not client:
            return []

        temp_file = f"/tmp/temp.{self.LSP_CONFIGS[language]['extensions'][0]}"

        if not client.document_uri:
            client.open_document(temp_file, code)
        else:
            client.update_document(code)

        return client.get_completions(line, character)

    def shutdown_all(self):
        for language, client in self.active_clients.items():
            client.shutdown()

        self.active_clients.clear()

    def shutdown(self, language: str):
        if language in self.active_clients:
            self.active_clients[language].shutdown()
            del self.active_clients[language]