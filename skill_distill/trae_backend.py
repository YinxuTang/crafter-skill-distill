"""Strict-JSON, non-interactive Trae CLI backend."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


class TraeCLIBackend:
    def __init__(
        self,
        model: str = "GPT-6.1-Sol",
        executable: str = "traex",
        workspace: Optional[str] = None,
        timeout_seconds: int = 600,
        log_dir: Optional[str] = None,
    ):
        self.model = model
        self.executable = executable
        self.workspace = str(Path(workspace or Path.cwd()).resolve())
        self.timeout_seconds = timeout_seconds
        self.log_dir = Path(log_dir).resolve() if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        value = text.strip()
        if value.startswith("```"):
            value = value.split("\n", 1)[1].rsplit("```", 1)[0]
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("Trae CLI response must be a JSON object")
        return parsed

    def generate(
        self,
        prompt: str,
        output_schema: str,
        audit_name: str,
    ) -> Dict[str, Any]:
        schema_path = str(Path(output_schema).resolve())
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as output_file:
            output_path = output_file.name
        # Prefer `traex -c model=...` (same as interactive CLI); `-m` fails
        # metadata resolution for several Sol models in this environment.
        command = [
            self.executable,
            "exec",
            "-c",
            f'model="{self.model}"',
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "--output-schema",
            schema_path,
            "--output-last-message",
            output_path,
            "--cd",
            self.workspace,
            "-",
        ]
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
                env=os.environ.copy(),
            )
            response_text = Path(output_path).read_text(encoding="utf-8")
            if self.log_dir:
                (self.log_dir / f"{audit_name}.prompt.txt").write_text(
                    prompt, encoding="utf-8"
                )
                (self.log_dir / f"{audit_name}.stdout.txt").write_text(
                    completed.stdout, encoding="utf-8"
                )
                (self.log_dir / f"{audit_name}.stderr.txt").write_text(
                    completed.stderr, encoding="utf-8"
                )
                (self.log_dir / f"{audit_name}.response.json").write_text(
                    response_text, encoding="utf-8"
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"traecli failed with exit {completed.returncode}: "
                    f"{completed.stderr[-2000:]}"
                )
            return self._parse_json(response_text)
        finally:
            Path(output_path).unlink(missing_ok=True)
