import os
import re
import ast
import tempfile
import shutil
import subprocess
from typing import Dict, List, Any, Optional
from pathlib import Path


class SKiDLExecutionError(Exception):
    pass


class SKiDLExecutor:
    def __init__(self, output_dir: Optional[str] = None):
        if output_dir is None:
            self.output_dir = tempfile.mkdtemp(prefix="skidl_exec_")
            self._temp_dir = True
        else:
            self.output_dir = output_dir
            self._temp_dir = False
            os.makedirs(output_dir, exist_ok=True)
    
    def validate_code(self, code: str) -> Dict[str, Any]:
        errors = []
        warnings = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
            return {"valid": False, "errors": errors, "warnings": warnings}
        has_skidl_import = "from skidl import" in code or "import skidl" in code
        if not has_skidl_import:
            errors.append("Missing SKiDL import statement")
        components = self.extract_components(code)
        if len(components) == 0:
            warnings.append("No components found in code")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}
    
    def extract_components(self, code: str) -> List[Dict[str, str]]:
        components = []
        pattern = r"(\w+)\s*=\s*Part\s*\(\s*['\"](\w+)['\"]\s*,\s*['\"](\w+)['\"]\s*(?:,\s*value\s*=\s*['\"]([^'\"]+)['\"])?\s*\)"
        matches = re.finditer(pattern, code)
        for match in matches:
            components.append({"name": match.group(1), "library": match.group(2), "part": match.group(3), "value": match.group(4) if match.group(4) else None})
        return components
    
    def extract_nets(self, code: str) -> List[str]:
        nets = []
        pattern = r"(\w+)\s*=\s*Net\s*\("
        matches = re.finditer(pattern, code)
        for match in matches:
            nets.append(match.group(1))
        return nets
    
    def _parse_warnings(self, output: str) -> List[str]:
        warnings = []
        for line in output.split("\n"):
            if "WARNING:" in line.upper() or "Warning:" in line:
                warnings.append(line.strip())
        return warnings
    
    def execute(self, code: str, circuit_name: str = "circuit") -> Dict[str, Any]:
        validation = self.validate_code(code)
        if not validation["valid"]:
            raise SKiDLExecutionError(f"Invalid code: {validation['errors']}")
        if "generate_netlist()" not in code:
            code += "\n\ngenerate_netlist()\n"
        code_file = os.path.join(self.output_dir, f"{circuit_name}.py")
        with open(code_file, "w") as f:
            f.write(code)
        try:
            result = subprocess.run(["python", code_file], cwd=self.output_dir, capture_output=True, text=True, timeout=30)
            output = result.stdout + result.stderr
            warnings = self._parse_warnings(output)
            if result.returncode != 0:
                raise SKiDLExecutionError(f"Execution failed: {result.stderr}")
            netlist_file = os.path.join(self.output_dir, f"{circuit_name}.net")
            if not os.path.exists(netlist_file):
                netlist_files = list(Path(self.output_dir).glob("*.net"))
                netlist_file = str(netlist_files[0]) if netlist_files else None
            return {"success": True, "netlist_file": netlist_file, "warnings": warnings, "output": output}
        except subprocess.TimeoutExpired:
            raise SKiDLExecutionError("Execution timed out after 30 seconds")
        except Exception as e:
            raise SKiDLExecutionError(f"Execution error: {str(e)}")
    
    def generate_netlist(self, code: str, circuit_name: str = "circuit") -> str:
        result = self.execute(code, circuit_name)
        if not result["success"]:
            raise SKiDLExecutionError("Netlist generation failed")
        if result["netlist_file"] is None:
            raise SKiDLExecutionError("No netlist file generated")
        return result["netlist_file"]
    
    def cleanup(self):
        if self._temp_dir and os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
