import datetime
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ProcessingState:
    is_processing: bool = False
    should_stop: bool = False
    current_file: Optional[str] = None
    progress: float = 0.0
    error: Optional[str] = None
    completion_message: Optional[str] = None
    log_messages: List[str] = field(default_factory=list)
    console_output: str = ""
    type_selections: Dict[str, str] = field(default_factory=dict)
    active_processes: List[subprocess.Popen] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add_log(self, message: str):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_messages.append(f"[{timestamp}] {message}")
        self.console_output += f"[{timestamp}] {message}\n"

    def add_process(self, process: subprocess.Popen):
        with self.lock:
            self.active_processes.append(process)

    def remove_process(self, process: subprocess.Popen):
        with self.lock:
            if process in self.active_processes:
                self.active_processes.remove(process)

    def terminate_all_processes(self):
        with self.lock:
            for process in self.active_processes:
                process.terminate()
            self.active_processes.clear()
