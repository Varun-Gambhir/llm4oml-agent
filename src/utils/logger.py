# ============================================================================
# File: src/utils/logger.py
# ============================================================================
"""Structured logging utility."""

import json
import logging
from pathlib import Path
from typing import Any, Dict
from datetime import datetime


class StructuredLogger:
    """Structured logger for agent execution."""
    
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup standard logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # JSON log for structured data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_log_path = self.log_dir / f"execution_{timestamp}.json"
        self.execution_log: Dict[str, Any] = {
            "start_time": datetime.now().isoformat(),
            "iterations": []
        }
    
    def log_iteration(
        self,
        iteration: int,
        node_name: str,
        data: Dict[str, Any]
    ):
        """Log an iteration event."""
        self.logger.info(f"[Iter {iteration}] {node_name}: {list(data.keys())}")
        
        # Ensure iteration slot exists
        while len(self.execution_log["iterations"]) < iteration:
            self.execution_log["iterations"].append({
                "iteration_number": len(self.execution_log["iterations"]) + 1
            })
        
        # Update iteration log
        iter_log = self.execution_log["iterations"][iteration - 1]
        iter_log.update(data)
    
    def log_error(self, message: str, exception: Exception):
        """Log an error."""
        self.logger.error(f"{message}: {exception}")
        self.execution_log["error"] = {
            "message": message,
            "exception": str(exception),
            "timestamp": datetime.now().isoformat()
        }
    
    def save(self):
        """Save structured log to JSON."""
        self.execution_log["end_time"] = datetime.now().isoformat()
        
        with open(self.json_log_path, "w", encoding="utf-8") as f:
            json.dump(self.execution_log, f, indent=2, default=str)
        
        self.logger.info(f"Execution log saved to {self.json_log_path}")
    
    def get_log(self) -> Dict[str, Any]:
        """Get current execution log."""
        return self.execution_log
