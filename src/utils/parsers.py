# ============================================================================
# File: src/utils/parsers.py
# ============================================================================
"""Parsing utilities for LLM outputs."""

import json
import re
from typing import Any, Dict, Optional


class ResponseParser:
    """Parse and clean LLM responses."""
    
    @staticmethod
    def extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from text with code fences."""
        # Try to find JSON in code fences
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return None
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    
    @staticmethod
    def clean_latex(text: str) -> str:
        """Clean LaTeX output from LLM."""
        # Remove markdown code fences
        text = re.sub(r'```latex\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        
        # Ensure proper document structure
        if '\\documentclass' not in text:
            text = '\\documentclass{article}\n\\begin{document}\n' + text + '\n\\end{document}'
        
        return text.strip()
    
    @staticmethod
    def extract_error_steps(text: str) -> Dict[str, list]:
        """
        Extract step-level error indices from feedback text.
        
        Expected format in feedback:
        - Hallucination Steps: [1, 5, 12]
        - Missing Steps: [3, 7]
        - ...
        """
        error_types = {
            'hallucinations': r'Hallucination Steps?:\s*\[([\d,\s]*)\]',
            'missing_steps': r'Missing Steps?:\s*\[([\d,\s]*)\]',
            'operator_errors': r'Operator Error Steps?:\s*\[([\d,\s]*)\]',
            'assumption_violations': r'Assumption Violation Steps?:\s*\[([\d,\s]*)\]',
            'flagged_steps': r'Flagged Steps?:\s*\[([\d,\s]*)\]'
        }
        
        error_sets = {}
        for error_type, pattern in error_types.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                steps_str = match.group(1)
                if steps_str.strip():
                    steps = [int(s.strip()) for s in steps_str.split(',') if s.strip()]
                    error_sets[error_type] = steps
                else:
                    error_sets[error_type] = []
            else:
                error_sets[error_type] = []
        
        return error_sets
