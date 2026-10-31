import json
import re
from pathlib import Path
from typing import Dict, Any

MEMORY_FILE = Path('memory.json')

SENSITIVE_PATTERNS = [
    r'\bpassword\b',
    r'\bpasscode\b',
    r'\botp\b',
    r'\bone[- ]time password\b',
    r'\brecovery code\b',
    r'\bpin\b',
    r'\b[0-9]{4,6}\b'
]


def load_memory() -> Dict[str, Any]:
    if not MEMORY_FILE.exists():
        return {}
    try:
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_memory(data: Dict[str, Any]) -> None:
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def is_sensitive(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pat, lowered) for pat in SENSITIVE_PATTERNS)


def remember_student_fact(student_id: str, fact: str) -> Dict[str, Any]:
    if is_sensitive(fact):
        return {
            'status': 'blocked',
            'message': 'Security Alert: Passwords, OTPs, and recovery codes cannot be saved to persistent memory.'
        }
    data = load_memory()
    facts = data.get(student_id, [])
    if fact not in facts:
        facts.append(fact)
    data[student_id] = facts
    save_memory(data)
    return {'status': 'success', 'message': f'Stored fact for {student_id}'}


def recall_student_memory(student_id: str) -> Dict[str, Any]:
    data = load_memory()
    facts = data.get(student_id, [])
    return {'student_id': student_id, 'facts': facts, 'count': len(facts)}
