import json
import re
from pathlib import Path
from typing import Dict, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from memory import remember_student_fact, recall_student_memory
from security import check_record_access, stage_approval_request

STUDENT_FILE = Path('data/students.json')
HANDBOOK_FILE = Path('data/student_handbook.txt')


def load_students_data() -> Dict[str, Any]:
    with open(STUDENT_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_fee_status(student_id: str, session_user_id: str, secure_mode: bool = False) -> Dict[str, Any]:
    if secure_mode:
        auth = check_record_access(session_user_id, student_id)
        if not auth['allowed']:
            return {'error': auth['error']}
    students = load_students_data()
    data = students.get(student_id)
    if not data:
        return {'error': f"Student ID '{student_id}' does not exist in records."}
    return {
        'student_id': student_id,
        'name': data['name'],
        'fee_status': data['fee_status'],
        'total_fee': data['total_fee'],
        'paid_amount': data['paid_amount'],
        'balance': data['balance']
    }


def get_scholarship_status(student_id: str, session_user_id: str, secure_mode: bool = False) -> Dict[str, Any]:
    if secure_mode:
        auth = check_record_access(session_user_id, student_id)
        if not auth['allowed']:
            return {'error': auth['error']}
    students = load_students_data()
    data = students.get(student_id)
    if not data:
        return {'error': f"Student ID '{student_id}' does not exist in records."}
    return {
        'student_id': student_id,
        'name': data['name'],
        'scholarship_status': data['scholarship_status']
    }


def _load_handbook_chunks():
    if not HANDBOOK_FILE.exists():
        return []
    with open(HANDBOOK_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    sections = re.split(r'(SECTION \d+: [^\n]+)', content)
    chunks = []
    for i in range(1, len(sections), 2):
        chunk_text = f"{sections[i].strip()}\n{sections[i+1].strip() if i+1 < len(sections) else ''}"
        chunks.append(chunk_text)
    return chunks


def search_student_policy(query: str, threshold: float = 0.20) -> Dict[str, Any]:
    chunks = _load_handbook_chunks()
    if not chunks:
        return {'evidence': [], 'message': 'Handbook missing.'}

    # Vectorize chunks and incoming query using standard English stop-words
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(chunks)
    query_vector = vectorizer.transform([query])

    # Compute cosine similarity
    scores = (tfidf_matrix @ query_vector.T).toarray().flatten()

    # Filter matches that meet the statistical relevance threshold (>= 0.20)
    ranked_indices = np.argsort(scores)[::-1]
    matches = [
        chunks[idx] for idx in ranked_indices if scores[idx] >= threshold
    ]

    if not matches:
        return {'evidence': [], 'message': 'No relevant policy found in the student handbook.'}
    return {'evidence': matches[:2], 'message': f'Retrieved {len(matches[:2])} section(s).'}


def create_support_request(description: str, session_user_id: str) -> Dict[str, Any]:
    return {'ticket_id': f'TCK-{abs(hash(description)) % 100000:05d}', 'requester': session_user_id, 'status': 'SUBMITTED'}


def request_programme_change(requested_program: str, session_user_id: str) -> Dict[str, Any]:
    req = stage_approval_request(student_id=session_user_id, action_type='PROGRAMME_CHANGE', payload={'requested_program': requested_program})
    return {'status': 'PENDING_APPROVAL', 'ticket': req['request_id'], 'message': f"Change request for '{requested_program}' submitted for staff review."}


TOOL_SCHEMAS = [
    {'type': 'function', 'function': {'name': 'get_fee_status', 'description': 'Lookup fee balance and payment status', 'parameters': {'type': 'object', 'properties': {'student_id': {'type': 'string'}}, 'required': ['student_id']}}},
    {'type': 'function', 'function': {'name': 'get_scholarship_status', 'description': 'Lookup scholarship status', 'parameters': {'type': 'object', 'properties': {'student_id': {'type': 'string'}}, 'required': ['student_id']}}},
    {'type': 'function', 'function': {'name': 'search_student_policy', 'description': 'Search handbook for policies', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}}},
    {'type': 'function', 'function': {'name': 'remember_student_fact', 'description': 'Persist useful student facts across sessions', 'parameters': {'type': 'object', 'properties': {'fact': {'type': 'string'}}, 'required': ['fact']}}},
    {'type': 'function', 'function': {'name': 'recall_student_memory', 'description': 'Recall student facts from memory', 'parameters': {'type': 'object', 'properties': {}, 'required': []}}},
    {'type': 'function', 'function': {'name': 'create_support_request', 'description': 'Open support ticket', 'parameters': {'type': 'object', 'properties': {'description': {'type': 'string'}}, 'required': ['description']}}},
    {'type': 'function', 'function': {'name': 'request_programme_change', 'description': 'Request programme change requiring staff approval', 'parameters': {'type': 'object', 'properties': {'requested_program': {'type': 'string'}}, 'required': ['requested_program']}}}
]