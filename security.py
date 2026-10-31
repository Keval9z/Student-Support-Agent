import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

STUDENT_FILE = Path('data/students.json')
USER_FILE = Path('data/demo_users.json')
STATE_FILE = Path('runtime_state.json')
AUDIT_FILE = Path('audit_log.json')


def hash_credential(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def create_demo_accounts() -> None:
    default_users = {
        'STU1001': {'hash': hash_credential('Student123!'), 'role': 'student'},
        'STU1002': {'hash': hash_credential('Student123!'), 'role': 'student'},
        'STU1003': {'hash': hash_credential('Student123!'), 'role': 'student'},
        'ADMIN': {'hash': hash_credential('Admin123!'), 'role': 'staff'}
    }
    should_populate = False
    if not USER_FILE.exists():
        should_populate = True
    else:
        try:
            with open(USER_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not data:  # file exists but JSON dictionary is empty
                should_populate = True
        except (json.JSONDecodeError, OSError):
            should_populate = True

    if should_populate:
        with open(USER_FILE, "w", encoding="utf-8") as f:
            json.dump(default_users, f, indent=2)


def authenticate_user(user_id: str, secret: str) -> Dict[str, Any]:
    create_demo_accounts()
    with open(USER_FILE, 'r', encoding='utf-8') as f:
        users = json.load(f)
    account = users.get(user_id)
    if not account or account.get('hash') != hash_credential(secret):
        return {'authenticated': False, 'role': None}
    return {'authenticated': True, 'role': account.get('role')}


def log_audit(actor_id: str, action: str, target_id: str, decision: str, outcome: str) -> None:
    records = []
    if AUDIT_FILE.exists():
        try:
            with open(AUDIT_FILE, 'r', encoding='utf-8') as f:
                records = json.load(f)
        except (json.JSONDecodeError, OSError):
            records = []
    records.append({
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'actor_id': actor_id,
        'action': action,
        'target_id': target_id,
        'decision': decision,
        'outcome': outcome
    })
    with open(AUDIT_FILE, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)


def check_record_access(requester_id: str, target_student_id: str) -> Dict[str, Any]:
    if requester_id == 'ADMIN' or requester_id == target_student_id:
        log_audit(requester_id, 'READ_RECORD', target_student_id, 'ALLOWED', 'ACCESS_GRANTED')
        return {'allowed': True, 'error': None}
    log_audit(requester_id, 'READ_RECORD', target_student_id, 'DENIED', 'CROSS_RECORD_VIOLATION')
    return {
        'allowed': False,
        'error': f"Authorization Error: Student '{requester_id}' is forbidden from inspecting records belonging to '{target_student_id}'."
    }


def load_state() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        return {'pending_approvals': {}}
    try:
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {'pending_approvals': {}}


def save_state(state: Dict[str, Any]) -> None:
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)


def stage_approval_request(student_id: str, action_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    state = load_state()
    request_id = f"REQ-{datetime.now().strftime('%y%m%d%H%M%S')}"
    item = {
        'request_id': request_id,
        'student_id': student_id,
        'action_type': action_type,
        'payload': payload,
        'status': 'PENDING',
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    state.setdefault('pending_approvals', {})[request_id] = item
    save_state(state)
    log_audit(student_id, action_type, student_id, 'STAGED', f'Pending approval ticket {request_id}')
    return item


def resolve_approval_request(request_id: str, reviewer_id: str, decision: str) -> Dict[str, Any]:
    if reviewer_id != 'ADMIN':
        log_audit(reviewer_id, 'RESOLVE_APPROVAL', request_id, 'DENIED', 'UNAUTHORIZED_REVIEWER')
        return {'success': False, 'message': 'Access Denied: Only staff (ADMIN) can approve or reject changes.'}
    state = load_state()
    req = state.get('pending_approvals', {}).get(request_id)
    if not req or req.get('status') != 'PENDING':
        return {'success': False, 'message': 'Request ID not found or already resolved.'}
    if decision.upper() == 'APPROVE':
        req['status'] = 'APPROVED'
        target = req['student_id']
        prog = req['payload'].get('requested_program')
        with open(STUDENT_FILE, 'r', encoding='utf-8') as f:
            students = json.load(f)
        if target in students and prog:
            students[target]['program'] = prog
            with open(STUDENT_FILE, 'w', encoding='utf-8') as f:
                json.dump(students, f, indent=2)
        save_state(state)
        log_audit(reviewer_id, 'APPROVE_CHANGE', target, 'APPROVED', f'Program updated to {prog}')
        return {'success': True, 'message': f'Request {request_id} approved. Record updated.'}
    elif decision.upper() == 'REJECT':
        req['status'] = 'REJECTED'
        save_state(state)
        log_audit(reviewer_id, 'REJECT_CHANGE', req['student_id'], 'REJECTED', 'Record unchanged')
        return {'success': True, 'message': f'Request {request_id} rejected. Record unmodified.'}
    return {'success': False, 'message': 'Invalid decision.'}
