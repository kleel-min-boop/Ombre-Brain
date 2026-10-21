"""Narrow authenticated B HTTP seam; no new storage, worker or MCP tools."""
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import math

from .. import _runtime as rt
from .._common import check_content_size, max_grow_input_bytes, max_grow_items
from .core import store_core


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def source_instant(value):
    value = datetime.fromisoformat(value)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('aware B timestamp required')
    return value.astimezone(timezone.utc)


def validate_daily(source):
    window = source['window']
    identity = {k: window[k] for k in ('start', 'end', 'timezone', 'settlement_hour')}
    start, end = source_instant(window['start']), source_instant(window['end'])
    zone = ZoneInfo(window['timezone'])
    hour = window['settlement_hour']
    local = start.astimezone(zone)
    if (type(hour) is not int or not 0 <= hour < 24
            or (local.hour, local.minute, local.second, local.microsecond) != (hour, 0, 0, 0)
            or end != (local + timedelta(days=1)).astimezone(timezone.utc)
            or source['work_id'] != 'b/memory-day/' + sha(canonical(identity))
            or window['id'] != source['work_id']):
        raise ValueError('invalid B daily window')
    budget = window['budget']
    if (set(budget) != {'max_source_chars', 'max_items', 'max_output_tokens',
                        'classification_timeout', 'write_timeout', 'model_attempts', 'write_attempts'}
            or any(type(budget[k]) is not int or budget[k] < 1
                   for k in ('max_source_chars', 'max_items', 'max_output_tokens'))
            or type(budget['model_attempts']) is not int or budget['model_attempts'] != 1
            or type(budget['write_attempts']) is not int or budget['write_attempts'] != 1
            or any(type(budget[k]) not in (int, float) or not math.isfinite(budget[k]) or budget[k] <= 0
                   for k in ('classification_timeout', 'write_timeout'))):
        raise ValueError('invalid daily B budgets')
    frozen = {k: source[k] for k in ('origin', 'unit', 'work_id', 'window', 'sources', 'exchange')}
    if (source.get('source_sha256') != sha(canonical(frozen))
            or len(canonical(dict(frozen, source_sha256=source['source_sha256']))) > budget['max_source_chars']):
        raise ValueError('daily source fingerprint/budget mismatch')
    refs, exchange = source['sources'], source['exchange']
    if not isinstance(refs, list) or not isinstance(exchange, list) or not refs or len(refs) != len(exchange):
        raise ValueError('invalid daily source coverage')
    seen, turns, order = set(), {}, []
    for ref, item in zip(refs, exchange):
        provenance = ref['provenance']
        at = source_instant(ref['created'])
        if (ref['source_id'] in seen or ref['source_id'] != 'home/message/' + ref['message_id']
                or provenance['message_id'] != ref['message_id']
                or not all(isinstance(provenance[k], str) and provenance[k] for k in
                           ('conversation_id', 'turn_id', 'message_id'))
                or ref['role'] not in ('user', 'zer') or ref['complete'] is not True or ref['source'] != 'user'
                or type(ref['source_order']) is not int or ref['source_order'] < 1
                or type(ref['carryover']) is not bool or ref['carryover'] != (at < start)
                or at >= end or not isinstance(item.get('content'), str)
                or sha(item['content']) != ref['source_sha256']
                or any(item.get(k) != ref[k] for k in ('source_id', 'role', 'created', 'source_order',
                                                      'complete', 'source', 'carryover', 'provenance'))):
            raise ValueError('daily B source mismatch')
        seen.add(ref['source_id'])
        turns.setdefault((provenance['conversation_id'], provenance['turn_id']), []).append(ref)
        order.append((at, ref['source_order']))
    if order != sorted(order):
        raise ValueError('daily B source out of order')
    for refs_in_turn in turns.values():
        if (refs_in_turn[0]['role'] != 'user' or sum(r['role'] == 'user' for r in refs_in_turn) != 1
                or not any(r['role'] == 'zer' for r in refs_in_turn)):
            raise ValueError('incomplete daily B turn')


def validate_source(source):
    if isinstance(source, dict) and source.get('unit') == 'daily-v1':
        if source.get('origin') != 'b_auto_derived':
            raise ValueError('invalid daily B origin')
        validate_daily(source)
        if len(canonical(source).encode('utf-8')) > max_grow_input_bytes():
            raise ValueError('B input exceeds existing provider limit')
        return
    if isinstance(source, dict) and 'unit' in source:
        raise ValueError('unknown B source unit')
    if (not isinstance(source, dict) or source.get('origin') != 'b_auto_derived'
            or not isinstance(source.get('work_id'), str)
            or not source['work_id'].startswith('b/memory-work/')):
        raise ValueError('invalid B source identity')
    refs, exchange = source.get('sources'), source.get('exchange')
    if not isinstance(refs, list) or not isinstance(exchange, list) or len(refs) != 2 or len(exchange) != 2:
        raise ValueError('invalid B exchange')
    if {r.get('role') for r in refs if isinstance(r, dict)} != {'user', 'zer'}:
        raise ValueError('invalid B roles')
    by_id = {r['source_id']: r for r in refs}
    if len(by_id) != 2:
        raise ValueError('duplicate B source')
    seen = set()
    for item in exchange:
        ref = by_id.get(item.get('source_id'))
        if (ref is None or item.get('role') != ref['role'] or not isinstance(item.get('content'), str)
                or sha(item['content']) != ref['source_sha256']
                or ref.get('source_id') != 'home/message/' + ref.get('message_id', '')
                or ref.get('provenance', {}).get('message_id') != ref['message_id']
                or source['work_id'] != 'b/memory-work/' + ref['provenance'].get('turn_id', '')):
            raise ValueError('B source mismatch')
        seen.add(item['source_id'])
    if len(seen) != 2:
        raise ValueError('incomplete B exchange')
    if len(canonical(source).encode('utf-8')) > max_grow_input_bytes():
        raise ValueError('B input exceeds existing provider limit')


async def classify(body):
    source = body['source']
    validate_source(source)
    limit = body['max_items']
    if type(limit) is not int or not 1 <= limit <= max_grow_items():
        raise ValueError('invalid B item limit')
    if source.get('unit') == 'daily-v1':
        if limit > source['window']['budget']['max_items']:
            raise ValueError('daily B item budget exceeded')
        return await rt.dehydrator.classify_b_auto(source['exchange'], max_items=limit, window=source['window'])
    return await rt.dehydrator.classify_b_auto(source['exchange'], max_items=limit)


async def write(payload):
    validate_source(payload)
    decision = payload['decision']
    if (payload.get('operation_id') != payload['work_id'] + '/write/1'
            or payload.get('decision_sha256') != sha(canonical(decision))
            or not isinstance(decision, dict) or set(decision) != {'action', 'event_shape', 'items'}
            or decision['action'] != 'save'):
        raise ValueError('invalid B operation')
    items = decision['items']
    if payload.get('unit') == 'daily-v1' and len(items) > payload['window']['budget']['max_items']:
        raise ValueError('daily B item budget exceeded')
    if (not isinstance(items, list) or not 1 <= len(items) <= max_grow_items()
            or decision['event_shape'] != ('single' if len(items) == 1 else 'multi')
            or any(not isinstance(i, str) or not i.strip() or check_content_size(i) for i in items)
            or len(canonical(payload).encode('utf-8')) > max_grow_input_bytes()):
        raise ValueError('invalid B final items')
    # Retain origin, operation, frozen decision and Home identities as immutable
    # evidence using Ombre's EXISTING content-addressed source store.
    source_ref = rt.source_store.put(canonical(payload))
    results = []
    for index, text in enumerate(items):
        ack = {}
        result = dict(index=index, status='unknown', provider_id=None)
        try:
            await store_core(text, [], 5, -1, -1, '',
                             source_refs=[dict(ref=source_ref, ranges=[])], _b_receipt=ack)
            bucket_id = ack.get('provider_id')
            # Read back existing storage by actual ID; never parse public titles.
            if isinstance(bucket_id, str) and re.fullmatch(r'[0-9a-f]{12}', bucket_id):
                bucket = await rt.bucket_mgr.get(bucket_id)
                if (bucket and bucket.get('id') == bucket_id
                        and any(r.get('ref') == source_ref
                                for r in bucket.get('metadata', {}).get('source_refs', []))):
                    result.update(status='confirmed', provider_id=bucket_id)
                    # The core's existing acknowledgement distinguishes a real
                    # create from an existing-target write. Merge proof comes
                    # only from the opt-in locked commit receipt, never an
                    # unlocked search or shadow event log.
                    result.update(version=sha(bucket['content']), previous_version=None,
                                  change_kind='unobserved_change', before_state='unobserved',
                                  change_evidence='preimage_unavailable')
                    if ack.get('merged') is False:
                        result['write_kind'] = 'created_new'
                        if bucket['content'] == text:
                            result.update(change_kind='created', before_state='absent_by_create',
                                          change_evidence='hold_create_exact_readback')
                        else:
                            result['change_evidence'] = 'create_readback_drift'
                    elif ack.get('merged') is True:
                        result['write_kind'] = 'updated_existing'
                        proof = ack.get('commit_evidence', {})
                        before, committed = proof.get('previous_version'), proof.get('version')
                        change = proof.get('change_kind')
                        if (proof.get('provider_id') == bucket_id
                                and proof.get('change_evidence') == 'locked_commit_readback'
                                and isinstance(before, str) and re.fullmatch('[0-9a-f]{64}', before)
                                and isinstance(committed, str) and re.fullmatch('[0-9a-f]{64}', committed)
                                and change == ('unchanged' if before == committed else 'updated')):
                            result.update(previous_version=before, committed_version=committed)
                            if result['version'] == committed:
                                result.update(change_kind=change, before_state='locked_preimage',
                                              change_evidence='locked_commit_readback')
                            else:
                                result['change_evidence'] = 'post_commit_readback_drift'
        except Exception:
            # Exception may follow a durable write. No retry, no claim of failure.
            pass
        results.append(result)
        if result['status'] != 'confirmed':
            # Keep already-confirmed receipts, stop fan-out after uncertainty.
            results.extend(dict(index=i, status='not_sent', provider_id=None)
                           for i in range(index + 1, len(items)))
            break
    status = ('confirmed' if all(r['status'] == 'confirmed' for r in results)
              else 'partial' if any(r['status'] == 'confirmed' for r in results) else 'unknown')
    receipt = dict(operation_id=payload['operation_id'], decision_sha256=payload['decision_sha256'],
                   status=status, results=results)
    if payload.get('unit') == 'daily-v1':
        receipt['source_sha256'] = payload['source_sha256']
    return receipt


async def read_bucket(body):
    """Current content/version only; get() does not activate or mutate a bucket."""
    bucket_id = body.get('bucket_id')
    if not isinstance(bucket_id, str) or re.fullmatch(r'[0-9a-f]{12}', bucket_id) is None:
        raise ValueError('invalid bucket ID')
    bucket = await rt.bucket_mgr.get(bucket_id)
    if bucket is None:
        return None
    content = bucket['content']
    return dict(bucket_id=bucket_id, version=sha(content), content=content)


def register(mcp, token_validators):
    # Installed only with this provider code; no credential discovery/new config.
    from starlette.responses import JSONResponse

    async def handle(request, operation):
        header = request.headers.get('Authorization', '')
        token = header[7:] if header.startswith('Bearer ') else ''
        if not token or not any(validate(token) for validate in token_validators):
            return JSONResponse({'error': 'unauthorized'}, status_code=401)
        try:
            raw = bytearray()
            async for chunk in request.stream():
                raw.extend(chunk)
                if len(raw) > max_grow_input_bytes():
                    return JSONResponse({'error': 'too_large'}, status_code=413)
            body = json.loads(raw)
            return JSONResponse(await operation(body))
        except (ValueError, TypeError, KeyError):
            return JSONResponse({'error': 'invalid_request_or_result'}, status_code=400)
        except Exception:
            return JSONResponse({'error': 'provider_unconfirmed'}, status_code=503)

    @mcp.custom_route('/b-memory/classify', methods=['POST'])
    async def classify_route(request):
        return await handle(request, classify)

    @mcp.custom_route('/b-memory/write', methods=['POST'])
    async def write_route(request):
        return await handle(request, write)

    @mcp.custom_route('/b-memory/read', methods=['POST'])
    async def read_route(request):
        return await handle(request, read_bucket)
