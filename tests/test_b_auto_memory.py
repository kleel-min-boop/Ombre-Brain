"""No-network synthetic tests of the actual Dehydrator/hold/HTTP B seam."""
import asyncio
import copy
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from starlette.applications import Starlette
from starlette.routing import Route

from dehydrator import Dehydrator
from ombrebrain.storage.source_store import SourceStore
from tools import _runtime as rt
from tools.hold import b_auto, core


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('live network forbidden in B memory tests')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket.socket, 'connect_ex', forbidden)


def source():
    refs, exchange = [], []
    for role, text in [('user', '今天一起去了青山公园看樱花。'), ('zer', '我也记得一起看樱花。')]:
        mid = 'fixture-' + role
        refs.append(dict(source_id='home/message/' + mid, message_id=mid, role=role,
            source_sha256=b_auto.sha(text), provenance=dict(message_id=mid,
                turn_id='fixture-turn', conversation_id='fixture-home')))
        exchange.append(dict(role=role, content=text, source_id='home/message/' + mid))
    return dict(origin='b_auto_derived', work_id='b/memory-work/fixture-turn', sources=refs, exchange=exchange)


def payload(items=None):
    items = ['一起去了青山公园看樱花。'] if items is None else items
    decision = dict(action='save', event_shape='single' if len(items) == 1 else 'multi', items=items)
    return dict(source(), decision=decision, operation_id='b/memory-work/fixture-turn/write/1',
                decision_sha256=b_auto.sha(b_auto.canonical(decision)))


@pytest.fixture
def runtime(bucket_mgr, monkeypatch):
    engine = object.__new__(Dehydrator)  # No clients, keys or cache DB initialized.
    engine.human = 'fixture-user'
    engine._require_api = MagicMock()
    engine._chat = AsyncMock(return_value=json.dumps(payload()['decision'], ensure_ascii=False))
    engine.analyze = AsyncMock(return_value=dict(domain=['生活'], tags=[],
        valence=.5, arousal=.3, suggested_name='fixture_title', importance=5))
    monkeypatch.setattr(rt, 'config', {'limits': {}, 'merge_threshold': 75})
    monkeypatch.setattr(rt, 'bucket_mgr', bucket_mgr)
    monkeypatch.setattr(rt, 'source_store', SourceStore(bucket_mgr.base_dir))
    monkeypatch.setattr(rt, 'dehydrator', engine)
    monkeypatch.setattr(rt, 'logger', MagicMock())
    monkeypatch.setattr(rt, 'decay_engine', SimpleNamespace(ensure_started=AsyncMock()))
    monkeypatch.setattr(rt, 'fire_webhook', None)
    # Only unrelated optional follow-ups are stubbed; hold/merge/storage are real.
    monkeypatch.setattr(core, 'check_plan_resolution', AsyncMock())
    monkeypatch.setattr(core, 'check_duplicate_for', AsyncMock())
    return engine, bucket_mgr


def app():
    class Registrar:
        def __init__(self):
            self.routes = []
        def custom_route(self, path, methods):
            def decorate(handler):
                self.routes.append(Route(path, handler, methods=methods))
                return handler
            return decorate
    registrar = Registrar()
    b_auto.register(registrar, (lambda token: token == 'fixture-token',))
    return Starlette(routes=registrar.routes)


@pytest.mark.asyncio
@pytest.mark.parametrize('items', [[], ['one'], ['one', 'two']])
async def test_real_dehydrator_decisions_have_no_writes(runtime, monkeypatch, items):
    engine, buckets = runtime
    decision = dict(action='save' if items else 'skip', items=items,
                    event_shape=('single' if len(items) == 1 else 'multi') if items else 'none')
    engine._chat.return_value = json.dumps(decision)
    spy = MagicMock(side_effect=AssertionError('classification must never store evidence'))
    monkeypatch.setattr(rt.source_store, 'put', spy)
    assert await b_auto.classify(dict(source=source(), max_items=3)) == decision
    assert await buckets.list_all() == []
    engine._chat.assert_awaited_once()
    assert 'Classify a completed User/Zer exchange' in engine._chat.call_args.args[0]
    spy.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('raw', ['', '{}', '{"action":"skip","event_shape":"none","items":["x"]}',
    '{"action":"save","event_shape":"single","items":["x","y"]}'])
async def test_invalid_classification_fails_closed(runtime, raw):
    engine, buckets = runtime
    engine._chat.return_value = raw
    with pytest.raises(ValueError):
        await b_auto.classify(dict(source=source(), max_items=3))
    assert await buckets.list_all() == []


@pytest.mark.asyncio
async def test_real_hold_create_merge_ids_and_retained_b_provenance(runtime):
    _, buckets = runtime
    request = payload()
    first = await b_auto.write(request)
    assert first['status'] == 'confirmed'
    assert first['results'][0]['write_kind'] == 'created_new'
    assert first['results'][0]['change_kind'] == 'created'
    assert first['results'][0]['previous_version'] is None
    assert first['results'][0]['version'] == b_auto.sha(request['decision']['items'][0])
    bid = first['results'][0]['provider_id']
    bucket = await buckets.get(bid)
    assert bucket['content'] == request['decision']['items'][0]
    assert bid != 'fixture_title'
    ref = bucket['metadata']['source_refs'][0]['ref']
    # Existing immutable source store holds B origin/IDs/operation, not a new DB.
    assert ref == rt.source_store.put(b_auto.canonical(request))
    second = copy.deepcopy(request)
    second['operation_id'] = second['operation_id'].replace('fixture-turn', 'another-turn')
    second['work_id'] = second['work_id'].replace('fixture-turn', 'another-turn')
    for s in second['sources']:
        s['provenance']['turn_id'] = 'another-turn'
    again = await b_auto.write(second)
    assert again['status'] == 'confirmed'
    assert again['results'][0]['provider_id'] == bid
    assert again['results'][0]['write_kind'] == 'updated_existing'
    assert again['results'][0]['change_kind'] == 'unchanged'
    assert again['results'][0]['previous_version'] == again['results'][0]['version']
    assert len(await buckets.list_all()) == 1
    assert len((await buckets.get(bid))['metadata']['source_refs']) == 2


@pytest.mark.asyncio
async def test_multi_partial_stops_after_uncertain_item(runtime, monkeypatch):
    original = b_auto.store_core
    calls = []
    async def uncertain(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 2:
            raise TimeoutError('after possible write')
        return await original(*args, **kwargs)
    monkeypatch.setattr(b_auto, 'store_core', uncertain)
    result = await b_auto.write(payload(['first factual event', 'second event', 'third event']))
    assert result['status'] == 'partial'
    assert [r['status'] for r in result['results']] == ['confirmed', 'unknown', 'not_sent']
    assert len(calls) == 2
    assert result['results'][0]['provider_id']


@pytest.mark.asyncio
async def test_unverified_id_or_missing_evidence_is_unknown(runtime, monkeypatch):
    async def title_only(*args, **kwargs):
        kwargs['_b_receipt']['provider_id'] = 'fixture_title'
        return '新建→fixture_title'
    monkeypatch.setattr(b_auto, 'store_core', title_only)
    result = await b_auto.write(payload())
    assert result['status'] == 'unknown'
    assert result['results'][0]['provider_id'] is None
    async def nonexistent(*args, **kwargs):
        kwargs['_b_receipt']['provider_id'] = 'abcdef123456'
    monkeypatch.setattr(b_auto, 'store_core', nonexistent)
    assert (await b_auto.write(payload()))['status'] == 'unknown'


@pytest.mark.asyncio
async def test_real_material_merge_captures_locked_preimage_and_committed_body(runtime, monkeypatch):
    engine, buckets = runtime
    bid = await buckets.create(content='old material fixture', domain=['生活'])
    before = await buckets.get(bid)
    async def search(*args, **kwargs):
        return [dict(await buckets.get(bid), score=100)]
    monkeypatch.setattr(buckets, 'search', search)
    engine.judge_same_event = AsyncMock(return_value=dict(same_event=True, confidence=1))
    result = await b_auto.write(payload(['new material fixture']))
    after = await buckets.get(bid)
    assert before['content'] != after['content']
    item = result['results'][0]
    assert item['status'] == 'confirmed' and item['provider_id'] == bid
    assert item['write_kind'] == 'updated_existing'
    assert item['change_kind'] == 'updated'
    assert item['previous_version'] == b_auto.sha(before['content'])
    assert item['before_state'] == 'locked_preimage'
    assert item['version'] == b_auto.sha(after['content'])


@pytest.mark.asyncio
async def test_create_readback_drift_does_not_fabricate_created_version(runtime, monkeypatch):
    _, buckets = runtime
    original = buckets.get
    async def drift(bucket_id):
        value = await original(bucket_id)
        if value:
            return dict(value, content='concurrent external edit fixture')
        return value
    monkeypatch.setattr(buckets, 'get', drift)
    result = await b_auto.write(payload())
    assert result['status'] == 'confirmed'
    assert result['results'][0]['change_kind'] == 'unobserved_change'
    assert result['results'][0]['change_evidence'] == 'create_readback_drift'


@pytest.mark.asyncio
async def test_bad_source_and_frozen_digest_rejected_before_store(runtime, monkeypatch):
    spy = MagicMock(side_effect=AssertionError('invalid requests must not write'))
    monkeypatch.setattr(rt.source_store, 'put', spy)
    for name in ('origin', 'operation_id', 'decision_sha256'):
        request = payload()
        request[name] = 'wrong'
        with pytest.raises(ValueError):
            await b_auto.write(request)
    request = payload()
    request['exchange'][0]['content'] = 'mutated'
    with pytest.raises(ValueError):
        await b_auto.write(request)
    spy.assert_not_called()


@pytest.mark.asyncio
async def test_http_auth_registration_and_no_mcp_tools(runtime):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app()), base_url='http://fixture') as client:
        request = dict(source=source(), max_items=3)
        assert (await client.post('/b-memory/classify', json=request)).status_code == 401
        assert (await client.post('/b-memory/write', json=payload())).status_code == 401
        headers = {'Authorization': 'Bearer fixture-token'}
        response = await client.post('/b-memory/classify', json=request, headers=headers)
        assert response.status_code == 200
        assert response.json()['action'] == 'save'
        response = await client.post('/b-memory/write', json=payload(), headers=headers)
        assert response.status_code == 200 and response.json()['status'] == 'confirmed'
        assert (await client.get('/b-memory/write', headers=headers)).status_code == 405


@pytest.mark.asyncio
async def test_review_read_interface_has_no_write_or_activation(runtime, monkeypatch):
    _, buckets = runtime
    bid = await buckets.create(content='isolated review body', domain=['fixture'])
    before = (await buckets.get(bid))
    monkeypatch.setattr(b_auto, 'store_core', AsyncMock(side_effect=AssertionError('no memory writes')))
    monkeypatch.setattr(rt.source_store, 'put', MagicMock(side_effect=AssertionError('no source writes')))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app()), base_url='http://fixture') as client:
        assert (await client.post('/b-memory/read', json={'bucket_id': bid})).status_code == 401
        response = await client.post('/b-memory/read', json={'bucket_id': bid},
                                     headers={'Authorization': 'Bearer fixture-token'})
        assert response.status_code == 200
        assert response.json() == dict(bucket_id=bid, content='isolated review body', version=b_auto.sha('isolated review body'))
    assert await buckets.get(bid) == before
    assert await b_auto.read_bucket({'bucket_id': '000000000000'}) is None
    with pytest.raises(ValueError):
        await b_auto.read_bucket({'bucket_id': '../invalid'})


@pytest.mark.asyncio
async def test_existing_hold_public_return_and_dream_archive_unchanged(runtime):
    _, buckets = runtime
    reply = await core.store_core('ordinary fixture', [], 5, -1, -1, '')
    assert isinstance(reply, str) and reply.startswith('新建→')
    archive = await core.store_core('dream fixture', [], 5, -1, -1, '', dream_archive=True)
    bid = archive.split('→')[1].split()[0]
    assert (await buckets.get(bid))['metadata']['dont_surface'] is True


@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get('ZER_HOME_TEST_SOURCE'), reason='separate Home checkout optional')
@pytest.mark.parametrize('outcome', ['confirmed', 'skip', 'multi', 'lost_ack', 'partial', 'merged_noop', 'merged_change'])
async def test_home_adapter_to_actual_provider_routes_and_restart(runtime, tmp_path, monkeypatch, outcome):
    home = Path(os.environ['ZER_HOME_TEST_SOURCE'])
    sys.path[:0] = [str(home / 'prototypes/claude_interactive_tmux'),
                    str(home / 'prototypes/claude_native_gateway')]
    from companion_memory_work import MemoryWork
    from companion_ombre_adapter import OmbreMemoryAdapter
    import test_companion_episodic_store as home_fixture
    fixture = home_fixture.EpisodicTests()
    fixture.setUp()
    try:
        engine, buckets = runtime
        items = [] if outcome == 'skip' else ['first fixture event', 'second fixture event'] if outcome in ('multi', 'partial') else ['one fixture event']
        engine._chat.return_value = json.dumps(dict(action='save' if items else 'skip', items=items,
            event_shape=('single' if len(items) == 1 else 'multi') if items else 'none'))
        engine._chat_once = AsyncMock(return_value=engine._chat.return_value)
        if outcome in ('merged_noop', 'merged_change'):
            bid = await buckets.create(content=items[0] if outcome == 'merged_noop' else 'old fixture event', domain=['生活'])
            async def search(*args, **kwargs):
                return [dict(await buckets.get(bid), score=100)]
            monkeypatch.setattr(buckets, 'search', search)
            engine.judge_same_event = AsyncMock(return_value=dict(same_event=True, confidence=1))
        if outcome == 'partial':
            original = b_auto.store_core
            count = 0
            async def partial(*args, **kwargs):
                nonlocal count
                count += 1
                if count == 2:
                    raise TimeoutError('fixture')
                return await original(*args, **kwargs)
            monkeypatch.setattr(b_auto, 'store_core', partial)
        rid = fixture.send('早上想买电影票' if outcome == 'multi' else '今天一起看樱花')
        fixture.finish(rid)
        if outcome == 'multi':
            for text in ('中午比较两家影院', '下午已经买了电影票', '晚上还收养了一只猫'):
                more = fixture.send(text)
                fixture.finish(more)
        from test_companion_memory_daily import freeze_current_day
        daily = freeze_current_day(fixture.host.memory_work)
        application = app()
        adapter = OmbreMemoryAdapter('http://fixture', 'fixture-token', timeout=1, max_response_bytes=8000)
        # Real B async adapter and real provider handlers, only HTTP I/O replaced
        # with in-process ASGI. No listener, network, model or production files.
        def in_process(path, body):
            async def request():
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application),
                                             base_url='http://fixture') as client:
                    response = await client.post(path, json=body, headers={'Authorization': 'Bearer fixture-token'})
                    response.raise_for_status()
                    if outcome == 'lost_ack' and path.endswith('/write'):
                        raise TimeoutError('HTTP acknowledgement lost after real local write')
                    return response.json()
            return asyncio.run(request())
        monkeypatch.setattr(adapter, '_post', in_process)
        result = await fixture.host.memory_work.run_one(adapter, max_source_chars=50000,
            max_items=3, classification_timeout=2, write_timeout=2)
        expected = {'skip': 'not_sent', 'lost_ack': 'unknown', 'partial': 'partial'}.get(outcome, 'confirmed')
        assert result['classification'] == 'completed' and result['delivery'] == expected
        assert result['id'] == daily['id']
        assert result['sources'][0]['provenance']['turn_id'] == rid
        if result['receipt']:
            assert result['receipt']['source_sha256'] == daily['source_sha256']
        if outcome in ('confirmed', 'multi', 'partial'):
            assert await rt.bucket_mgr.get(result['receipt']['results'][0]['provider_id'])
        if outcome == 'multi':
            assert daily['source_count'] == 8
            engine._chat_once.assert_awaited_once()
            material = json.loads(engine._chat_once.call_args.args[1])
            assert [i['content'] for i in material['exchange'] if i['role'] == 'user'] == [
                '早上想买电影票', '中午比较两家影院', '下午已经买了电影票', '晚上还收养了一只猫']
            for item in result['receipt']['results']:
                bucket = await buckets.get(item['provider_id'])
                evidence = json.loads(rt.source_store.read(bucket['metadata']['source_refs'][0]['ref']))
                assert evidence['sources'] == daily['sources']
                assert evidence['source_sha256'] == daily['source_sha256']
                for ref in evidence['sources']:
                    message = fixture.state.db.execute('SELECT content FROM messages WHERE id=?',
                                                       (ref['message_id'],)).fetchone()
                    assert b_auto.sha(message['content']) == ref['source_sha256']
        bucket_count = len(await buckets.list_all())
        assert bucket_count == {'skip': 0, 'multi': 2}.get(outcome, 1)
        from companion_memory_review import MemoryReview
        review = MemoryReview(fixture.state)
        expected_pending = {'confirmed': 1, 'multi': 2, 'partial': 1, 'merged_change': 1}.get(outcome, 0)
        assert review.report(now=__import__('time').time())['backlog']['pending'] == expected_pending
        if outcome == 'merged_noop':
            assert review.report(now=__import__('time').time())['observations']['unchanged'] == 1
        reopened = home_fixture.DisplayState(fixture.state.root)
        try:
            assert await MemoryWork(reopened).run_one(adapter, max_source_chars=8000,
                max_items=3, classification_timeout=2, write_timeout=2) is None
            assert len(await buckets.list_all()) == bucket_count
            assert MemoryReview(reopened).report(now=__import__('time').time())['backlog']['pending'] == expected_pending
        finally:
            reopened.db.close()
    finally:
        fixture.doCleanups()


async def merge_target(runtime, monkeypatch):
    engine, buckets = runtime
    bid = await buckets.create(content='initial fixture body', domain=['生活'])
    async def search(*args, **kwargs):
        return [dict(await buckets.get(bid), score=100)]
    monkeypatch.setattr(buckets, 'search', search)
    engine.judge_same_event = AsyncMock(return_value=dict(same_event=True, confidence=1))
    return bid, buckets


@pytest.mark.asyncio
async def test_concurrent_updates_serialize_and_capture_each_actual_commit(runtime, monkeypatch):
    bid, buckets = await merge_target(runtime, monkeypatch)
    results = await asyncio.gather(b_auto.write(payload(['first change'])),
                                   b_auto.write(payload(['second change'])))
    assert [r['status'] for r in results] == ['confirmed', 'confirmed']
    first, second = [r['results'][0] for r in results]
    # The existing merge-target lock serializes different content fan-in.
    assert first['previous_version'] == b_auto.sha('initial fixture body')
    assert first['version'] == second['previous_version']
    assert second['version'] == b_auto.sha((await buckets.get(bid))['content'])
    assert first['change_kind'] == second['change_kind'] == 'updated'
    assert len(await buckets.list_all()) == 1
    # Each task's optional evidence sink was reset, including on conflict.
    from tools._common import _b_write_receipt
    assert _b_write_receipt.get() is None


@pytest.mark.asyncio
async def test_sequential_updates_use_actual_latest_preimage(runtime, monkeypatch):
    bid, buckets = await merge_target(runtime, monkeypatch)
    first = (await b_auto.write(payload(['first change'])))['results'][0]
    second = (await b_auto.write(payload(['second change'])))['results'][0]
    assert first['provider_id'] == second['provider_id'] == bid
    assert first['version'] == second['previous_version']
    assert first['previous_version'] != second['version']
    assert second['change_kind'] == 'updated'
    assert 'initial fixture body' not in json.dumps(second)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['false', 'before', 'after'])
async def test_update_failure_cannot_fallback_create_or_replay(runtime, monkeypatch, failure):
    bid, buckets = await merge_target(runtime, monkeypatch)
    original = buckets._update_locked
    calls = []
    async def update(*args, **kwargs):
        calls.append(args[0])
        if failure == 'false':
            return False
        if failure == 'after':
            await original(*args, **kwargs)
        raise OSError('fixture commit uncertainty')
    monkeypatch.setattr(buckets, '_update_locked', update)
    result = await b_auto.write(payload(['new change', 'must not send']))
    assert result['status'] == 'unknown'
    assert [r['status'] for r in result['results']] == ['unknown', 'not_sent']
    assert calls == [bid]
    assert len(await buckets.list_all()) == 1
    assert ('new change' in (await buckets.get(bid))['content']) == (failure == 'after')


@pytest.mark.asyncio
@pytest.mark.parametrize('readback', ['locked_failure', 'later_drift', 'later_failure'])
async def test_failed_or_drifted_post_commit_read_is_not_update_proof(runtime, monkeypatch, readback):
    bid, buckets = await merge_target(runtime, monkeypatch)
    original_get, original_update = buckets.get, buckets._update_locked
    reads = None
    outer_read = False
    original_store = b_auto.store_core
    async def store(*args, **kwargs):
        nonlocal outer_read
        result = await original_store(*args, **kwargs)
        outer_read = True
        return result
    monkeypatch.setattr(b_auto, "store_core", store)
    async def update(*args, **kwargs):
        nonlocal reads
        committed = await original_update(*args, **kwargs)
        reads = 0
        return committed
    async def get(key):
        nonlocal reads
        result = await original_get(key)
        if reads is not None:
            reads += 1
            if (readback == 'locked_failure' and reads == 1
                    or readback == 'later_failure' and outer_read):
                raise OSError('fixture read failed')
            if readback == 'later_drift' and outer_read:
                result = dict(result, content='different later body')
        return result
    monkeypatch.setattr(buckets, '_update_locked', update)
    monkeypatch.setattr(buckets, 'get', get)
    result = await b_auto.write(payload(['new change']))
    item = result['results'][0]
    if readback == 'later_failure':
        assert item['status'] == 'unknown'
    else:
        assert item['status'] == 'confirmed'
        assert item['change_kind'] == 'unobserved_change'
        if readback == 'later_drift':
            assert item['change_evidence'] == 'post_commit_readback_drift'
    assert 'new change' in (await original_get(bid))['content']
    assert len(await buckets.list_all()) == 1


@pytest.mark.asyncio
async def test_external_preimage_conflict_stops_without_retry_or_create(runtime, monkeypatch):
    bid, buckets = await merge_target(runtime, monkeypatch)
    calls = 0
    async def concurrent_edit(*args):
        nonlocal calls
        calls += 1
        await buckets.update(bid, content='independent edit')
        return dict(same_event=True, confidence=1)
    rt.dehydrator.judge_same_event = concurrent_edit
    result = await b_auto.write(payload(['must not overwrite']))
    assert result['status'] == 'unknown'
    assert calls == 1
    assert (await buckets.get(bid))['content'] == 'independent edit'
    assert len(await buckets.list_all()) == 1
