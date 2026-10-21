"""Daily source contract and actual isolated hold reuse; every model is fake."""
import copy
from datetime import datetime, timedelta, timezone
import json
from unittest.mock import AsyncMock

import pytest

from .test_b_auto_memory import runtime, no_network
from tools.hold import b_auto
from tools import _runtime as rt


def daily_source():
    start = datetime.fromisoformat('2026-10-01T03:00:00+08:00').astimezone(timezone.utc)
    identity = dict(start=start.isoformat(), end=(start+timedelta(days=1)).isoformat(),
                    timezone='Asia/Shanghai',settlement_hour=3)
    wid = 'b/memory-day/' + b_auto.sha(b_auto.canonical(identity))
    budget = dict(max_source_chars=50000,max_items=5,max_output_tokens=1024,
                  classification_timeout=1,write_timeout=1,model_attempts=1,write_attempts=1)
    refs, exchange = [], []
    events = [('早上想买电影票', '可以先比较',4,'home-one'),
              ('中午比较了两个影院', '还没买，不确定',9,'home-one'),
              ('下午已经买到票，实际去了电影节', '记得这个晚上',14,'home-one'),
              ('另一个窗口：今天收养猫了', '一件独立的新事',15,'home-two')]
    for n,(user,zer,hour,conversation) in enumerate(events):
        for j,(role,text) in enumerate((('user',user),('zer',zer))):
            mid=f'fixture-{n}-{j}'
            meta=dict(source_id='home/message/'+mid,role=role,
                created=(start+timedelta(hours=hour,seconds=j)).isoformat(),source_order=n*2+j+1,
                complete=True,source='user',carryover=False,
                provenance=dict(conversation_id=conversation,turn_id=f'turn-{n}',message_id=mid))
            refs.append(dict(meta,message_id=mid,source_sha256=b_auto.sha(text)))
            exchange.append(dict(meta,content=text))
    source=dict(origin='b_auto_derived',unit='daily-v1',work_id=wid,
                window=dict(identity,id=wid,budget=budget),sources=refs,exchange=exchange)
    resign(source)
    return source


def resign(source):
    frozen={k:source[k] for k in ('origin','unit','work_id','window','sources','exchange')}
    source['source_sha256']=b_auto.sha(b_auto.canonical(frozen))


def write_payload(source, items):
    decision=dict(action='save',event_shape='single' if len(items)==1 else 'multi',items=items)
    return dict(source,decision=decision,operation_id=source['work_id']+'/write/1',
                decision_sha256=b_auto.sha(b_auto.canonical(decision)))


@pytest.mark.asyncio
async def test_daily_one_classification_sees_full_evolving_events_and_separate_origins(runtime):
    engine,buckets=runtime
    source=daily_source()
    decision=dict(action='save',event_shape='multi',items=['比较两家影院后已买票并去了电影节','今天收养了猫'])
    engine._chat_once=AsyncMock(return_value=json.dumps(decision,ensure_ascii=False))
    result=await b_auto.classify(dict(source=source,max_items=5))
    assert result==decision
    engine._chat_once.assert_awaited_once()
    engine._chat.assert_not_awaited()
    call=engine._chat_once.call_args
    assert call.kwargs['max_tokens']==1024
    material=json.loads(call.args[1])
    assert material['exchange']==source['exchange']  # Including the late outcome.
    assert len(material['exchange'])==8
    assert {i['provenance']['conversation_id'] for i in material['exchange']}=={'home-one','home-two'}
    assert 'ONE continuous daily settlement window' in call.args[0]
    assert 'Never generate Zer meaning/why_remembered' in call.args[0]
    assert await buckets.list_all()==[]
    receipt=await b_auto.write(write_payload(source,result['items']))
    assert receipt['status']=='confirmed' and len(receipt['results'])==2
    assert receipt['source_sha256']==source['source_sha256']
    for item in receipt['results']:
        assert item['change_kind']=='created'
        bucket=await buckets.get(item['provider_id'])
        evidence=json.loads(rt.source_store.read(bucket['metadata']['source_refs'][0]['ref']))
        assert evidence['work_id']==source['work_id']
        assert evidence['source_sha256']==source['source_sha256']
        assert evidence['sources']==source['sources']
        assert evidence['exchange']==source['exchange']


@pytest.mark.asyncio
async def test_daily_failure_has_no_classification_retry_or_write(runtime):
    engine,buckets=runtime
    engine._chat_once=AsyncMock(side_effect=TimeoutError('fake transient'))
    with pytest.raises(TimeoutError):
        await b_auto.classify(dict(source=daily_source(),max_items=5))
    engine._chat_once.assert_awaited_once()
    engine._chat.assert_not_awaited()
    assert await buckets.list_all()==[]


@pytest.mark.asyncio
async def test_daily_skip_no_write_and_no_private_judgment_fields(runtime):
    engine,buckets=runtime
    engine._chat_once=AsyncMock(return_value='{"action":"skip","event_shape":"none","items":[]}')
    assert (await b_auto.classify(dict(source=daily_source(),max_items=5)))['action']=='skip'
    engine._chat_once.return_value='{"action":"save","event_shape":"single","items":["fake"],"meaning":"mine"}'
    with pytest.raises(ValueError):
        await b_auto.classify(dict(source=daily_source(),max_items=5))
    assert await buckets.list_all()==[]


@pytest.mark.asyncio
@pytest.mark.parametrize('damage',['body','timestamp','order','incomplete','window','fingerprint',
                                   'duplicate','turn','budget','attempts','item_limit'])
async def test_daily_corrupt_source_fails_before_any_model_or_write(runtime,damage):
    engine,buckets=runtime
    engine._chat_once=AsyncMock()
    source=daily_source()
    if damage=='body': source['exchange'][-1]['content']='tampered'
    if damage=='timestamp': source['sources'][-1]['created']=source['window']['end']
    if damage=='order':
        source['sources'].reverse(); source['exchange'].reverse(); resign(source)
    if damage=='incomplete':
        source['sources'][-1]['complete']=False;source['exchange'][-1]['complete']=False;resign(source)
    if damage=='window': source['window']['start']='2026-10-01T03:00:00'
    if damage=='fingerprint': source['source_sha256']='0'*64
    if damage=='duplicate': source['sources'][-1]=copy.deepcopy(source['sources'][0]);resign(source)
    if damage=='turn':
        source['sources']=source['sources'][:-1];source['exchange']=source['exchange'][:-1];resign(source)
    if damage=='budget': source['window']['budget']['max_source_chars']=10;resign(source)
    if damage=='attempts': source['window']['budget']['model_attempts']=2;resign(source)
    if damage=='item_limit': source['window']['budget']['max_items']=1;resign(source)
    with pytest.raises(ValueError):
        await b_auto.classify(dict(source=source,max_items=5))
    engine._chat_once.assert_not_awaited()
    assert await buckets.list_all()==[]


@pytest.mark.asyncio
async def test_daily_carryover_accepts_real_older_times_only_when_explicit(runtime):
    engine,_=runtime
    source=daily_source()
    for ref,item in zip(source['sources'][:2],source['exchange'][:2]):
        ref['created']=item['created']='2026-09-30T18:59:59+00:00'
        ref['carryover']=item['carryover']=True
    resign(source)
    b_auto.validate_source(source)
    source['sources'][0]['carryover']=source['exchange'][0]['carryover']=False
    resign(source)
    with pytest.raises(ValueError):
        b_auto.validate_source(source)
