import asyncio
from datetime import datetime

import pytest
from infra.notify import order

# python3 -m pytest -q tests/test_notify_order.py

class DummyLoop:
    def __init__(self):
        self.tasks = []

    def create_task(self, coro):
        self.tasks.append(coro)
        return coro


class FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        self._response = type("Response", (), {"status_code": 200, "text": "OK"})()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        return self._response


async def dummy_play_sound():
    return "played"


def make_test_event():
    return order.OrderEvent(
        ts=datetime(2026, 4, 15, 12, 0, 0),
        symbol="AAPL",
        action="OPEN",
        side="LONG",
        size=1,
        price=150.0,
        value=150.0,
        reason="signal",
        status="FILLED",
    )


def test_notify_order_v1_skips_sound_when_persist_disabled(monkeypatch, capsys):
    event = make_test_event()
    

    def fail_get_loop():
        raise AssertionError("get_running_loop should not be called")

    monkeypatch.setattr(asyncio, "get_running_loop", fail_get_loop)

    order.notify_order_v1(event, position_mgr=None)

    captured = capsys.readouterr()
    assert "ORDER OPEN LONG" in captured.out
    assert "Symbol: AAPL" in captured.out
    assert "Price: 150.00" in captured.out


def test_notify_order_v1_schedules_play_sound(monkeypatch):
    event = make_test_event()
    
    monkeypatch.setattr(order, "play_sound", dummy_play_sound)

    fake_loop = DummyLoop()
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: fake_loop)

    order.notify_order_v1(event, position_mgr=None)

    assert len(fake_loop.tasks) == 1
    assert asyncio.iscoroutine(fake_loop.tasks[0])
    assert asyncio.run(fake_loop.tasks[0]) == "played"


def test_notify_order_schedules_persist_and_notify(monkeypatch):
    event = make_test_event()
   
    monkeypatch.setattr(order, "play_sound", dummy_play_sound)
    monkeypatch.setattr(order, "httpx", type("httpx", (), {"AsyncClient": FakeAsyncClient}))

    fake_loop = DummyLoop()
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: fake_loop)

    order.notify_order(event, position_mgr=None)

    assert len(fake_loop.tasks) == 1
    coro = fake_loop.tasks[0]
    assert asyncio.iscoroutine(coro)
    assert asyncio.run(coro) is None


def test_notify_order_no_task_when_not_filled(monkeypatch):
    event = make_test_event()
    event.status = "PENDING"


    def fail_get_loop():
        raise AssertionError("get_running_loop should not be called")

    monkeypatch.setattr(asyncio, "get_running_loop", fail_get_loop)

    order.notify_order(event, position_mgr=None)
