from web import _shared as sh
from web import letters


class Request:
    def __init__(self, headers=None): self.headers = headers or {}


def test_observer_token_is_letter_only_and_constant_time_path(monkeypatch):
    monkeypatch.setattr(sh, "_is_authenticated", lambda request: False)
    monkeypatch.setenv("OMBRE_HOME_LETTER_OBSERVER_TOKEN", "observer-secret")
    assert sh._require_letter_human_auth(Request({"X-Ombre-Home-Letter-Token":"observer-secret"})) is None
    assert sh._require_letter_human_auth(Request({"X-Ombre-Home-Letter-Token":"wrong"})).status_code == 401
    assert sh._require_letter_human_auth(Request({"Authorization":"Bearer observer-secret"})).status_code == 401
    assert sh._require_auth(Request({"X-Ombre-Home-Letter-Token":"observer-secret"})).status_code == 401


def test_browser_session_remains_valid_without_observer_token(monkeypatch):
    monkeypatch.setattr(sh, "_is_authenticated", lambda request: True)
    assert sh._require_letter_human_auth(Request()) is None


def test_only_letter_dashboard_routes_use_observer_helper():
    source = open(letters.__file__, encoding="utf-8").read()
    assert source.count("_require_letter_human_auth(request)") == 4
    assert "_require_auth(request)" not in source
