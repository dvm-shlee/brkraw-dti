"""Smoke tests: the hook imports against the installed brkraw and keeps the
converter-hook contract."""

import inspect


def test_hook_imports_with_installed_brkraw():
    from brkraw_dti import hook  # every brkraw name the hook uses must exist

    assert set(hook.HOOK) == {"convert"}


def test_convert_accepts_kwargs():
    from brkraw_dti.hook import HOOK

    kinds = {p.kind for p in inspect.signature(HOOK["convert"]).parameters.values()}
    assert inspect.Parameter.VAR_KEYWORD in kinds
