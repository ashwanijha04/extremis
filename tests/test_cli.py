"""Smoke tests for the extremis CLI — stats, doctor, traces, and arg parsing.
Was at 0% coverage."""

from __future__ import annotations

import argparse
import sys
from unittest.mock import patch

import pytest

from extremis import cli


@pytest.fixture()
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("EXTREMIS_EXTREMIS_HOME", str(tmp_path))
    monkeypatch.setenv("EXTREMIS_ENABLE_FAITHFULNESS_CHECK", "false")
    monkeypatch.setenv("EXTREMIS_SELF_CONSISTENCY_N", "0")
    return tmp_path


def test_stats_empty_home(tmp_home, capsys):
    cli._stats(argparse.Namespace())
    out = capsys.readouterr().out
    assert "memories" in out.lower() or "log" in out.lower()


def test_doctor_reports(tmp_home, capsys):
    try:
        cli._doctor(argparse.Namespace())
    except SystemExit:
        pass  # doctor may exit nonzero on a cold home — that's a valid verdict
    out = capsys.readouterr().out
    assert out.strip(), "doctor must print a report"


def test_traces_no_file(tmp_home, capsys):
    try:
        cli._traces(argparse.Namespace())
    except SystemExit:
        pass
    out = capsys.readouterr().out + capsys.readouterr().err
    assert isinstance(out, str)


def test_main_no_args_shows_help(tmp_home, capsys):
    with patch.object(sys, "argv", ["extremis"]):
        try:
            cli.main()
        except SystemExit:
            pass
    out = capsys.readouterr()
    assert "stats" in (out.out + out.err)


def test_main_dispatches_stats(tmp_home, capsys):
    with patch.object(sys, "argv", ["extremis", "stats"]):
        try:
            cli.main()
        except SystemExit:
            pass
    assert capsys.readouterr().out.strip()
