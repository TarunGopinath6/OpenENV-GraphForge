"""Tests for BodyTemplateLibrary."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from engine.body_template_library import BodyTemplateLibrary, TemplateNotFoundError, TemplateArgError


@pytest.fixture
def btl():
    return BodyTemplateLibrary()


def test_library_has_25_templates(btl):
    assert len(btl.list_templates()) == 25


def test_render_add_two(btl):
    src = btl.render("add_two", {"a": "x", "b": "y"})
    assert "x + y" in src


def test_render_identity(btl):
    src = btl.render("identity", {"param": "val"})
    assert "return val" in src


def test_render_missing_required_arg_raises(btl):
    with pytest.raises(TemplateArgError):
        btl.render("add_two", {"a": "x"})  # missing 'b'


def test_render_unknown_template_raises(btl):
    with pytest.raises(TemplateNotFoundError):
        btl.render("nonexistent_template", {})


def test_validate_args_returns_errors(btl):
    errors = btl.validate_args("multiply", {"a": "x"})
    assert any("b" in e for e in errors)


def test_validate_args_ok(btl):
    errors = btl.validate_args("identity", {"param": "v"})
    assert errors == []


def test_get_expected_return_type(btl):
    rt = btl.get_expected_return_type("filter_list")
    assert rt == "list"
