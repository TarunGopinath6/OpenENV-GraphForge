"""Body template library: 25 deterministic code patterns for function bodies."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class TemplateNotFoundError(KeyError):
    pass


class TemplateArgError(ValueError):
    pass


@dataclass
class BodyTemplate:
    template_id: str
    description: str
    required_args: List[str]
    optional_args: Dict[str, Any]
    source_pattern: str          # Python source with {arg_name} substitutions
    expected_return_type: str    # indicative return type


_TEMPLATES: List[BodyTemplate] = [
    BodyTemplate(
        "identity",
        "Return the first parameter unchanged",
        ["param"],
        {},
        "return {param}",
        "T",
    ),
    BodyTemplate(
        "constant",
        "Return a literal constant value",
        ["value"],
        {},
        "return {value}",
        "Any",
    ),
    BodyTemplate(
        "add_two",
        "Add two parameters",
        ["a", "b"],
        {},
        "return {a} + {b}",
        "Any",
    ),
    BodyTemplate(
        "multiply",
        "Multiply two parameters",
        ["a", "b"],
        {},
        "return {a} * {b}",
        "Any",
    ),
    BodyTemplate(
        "filter_list",
        "Filter a list with a predicate function",
        ["iterable", "predicate"],
        {},
        "return [x for x in {iterable} if {predicate}(x)]",
        "list",
    ),
    BodyTemplate(
        "map_list",
        "Map a transform function over a list",
        ["iterable", "transform"],
        {},
        "return [{transform}(x) for x in {iterable}]",
        "list",
    ),
    BodyTemplate(
        "dict_get",
        "Get a value from a dict with optional default",
        ["mapping", "key"],
        {"default": "None"},
        "return {mapping}.get({key}, {default})",
        "Any",
    ),
    BodyTemplate(
        "conditional",
        "Return one of two values based on a condition",
        ["condition", "if_true", "if_false"],
        {},
        "return {if_true} if {condition} else {if_false}",
        "Any",
    ),
    BodyTemplate(
        "try_except",
        "Call a target, return fallback on any exception",
        ["target_call", "fallback"],
        {},
        textwrap.dedent("""\
            try:
                return {target_call}
            except Exception:
                return {fallback}"""),
        "Any",
    ),
    BodyTemplate(
        "chain_call",
        "Apply func1 then func2 in sequence",
        ["func1", "func2", "arg"],
        {},
        "return {func2}({func1}({arg}))",
        "Any",
    ),
    BodyTemplate(
        "zip_combine",
        "Zip two iterables into a list of tuples",
        ["a", "b"],
        {},
        "return list(zip({a}, {b}))",
        "list",
    ),
    BodyTemplate(
        "string_format",
        "Format a template string with kwargs",
        ["template_str", "kwargs"],
        {},
        "return {template_str}.format(**{kwargs})",
        "str",
    ),
    BodyTemplate(
        "dataclass_constructor",
        "Construct a dataclass from keyword arguments",
        ["cls_name", "kwargs"],
        {},
        "return {cls_name}(**{kwargs})",
        "Any",
    ),
    BodyTemplate(
        "validate_and_pass",
        "Assert a condition then return the value",
        ["condition", "value", "error_msg"],
        {"error_msg": '"Validation failed"'},
        textwrap.dedent("""\
            if not ({condition}):
                raise ValueError({error_msg})
            return {value}"""),
        "Any",
    ),
    BodyTemplate(
        "accumulate",
        "Accumulate items via an expression into a list",
        ["iterable", "expr"],
        {},
        textwrap.dedent("""\
            result = []
            for x in {iterable}:
                result.append({expr})
            return result"""),
        "list",
    ),
    BodyTemplate(
        "recursive_sum",
        "Recursively sum a list (base case + recursive call)",
        ["lst", "func_name"],
        {},
        textwrap.dedent("""\
            if not {lst}:
                return 0
            return {lst}[0] + {func_name}({lst}[1:])"""),
        "int",
    ),
    BodyTemplate(
        "memoized_call",
        "Wrap a function call with lru_cache memoization",
        ["target_call"],
        {},
        textwrap.dedent("""\
            import functools
            @functools.lru_cache(maxsize=128)
            def _memo(*args, **kwargs):
                return {target_call}
            return _memo"""),
        "Any",
    ),
    BodyTemplate(
        "pipeline_stage",
        "Apply a sequence of callables to a value",
        ["value", "stages"],
        {},
        textwrap.dedent("""\
            result = {value}
            for stage in {stages}:
                result = stage(result)
            return result"""),
        "Any",
    ),
    BodyTemplate(
        "batch_process",
        "Process a list in fixed-size chunks",
        ["items", "chunk_size", "processor"],
        {},
        textwrap.dedent("""\
            results = []
            for i in range(0, len({items}), {chunk_size}):
                chunk = {items}[i:i + {chunk_size}]
                results.extend({processor}(chunk))
            return results"""),
        "list",
    ),
    BodyTemplate(
        "flatten",
        "Flatten one level of nesting from a list of lists",
        ["nested"],
        {},
        "return [item for sublist in {nested} for item in sublist]",
        "list",
    ),
    BodyTemplate(
        "group_by",
        "Group a list of items by a key function",
        ["items", "key_func"],
        {},
        textwrap.dedent("""\
            groups: dict = {}
            for item in {items}:
                k = {key_func}(item)
                groups.setdefault(k, []).append(item)
            return groups"""),
        "dict",
    ),
    BodyTemplate(
        "sorted_by",
        "Sort a list by a key function with optional reverse",
        ["items", "key_func"],
        {"reverse": "False"},
        "return sorted({items}, key={key_func}, reverse={reverse})",
        "list",
    ),
    BodyTemplate(
        "deduplicate",
        "Remove duplicates while preserving insertion order",
        ["iterable"],
        {},
        "return list(dict.fromkeys({iterable}))",
        "list",
    ),
    BodyTemplate(
        "safe_divide",
        "Divide two numbers, returning fallback on ZeroDivisionError",
        ["numerator", "denominator"],
        {"fallback": "0.0"},
        textwrap.dedent("""\
            if {denominator} == 0:
                return {fallback}
            return {numerator} / {denominator}"""),
        "float",
    ),
    BodyTemplate(
        "type_coerce",
        "Cast a value to a target type, returning fallback on failure",
        ["value", "target_type"],
        {"fallback": "None"},
        textwrap.dedent("""\
            try:
                return {target_type}({value})
            except (ValueError, TypeError):
                return {fallback}"""),
        "Any",
    ),
]


class BodyTemplateLibrary:
    """Registry of 25 reusable body templates."""

    def __init__(self) -> None:
        self._templates: Dict[str, BodyTemplate] = {t.template_id: t for t in _TEMPLATES}

    def list_templates(self) -> List[str]:
        return list(self._templates)

    def get_template(self, template_id: str) -> BodyTemplate:
        if template_id not in self._templates:
            raise TemplateNotFoundError(f"Unknown template: {template_id!r}")
        return self._templates[template_id]

    def get_expected_return_type(self, template_id: str) -> str:
        return self.get_template(template_id).expected_return_type

    def validate_args(self, template_id: str, args: Dict[str, Any]) -> List[str]:
        """Return error strings for missing required args (empty = valid)."""
        tmpl = self.get_template(template_id)
        errors: List[str] = []
        for req in tmpl.required_args:
            if req not in args:
                errors.append(f"Template {template_id!r} requires arg {req!r}")
        return errors

    def render(self, template_id: str, args: Dict[str, Any]) -> str:
        """Render a template to a Python source string (function body).

        Args are substituted with str.format_map. Missing optional args use
        their defaults; missing required args raise TemplateArgError.
        """
        tmpl = self.get_template(template_id)

        errors = self.validate_args(template_id, args)
        if errors:
            raise TemplateArgError("; ".join(errors))

        merged = dict(tmpl.optional_args)
        merged.update(args)

        try:
            source = tmpl.source_pattern.format_map(merged)
        except KeyError as e:
            raise TemplateArgError(f"Template {template_id!r} missing arg {e}") from e

        return source
