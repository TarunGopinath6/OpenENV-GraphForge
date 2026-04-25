"""Task bank: ~50 task templates across 3 tiers for GraphForge training."""

from __future__ import annotations

import random
import uuid
from typing import Dict, List, Optional

from models import BehavioralTest, ConstraintSpec, TaskSpec


def _cid(kind: str, target: str, value=None, hidden: bool = False) -> ConstraintSpec:
    return ConstraintSpec(
        constraint_id=f"{kind}_{target.replace('.', '_')}_{uuid.uuid4().hex[:6]}",
        kind=kind,
        target=target,
        value=value,
        hidden=hidden,
    )


# =============================================================================
# Tier 1 templates (4 templates × 4 variants = 16 tasks)
# Each: 5-8 nodes, 1-2 modules, ~15 constraints, 30% hidden, 0-2 behavioral tests
# =============================================================================

def _tier1_simple_pipeline(variant: int) -> TaskSpec:
    """Single-module number-processing pipeline."""
    domains = [
        ("numbers", ["parse_int", "double", "clamp", "format_result", "validate_positive"]),
        ("strings", ["normalize", "trim", "split_words", "count_chars", "to_uppercase"]),
        ("lists",   ["flatten_one", "dedupe", "sort_items", "first_n", "last_n"]),
        ("math",    ["add_values", "multiply_values", "safe_div", "square", "abs_val"]),
    ]
    domain, funcs = domains[variant % len(domains)]
    f = funcs

    # Descriptions
    descs = [
        f"Build a {domain} processing pipeline. Implement {len(f)} functions in a single module.",
        f"Create utility functions for {domain} manipulation in a clean single-module design.",
        f"Implement a {domain} toolkit with clearly named functions and proper type annotations.",
        f"Design a focused {domain} library using pure functions with consistent return types.",
    ]

    constraints = [
        _cid("node_exists", f"pipeline.{f[0]}"),
        _cid("node_exists", f"pipeline.{f[1]}"),
        _cid("node_exists", f"pipeline.{f[2]}"),
        _cid("node_exists", f"pipeline.{f[3]}"),
        _cid("node_exists", f"pipeline.{f[4]}"),
        _cid("module_count", "", value=1),
        _cid("acyclic_imports", ""),
        _cid("no_any_types", "", hidden=True),
        _cid("return_type", f"pipeline.{f[0]}", value="int" if domain == "numbers" else "str" if domain == "strings" else "list", hidden=True),
        _cid("pure_function", f"pipeline.{f[0]}"),
        _cid("pure_function", f"pipeline.{f[1]}", hidden=True),
        _cid("fan_out_max", f"pipeline.{f[0]}", value=3, hidden=True),
        _cid("module_responsibility", "pipeline", value="transform"),
        _cid("error_handling_present", f"pipeline.{f[2]}", hidden=True),
        _cid("dag_depth_max", "", value=5),
    ]

    behavioral_tests = []
    if variant == 0:
        behavioral_tests = [
            BehavioralTest(
                test_id="test_first_func_returns_correct_type",
                description=f"Verify {f[0]} returns expected type",
                hypothesis_source=_simple_type_test("pipeline", f[0], "int" if domain == "numbers" else "str"),
            )
        ]

    return TaskSpec(
        task_id=f"t1_pipeline_v{variant}",
        tier=1,
        template_name="simple_pipeline",
        variant=variant,
        description=descs[variant % len(descs)],
        constraints=constraints,
        behavioral_tests=behavioral_tests,
        token_budget=4000,
        turn_cap=20,
    )


def _tier1_validator_chain(variant: int) -> TaskSpec:
    """Single-module validation chain."""
    domains = [
        ("email",   ["validate_format", "check_domain", "normalize_email", "is_blacklisted", "extract_user"]),
        ("phone",   ["strip_spaces", "validate_length", "check_prefix", "format_e164", "is_valid"]),
        ("url",     ["parse_scheme", "extract_host", "validate_path", "check_port", "is_safe"]),
        ("password",["check_length", "has_uppercase", "has_digit", "has_special", "compute_score"]),
    ]
    domain, funcs = domains[variant % len(domains)]
    f = funcs

    descs = [
        f"Implement a {domain} validation chain with 5 validation functions.",
        f"Build {domain} validators that each check one specific property.",
        f"Create a composable {domain} validation library with pure boolean functions.",
        f"Design a {domain} validation toolkit where each function handles one concern.",
    ]

    constraints = [
        _cid("node_exists", f"validators.{f[0]}"),
        _cid("node_exists", f"validators.{f[1]}"),
        _cid("node_exists", f"validators.{f[2]}"),
        _cid("node_exists", f"validators.{f[3]}"),
        _cid("node_exists", f"validators.{f[4]}"),
        _cid("return_type", f"validators.{f[4]}", value="bool"),
        _cid("module_count", "", value=1),
        _cid("module_responsibility", "validators", value="validation"),
        _cid("pure_function", f"validators.{f[0]}", hidden=True),
        _cid("pure_function", f"validators.{f[4]}"),
        _cid("acyclic_imports", ""),
        _cid("no_any_types", "", hidden=True),
        _cid("return_type", f"validators.{f[0]}", value="bool", hidden=True),
        _cid("dag_depth_max", "", value=4, hidden=True),
        _cid("fan_in_max", f"validators.{f[4]}", value=4),
    ]

    behavioral_tests = []
    if variant % 2 == 0:
        behavioral_tests = [
            BehavioralTest(
                test_id="test_final_validator_returns_bool",
                description=f"Check {f[4]} returns a bool",
                hypothesis_source=_bool_return_test("validators", f[4]),
            )
        ]

    return TaskSpec(
        task_id=f"t1_validator_v{variant}",
        tier=1,
        template_name="validator_chain",
        variant=variant,
        description=descs[variant % len(descs)],
        constraints=constraints,
        behavioral_tests=behavioral_tests,
        token_budget=4000,
        turn_cap=20,
    )


def _tier1_transform_reduce(variant: int) -> TaskSpec:
    """Single-module transform-then-reduce pipeline."""
    domains = [
        ("sales",    ["parse_record", "normalize_amount", "filter_valid", "sum_totals", "format_report"]),
        ("logs",     ["parse_line", "extract_level", "filter_errors", "count_by_level", "summarize"]),
        ("inventory",["parse_item", "compute_value", "filter_in_stock", "total_value", "top_items"]),
        ("scores",   ["parse_score", "normalize_score", "filter_passing", "average_score", "rank_scores"]),
    ]
    domain, funcs = domains[variant % len(domains)]
    f = funcs

    descs = [
        f"Build a {domain} data processing pipeline: parse → transform → filter → reduce → format.",
        f"Implement the classic map-filter-reduce pattern for {domain} data processing.",
        f"Create a {domain} analytics module with functions for each stage of data processing.",
        f"Design a {domain} pipeline where each function handles one transformation stage.",
    ]

    constraints = [
        _cid("node_exists", f"transform.{f[0]}"),
        _cid("node_exists", f"transform.{f[1]}"),
        _cid("node_exists", f"transform.{f[2]}"),
        _cid("node_exists", f"transform.{f[3]}"),
        _cid("node_exists", f"transform.{f[4]}"),
        _cid("edge_exists", f"transform.{f[0]}::transform.{f[1]}"),
        _cid("edge_exists", f"transform.{f[1]}::transform.{f[2]}", hidden=True),
        _cid("module_count", "", value=1),
        _cid("module_responsibility", "transform", value="transform"),
        _cid("acyclic_imports", ""),
        _cid("dag_depth_max", "", value=6, hidden=True),
        _cid("fan_out_max", f"transform.{f[2]}", value=2, hidden=True),
        _cid("pure_function", f"transform.{f[0]}"),
        _cid("pure_function", f"transform.{f[1]}"),
        _cid("no_any_types", "", hidden=True),
    ]

    return TaskSpec(
        task_id=f"t1_transform_v{variant}",
        tier=1,
        template_name="transform_reduce",
        variant=variant,
        description=descs[variant % len(descs)],
        constraints=constraints,
        behavioral_tests=[],
        token_budget=4000,
        turn_cap=20,
    )


def _tier1_config_loader(variant: int) -> TaskSpec:
    """Single-module config loading with validation."""
    domains = [
        ("app",     ["load_defaults", "read_file", "parse_json", "validate_keys", "merge_config"]),
        ("db",      ["load_defaults", "read_env", "parse_dsn", "validate_schema", "build_url"]),
        ("api",     ["load_defaults", "read_headers", "parse_auth", "validate_token", "build_client"]),
        ("service", ["load_defaults", "read_yaml", "parse_settings", "validate_required", "apply_overrides"]),
    ]
    domain, funcs = domains[variant % len(domains)]
    f = funcs

    descs = [
        f"Implement a {domain} configuration loader with defaults, file reading, parsing, validation, and merging.",
        f"Build a robust {domain} config system that loads from files, validates keys, and merges with defaults.",
        f"Create a {domain} configuration module with clear separation between loading, parsing, and validation.",
        f"Design a composable {domain} config loader where each concern is handled by its own function.",
    ]

    constraints = [
        _cid("node_exists", f"config.{f[0]}"),
        _cid("node_exists", f"config.{f[2]}"),
        _cid("node_exists", f"config.{f[3]}"),
        _cid("node_exists", f"config.{f[4]}"),
        _cid("return_type", f"config.{f[0]}", value="dict"),
        _cid("return_type", f"config.{f[3]}", value="bool"),
        _cid("module_count", "", value=1),
        _cid("module_responsibility", "config", value="io"),
        _cid("error_handling_present", f"config.{f[2]}"),
        _cid("error_handling_present", f"config.{f[3]}", hidden=True),
        _cid("acyclic_imports", ""),
        _cid("no_any_types", "", hidden=True),
        _cid("fan_out_max", f"config.{f[4]}", value=3, hidden=True),
        _cid("dag_depth_max", "", value=5, hidden=True),
        _cid("entrypoint_exists", f"config.{f[4]}", value=None),
    ]

    behavioral_tests = [
        BehavioralTest(
            test_id="test_load_defaults_returns_dict",
            description=f"Check {f[0]} returns a dict",
            hypothesis_source=_dict_return_test("config", f[0]),
        )
    ]

    return TaskSpec(
        task_id=f"t1_config_v{variant}",
        tier=1,
        template_name="config_loader",
        variant=variant,
        description=descs[variant % len(descs)],
        constraints=constraints,
        behavioral_tests=behavioral_tests,
        token_budget=4000,
        turn_cap=20,
    )


# =============================================================================
# Tier 2 templates (6 templates × 4 variants = 24 tasks)
# Each: 10-18 nodes, 3-4 modules, ~30 constraints, 35% hidden, 3-5 behavioral tests
# =============================================================================

def _tier2_data_pipeline(variant: int) -> TaskSpec:
    """Multi-module ETL data pipeline."""
    domains = [
        ("csv",    "ingest", "transform", "storage",
         ["read_file","parse_header","parse_row","validate_row"],
         ["normalize_field","type_cast","apply_transform","filter_nulls"],
         ["write_record","flush_batch","compute_stats","close_writer"]),
        ("json",   "ingest", "transform", "storage",
         ["read_stream","parse_object","validate_schema","flatten_nested"],
         ["extract_fields","map_values","deduplicate","sort_records"],
         ["build_index","write_batch","compute_checksum","finalize"]),
        ("xml",    "ingest", "transform", "storage",
         ["open_file","parse_element","validate_structure","extract_text"],
         ["clean_whitespace","normalize_attrs","merge_siblings","tag_records"],
         ["serialize_row","append_output","verify_output","close_output"]),
        ("tsv",    "ingest", "transform", "storage",
         ["read_tsv","detect_delimiter","parse_columns","check_encoding"],
         ["strip_quotes","convert_types","fill_defaults","validate_types"],
         ["format_output","write_lines","build_summary","close_file"]),
    ]
    domain, mod_a, mod_b, mod_c, fa, fb, fc = domains[variant % len(domains)]

    descs = [
        f"Build a {domain} ETL pipeline with three modules: {mod_a}, {mod_b}, and {mod_c}.",
        f"Implement a {domain} data processing system split across ingestion, transformation, and storage layers.",
        f"Create a {domain} pipeline following strict module boundaries between reading, processing, and writing.",
        f"Design a {domain} data pipeline where each module has a clear responsibility and no circular imports.",
    ]

    constraints = []
    for name in fa:
        constraints.append(_cid("node_exists", f"{mod_a}.{name}"))
    for name in fb:
        constraints.append(_cid("node_exists", f"{mod_b}.{name}"))
    for name in fc:
        constraints.append(_cid("node_exists", f"{mod_c}.{name}"))
    constraints += [
        _cid("module_count", "", value=3),
        _cid("module_responsibility", mod_a, value="io"),
        _cid("module_responsibility", mod_b, value="transform", hidden=True),
        _cid("module_responsibility", mod_c, value="io"),
        _cid("acyclic_imports", ""),
        _cid("edge_exists", f"{mod_a}.{fa[-1]}::{mod_b}.{fb[0]}", hidden=True),
        _cid("edge_exists", f"{mod_b}.{fb[-1]}::{mod_c}.{fc[0]}", hidden=True),
        _cid("no_any_types", "", hidden=True),
        _cid("type_consistency", "", hidden=True),
        _cid("dag_depth_max", "", value=8, hidden=True),
        _cid("fan_out_max", f"{mod_b}.{fb[0]}", value=4, hidden=True),
        _cid("pure_function", f"{mod_b}.{fb[0]}"),
        _cid("pure_function", f"{mod_b}.{fb[1]}", hidden=True),
        _cid("error_handling_present", f"{mod_a}.{fa[0]}"),
        _cid("error_handling_present", f"{mod_c}.{fc[0]}", hidden=True),
    ]

    behavioral_tests = [
        BehavioralTest(
            test_id="test_parse_returns_list",
            description=f"Check {fa[1]} returns a list",
            hypothesis_source=_list_return_test(mod_a, fa[1]),
        ),
        BehavioralTest(
            test_id="test_transform_idempotent",
            description=f"Check {fb[0]} is deterministic",
            hypothesis_source=_simple_type_test(mod_b, fb[0], "str"),
        ),
        BehavioralTest(
            test_id="test_validate_returns_bool",
            description=f"Check {fa[2]} returns bool",
            hypothesis_source=_bool_return_test(mod_a, fa[2]),
        ),
    ]

    return TaskSpec(
        task_id=f"t2_datapipeline_v{variant}",
        tier=2,
        template_name="data_pipeline",
        variant=variant,
        description=descs[variant % len(descs)],
        constraints=constraints,
        behavioral_tests=behavioral_tests,
        token_budget=8000,
        turn_cap=35,
    )


def _tier2_validator_chain(variant: int) -> TaskSpec:
    """Multi-module validation with rules and reporting."""
    domains = [
        ("form",    ["parse_field","strip_html","normalize_space"],
                    ["rule_required","rule_min_len","rule_max_len","rule_pattern"],
                    ["collect_errors","format_message","build_report"]),
        ("api_req", ["parse_body","decode_json","normalize_keys"],
                    ["check_auth","check_rate","check_schema","check_quota"],
                    ["aggregate_errors","map_codes","serialize_response"]),
        ("config",  ["read_source","tokenize_values","cast_types"],
                    ["rule_type","rule_range","rule_enum","rule_required"],
                    ["gather_issues","format_issue","emit_report"]),
        ("event",   ["parse_event","extract_fields","normalize_types"],
                    ["check_version","check_source","check_payload","check_timestamp"],
                    ["collect_violations","describe_violation","build_audit"]),
    ]
    domain, fa, fb, fc = domains[variant % len(domains)]

    descs = [
        f"Build a {domain} validation system with parsing, rule checking, and error reporting modules.",
        f"Implement a three-module {domain} validator: input parsing, rule application, and report generation.",
        f"Create a composable {domain} validation framework where rules and reporting are in separate modules.",
        f"Design a {domain} validation pipeline with clean module boundaries for each validation stage.",
    ]

    constraints = []
    for name in fa:
        constraints.append(_cid("node_exists", f"parser.{name}"))
    for name in fb:
        constraints.append(_cid("node_exists", f"rules.{name}"))
    for name in fc:
        constraints.append(_cid("node_exists", f"reporter.{name}"))
    constraints += [
        _cid("module_count", "", value=3),
        _cid("module_responsibility", "parser", value="transform"),
        _cid("module_responsibility", "rules", value="validation", hidden=True),
        _cid("module_responsibility", "reporter", value="io"),
        _cid("return_type", f"rules.{fb[0]}", value="bool"),
        _cid("return_type", f"rules.{fb[1]}", value="bool", hidden=True),
        _cid("pure_function", f"rules.{fb[0]}"),
        _cid("pure_function", f"rules.{fb[1]}", hidden=True),
        _cid("acyclic_imports", ""),
        _cid("no_any_types", "", hidden=True),
        _cid("fan_out_max", f"reporter.{fc[-1]}", value=5, hidden=True),
        _cid("edge_exists", f"parser.{fa[-1]}::rules.{fb[0]}", hidden=True),
        _cid("error_handling_present", f"reporter.{fc[-1]}", hidden=True),
        _cid("dag_depth_max", "", value=7, hidden=True),
        _cid("entrypoint_exists", f"reporter.{fc[-1]}", value=None),
    ]

    behavioral_tests = [
        BehavioralTest(
            test_id="test_rule_returns_bool",
            description=f"Check {fb[0]} returns bool",
            hypothesis_source=_bool_return_test("rules", fb[0]),
        ),
        BehavioralTest(
            test_id="test_reporter_returns_dict",
            description=f"Check {fc[-1]} returns dict",
            hypothesis_source=_dict_return_test("reporter", fc[-1]),
        ),
        BehavioralTest(
            test_id="test_parser_returns_expected",
            description=f"Check {fa[0]} output type",
            hypothesis_source=_simple_type_test("parser", fa[0], "str"),
        ),
    ]

    return TaskSpec(
        task_id=f"t2_validator_v{variant}",
        tier=2,
        template_name="validator_chain",
        variant=variant,
        description=descs[variant % len(descs)],
        constraints=constraints,
        behavioral_tests=behavioral_tests,
        token_budget=8000,
        turn_cap=35,
    )


def _tier2_event_router(variant: int) -> TaskSpec:
    """Multi-module event routing system."""
    domains = [
        ("webhook", ["parse_payload","verify_signature","extract_event_type"],
                    ["route_created","route_updated","route_deleted","route_default"],
                    ["format_ack","log_event","emit_response"]),
        ("message", ["decode_message","validate_format","extract_routing_key"],
                    ["handle_order","handle_payment","handle_shipment","handle_cancel"],
                    ["build_reply","publish_result","record_metrics"]),
        ("sensor",  ["parse_reading","validate_range","classify_event"],
                    ["handle_alert","handle_warning","handle_normal","handle_unknown"],
                    ["format_notification","store_event","trigger_action"]),
        ("command", ["parse_command","validate_syntax","extract_intent"],
                    ["exec_create","exec_update","exec_delete","exec_query"],
                    ["build_response","log_audit","send_reply"]),
    ]
    domain, fa, fb, fc = domains[variant % len(domains)]

    descs = [
        f"Build a {domain} event routing system: parse and classify events, dispatch to handlers, emit responses.",
        f"Implement a three-module {domain} router with ingestion, dispatch, and output stages.",
        f"Create a {domain} event processing system where routing logic is isolated from I/O.",
        f"Design a {domain} message router with clear dispatch table and isolated output formatting.",
    ]

    constraints = []
    for name in fa: constraints.append(_cid("node_exists", f"ingestion.{name}"))
    for name in fb: constraints.append(_cid("node_exists", f"handlers.{name}"))
    for name in fc: constraints.append(_cid("node_exists", f"output.{name}"))
    constraints += [
        _cid("module_count", "", value=3),
        _cid("module_responsibility", "ingestion", value="transform"),
        _cid("module_responsibility", "handlers", value="orchestration", hidden=True),
        _cid("module_responsibility", "output", value="io"),
        _cid("acyclic_imports", ""),
        _cid("edge_exists", f"ingestion.{fa[-1]}::handlers.{fb[0]}", hidden=True),
        _cid("edge_exists", f"handlers.{fb[0]}::output.{fc[0]}", hidden=True),
        _cid("dag_depth_max", "", value=8, hidden=True),
        _cid("fan_in_max", f"output.{fc[0]}", value=4, hidden=True),
        _cid("no_any_types", "", hidden=True),
        _cid("error_handling_present", f"ingestion.{fa[0]}"),
        _cid("error_handling_present", f"handlers.{fb[-1]}", hidden=True),
        _cid("pure_function", f"ingestion.{fa[0]}"),
        _cid("pure_function", f"ingestion.{fa[2]}", hidden=True),
        _cid("entrypoint_exists", f"ingestion.{fa[0]}", value=None),
    ]

    behavioral_tests = [
        BehavioralTest(
            test_id="test_parse_input",
            description=f"Check {fa[0]} returns expected type",
            hypothesis_source=_simple_type_test("ingestion", fa[0], "str"),
        ),
        BehavioralTest(
            test_id="test_default_handler_exists",
            description=f"Verify {fb[-1]} is callable",
            hypothesis_source=_callable_test("handlers", fb[-1]),
        ),
        BehavioralTest(
            test_id="test_output_format",
            description=f"Check {fc[0]} returns a string",
            hypothesis_source=_simple_type_test("output", fc[0], "str"),
        ),
    ]

    return TaskSpec(
        task_id=f"t2_eventrouter_v{variant}",
        tier=2,
        template_name="event_router",
        variant=variant,
        description=descs[variant % len(descs)],
        constraints=constraints,
        behavioral_tests=behavioral_tests,
        token_budget=8000,
        turn_cap=35,
    )


def _tier2_config_loader(variant: int) -> TaskSpec:
    """Multi-module config loading with schema validation."""
    domains = [
        ("app",    ["read_file","detect_format","parse_content"],
                   ["check_required","check_types","check_ranges","coerce_values"],
                   ["merge_layers","apply_env","resolve_refs","build_final"]),
        ("infra",  ["scan_sources","load_source","decode_content"],
                   ["validate_fields","validate_values","check_dependencies","normalize"],
                   ["merge_all","substitute_vars","expand_paths","finalize_config"]),
        ("plugin", ["discover_plugins","read_manifest","load_settings"],
                   ["check_version","check_conflicts","check_permissions","validate_schema"],
                   ["merge_plugin_config","apply_defaults","lock_config","export_config"]),
        ("multi",  ["list_configs","read_config","deserialize"],
                   ["schema_check","range_check","enum_check","cross_check"],
                   ["layer_merge","env_override","secret_inject","config_output"]),
    ]
    domain, fa, fb, fc = domains[variant % len(domains)]

    descs = [
        f"Build a {domain} configuration system with source reading, schema validation, and config merging.",
        f"Implement a multi-layer {domain} config loader across three modules: reading, validation, merging.",
        f"Create a {domain} configuration framework with clear boundaries between I/O, validation, and assembly.",
        f"Design a robust {domain} config system where validation rules are isolated and composable.",
    ]

    constraints = []
    for name in fa: constraints.append(_cid("node_exists", f"reader.{name}"))
    for name in fb: constraints.append(_cid("node_exists", f"schema.{name}"))
    for name in fc: constraints.append(_cid("node_exists", f"assembler.{name}"))
    constraints += [
        _cid("module_count", "", value=3),
        _cid("module_responsibility", "reader", value="io"),
        _cid("module_responsibility", "schema", value="validation", hidden=True),
        _cid("module_responsibility", "assembler", value="orchestration"),
        _cid("return_type", f"schema.{fb[0]}", value="bool"),
        _cid("return_type", f"schema.{fb[1]}", value="bool", hidden=True),
        _cid("return_type", f"assembler.{fc[-1]}", value="dict", hidden=True),
        _cid("acyclic_imports", ""),
        _cid("dag_depth_max", "", value=8, hidden=True),
        _cid("error_handling_present", f"reader.{fa[0]}"),
        _cid("error_handling_present", f"assembler.{fc[-1]}", hidden=True),
        _cid("no_any_types", "", hidden=True),
        _cid("pure_function", f"schema.{fb[0]}", hidden=True),
        _cid("entrypoint_exists", f"assembler.{fc[-1]}", value=None),
        _cid("fan_out_max", f"assembler.{fc[-1]}", value=5, hidden=True),
    ]

    behavioral_tests = [
        BehavioralTest(
            test_id="test_validator_returns_bool",
            description=f"Check {fb[0]} returns bool",
            hypothesis_source=_bool_return_test("schema", fb[0]),
        ),
        BehavioralTest(
            test_id="test_final_output_is_dict",
            description=f"Check {fc[-1]} returns dict",
            hypothesis_source=_dict_return_test("assembler", fc[-1]),
        ),
        BehavioralTest(
            test_id="test_reader_not_empty",
            description=f"Check {fa[0]} does not return empty",
            hypothesis_source=_nonempty_test("reader", fa[0]),
        ),
        BehavioralTest(
            test_id="test_schema_check_rejects_empty",
            description=f"Check {fb[0]} rejects empty input gracefully",
            hypothesis_source=_bool_return_test("schema", fb[0]),
        ),
    ]

    return TaskSpec(
        task_id=f"t2_config_v{variant}",
        tier=2,
        template_name="config_loader",
        variant=variant,
        description=descs[variant % len(descs)],
        constraints=constraints,
        behavioral_tests=behavioral_tests,
        token_budget=8000,
        turn_cap=35,
    )


def _tier2_batch_processor(variant: int) -> TaskSpec:
    """Multi-module batch processing system."""
    domains = [
        ("image",   ["read_images","detect_format","decode_bytes"],
                    ["resize","convert_color","normalize_pixels","apply_filter"],
                    ["encode_output","write_batch","log_stats"]),
        ("document",["read_docs","detect_encoding","tokenize"],
                    ["clean_text","extract_entities","compute_tfidf","score_doc"],
                    ["serialize","write_index","update_stats"]),
        ("records", ["fetch_records","decode_records","shard_batch"],
                    ["validate_record","transform_record","enrich_record","dedupe_record"],
                    ["write_records","commit_batch","publish_metrics"]),
        ("audio",   ["read_chunks","detect_format","decode_audio"],
                    ["normalize_amplitude","trim_silence","extract_features","classify"],
                    ["encode_output","write_processed","emit_events"]),
    ]
    domain, fa, fb, fc = domains[variant % len(domains)]

    descs = [
        f"Build a {domain} batch processor with reading, transformation, and output modules.",
        f"Implement a {domain} processing pipeline that handles items in batches across three modules.",
        f"Create a {domain} batch system where each processing stage is an isolated module.",
        f"Design a {domain} processor with clear module separation for I/O, computation, and output.",
    ]

    constraints = []
    for name in fa: constraints.append(_cid("node_exists", f"reader.{name}"))
    for name in fb: constraints.append(_cid("node_exists", f"processor.{name}"))
    for name in fc: constraints.append(_cid("node_exists", f"writer.{name}"))
    constraints += [
        _cid("module_count", "", value=3),
        _cid("module_responsibility", "reader", value="io"),
        _cid("module_responsibility", "processor", value="transform", hidden=True),
        _cid("module_responsibility", "writer", value="io"),
        _cid("edge_exists", f"reader.{fa[-1]}::processor.{fb[0]}", hidden=True),
        _cid("acyclic_imports", ""),
        _cid("pure_function", f"processor.{fb[0]}"),
        _cid("pure_function", f"processor.{fb[1]}", hidden=True),
        _cid("error_handling_present", f"reader.{fa[0]}"),
        _cid("error_handling_present", f"writer.{fc[0]}", hidden=True),
        _cid("no_any_types", "", hidden=True),
        _cid("dag_depth_max", "", value=8, hidden=True),
        _cid("fan_out_max", f"processor.{fb[0]}", value=4, hidden=True),
        _cid("entrypoint_exists", f"writer.{fc[0]}", value=None),
        _cid("type_consistency", "", hidden=True),
    ]

    behavioral_tests = [
        BehavioralTest(
            test_id="test_processor_pure",
            description=f"Check {fb[0]} is deterministic",
            hypothesis_source=_simple_type_test("processor", fb[0], "str"),
        ),
        BehavioralTest(
            test_id="test_reader_returns_list",
            description=f"Check {fa[0]} returns list",
            hypothesis_source=_list_return_test("reader", fa[0]),
        ),
        BehavioralTest(
            test_id="test_writer_produces_output",
            description=f"Check {fc[0]} returns expected type",
            hypothesis_source=_simple_type_test("writer", fc[0], "str"),
        ),
    ]

    return TaskSpec(
        task_id=f"t2_batch_v{variant}",
        tier=2,
        template_name="batch_processor",
        variant=variant,
        description=descs[variant % len(descs)],
        constraints=constraints,
        behavioral_tests=behavioral_tests,
        token_budget=8000,
        turn_cap=35,
    )


def _tier2_transform_reduce(variant: int) -> TaskSpec:
    """Multi-module transform-reduce with aggregation."""
    domains = [
        ("metrics",  ["collect_raw","parse_metric","tag_metric"],
                     ["normalize","compute_rate","apply_aggregation","smooth"],
                     ["write_ts","compute_summary","emit_alerts"]),
        ("analytics",["fetch_events","parse_event","enrich_event"],
                     ["segment","filter_cohort","compute_funnel","compute_retention"],
                     ["build_report","export_csv","send_notification"]),
        ("finance",  ["fetch_transactions","parse_amount","categorize"],
                     ["compute_balance","compute_avg","detect_anomaly","classify_spend"],
                     ["generate_statement","export_data","archive_records"]),
        ("telemetry",["receive_spans","parse_span","correlate_spans"],
                     ["compute_latency","detect_outlier","aggregate_p99","tag_error"],
                     ["publish_dashboard","store_traces","trigger_alert"]),
    ]
    domain, fa, fb, fc = domains[variant % len(domains)]

    descs = [
        f"Build a {domain} analytics pipeline: collect raw data, transform and aggregate, then output results.",
        f"Implement a {domain} reduce pipeline across ingestion, computation, and output modules.",
        f"Create a three-module {domain} processing system with clear data flow from input to output.",
        f"Design a {domain} data pipeline where computation is isolated from I/O in all three modules.",
    ]

    constraints = []
    for name in fa: constraints.append(_cid("node_exists", f"ingest.{name}"))
    for name in fb: constraints.append(_cid("node_exists", f"compute.{name}"))
    for name in fc: constraints.append(_cid("node_exists", f"output.{name}"))
    constraints += [
        _cid("module_count_range", "", value=[3, 4]),
        _cid("module_responsibility", "ingest", value="io"),
        _cid("module_responsibility", "compute", value="transform"),
        _cid("acyclic_imports", ""),
        _cid("edge_exists", f"ingest.{fa[-1]}::compute.{fb[0]}", hidden=True),
        _cid("no_any_types", "", hidden=True),
        _cid("pure_function", f"compute.{fb[0]}"),
        _cid("pure_function", f"compute.{fb[1]}", hidden=True),
        _cid("error_handling_present", f"ingest.{fa[0]}"),
        _cid("fan_out_max", f"compute.{fb[2]}", value=3, hidden=True),
        _cid("dag_depth_max", "", value=9, hidden=True),
        _cid("entrypoint_exists", f"output.{fc[0]}", value=None),
        _cid("type_consistency", "", hidden=True),
        _cid("internal_only", f"compute.{fb[1]}", hidden=True),
        _cid("module_size_max", "compute", value=6, hidden=True),
    ]

    behavioral_tests = [
        BehavioralTest(
            test_id="test_compute_returns_number",
            description=f"Check {fb[0]} returns a number",
            hypothesis_source=_numeric_return_test("compute", fb[0]),
        ),
        BehavioralTest(
            test_id="test_output_format",
            description=f"Check {fc[0]} returns str",
            hypothesis_source=_simple_type_test("output", fc[0], "str"),
        ),
        BehavioralTest(
            test_id="test_ingest_pure",
            description=f"Check {fa[1]} is deterministic",
            hypothesis_source=_simple_type_test("ingest", fa[1], "str"),
        ),
        BehavioralTest(
            test_id="test_aggregation_stable",
            description=f"Check {fb[2]} produces consistent output",
            hypothesis_source=_numeric_return_test("compute", fb[2]),
        ),
    ]

    return TaskSpec(
        task_id=f"t2_transformreduce_v{variant}",
        tier=2,
        template_name="transform_reduce",
        variant=variant,
        description=descs[variant % len(descs)],
        constraints=constraints,
        behavioral_tests=behavioral_tests,
        token_budget=8000,
        turn_cap=35,
    )


# =============================================================================
# Tier 3 templates (4 templates × 3 variants = 12 tasks)
# Each: 18-30 nodes, 5-7 modules, ~50 constraints, 40% hidden, 5-8 behavioral tests
# =============================================================================

def _tier3_microservice(variant: int) -> TaskSpec:
    """Full 5-module microservice with auth, validation, business logic, storage, api."""
    domains = [
        "user_registration",
        "order_processing",
        "payment_gateway",
    ]
    domain = domains[variant % len(domains)]

    descs = [
        f"Implement a {domain} microservice with authentication, input validation, business logic, persistence, and API layers.",
        f"Build a complete {domain} service split across auth, validation, core logic, storage, and HTTP API modules.",
        f"Design a {domain} backend with strict module boundaries: each layer has exactly one responsibility.",
    ]

    auth_funcs = ["verify_token", "decode_claims", "check_permissions", "refresh_token"]
    val_funcs  = ["parse_request", "validate_fields", "check_uniqueness", "normalize_input", "sanitize"]
    core_funcs = ["execute_operation", "apply_rules", "compute_result", "emit_events", "rollback_on_error"]
    store_funcs= ["begin_tx", "persist_entity", "query_entity", "commit_tx", "rollback_tx"]
    api_funcs  = ["parse_http", "authenticate", "dispatch", "format_response", "handle_error"]

    constraints = []
    for name in auth_funcs: constraints.append(_cid("node_exists", f"auth.{name}"))
    for name in val_funcs:  constraints.append(_cid("node_exists", f"validation.{name}"))
    for name in core_funcs: constraints.append(_cid("node_exists", f"core.{name}"))
    for name in store_funcs:constraints.append(_cid("node_exists", f"storage.{name}"))
    for name in api_funcs:  constraints.append(_cid("node_exists", f"api.{name}"))
    constraints += [
        _cid("module_count", "", value=5),
        _cid("module_responsibility", "auth", value="validation", hidden=True),
        _cid("module_responsibility", "validation", value="validation"),
        _cid("module_responsibility", "core", value="orchestration", hidden=True),
        _cid("module_responsibility", "storage", value="io"),
        _cid("module_responsibility", "api", value="io"),
        _cid("acyclic_imports", ""),
        _cid("dag_depth_max", "", value=12, hidden=True),
        _cid("return_type", "auth.verify_token", value="bool"),
        _cid("return_type", "storage.persist_entity", value="bool", hidden=True),
        _cid("pure_function", "auth.decode_claims"),
        _cid("pure_function", "validation.parse_request", hidden=True),
        _cid("pure_function", "validation.validate_fields", hidden=True),
        _cid("error_handling_present", "storage.persist_entity"),
        _cid("error_handling_present", "storage.begin_tx", hidden=True),
        _cid("error_handling_present", "api.handle_error"),
        _cid("edge_exists", "api.authenticate::auth.verify_token", hidden=True),
        _cid("edge_exists", "api.dispatch::core.execute_operation", hidden=True),
        _cid("edge_exists", "core.execute_operation::storage.persist_entity", hidden=True),
        _cid("edge_exists", "core.execute_operation::validation.check_uniqueness", hidden=True),
        _cid("internal_only", "storage.begin_tx", hidden=True),
        _cid("internal_only", "storage.commit_tx", hidden=True),
        _cid("fan_in_max", "core.execute_operation", value=2, hidden=True),
        _cid("fan_out_max", "core.execute_operation", value=5, hidden=True),
        _cid("no_any_types", "", hidden=True),
        _cid("type_consistency", "", hidden=True),
        _cid("module_size_max", "auth", value=6),
        _cid("module_size_max", "storage", value=6, hidden=True),
        _cid("entrypoint_exists", "api.dispatch", value=None),
        _cid("error_handling_absent", "auth.decode_claims", hidden=True),
        _cid("error_handling_absent", "validation.parse_request", hidden=True),
    ]

    behavioral_tests = [
        BehavioralTest(test_id="test_verify_returns_bool", description="auth.verify_token returns bool",
            hypothesis_source=_bool_return_test("auth", "verify_token")),
        BehavioralTest(test_id="test_persist_returns_bool", description="storage.persist_entity returns bool",
            hypothesis_source=_bool_return_test("storage", "persist_entity")),
        BehavioralTest(test_id="test_parse_request_pure", description="validation.parse_request is deterministic",
            hypothesis_source=_simple_type_test("validation", "parse_request", "dict")),
        BehavioralTest(test_id="test_core_callable", description="core.execute_operation is callable",
            hypothesis_source=_callable_test("core", "execute_operation")),
        BehavioralTest(test_id="test_api_response_format", description="api.format_response returns str",
            hypothesis_source=_simple_type_test("api", "format_response", "str")),
        BehavioralTest(test_id="test_decode_claims_pure", description="auth.decode_claims is pure",
            hypothesis_source=_simple_type_test("auth", "decode_claims", "dict")),
    ]

    return TaskSpec(
        task_id=f"t3_microservice_v{variant}",
        tier=3,
        template_name="microservice",
        variant=variant,
        description=descs[variant % len(descs)],
        constraints=constraints,
        behavioral_tests=behavioral_tests,
        token_budget=14000,
        turn_cap=55,
    )


def _tier3_query_engine(variant: int) -> TaskSpec:
    """6-module query engine: parse, plan, optimize, execute, cache, output."""
    domains = ["sql_subset", "graph_query", "filter_dsl"]
    domain = domains[variant % len(domains)]

    descs = [
        f"Build a {domain} query engine with 6 modules: tokenizer, parser, planner, optimizer, executor, and formatter.",
        f"Implement a complete {domain} query processor split across lexing, parsing, planning, optimization, execution, and output.",
        f"Design a {domain} query system where each processing stage is fully isolated in its own module.",
    ]

    lex_funcs    = ["tokenize", "classify_token", "handle_literal", "handle_operator"]
    parse_funcs  = ["parse_tokens", "build_ast", "validate_syntax", "extract_clauses"]
    plan_funcs   = ["build_plan", "estimate_cost", "choose_strategy", "annotate_plan"]
    opt_funcs    = ["apply_rules", "push_predicates", "eliminate_duplicates", "reorder_joins"]
    exec_funcs   = ["execute_plan", "fetch_data", "apply_filter", "compute_projection", "sort_results"]
    fmt_funcs    = ["serialize_result", "paginate", "format_error", "build_metadata"]

    constraints = []
    for name in lex_funcs:   constraints.append(_cid("node_exists", f"lexer.{name}"))
    for name in parse_funcs: constraints.append(_cid("node_exists", f"parser.{name}"))
    for name in plan_funcs:  constraints.append(_cid("node_exists", f"planner.{name}"))
    for name in opt_funcs:   constraints.append(_cid("node_exists", f"optimizer.{name}"))
    for name in exec_funcs:  constraints.append(_cid("node_exists", f"executor.{name}"))
    for name in fmt_funcs:   constraints.append(_cid("node_exists", f"formatter.{name}"))
    constraints += [
        _cid("module_count", "", value=6),
        _cid("module_responsibility", "lexer", value="transform", hidden=True),
        _cid("module_responsibility", "parser", value="transform"),
        _cid("module_responsibility", "planner", value="orchestration", hidden=True),
        _cid("module_responsibility", "optimizer", value="transform", hidden=True),
        _cid("module_responsibility", "executor", value="io"),
        _cid("module_responsibility", "formatter", value="io"),
        _cid("acyclic_imports", ""),
        _cid("dag_depth_max", "", value=15, hidden=True),
        _cid("edge_exists", "lexer.tokenize::parser.parse_tokens", hidden=True),
        _cid("edge_exists", "parser.build_ast::planner.build_plan", hidden=True),
        _cid("edge_exists", "planner.build_plan::optimizer.apply_rules", hidden=True),
        _cid("edge_exists", "optimizer.apply_rules::executor.execute_plan", hidden=True),
        _cid("pure_function", "lexer.tokenize", hidden=True),
        _cid("pure_function", "parser.build_ast"),
        _cid("pure_function", "optimizer.apply_rules"),
        _cid("pure_function", "optimizer.push_predicates", hidden=True),
        _cid("no_any_types", "", hidden=True),
        _cid("type_consistency", "", hidden=True),
        _cid("fan_out_max", "planner.build_plan", value=3, hidden=True),
        _cid("fan_out_max", "executor.execute_plan", value=4, hidden=True),
        _cid("internal_only", "planner.annotate_plan", hidden=True),
        _cid("internal_only", "optimizer.eliminate_duplicates", hidden=True),
        _cid("error_handling_present", "executor.execute_plan"),
        _cid("error_handling_present", "executor.fetch_data", hidden=True),
        _cid("module_size_max", "lexer", value=5, hidden=True),
        _cid("module_size_max", "optimizer", value=5),
        _cid("entrypoint_exists", "executor.execute_plan", value=None),
        _cid("return_type", "lexer.tokenize", value="list"),
        _cid("return_type", "optimizer.apply_rules", value="dict", hidden=True),
    ]

    behavioral_tests = [
        BehavioralTest(test_id="test_tokenize_returns_list", description="lexer.tokenize returns list",
            hypothesis_source=_list_return_test("lexer", "tokenize")),
        BehavioralTest(test_id="test_plan_returns_dict", description="planner.build_plan returns dict",
            hypothesis_source=_dict_return_test("planner", "build_plan")),
        BehavioralTest(test_id="test_optimizer_pure", description="optimizer.apply_rules is pure",
            hypothesis_source=_simple_type_test("optimizer", "apply_rules", "dict")),
        BehavioralTest(test_id="test_formatter_returns_str", description="formatter.serialize_result returns str",
            hypothesis_source=_simple_type_test("formatter", "serialize_result", "str")),
        BehavioralTest(test_id="test_classify_token_bool", description="lexer.classify_token returns bool or str",
            hypothesis_source=_callable_test("lexer", "classify_token")),
        BehavioralTest(test_id="test_error_handling_present", description="executor.execute_plan handles exceptions",
            hypothesis_source=_callable_test("executor", "execute_plan")),
        BehavioralTest(test_id="test_parser_deterministic", description="parser.build_ast is deterministic",
            hypothesis_source=_simple_type_test("parser", "build_ast", "dict")),
    ]

    return TaskSpec(
        task_id=f"t3_queryengine_v{variant}",
        tier=3,
        template_name="query_engine",
        variant=variant,
        description=descs[variant % len(descs)],
        constraints=constraints,
        behavioral_tests=behavioral_tests,
        token_budget=14000,
        turn_cap=55,
    )


def _tier3_workflow_engine(variant: int) -> TaskSpec:
    """5-module workflow engine with task scheduling, state management, execution, retry, monitoring."""
    domains = ["job_scheduler", "dag_runner", "pipeline_orchestrator"]
    domain = domains[variant % len(domains)]

    descs = [
        f"Build a {domain} with five modules: task definition, scheduler, executor, state tracker, and monitor.",
        f"Implement a complete {domain} where task scheduling, execution, and monitoring are isolated modules.",
        f"Design a {domain} system with clear separation between task definition, orchestration, and state management.",
    ]

    defn_funcs  = ["define_task", "validate_task", "compute_dependencies", "serialize_task"]
    sched_funcs = ["build_schedule", "topological_sort", "assign_priority", "check_resources"]
    exec_funcs  = ["start_task", "run_task", "collect_result", "handle_failure", "retry_task"]
    state_funcs = ["init_state", "update_state", "query_state", "finalize_state"]
    mon_funcs   = ["record_event", "compute_metrics", "check_health", "emit_alert", "generate_report"]

    constraints = []
    for name in defn_funcs:  constraints.append(_cid("node_exists", f"definition.{name}"))
    for name in sched_funcs: constraints.append(_cid("node_exists", f"scheduler.{name}"))
    for name in exec_funcs:  constraints.append(_cid("node_exists", f"executor.{name}"))
    for name in state_funcs: constraints.append(_cid("node_exists", f"state.{name}"))
    for name in mon_funcs:   constraints.append(_cid("node_exists", f"monitor.{name}"))
    constraints += [
        _cid("module_count", "", value=5),
        _cid("module_responsibility", "definition", value="transform"),
        _cid("module_responsibility", "scheduler", value="orchestration", hidden=True),
        _cid("module_responsibility", "executor", value="orchestration"),
        _cid("module_responsibility", "state", value="io", hidden=True),
        _cid("module_responsibility", "monitor", value="io"),
        _cid("acyclic_imports", ""),
        _cid("dag_depth_max", "", value=12, hidden=True),
        _cid("pure_function", "definition.validate_task"),
        _cid("pure_function", "scheduler.topological_sort", hidden=True),
        _cid("pure_function", "definition.compute_dependencies", hidden=True),
        _cid("error_handling_present", "executor.run_task"),
        _cid("error_handling_present", "executor.retry_task", hidden=True),
        _cid("error_handling_present", "state.update_state", hidden=True),
        _cid("edge_exists", "scheduler.build_schedule::executor.start_task", hidden=True),
        _cid("edge_exists", "executor.run_task::state.update_state", hidden=True),
        _cid("edge_exists", "executor.collect_result::monitor.record_event", hidden=True),
        _cid("internal_only", "state.init_state", hidden=True),
        _cid("fan_in_max", "state.update_state", value=3, hidden=True),
        _cid("fan_out_max", "executor.run_task", value=4, hidden=True),
        _cid("no_any_types", "", hidden=True),
        _cid("type_consistency", "", hidden=True),
        _cid("return_type", "definition.validate_task", value="bool"),
        _cid("return_type", "monitor.generate_report", value="dict", hidden=True),
        _cid("entrypoint_exists", "executor.start_task", value=None),
        _cid("module_size_max", "state", value=5, hidden=True),
        _cid("module_size_max", "definition", value=5),
    ]

    behavioral_tests = [
        BehavioralTest(test_id="test_validate_task_bool", description="definition.validate_task returns bool",
            hypothesis_source=_bool_return_test("definition", "validate_task")),
        BehavioralTest(test_id="test_topo_sort_list", description="scheduler.topological_sort returns list",
            hypothesis_source=_list_return_test("scheduler", "topological_sort")),
        BehavioralTest(test_id="test_run_task_callable", description="executor.run_task is callable",
            hypothesis_source=_callable_test("executor", "run_task")),
        BehavioralTest(test_id="test_state_query_dict", description="state.query_state returns dict",
            hypothesis_source=_dict_return_test("state", "query_state")),
        BehavioralTest(test_id="test_monitor_report_dict", description="monitor.generate_report returns dict",
            hypothesis_source=_dict_return_test("monitor", "generate_report")),
        BehavioralTest(test_id="test_metrics_numeric", description="monitor.compute_metrics returns a number",
            hypothesis_source=_numeric_return_test("monitor", "compute_metrics")),
    ]

    return TaskSpec(
        task_id=f"t3_workflow_v{variant}",
        tier=3,
        template_name="workflow_engine",
        variant=variant,
        description=descs[variant % len(descs)],
        constraints=constraints,
        behavioral_tests=behavioral_tests,
        token_budget=14000,
        turn_cap=55,
    )


def _tier3_plugin_system(variant: int) -> TaskSpec:
    """7-module plugin/extension system."""
    domains = ["ide_extension", "cms_plugin", "analytics_addon"]
    domain = domains[variant % len(domains)]

    descs = [
        f"Build a {domain} plugin system with discovery, loading, validation, registry, lifecycle, api, and events modules.",
        f"Implement a complete {domain} plugin framework spanning 7 modules with strict isolation.",
        f"Design a {domain} extension system where each plugin lifecycle stage is handled by a dedicated module.",
    ]

    disc_funcs   = ["scan_directory", "find_manifests", "read_manifest", "filter_compatible"]
    load_funcs   = ["load_module", "instantiate_plugin", "inject_dependencies"]
    val_funcs    = ["validate_manifest", "check_interface", "verify_permissions"]
    reg_funcs    = ["register_plugin", "deregister_plugin", "lookup_plugin", "list_plugins"]
    life_funcs   = ["on_enable", "on_disable", "on_update", "on_error"]
    api_funcs    = ["call_hook", "broadcast_event", "get_plugin_api"]
    event_funcs  = ["emit_event", "subscribe", "unsubscribe", "dispatch_event"]

    constraints = []
    for name in disc_funcs:  constraints.append(_cid("node_exists", f"discovery.{name}"))
    for name in load_funcs:  constraints.append(_cid("node_exists", f"loader.{name}"))
    for name in val_funcs:   constraints.append(_cid("node_exists", f"validator.{name}"))
    for name in reg_funcs:   constraints.append(_cid("node_exists", f"registry.{name}"))
    for name in life_funcs:  constraints.append(_cid("node_exists", f"lifecycle.{name}"))
    for name in api_funcs:   constraints.append(_cid("node_exists", f"pluginapi.{name}"))
    for name in event_funcs: constraints.append(_cid("node_exists", f"events.{name}"))
    constraints += [
        _cid("module_count", "", value=7),
        _cid("module_responsibility", "discovery", value="io"),
        _cid("module_responsibility", "loader", value="orchestration", hidden=True),
        _cid("module_responsibility", "validator", value="validation"),
        _cid("module_responsibility", "registry", value="orchestration"),
        _cid("module_responsibility", "lifecycle", value="orchestration", hidden=True),
        _cid("module_responsibility", "pluginapi", value="orchestration"),
        _cid("module_responsibility", "events", value="io"),
        _cid("acyclic_imports", ""),
        _cid("dag_depth_max", "", value=14, hidden=True),
        _cid("pure_function", "validator.validate_manifest"),
        _cid("pure_function", "validator.check_interface", hidden=True),
        _cid("pure_function", "discovery.filter_compatible", hidden=True),
        _cid("return_type", "validator.validate_manifest", value="bool"),
        _cid("return_type", "validator.check_interface", value="bool", hidden=True),
        _cid("error_handling_present", "loader.load_module"),
        _cid("error_handling_present", "registry.register_plugin", hidden=True),
        _cid("edge_exists", "discovery.filter_compatible::loader.load_module", hidden=True),
        _cid("edge_exists", "loader.instantiate_plugin::validator.validate_manifest", hidden=True),
        _cid("edge_exists", "loader.instantiate_plugin::registry.register_plugin", hidden=True),
        _cid("edge_exists", "lifecycle.on_enable::events.emit_event", hidden=True),
        _cid("internal_only", "loader.inject_dependencies", hidden=True),
        _cid("fan_in_max", "registry.register_plugin", value=2, hidden=True),
        _cid("fan_out_max", "pluginapi.call_hook", value=3, hidden=True),
        _cid("no_any_types", "", hidden=True),
        _cid("type_consistency", "", hidden=True),
        _cid("module_size_max", "validator", value=4, hidden=True),
        _cid("module_size_max", "lifecycle", value=5),
        _cid("entrypoint_exists", "pluginapi.call_hook", value=None),
        _cid("error_handling_absent", "validator.validate_manifest", hidden=True),
    ]

    behavioral_tests = [
        BehavioralTest(test_id="test_validate_manifest_bool", description="validator.validate_manifest returns bool",
            hypothesis_source=_bool_return_test("validator", "validate_manifest")),
        BehavioralTest(test_id="test_list_plugins_list", description="registry.list_plugins returns list",
            hypothesis_source=_list_return_test("registry", "list_plugins")),
        BehavioralTest(test_id="test_scan_directory_list", description="discovery.scan_directory returns list",
            hypothesis_source=_list_return_test("discovery", "scan_directory")),
        BehavioralTest(test_id="test_emit_event_callable", description="events.emit_event is callable",
            hypothesis_source=_callable_test("events", "emit_event")),
        BehavioralTest(test_id="test_filter_compatible_list", description="discovery.filter_compatible returns list",
            hypothesis_source=_list_return_test("discovery", "filter_compatible")),
        BehavioralTest(test_id="test_check_interface_bool", description="validator.check_interface returns bool",
            hypothesis_source=_bool_return_test("validator", "check_interface")),
        BehavioralTest(test_id="test_call_hook_callable", description="pluginapi.call_hook is callable",
            hypothesis_source=_callable_test("pluginapi", "call_hook")),
        BehavioralTest(test_id="test_registry_lookup", description="registry.lookup_plugin returns str or None",
            hypothesis_source=_callable_test("registry", "lookup_plugin")),
    ]

    return TaskSpec(
        task_id=f"t3_plugin_v{variant}",
        tier=3,
        template_name="plugin_system",
        variant=variant,
        description=descs[variant % len(descs)],
        constraints=constraints,
        behavioral_tests=behavioral_tests,
        token_budget=14000,
        turn_cap=55,
    )


# =============================================================================
# Behavioral test source helpers
# =============================================================================

def _simple_type_test(module: str, func: str, expected_type: str) -> str:
    return f"""\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

def test_{func}_callable():
    import {module}
    assert callable(getattr({module}, '{func}', None)), "{func} must be callable"
"""


def _bool_return_test(module: str, func: str) -> str:
    return f"""\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

def test_{func}_callable():
    import {module}
    assert callable(getattr({module}, '{func}', None)), "{func} must be callable"
"""


def _dict_return_test(module: str, func: str) -> str:
    return f"""\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

def test_{func}_callable():
    import {module}
    assert callable(getattr({module}, '{func}', None)), "{func} must be callable"
"""


def _list_return_test(module: str, func: str) -> str:
    return f"""\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

def test_{func}_callable():
    import {module}
    assert callable(getattr({module}, '{func}', None)), "{func} must be callable"
"""


def _callable_test(module: str, func: str) -> str:
    return f"""\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

def test_{func}_callable():
    import {module}
    assert callable(getattr({module}, '{func}', None)), "{func} must be callable"
"""


def _nonempty_test(module: str, func: str) -> str:
    return f"""\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

def test_{func}_callable():
    import {module}
    assert callable(getattr({module}, '{func}', None)), "{func} must be callable"
"""


def _numeric_return_test(module: str, func: str) -> str:
    return f"""\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

def test_{func}_callable():
    import {module}
    assert callable(getattr({module}, '{func}', None)), "{func} must be callable"
"""


# =============================================================================
# TaskBank
# =============================================================================

# Registry: tier -> list of (template_fn, num_variants)
_TIER1_TEMPLATES = [
    (_tier1_simple_pipeline, 4),
    (_tier1_validator_chain, 4),
    (_tier1_transform_reduce, 4),
    (_tier1_config_loader, 4),
]
_TIER2_TEMPLATES = [
    (_tier2_data_pipeline, 4),
    (_tier2_validator_chain, 4),
    (_tier2_event_router, 4),
    (_tier2_config_loader, 4),
    (_tier2_batch_processor, 4),
    (_tier2_transform_reduce, 4),
]
_TIER3_TEMPLATES = [
    (_tier3_microservice, 3),
    (_tier3_query_engine, 3),
    (_tier3_workflow_engine, 3),
    (_tier3_plugin_system, 3),
]


class TaskBank:
    """Registry of ~50 task specs across three tiers."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._cache: Dict[str, TaskSpec] = {}
        self._build_index()

    def _build_index(self) -> None:
        for tier, templates in [
            (1, _TIER1_TEMPLATES),
            (2, _TIER2_TEMPLATES),
            (3, _TIER3_TEMPLATES),
        ]:
            for fn, num_variants in templates:
                for v in range(num_variants):
                    spec = fn(v)
                    self._cache[spec.task_id] = spec

    def sample(self, tier: int, seed: Optional[int] = None) -> TaskSpec:
        """Sample a random task from the given tier."""
        rng = random.Random(seed) if seed is not None else self._rng
        tier_tasks = [t for t in self._cache.values() if t.tier == tier]
        if not tier_tasks:
            raise ValueError(f"No tasks for tier {tier}")
        return rng.choice(tier_tasks)

    def get_task(self, task_id: str) -> TaskSpec:
        if task_id not in self._cache:
            raise KeyError(f"Unknown task_id: {task_id!r}")
        return self._cache[task_id]

    def list_tasks(self, tier: Optional[int] = None) -> List[str]:
        if tier is None:
            return list(self._cache)
        return [tid for tid, t in self._cache.items() if t.tier == tier]

    def __len__(self) -> int:
        return len(self._cache)
