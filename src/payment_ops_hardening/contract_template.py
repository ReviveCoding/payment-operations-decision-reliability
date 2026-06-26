from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from payment_ops_hardening.atomic_io import atomic_write_json
from payment_ops_hardening.release_contract import load_release_contract
from payment_ops_hardening.strict_json import StrictJSONError, read_strict_json

_TOKEN = re.compile(r"__[A-Z0-9_]+__")


class ContractTemplateError(RuntimeError):
    """Raised when an integration contract template cannot be rendered safely."""


def _replace(value: Any, replacements: Mapping[str, object]) -> Any:
    if isinstance(value, str):
        if value in replacements:
            return replacements[value]
        result = value
        for token, replacement in replacements.items():
            if not isinstance(replacement, (str, int)):
                raise ContractTemplateError(
                    f"replacement for {token!r} must be a string or integer"
                )
            result = result.replace(token, str(replacement))
        return result
    if isinstance(value, list):
        return [_replace(item, replacements) for item in value]
    if isinstance(value, dict):
        rendered: dict[str, Any] = {}
        for key, item in value.items():
            rendered_key = _replace(str(key), replacements)
            if not isinstance(rendered_key, str):
                raise ContractTemplateError(
                    "rendered JSON object keys must remain strings"
                )
            if rendered_key in rendered:
                raise ContractTemplateError(
                    f"template rendering produced duplicate key: {rendered_key!r}"
                )
            rendered[rendered_key] = _replace(item, replacements)
        return rendered
    return value


def render_release_contract_template(
    template_path: str | Path,
    output_path: str | Path,
    replacements: Mapping[str, object],
) -> dict:
    """Render, validate, and atomically write an independent release contract."""
    source = Path(template_path)
    try:
        template = read_strict_json(source)
    except StrictJSONError as exc:
        raise ContractTemplateError(
            f"invalid contract template: {source}: {exc}"
        ) from exc
    if not isinstance(template, dict):
        raise ContractTemplateError("contract template must contain a JSON object")
    variables = template.pop("template_variables", None)
    if not isinstance(variables, list) or not variables:
        raise ContractTemplateError("template_variables must be a non-empty list")
    if any(
        not isinstance(item, str) or not _TOKEN.fullmatch(item) for item in variables
    ):
        raise ContractTemplateError("template_variables contains an invalid token")
    missing = sorted(set(variables) - set(replacements))
    unexpected = sorted(set(replacements) - set(variables))
    if missing or unexpected:
        raise ContractTemplateError(
            f"template replacement mismatch; missing={missing}, unexpected={unexpected}"
        )

    rendered = _replace(template, replacements)
    unresolved = sorted(set(_TOKEN.findall(str(rendered))))
    if unresolved:
        raise ContractTemplateError(f"unresolved template variables: {unresolved}")
    destination = Path(output_path)
    atomic_write_json(destination, rendered)
    try:
        return load_release_contract(destination)
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise ContractTemplateError(
            f"rendered release contract is invalid: {exc}"
        ) from exc
