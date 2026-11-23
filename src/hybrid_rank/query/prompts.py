# prompts.py
# The query decomposition prompt and its JSON output schema. Nothing here is dataset-specific: the catalog
# description and filter fields come from the dataset's adapter, and each field's typical values from the
# catalog itself (FilterIndex.catalog_notes).

from typing import Any, Dict, Mapping, Sequence

from hybrid_rank.schema import FilterField

SYSTEM_TEMPLATE = """\
You turn a shopper's search request into a search query and hard filters for a product catalog.

The catalog: {catalog}

Fields you can filter on:
{fields}

Return:
- semantic_query: the request rewritten as a concise product search query. Keep every detail about the \
item wanted (type, features, materials, uses, sizes, brands, and any product it should go with) and drop \
conversational filler ("Can you recommend", "I'm looking for").
- filters: a value for a field only when the shopper explicitly requires it of the item they want. \
Otherwise leave text fields as an empty list and number bounds as null.

A hard filter removes every product with a different value, so a wrong filter hides the right answer. \
When in doubt, leave the field empty:
- Don't filter on products mentioned as context: items the shopper already owns, or wants this one to go \
with, fit, or replace. "A case that fits my Garmin watch" has no brand filter.
- Don't infer values the shopper didn't state: "affordable" is not a price, and "for my daughter" is not \
a color.
- Use the shopper's own words, no narrower than they said them. List several values for a field only when \
the shopper accepts any of them ("red or blue")."""

MATCH_NOTE = {
    "contains": "matches catalog values containing all of its words",
    "prefix": "matches catalog values that start with it",
}


def _field_line(field: FilterField, note: str) -> str:
    kind = "number: min and/or max" if field.kind == "number" else f"text, {MATCH_NOTE[field.match]}"
    return f"- {field.name} ({kind}): {field.description}" + (f" {note}" if note else "")


def system_prompt(catalog: str, fields: Sequence[FilterField], notes: Mapping[str, str]) -> str:
    lines = "\n".join(_field_line(f, notes.get(f.name, "")) for f in fields)
    return SYSTEM_TEMPLATE.format(catalog=catalog, fields=lines)


def output_schema(fields: Sequence[FilterField]) -> Dict[str, Any]:
    """Every field is required, so the model answers each one: [] / null means no filter."""
    number_or_null = {"anyOf": [{"type": "number"}, {"type": "null"}]}
    properties: Dict[str, Any] = {}
    for f in fields:
        if f.kind == "number":
            properties[f.name] = {
                "type": "object",
                "properties": {"min": number_or_null, "max": number_or_null},
                "required": ["min", "max"],
                "additionalProperties": False,
            }
        else:
            properties[f.name] = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "semantic_query": {"type": "string"},
            "filters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
        "required": ["semantic_query", "filters"],
        "additionalProperties": False,
    }
