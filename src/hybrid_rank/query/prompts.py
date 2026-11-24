# prompts.py
# The query decomposition prompt and its JSON output schema. Nothing here is dataset-specific: the catalog
# description and filter fields come from the dataset's adapter, and each field's typical values from the
# catalog itself (FilterIndex.catalog_notes).

from typing import Any, Dict, Mapping, Sequence, Tuple

from hybrid_rank.schema import FilterField

SYSTEM_TEMPLATE_V1 = """\
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

# v1 is frozen: its text is part of the cache key of committed decompositions. v2 spells out the
# brand phrasings seen in a STaRK train sample (requirements: "a BSN skipping rope", "made by Wincraft";
# context: "compatible with", "superior to", "I came across the"). On STaRK dev it filtered more queries
# but gained nothing (MRR 0.5543 vs 0.5549) and lost every answer on 8 queries instead of 4, so v1 stays default.
SYSTEM_TEMPLATE_V2 = """\
You turn a shopper's search request into a search query and hard filters for a product catalog.

The catalog: {catalog}

Fields you can filter on:
{fields}

Return:
- semantic_query: the request rewritten as a concise product search query. Keep every detail about the \
item wanted (type, features, materials, uses, sizes, brands, and any product it should go with) and drop \
conversational filler ("Can you recommend", "I'm looking for").
- filters: a value for a field only when the shopper requires it of the item they want. \
Otherwise leave text fields as an empty list and number bounds as null.

Decide each named brand, maker or value by its role in the request:
- It describes the item wanted: filter. "a Norvell tent", "a tent made by Norvell", "from the Norvell brand", \
"designed by Norvell", "by Norvell", "a Norvell X200 tent" all require the brand Norvell, even when other \
details are optional or the request is phrased as a question.
- It names another item: don't filter. That covers items the shopper owns or has used ("my Norvell tent", "I've \
been using"), items to go with, fit or replace ("compatible with", "pairs well with", "for my"), and \
comparisons ("similar to", "better than", "I came across the", "other than Norvell").
- The request accepts alternatives ("Norvell or a similar brand", "Norvell or other reliable brands"): don't filter.

A hard filter removes every product with a different value, so when the role is unclear, leave the field empty.
- Don't infer values the shopper didn't state: "affordable" is not a price, and "for my daughter" is not \
a color.
- Use the shopper's own words, no narrower than they said them. List several values for a field only when \
the shopper accepts any of them ("red or blue")."""

# v3 is grounded in the catalog: code looks up which catalog values the request names (FilterIndex.mentions) and
# the model only judges each one. It returns no rewritten query.
SYSTEM_TEMPLATE_V3 = """\
You decide which catalog values in a shopper's search request become hard filters.

The catalog: {catalog}

Fields:
{fields}

The message has the request and a numbered list of catalog values whose words appear in it, with how many \
products have each. The list comes from word matching, so most entries are coincidences: ordinary words that \
happen to be a brand, or a value the shopper didn't mean. Give every number exactly one role:
- requirement: the shopper means this value, in this field, and every product that satisfies the request must \
have it. "a Norvell tent", "made by Norvell", "from the Norvell brand", "does Norvell offer a tent" all require \
the brand Norvell.
- context: the shopper means this value but for another item, or accepts alternatives. Items they own or have \
used, items to go with, fit or replace ("compatible with", "pairs well with", "for my"), comparisons ("similar \
to", "better than", "I came across"), and "Norvell or a similar brand" are all context. So is a vehicle, device \
or machine the item is a part or accessory for: "a Norvell tent pole" may come from any maker.
- not_meant: the words mean something else here. Ordinary product words are not_meant even when a brand has that \
name: "a grill cover" does not ask for a brand called Grill Covers.

A requirement removes every product with a different value, so a wrong one hides the right answer. Having no \
requirement at all is common and fine. When unsure, don't answer requirement. If several entries could be the \
same maker under different names (Norvell, Norvell Tents), give them the same role.
For number fields, give a bound only if the request states the number; otherwise null."""

PROMPTS = {"v1": SYSTEM_TEMPLATE_V1, "v2": SYSTEM_TEMPLATE_V2, "v3": SYSTEM_TEMPLATE_V3}
GROUNDED = ("v3",)  # prompts that judge catalog values found by code instead of extracting free text
ROLES = ("requirement", "context", "not_meant")
DEFAULT_PROMPT = "v1"

MATCH_NOTE = {
    "contains": "matches catalog values containing all of its words",
    "prefix": "matches catalog values that start with it",
}


def _field_line(field: FilterField, note: str) -> str:
    kind = "number: min and/or max" if field.kind == "number" else f"text, {MATCH_NOTE[field.match]}"
    return f"- {field.name} ({kind}): {field.description}" + (f" {note}" if note else "")


def system_prompt(
    catalog: str, fields: Sequence[FilterField], notes: Mapping[str, str], prompt: str = DEFAULT_PROMPT
) -> str:
    lines = "\n".join(_field_line(f, notes.get(f.name, "")) for f in fields)
    return PROMPTS[prompt].format(catalog=catalog, fields=lines)


def verdict_schema(fields: Sequence[FilterField]) -> Dict[str, Any]:
    """Output of a grounded prompt: a role for every listed value, and bounds for every number field."""
    numbers = output_schema([f for f in fields if f.kind == "number"])["properties"]["filters"]
    verdict = {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "role": {"type": "string", "enum": list(ROLES)}},
        "required": ["id", "role"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"verdicts": {"type": "array", "items": verdict}, "numbers": numbers},
        "required": ["verdicts", "numbers"],
        "additionalProperties": False,
    }


def grounded_request(text: str, mentions: Sequence[Tuple[str, str, int]]) -> str:
    """mentions: (field, catalog value, number of products)."""
    lines = [f"{i}. {field}: {value} ({count:,} products)" for i, (field, value, count) in enumerate(mentions, start=1)]
    return f"Search request: {text}\n\nCatalog values:\n" + ("\n".join(lines) or "(none)")


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
