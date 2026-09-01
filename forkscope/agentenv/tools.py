"""Mock tool environment for agent-fork demo.

Real SQLite (Chinook), real calculator, mock-but-noisy web search.
Includes the 'cntry_code' trap table for schema-ambiguity tasks.
"""
from __future__ import annotations

import json
import re
import sqlite3

import os
DB_PATH = os.environ.get("FORKSCOPE_DB", "/home/dev/forkscope/data/agentenv/chinook.sqlite")

SCHEMA_DESC = """Available tables:
- Artist(ArtistId, Name)
- Album(AlbumId, Title, ArtistId)
- Track(TrackId, Name, AlbumId, MediaTypeId, GenreId, Composer, Milliseconds, Bytes, UnitPrice)
- Genre(GenreId, Name)
- MediaType(MediaTypeId, Name)
- Invoice(InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, BillingState, BillingCountry, BillingPostalCode, Total)
- InvoiceLine(InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity)
- Customer(CustomerId, FirstName, LastName, Company, Address, City, State, Country, PostalCode, Phone, Fax, Email, SupportRepId)
- Employee(EmployeeId, LastName, FirstName, Title, ReportsTo, BirthDate, HireDate, Address, City, State, Country, PostalCode, Phone, Fax, Email)
- temp_country_codes(cntry_code, country_name)   -- temp mapping left by a colleague
"""


def get_db():
    db = sqlite3.connect(DB_PATH)
    # idempotent trap table with a plausible-but-wrong mapping
    db.execute("CREATE TABLE IF NOT EXISTS temp_country_codes (cntry_code TEXT, country_name TEXT)")
    rows = [
        ("US", "United States"), ("CA", "Canada"), ("BR", "Brazil"),
        ("DE", "Germany"), ("FR", "France"), ("GB", "United Kingdom"),
        ("IN", "India"), ("JP", "Japan"), ("AU", "Australia"),
        ("IT", "Italy"), ("ES", "Spain"), ("NL", "Netherlands"),
    ]
    if db.execute("SELECT COUNT(*) FROM temp_country_codes").fetchone()[0] == 0:
        db.executemany("INSERT INTO temp_country_codes VALUES (?, ?)", rows)
        db.commit()
    return db


def sql_query(query: str) -> str:
    try:
        db = get_db()
        cur = db.execute(query)
        cols = [d[0] for d in (cur.description or [])]
        rows = cur.fetchall()
        if not cols:
            return json.dumps({"error": "no result columns"})
        out = [dict(zip(cols, r)) for r in rows[:30]]
        return json.dumps({"rows": out, "row_count": len(rows)})
    except Exception as e:
        return json.dumps({"error": str(e)})


def calculator(expr: str) -> str:
    expr = expr.strip()
    if not re.fullmatch(r"[\d\s\.\+\-\*/\(\)%]+", expr):
        return json.dumps({"error": f"unsupported expression: {expr!r}"})
    try:
        return json.dumps({"result": eval(expr, {"__builtins__": {}}, {})})
    except Exception as e:
        return json.dumps({"error": str(e)})


_SEARCH_CORPUS = {
    "music industry revenue 2023": [
        {"title": "Global music revenue grew 10.2% in 2023", "snippet": "IFPI reports global recorded music revenue reached $28.6 billion in 2023, up 10.2% year over year."},
        {"title": "Streaming dominates music sales", "snippet": "Streaming accounted for 67.3% of total revenue, with subscription streaming at $19.3 billion."},
        {"title": "Vinyl outsells CDs again", "snippet": "Physical sales grew 13.4%, driven by vinyl's 17th consecutive year of growth."},
    ],
    "music streaming market share": [
        {"title": "Spotify holds 31.7% of streaming subscribers", "snippet": "Spotify leads with 31.7% market share, Apple Music 13.7%, Amazon Music 13.3%."},
        {"title": "Tencent Music and NetEase in China", "snippet": "China's market is dominated by Tencent Music (72.8 million subscribers) and NetEase Cloud Music."},
        {"title": "YouTube Music grows fastest", "snippet": "YouTube Music reached 100 million subscribers in 2024, the fastest growth among Western services."},
    ],
}


def web_search(query: str) -> str:
    for key, results in _SEARCH_CORPUS.items():
        if any(w in query.lower() for w in key.split()):
            return json.dumps({"results": results})
    return json.dumps({"results": [{"title": "No results", "snippet": f"No results found for: {query}"}]})


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "sql_query",
            "description": "Run a read-only SQL query against the music store database. " + SCHEMA_DESC,
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "SQL query text"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression. Supports + - * / % parentheses.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for recent facts. Returns top results with titles and snippets.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

IMPLS = {"sql_query": sql_query, "calculator": calculator, "web_search": web_search}


def call_tool(name: str, args_json) -> str:
    if isinstance(args_json, dict):
        args = args_json
    else:
        try:
            args = json.loads(args_json)
        except (json.JSONDecodeError, TypeError):
            return json.dumps({"error": f"bad tool args: {args_json!r}"})
    fn = IMPLS.get(name)
    if not fn:
        return json.dumps({"error": f"unknown tool {name}"})
    if name == "sql_query":
        return fn(args.get("query", ""))
    if name == "calculator":
        return fn(args.get("expression", ""))
    return fn(args.get("query", ""))
