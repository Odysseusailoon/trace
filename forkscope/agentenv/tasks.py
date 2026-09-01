"""10 agent tasks over the Chinook DB. Each has an executable gold answer."""
from __future__ import annotations

import json
import os
import sqlite3

DB = os.environ.get("FORKSCOPE_DB", "/home/dev/forkscope/data/agentenv/chinook.sqlite")


def q(sql: str):
    db = sqlite3.connect(DB)
    return db.execute(sql).fetchall()


def gold(task_id: str):
    """Executable gold answers, computed directly."""
    if task_id == "t1_top_genre_share":
        total = q("SELECT COUNT(*) FROM Track")[0][0]
        top = q("SELECT g.Name, COUNT(*) c FROM Track t JOIN Genre g USING(GenreId) GROUP BY g.Name ORDER BY c DESC LIMIT 1")[0]
        return {"answer": round(top[1] / total, 4), "genre": top[0], "count": top[1], "total": total}
    if task_id == "t2_country_trap":
        # correct: use Customer.Country directly
        rows = q("SELECT Country, COUNT(*) c FROM Customer GROUP BY Country ORDER BY c DESC LIMIT 3")
        return {"top3": rows}
    if task_id == "t3_invoice_avg_top":
        rows = q("SELECT BillingCountry, AVG(Total) a FROM Invoice GROUP BY BillingCountry ORDER BY a DESC LIMIT 1")[0]
        return {"country": rows[0], "avg": round(rows[1], 2)}
    if task_id == "t4_artist_album_ratio":
        r = q("SELECT COUNT(DISTINCT ArtistId) FROM Album")[0][0], q("SELECT COUNT(*) FROM Album")[0][0]
        return {"ratio": round(r[1] / r[0], 3)}
    if task_id == "t5_usa_rock_revenue":
        r = q("""SELECT ROUND(SUM(il.UnitPrice*il.Quantity),2) FROM InvoiceLine il
                 JOIN Invoice i USING(InvoiceId) JOIN Track t USING(TrackId) JOIN Genre g USING(GenreId)
                 WHERE i.BillingCountry='USA' AND g.Name='Rock'""")[0][0]
        return {"revenue": r}
    if task_id == "t6_employee_span":
        r = q("""SELECT e.LastName, COUNT(s.EmployeeId) FROM Employee e
                 LEFT JOIN Employee s ON s.ReportsTo = e.EmployeeId
                 GROUP BY e.EmployeeId ORDER BY 2 DESC LIMIT 1""")[0]
        return {"name": r[0], "reports": r[1]}
    if task_id == "t7_avg_track_len_min":
        ms = q("SELECT AVG(Milliseconds) FROM Track")[0][0]
        return {"minutes": round(ms / 60000, 2)}
    if task_id == "t8_second_city_customers":
        r = q("SELECT City, COUNT(*) c FROM Customer GROUP BY City ORDER BY c DESC, City LIMIT 2")
        return {"second": r[1]}
    if task_id == "t9_rock_vs_metal_diff":
        r = q("""SELECT g.Name, COUNT(*) FROM Track t JOIN Genre g USING(GenreId)
                 WHERE g.Name IN ('Rock','Metal') GROUP BY g.Name""")
        d = dict(r)
        return {"diff": abs(d.get("Rock", 0) - d.get("Metal", 0)), "counts": d}
    if task_id == "t10_search_plus_calc":
        return {"pct_of_28_6b": round(19.3 / 28.6 * 100, 1)}


TASKS = [
    {"id": "t1_top_genre_share", "kind": "multi-step+calc",
     "prompt": "What share (as a percentage, 1 decimal) of all tracks in the music store belongs to the most common genre? Use the database and calculator."},
    {"id": "t2_country_trap", "kind": "schema-ambiguity",
     "prompt": "Which 3 countries have the most customers in the store, and how many customers does each have? (There may be helper tables in the database.)"},
    {"id": "t3_invoice_avg_top", "kind": "multi-step",
     "prompt": "Which billing country has the highest average invoice total, and what is that average (2 decimals)?"},
    {"id": "t4_artist_album_ratio", "kind": "multi-step+calc",
     "prompt": "On average, how many albums per artist are in the store? Give the ratio to 3 decimals."},
    {"id": "t5_usa_rock_revenue", "kind": "multi-step",
     "prompt": "How much revenue did Rock tracks generate from invoices billed to the USA? Give the exact total (2 decimals)."},
    {"id": "t6_employee_span", "kind": "multi-step",
     "prompt": "Which employee has the most direct reports, and how many?"},
    {"id": "t7_avg_track_len_min", "kind": "calc",
     "prompt": "What is the average track length in minutes (2 decimals)?"},
    {"id": "t8_second_city_customers", "kind": "ranking-off-by-one",
     "prompt": "Which city has the SECOND most customers, and how many? (Order by count descending, then city name ascending.)"},
    {"id": "t9_rock_vs_metal_diff", "kind": "multi-step+calc",
     "prompt": "How many more tracks are in the Rock genre than in the Metal genre?"},
    {"id": "t10_search_plus_calc", "kind": "web+calc",
     "prompt": "Using web search: subscription streaming revenue in 2023 as a percentage of total global recorded music revenue. Give 1 decimal."},
]


def tasks_with_gold():
    return [{**t, "gold": gold(t["id"])} for t in TASKS]


if __name__ == "__main__":
    for t in tasks_with_gold():
        print(t["id"], "->", t["gold"])
