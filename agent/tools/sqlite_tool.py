# agent/tools/sqlite_tool.py
import sqlite3
from typing import Tuple, Any, Dict

class SQLiteTool:
    def __init__(self, path: str):
        self.path = path

    def connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def run(self, sql: str, params: Tuple=()):
        conn = self.connect()
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            if sql.strip().lower().startswith("select"):
                rows = cur.fetchall()
                columns = [c[0] for c in cur.description] if cur.description else []
                return {"columns": columns, "rows": [dict(r) for r in rows], "error": None}
            else:
                conn.commit()
                return {"columns": [], "rows": [], "error": None}
        except Exception as e:
            return {"columns": [], "rows": [], "error": str(e)}
        finally:
            conn.close()

    def schema(self):
        # return list of tables and CREATE statements
        conn = self.connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT name, type, sql FROM sqlite_master WHERE type IN ('table','view') ORDER BY name;")
            rows = cur.fetchall()
            return {"rows": [dict(r) for r in rows], "error": None}
        except Exception as e:
            return {"rows": [], "error": str(e)}
        finally:
            conn.close()
