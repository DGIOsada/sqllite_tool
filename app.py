from __future__ import annotations

import csv
import os
import re
import sqlite3
import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from flask import Flask, render_template, request, redirect, url_for, flash, session,send_file, Response
from slugify import slugify


APP_ROOT = Path(__file__).parent
DATA_DIR = APP_ROOT / "data"
UPLOAD_DIR = APP_ROOT / "uploads"
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")


# -----------------------------
# Helpers
# -----------------------------
SQLITE_TYPES = ["TEXT", "INTEGER", "REAL", "BOOLEAN", "DATE", "DATETIME"]

def safe_sql_identifier(name: str) -> str:
    """
    Convert arbitrary header to safe snake_case ASCII identifier.
    - Allows user override later.
    """
    base = slugify(name, separator="_")  # japanese -> romaji-ish or stripped
    base = re.sub(r"[^a-zA-Z0-9_]", "_", base).strip("_")
    if not base:
        base = "col"
    if re.match(r"^\d", base):
        base = f"c_{base}"
    return base.lower()


def norm_key(s: str) -> str:
    if s is None:
        return ""
    return str(s).replace("\r", "").replace("\n", "").replace("\u3000", " ").strip()


def infer_column_type(samples: List[str]) -> str:
    """
    Very lightweight inference from sample values.
    User can override in UI.
    """
    cleaned = [s.strip() for s in samples if s is not None and str(s).strip() != ""]
    if not cleaned:
        return "TEXT"

    def is_int(x: str) -> bool:
        try:
            int(x)
            return True
        except:
            return False

    def is_float(x: str) -> bool:
        try:
            float(x)
            return True
        except:
            return False

    def is_date(x: str) -> bool:
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
            try:
                datetime.strptime(x, fmt)
                return True
            except:
                pass
        return False

    def is_datetime(x: str) -> bool:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                datetime.strptime(x, fmt)
                return True
            except:
                pass
        return False

    if all(is_int(x) for x in cleaned):
        return "INTEGER"
    if all(is_float(x) for x in cleaned):
        return "REAL"
    if all(x.lower() in ("true", "false", "0", "1", "yes", "no", "y", "n") for x in cleaned):
        return "BOOLEAN"
    if all(is_datetime(x) for x in cleaned):
        return "DATETIME"
    if all(is_date(x) for x in cleaned):
        return "DATE"
    return "TEXT"


def try_coerce_value(val: str, col_type: str) -> tuple[bool, Any, str]:
    """
    returns (ok, converted_value, reason)
    """
    if val is None:
        return True, None, ""
    s = str(val).strip()
    if s == "":
        return True, None, ""

    try:
        if col_type == "INTEGER":
            return True, int(s), ""
        if col_type == "REAL":
            return True, float(s), ""
        if col_type == "BOOLEAN":
            sl = s.lower()
            if sl in ("true", "yes", "y", "1"):
                return True, 1, ""
            if sl in ("false", "no", "n", "0"):
                return True, 0, ""
            return False, None, "booleanに変換できません"
        if col_type == "DATE":
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
                try:
                    return True, datetime.strptime(s, fmt).strftime("%Y-%m-%d"), ""
                except:
                    pass
            return False, None, "dateに変換できません（YYYY-MM-DD等）"
        if col_type == "DATETIME":
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                try:
                    return True, datetime.strptime(s, fmt).strftime("%Y-%m-%d %H:%M:%S"), ""
                except:
                    pass
            return False, None, "datetimeに変換できません"
        # TEXT
        return True, s, ""
    except Exception as e:
        return False, None, str(e)


def coerce_value(val: str, col_type: str) -> Any:
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None

    try:
        if col_type == "INTEGER":
            return int(s)
        if col_type == "REAL":
            return float(s)
        if col_type == "BOOLEAN":
            sl = s.lower()
            if sl in ("true", "yes", "y", "1"):
                return 1
            if sl in ("false", "no", "n", "0"):
                return 0
            return None
        if col_type == "DATE":
            # Normalize to YYYY-MM-DD when possible
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
                try:
                    return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
                except:
                    pass
            return s
        if col_type == "DATETIME":
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                try:
                    return datetime.strptime(s, fmt).strftime("%Y-%m-%d %H:%M:%S")
                except:
                    pass
            return s
    except:
        return None

    return s  # TEXT


def connect_db(db_name: str) -> sqlite3.Connection:
    db_path = DATA_DIR / f"{db_name}.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def is_safe_select_sql(sql: str) -> tuple[bool, str]:
    s = (sql or "").strip()
    if not s:
        return False, "SQLが空です"
    # 複文禁止（; が途中にあるものを拒否）
    if ";" in s.rstrip(";"):
        return False, "複数ステートメントは不可です（;禁止）"
    # 先頭が SELECT or WITH 以外を拒否
    head = re.sub(r"^\s+", "", s, flags=re.DOTALL).lower()
    if not (head.startswith("select") or head.startswith("with")):
        return False, "SELECT / WITH のみ実行可能です"
    # 危険ワード拒否（保険）
    banned = ["insert", "update", "delete", "drop", "alter", "create", "attach", "detach", "pragma", "vacuum", "reindex"]
    low = head.lower()
    if any(re.search(rf"\b{w}\b", low) for w in banned):
        return False, "更新系SQLは禁止です"
    return True, ""


def read_csv_header_and_samples(csv_path: Path, sample_rows: int = 50) -> Tuple[List[str], List[List[str]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for i, row in enumerate(reader):
            rows.append(row)
            if i + 1 >= sample_rows:
                break
    return header, rows


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def index():
    return render_template("index.html")


@app.post("/start")
def start():
    db_name = request.form.get("db_name", "").strip()
    table_name = request.form.get("table_name", "").strip()
    if not db_name:
        flash("DB名を入力してください", "error")
        return redirect(url_for("index"))
    if not table_name:
        flash("テーブル名を入力してください", "error")
        return redirect(url_for("index"))

    # Basic sanitize for filenames + sqlite identifiers
    db_name_safe = safe_sql_identifier(db_name)
    table_name_safe = safe_sql_identifier(table_name)

    file = request.files.get("csv_file")
    if not file or file.filename == "":
        flash("CSVファイルを選択してください", "error")
        return redirect(url_for("index"))

    csv_path = UPLOAD_DIR / f"{db_name_safe}__{table_name_safe}__upload.csv"
    file.save(csv_path)

    header, sample_rows = read_csv_header_and_samples(csv_path)
    # Build column proposals
    # collect per-col samples
    per_col_samples: List[List[str]] = [[] for _ in header]
    for r in sample_rows:
        for i in range(len(header)):
            per_col_samples[i].append(r[i] if i < len(r) else "")

    cols = []
    used = set()
    for i, h in enumerate(header):
        proposed = safe_sql_identifier(h)
        # de-dup
        base = proposed
        n = 2
        while proposed in used:
            proposed = f"{base}_{n}"
            n += 1
        used.add(proposed)

        cols.append({
            "display_name": norm_key(h),
            "column_name": proposed,
            "type": infer_column_type(per_col_samples[i]),
            "not_null": False,
            "unique": False,
            "pk": False,
        })

    session["db_name"] = db_name_safe
    session["table_name"] = table_name_safe
    session["csv_path"] = str(csv_path)
    session["cols"] = cols

    return redirect(url_for("schema"))


@app.get("/schema")
def schema():
    cols = session.get("cols")
    if not cols:
        return redirect(url_for("index"))
    return render_template(
        "schema.html",
        db_name=session["db_name"],
        table_name=session["table_name"],
        cols=cols,
        sqlite_types=SQLITE_TYPES,
    )


@app.post("/create")
def create():
    db_name = session.get("db_name")
    table_name = session.get("table_name")
    csv_path = session.get("csv_path")
    if not (db_name and table_name and csv_path):
        return redirect(url_for("index"))

    # read edited schema from form
    cols: List[Dict[str, Any]] = []
    idx = 0
    while True:
        dn = request.form.get(f"display_name_{idx}")
        if dn is None:
            break
        cn = request.form.get(f"column_name_{idx}", "").strip()
        ct = request.form.get(f"type_{idx}", "TEXT").strip().upper()
        pk = request.form.get(f"pk_{idx}") == "on"
        nn = request.form.get(f"not_null_{idx}") == "on"
        uq = request.form.get(f"unique_{idx}") == "on"

        cn = safe_sql_identifier(cn) if cn else safe_sql_identifier(dn)
        if ct not in SQLITE_TYPES:
            ct = "TEXT"

        cols.append({
            "display_name": dn,
            "column_name": cn,
            "type": ct,
            "pk": pk,
            "not_null": nn,
            "unique": uq,
        })
        idx += 1

    # ensure at most one PK (SQLite supports composite, but keep simple)
    pk_cols = [c for c in cols if c["pk"]]
    if len(pk_cols) > 1:
        flash("PK（主キー）は1列だけにしてください（簡易版の制約）", "error")
        return redirect(url_for("schema"))

    # create table
    conn = connect_db(db_name)
    try:
        col_defs = []
        for c in cols:
            parts = [f'"{c["column_name"]}"', c["type"]]
            if c["pk"]:
                parts.append("PRIMARY KEY")
            if c["not_null"]:
                parts.append("NOT NULL")
            if c["unique"]:
                parts.append("UNIQUE")
            col_defs.append(" ".join(parts))

        ddl = f'CREATE TABLE IF NOT EXISTS "{table_name}" (\n  ' + ",\n  ".join(col_defs) + "\n);"
        conn.execute(ddl)

        # --- Validate CSV before import ---
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)

            bad_cells = []
            all_rows = []

            for row_idx, row in enumerate(reader, start=2):  # header=1行目 → データは2行目から
                all_rows.append(row)

                for c in cols:
                    ok, _, reason = try_coerce_value(row.get(c["display_name"], ""), c["type"])
                    if not ok:
                        bad_cells.append({
                            "row": row_idx,
                            "col_display": c["display_name"],
                            "col_db": c["column_name"],
                            "type": c["type"],
                            "value": row.get(c["display_name"], ""),
                            "reason": reason,
                        })

        # NGがあれば validate へ（INSERTはしない）
        if bad_cells:
            session["bad_cells"] = bad_cells[:300]  # 表示上限
            tsv_lines = ["row\tcolumn\tvalue"]
            for b in session["bad_cells"]:
                tsv_lines.append(f'{b["row"]}\t{b["col_display"]}\t{b["value"]}')
            session["fix_tsv"] = "\n".join(tsv_lines)
            return redirect(url_for("validate"))

        # --- Import CSV (only if valid) ---
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            insert_cols = [c["column_name"] for c in cols]
            placeholders = ",".join(["?"] * len(insert_cols))
            sql = f'INSERT INTO "{table_name}" (' + ",".join([f'"{c}"' for c in insert_cols]) + f") VALUES ({placeholders})"

            rows_inserted = 0
            for row in reader:
                # ★ CSVのヘッダーキーを正規化（改行・CR除去）
                row_n = {norm_key(k): v for k, v in row.items()}

                values = []
                for c in cols:
                    # ★ display_name 側も正規化して参照
                    display = norm_key(c["display_name"])

                    ok, v, _ = try_coerce_value(row_n.get(display, ""), c["type"])
                    values.append(v if ok else None)

                conn.execute(sql, values)
                rows_inserted += 1





        conn.commit()
    finally:
        conn.close()

    # cleanup session
    session.pop("cols", None)

    return render_template(
        "done.html",
        db_name=db_name,
        table_name=table_name,
        rows="(CSV読み込み完了)",
    )

@app.get("/validate")
def validate():
    bad_cells = session.get("bad_cells", [])
    fix_tsv = session.get("fix_tsv", "")
    if not bad_cells:
        return redirect(url_for("schema"))
    return render_template("validate.html",
                           db_name=session["db_name"],
                           table_name=session["table_name"],
                           bad_cells=bad_cells,
                           fix_tsv=fix_tsv)


@app.get("/sql")
def sql_console():
    dbs = sorted([p.stem for p in DATA_DIR.glob("*.db")])
    selected_db = request.args.get("db") or (session.get("db_name") if session.get("db_name") in dbs else (dbs[0] if dbs else ""))

    tables = []
    if selected_db:
        conn = connect_db(selected_db)
        try:
            tables = [r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )]
        finally:
            conn.close()

    # ★ resultはsessionから取らない
    result = None

    # ★ デフォルトは必要ならここ（でも上書きされない）
    sql_text = session.get("sql_text") or "SELECT * FROM jinji_master LIMIT 100"

    return render_template("sql.html",
                           dbs=dbs,
                           selected_db=selected_db,
                           tables=tables,
                           sql_text=sql_text,
                           result=result)



@app.get("/sql/export")
def sql_export():
    fmt = (request.args.get("fmt") or "csv").lower()
    db = session.get("sql_last_db")
    sql_text = session.get("sql_last_sql")

    if not db or not sql_text:
        flash("エクスポート対象がありません。先にSQLを実行してください。", "error")
        return redirect(url_for("sql_console"))

    ok, msg = is_safe_select_sql(sql_text)
    if not ok:
        flash(msg, "error")
        return redirect(url_for("sql_console", db=db))

    conn = connect_db(db)
    try:
        cur = conn.execute(sql_text)
        cols = [d[0] for d in cur.description] if cur.description else []

        import io
        output = io.StringIO()

        if fmt == "tsv":
            writer = csv.writer(output, delimiter="\t", lineterminator="\n")
            filename = "result.tsv"
            mimetype = "text/tab-separated-values"
        else:
            writer = csv.writer(output, lineterminator="\n")
            filename = "result.csv"
            mimetype = "text/csv"

        writer.writerow(cols)

        # ★ここがポイント：プレビューじゃなく、クエリ結果を最後まで書き出す
        for row in cur:
            writer.writerow(list(row))

        data = output.getvalue().encode("utf-8-sig")  # Excel配慮
        return Response(
            data,
            mimetype=mimetype,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    finally:
        conn.close()


@app.post("/sql/table_action")
def table_action():
    db = request.form.get("db")
    table = request.form.get("table")
    action = request.form.get("action")

    if not db or not table:
        flash("DBまたはテーブルが指定されていません", "error")
        return redirect(url_for("sql_console"))

    # テーブル名の最低限バリデーション
    if not re.match(r"^[A-Za-z0-9_]+$", table):
        flash("不正なテーブル名です", "error")
        return redirect(url_for("sql_console", db=db))

    conn = connect_db(db)
    try:
        if action == "truncate":
            conn.execute(f'DELETE FROM "{table}"')
            conn.commit()
            flash(f"テーブル {table} を空にしました", "success")

        elif action == "drop":
            if request.form.get("confirm_drop") != "yes":
                flash("削除確認にチェックしてください", "error")
            else:
                conn.execute(f'DROP TABLE "{table}"')
                conn.commit()
                flash(f"テーブル {table} を削除しました", "success")

        else:
            flash("不明な操作です", "error")

    except Exception as e:
        flash(f"操作エラー: {e}", "error")
    finally:
        conn.close()

    return redirect(url_for("sql_console", db=db))


@app.post("/apply_fixes")
def apply_fixes():
    """
    fix_tsv: 行番号・列名・値 を受け取ってCSV内容（session内のall_rowsではなく再読み）に反映する簡易版
    """
    csv_path = Path(session["csv_path"])
    fixes_text = request.form.get("fix_tsv", "")

    # fixes parse (TSV)
    fixes = []
    for i, line in enumerate(fixes_text.splitlines()):
        if i == 0:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        fixes.append((int(parts[0]), parts[1], parts[2]))

    # CSV再読込→修正→一時CSVとして上書き保存（簡易）
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    # row番号は “CSV上の行番号” なので、rows index へ変換（row 2 => index 0）
    for row_no, col_name, new_val in fixes:
        idx = row_no - 2
        if 0 <= idx < len(rows) and col_name in rows[idx]:
            rows[idx][col_name] = new_val

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 修正後は create をもう一度（簡易）— schemaへ戻して実行でもOK
    return redirect(url_for("schema"))


@app.post("/sql/run")
def sql_run():
    db = request.form.get("db", "").strip()
    sql_text = request.form.get("sql_text", "")

    # ★ 全角スペース除去＋trim（地味に効く）
    sql_text = sql_text.replace("\u3000", " ").strip()

    # ★ SQLは保存（ここだけsession使う）
    session["sql_text"] = sql_text
    session["sql_last_db"] = db
    session["sql_last_sql"] = sql_text  # export用

    dbs = sorted([p.stem for p in DATA_DIR.glob("*.db")])
    selected_db = db if db in dbs else (dbs[0] if dbs else "")

    tables = []
    if selected_db:
        conn = connect_db(selected_db)
        try:
            tables = [r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )]
        finally:
            conn.close()

    if not selected_db:
        flash("DBが見つかりません", "error")
        return render_template("sql.html", dbs=dbs, selected_db="", tables=[], sql_text=sql_text, result=None)

    ok, msg = is_safe_select_sql(sql_text)
    if not ok:
        flash(msg, "error")
        return render_template("sql.html", dbs=dbs, selected_db=selected_db, tables=tables, sql_text=sql_text, result=None)

    conn = connect_db(selected_db)
    try:
        cur = conn.execute(sql_text)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(200)  # プレビューだけ200
        result = {"cols": cols, "rows": [list(r) for r in rows], "count": len(rows)}
        return render_template("sql.html", dbs=dbs, selected_db=selected_db, tables=tables, sql_text=sql_text, result=result)
    except Exception as e:
        flash(f"SQL実行エラー: {e}", "error")
        return render_template("sql.html", dbs=dbs, selected_db=selected_db, tables=tables, sql_text=sql_text, result=None)
    finally:
        conn.close()



if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)

