# SQLite Data Analyze Tool

CSV / TSV / Spreadsheet 由来のデータを取り込み、  
SQLite + Flask で **SQL実行・JOIN・CSV/TSVエクスポート**ができる軽量ツールです。

---

## Features
- CSVアップロード → SQLiteテーブル作成
- カラム名・データ型のGUI定義
- 型不一致データの検出・修正
- SELECT専用SQLコンソール
- JOIN対応
- 結果を CSV / TSV でエクスポート
- テーブルの中身削除 / テーブル削除（安全ガード付き）

---

## Directory Structure
```text
.
├── app.py
├── templates/
│   ├── index.html
│   ├── schema.html
│   ├── validate.html
│   └── sql.html
├── data/              # SQLite DB (gitignore)
├── requirements.txt
├── .gitignore
└── README.md


## Setup

### 1. Python仮想環境
```bash
python3 -m venv .venv
source .venv/bin/activate

## Setup

### 1. Python仮想環境
```bash
python3 -m venv .venv
source .venv/bin/activate

python app.py

http://127.0.0.1:5000



SELECT *
FROM jinji_master jm
LEFT JOIN attendance a
  ON TRIM(CAST(a.stuff_id AS TEXT)) = TRIM(jm.stuff_id)
LEFT JOIN floor_seat fs
  ON TRIM(jm.seat_no) = TRIM(fs.seat_id)


---

## コミット手順（このまま実行してください）

```bash
git status
git add README.md
git commit -m "Update README with setup and SQL example"
git push

