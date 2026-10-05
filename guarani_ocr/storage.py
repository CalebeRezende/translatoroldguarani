"""Armazenamento: banco SQLite local + sincronização opcional com Google Drive.

No Drive, cada página vira:
  - uma imagem PNG processada e um .txt com a transcrição, numa pasta;
  - uma linha numa Planilha Google (o "banco de dados" consultável).
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(os.getenv("OCR_DATA_DIR", "data"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    book            TEXT,
    source_name     TEXT NOT NULL,
    page_index      INTEGER NOT NULL,
    engine          TEXT,
    confidence      REAL,
    raw_text        TEXT,
    corrected_text  TEXT,
    modern_text     TEXT,
    lines_json      TEXT,
    image_path      TEXT,
    drive_image_id  TEXT,
    drive_text_id   TEXT,
    created_at      TEXT,
    updated_at      TEXT,
    UNIQUE(source_name, page_index)
);
"""

SHEET_HEADER = ["id", "livro", "arquivo", "página", "motor", "confiança",
                "texto_corrigido", "texto_modernizado", "imagem_drive",
                "texto_drive", "atualizado_em"]


def page_name(row) -> str:
    """Nome curto da página, ex.: "prologo_p1" (arquivo prologo.jpg, 2ª página)."""
    return f"{Path(row['source_name']).stem}_p{row['page_index']}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or DATA_DIR / "ocr.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        (self.path.parent / "pages").mkdir(exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def image_path_for(self, source_name: str, page_index: int) -> Path:
        stem = Path(source_name).stem
        return self.path.parent / "pages" / f"{stem}_p{page_index}.png"

    def upsert(self, *, source_name: str, page_index: int, **fields) -> int:
        """Insere ou atualiza a página (chave: arquivo + índice). Retorna o id."""
        if "lines" in fields:
            fields["lines_json"] = json.dumps(fields.pop("lines"), ensure_ascii=False)
        fields["updated_at"] = _now()
        row = self.conn.execute(
            "SELECT id FROM pages WHERE source_name=? AND page_index=?",
            (source_name, page_index)).fetchone()
        if row:
            sets = ", ".join(f"{k}=?" for k in fields)
            self.conn.execute(f"UPDATE pages SET {sets} WHERE id=?",
                              (*fields.values(), row["id"]))
            page_id = row["id"]
        else:
            fields.update(source_name=source_name, page_index=page_index,
                          created_at=fields["updated_at"])
            cols = ", ".join(fields)
            marks = ", ".join("?" for _ in fields)
            cur = self.conn.execute(f"INSERT INTO pages ({cols}) VALUES ({marks})",
                                    tuple(fields.values()))
            page_id = cur.lastrowid
        self.conn.commit()
        return page_id

    def get(self, page_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM pages WHERE id=?", (page_id,)).fetchone()

    def all(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM pages ORDER BY book, source_name, page_index").fetchall()

    def corrected(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM pages WHERE corrected_text IS NOT NULL "
            "AND corrected_text != ''").fetchall()


class DriveSync:
    """Envia páginas para o Google Drive e registra numa Planilha Google.

    Configuração (variáveis de ambiente):
      GOOGLE_CLIENT_SECRET  caminho do credentials.json (OAuth "App para computador")
      DRIVE_FOLDER_ID       id da pasta no Drive onde salvar imagens e textos
      SHEET_ID              id da Planilha Google usada como banco de dados
    """

    SCOPES = ["https://www.googleapis.com/auth/drive.file",
              "https://www.googleapis.com/auth/spreadsheets"]

    def __init__(self):
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        token = DATA_DIR / "google_token.json"
        creds = None
        if token.exists():
            creds = Credentials.from_authorized_user_file(str(token), self.SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                secret = os.getenv("GOOGLE_CLIENT_SECRET", "credentials.json")
                flow = InstalledAppFlow.from_client_secrets_file(secret, self.SCOPES)
                creds = flow.run_local_server(port=0)
            token.write_text(creds.to_json())
        self.drive = build("drive", "v3", credentials=creds)
        self.sheets = build("sheets", "v4", credentials=creds)
        self.folder_id = os.environ["DRIVE_FOLDER_ID"]
        self.sheet_id = os.getenv("SHEET_ID")

    @staticmethod
    def configured() -> bool:
        return bool(os.getenv("DRIVE_FOLDER_ID"))

    def _put_file(self, name: str, data: bytes, mime: str, file_id: str | None) -> str:
        from googleapiclient.http import MediaInMemoryUpload

        media = MediaInMemoryUpload(data, mimetype=mime)
        if file_id:
            self.drive.files().update(fileId=file_id, media_body=media).execute()
            return file_id
        meta = {"name": name, "parents": [self.folder_id]}
        return self.drive.files().create(body=meta, media_body=media,
                                         fields="id").execute()["id"]

    def _upsert_sheet_row(self, values: list) -> None:
        if not self.sheet_id:
            return
        api = self.sheets.spreadsheets().values()
        col = api.get(spreadsheetId=self.sheet_id, range="A:A").execute().get("values", [])
        if not col:
            api.update(spreadsheetId=self.sheet_id, range="A1",
                       valueInputOption="RAW", body={"values": [SHEET_HEADER]}).execute()
            col = [["id"]]
        ids = [r[0] if r else "" for r in col]
        if str(values[0]) in ids:
            row = ids.index(str(values[0])) + 1
            api.update(spreadsheetId=self.sheet_id, range=f"A{row}",
                       valueInputOption="RAW", body={"values": [values]}).execute()
        else:
            api.append(spreadsheetId=self.sheet_id, range="A1",
                       valueInputOption="RAW", insertDataOption="INSERT_ROWS",
                       body={"values": [values]}).execute()

    def push(self, db: Database, page_id: int) -> None:
        p = db.get(page_id)
        stem = f"{Path(p['source_name']).stem}_p{p['page_index']}"
        img_id = self._put_file(f"{stem}.png", Path(p["image_path"]).read_bytes(),
                                "image/png", p["drive_image_id"])
        text = p["corrected_text"] or p["raw_text"] or ""
        txt_id = self._put_file(f"{stem}.txt", text.encode("utf-8"),
                                "text/plain", p["drive_text_id"])
        db.upsert(source_name=p["source_name"], page_index=p["page_index"],
                  drive_image_id=img_id, drive_text_id=txt_id)
        link = "https://drive.google.com/file/d/{}/view"
        self._upsert_sheet_row([
            p["id"], p["book"] or "", p["source_name"], p["page_index"], p["engine"],
            round(p["confidence"], 1) if p["confidence"] is not None else "",
            text[:49000], (p["modern_text"] or "")[:49000],  # limite de célula: 50k
            link.format(img_id), link.format(txt_id), _now(),
        ])
