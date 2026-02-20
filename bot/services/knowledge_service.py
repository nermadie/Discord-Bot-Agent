import os
import re
import json
import ast
import asyncio
import sqlite3
from datetime import datetime, time, timedelta
from urllib.parse import quote_plus

import pytz
import aiohttp
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request

from bot.config.settings import *
from bot.state.runtime import *
from tools.constants import WEATHER_DEFAULT_LOCATION, WEATHER_FORECAST_MAX_DAYS


SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/tasks",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
]

VIETNAM_HOLIDAYS = {
    "01-01": "Tết Dương lịch",
    "02-14": "Valentine",
    "03-08": "Quốc tế Phụ nữ",
    "04-30": "Giải phóng miền Nam",
    "05-01": "Quốc tế Lao động",
    "06-01": "Quốc tế Thiếu nhi",
    "09-02": "Quốc khánh Việt Nam",
    "10-20": "Ngày Phụ nữ Việt Nam",
    "11-20": "Ngày Nhà giáo Việt Nam",
    "12-24": "Giáng sinh",
    "12-25": "Giáng sinh",
}

SPECIAL_COUNTDOWNS = {
    "tet2026": {
        "name": "Tết Nguyên Đán 2026",
        "datetime": "2026-01-29 00:00:00",
        "emoji": "🧧",
        "milestones": [3600, 1800, 900, 600, 300, 60, 30, 10, 5, 4, 3, 2, 1, 0],
    },
    "newyear": {
        "name": "Năm Mới 2026",
        "datetime": "2026-01-01 00:00:00",
        "emoji": "🎆",
        "milestones": [3600, 1800, 900, 60, 30, 10, 5, 4, 3, 2, 1, 0],
    },
}

LUNAR_TET_DATES = {
    2025: (1, 29),
    2026: (2, 17),
    2027: (2, 6),
    2028: (1, 26),
    2029: (2, 13),
    2030: (2, 3),
}


class KnowledgeBot:
    """Facade service for AI, calendar/tasks, weather, countdown, and study features."""

    def __init__(self):
        """Initialize timezone and lazy Google service clients."""
        self.timezone = pytz.timezone("Asia/Ho_Chi_Minh")
        self._calendar_service = None
        self._tasks_service = None
        self._gmail_service = None
        self._mail_llm_rotate_index = 0

    def _default_countdown_milestones(self):
        """Return default milestone seconds for countdown notifications."""
        minute_milestones = [300, 240, 180, 120]
        second_milestones = list(range(60, -1, -1))
        return minute_milestones + second_milestones

    async def _call_ai_with_fallback(
        self,
        messages,
        primary_model,
        fallback_models,
        temperature=0.1,
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout_seconds=None,
    ):
        """Call AI endpoint with model fallback chain and normalized response."""
        url = "https://models.github.ai/inference/chat/completions"
        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Content-Type": "application/json",
        }

        models = [primary_model] + [m for m in fallback_models if m != primary_model]
        errors = []

        effective_timeout = (
            timeout_seconds
            if timeout_seconds and timeout_seconds > 0
            else AI_REQUEST_TIMEOUT_SECONDS
        )
        client_timeout = aiohttp.ClientTimeout(total=effective_timeout)

        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            for model in models:
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }
                if max_tokens:
                    payload["max_tokens"] = max_tokens

                try:
                    async with session.post(
                        url, headers=headers, json=payload
                    ) as response:
                        raw_text = await response.text()
                        if raw_text.strip():
                            try:
                                data = json.loads(raw_text)
                            except Exception:
                                data = raw_text
                        else:
                            data = {}

                        if response.status == 200:
                            if isinstance(data, dict) and data.get("choices"):
                                content = self._normalize_model_content(
                                    data["choices"][0]["message"].get("content", "")
                                )
                                return {
                                    "ok": True,
                                    "content": content,
                                    "model": model,
                                }

                            if isinstance(data, list):
                                return {
                                    "ok": True,
                                    "content": data,
                                    "model": model,
                                }

                            if isinstance(data, dict):
                                generic_content = data.get("content") or data.get(
                                    "message"
                                )
                                if generic_content:
                                    return {
                                        "ok": True,
                                        "content": self._normalize_model_content(
                                            generic_content
                                        ),
                                        "model": model,
                                    }

                            if isinstance(data, str) and data.strip():
                                return {
                                    "ok": True,
                                    "content": data.strip(),
                                    "model": model,
                                }

                        errors.append(
                            f"{model}: HTTP {response.status} - {str(data)[:800]}"
                        )
                except asyncio.TimeoutError:
                    errors.append(f"{model}: request timeout sau {effective_timeout}s")
                except Exception as e:
                    errors.append(f"{model}: {str(e)}")

        return {
            "ok": False,
            "error": (
                "Hệ thống AI phản hồi quá chậm hoặc lỗi endpoint. "
                + " | ".join(errors[:3])
            ),
            "model": None,
            "content": None,
        }

    def _normalize_model_content(self, content):
        """Normalize provider content payloads into plain text."""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and item.get("text"):
                        parts.append(str(item.get("text")))
                    elif item.get("text"):
                        parts.append(str(item.get("text")))
                    elif item.get("content"):
                        parts.append(str(item.get("content")))
                else:
                    parts.append(str(item))
            return "\n".join([p for p in parts if p]).strip()

        if isinstance(content, dict):
            if content.get("text"):
                return str(content.get("text"))
            if content.get("content"):
                return str(content.get("content"))

        return str(content or "")

    def _extract_json_block(self, text):
        """Extract and parse best-effort JSON object from model output."""
        if not text:
            return None

        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

        try:
            return json.loads(text)
        except Exception:
            return None

    def _strip_think_block(self, text):
        """Remove hidden reasoning blocks from model output."""
        if not text:
            return ""
        cleaned = re.sub(
            r"<think>.*?</think>", "", str(text), flags=re.DOTALL | re.IGNORECASE
        )
        return cleaned.strip()

    def _extract_visible_reasoning_message(self, content):
        """Extract user-visible assistant reasoning text from mixed payload formats."""
        if content is None:
            return ""

        parsed = None
        if isinstance(content, list):
            parsed = content
        elif isinstance(content, str):
            stripped = content.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = json.loads(stripped)
                except Exception:
                    parsed = None

        if isinstance(parsed, list):
            assistant_messages = [
                str(item.get("message", "")).strip()
                for item in parsed
                if isinstance(item, dict) and item.get("role") == "assistant"
            ]
            if assistant_messages:
                return self._strip_think_block(assistant_messages[-1])

        return self._strip_think_block(content)

    async def _extract_single_image_information(
        self,
        image_url,
        user_prompt="",
        username="User",
        image_index=1,
        total_images=1,
    ):
        """Extract structured text information from a single image URL."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý trích xuất thông tin từ ảnh. "
                    "Mô tả ngắn gọn nội dung chính, chữ trong ảnh, số liệu quan trọng, "
                    "và các điểm cần chú ý. Trả lời tiếng Việt."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Người dùng: {username}\n"
                            f"Ảnh {image_index}/{total_images}.\n"
                            f"Yêu cầu: {user_prompt or 'Trích xuất thông tin quan trọng từ ảnh này.'}"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]

        return await self._call_ai_with_fallback(
            messages,
            VISION_MODEL_PRIMARY,
            VISION_MODEL_FALLBACKS,
            temperature=0.1,
            max_tokens=MAX_OUTPUT_TOKENS,
        )

    async def _extract_images_information(
        self, image_urls, user_prompt="", username="User"
    ):
        """Extract information from multiple images and return per-image results."""
        extracted_items = []
        models_used = []

        for idx, image_url in enumerate(image_urls, start=1):
            vision_result = await self._extract_single_image_information(
                image_url=image_url,
                user_prompt=user_prompt,
                username=username,
                image_index=idx,
                total_images=len(image_urls),
            )

            if vision_result.get("ok"):
                extracted_text = (vision_result.get("content") or "").strip()
                extracted_items.append(
                    {
                        "index": idx,
                        "url": image_url,
                        "text": extracted_text[:2000],
                        "model": vision_result.get("model"),
                        "ok": True,
                    }
                )
                if vision_result.get("model"):
                    models_used.append(vision_result.get("model"))
            else:
                extracted_items.append(
                    {
                        "index": idx,
                        "url": image_url,
                        "text": "",
                        "model": None,
                        "ok": False,
                        "error": vision_result.get("error", "Unknown error"),
                    }
                )

        deduped_models = []
        seen = set()
        for model in models_used:
            if model not in seen:
                deduped_models.append(model)
                seen.add(model)

        return {
            "items": extracted_items,
            "models": deduped_models,
        }

    async def chat(
        self, user_prompt, username="User", image_urls=None, prior_context=""
    ):
        """Generate chat response with optional image extraction and prior context."""
        image_urls = image_urls or []
        image_context = ""
        image_extractions = []
        vision_models = []

        if image_urls:
            extraction_result = await self._extract_images_information(
                image_urls=image_urls,
                user_prompt=user_prompt,
                username=username,
            )
            image_extractions = extraction_result.get("items", [])
            vision_models = extraction_result.get("models", [])

            context_lines = []
            for item in image_extractions:
                if item.get("ok") and item.get("text"):
                    context_lines.append(f"[Ảnh {item['index']}] {item['text']}")
                else:
                    context_lines.append(
                        f"[Ảnh {item['index']}] Không trích xuất được nội dung"
                    )

            image_context = (
                "\n\nThông tin trích xuất từ ảnh đính kèm (xử lý từng ảnh):\n"
                + "\n".join([f"- {line}" for line in context_lines[:6]])
            )

        prior_context_text = ""
        if prior_context:
            prior_context_text = (
                "\n\nNgữ cảnh chat trước đó (do người dùng chọn):\n"
                f"{prior_context[:6000]}"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý AI thân thiện, trả lời rõ ràng, súc tích, tiếng Việt tự nhiên. "
                    "Nếu có URL ảnh thì dùng như ngữ cảnh tham chiếu khi trả lời."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Người dùng: {username}\n\n"
                    f"Câu hỏi: {user_prompt or 'Hãy phân tích ảnh đính kèm nếu có.'}"
                    f"{image_context}"
                    f"{prior_context_text}"
                ),
            },
        ]

        ai_result = await self._call_ai_with_fallback(
            messages,
            CHAT_MODEL_PRIMARY,
            CHAT_MODEL_FALLBACKS,
            temperature=0.1,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        ai_result["vision_models"] = vision_models
        ai_result["image_extractions"] = image_extractions
        return ai_result

    async def reasoning(
        self,
        user_prompt,
        username="User",
    ):
        """Run reasoning-oriented answer flow using dedicated reasoning models."""
        system_prompt = (
            "Bạn là trợ lý reasoning. Trả lời bằng tiếng Việt rõ ràng, dễ đọc, không dùng LaTeX. "
            "Khi có công thức, hãy viết dạng văn bản thường. "
            "Định dạng theo từng dòng ngắn, nhấn mạnh ý chính bằng chữ đậm Markdown."
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (f"Người dùng: {username}\n\n" f"Bài toán: {user_prompt}"),
            },
        ]

        ai_result = await self._call_ai_with_fallback(
            messages,
            REASONING_MODEL_PRIMARY,
            REASONING_MODEL_FALLBACKS,
            temperature=0.1,
            max_tokens=MAX_OUTPUT_TOKENS,
            timeout_seconds=REASONING_REQUEST_TIMEOUT_SECONDS,
        )

        if not ai_result["ok"]:
            return ai_result

        visible = self._extract_visible_reasoning_message(ai_result["content"])
        return {
            "ok": True,
            "content": visible,
            "raw_content": ai_result["content"],
            "model": ai_result["model"],
        }

    def _get_credentials(self):
        """Load and refresh OAuth credentials from local token store."""
        creds = None
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                with open("token.json", "w") as token:
                    token.write(creds.to_json())
            else:
                return None
        return creds

    def _get_calendar_service(self):
        """Get or lazily initialize Google Calendar API client."""
        if not self._calendar_service:
            creds = self._get_credentials()
            if not creds:
                return None
            self._calendar_service = build("calendar", "v3", credentials=creds)
        return self._calendar_service

    def _get_tasks_service(self):
        """Get or lazily initialize Google Tasks API client."""
        if not self._tasks_service:
            creds = self._get_credentials()
            if not creds:
                return None
            self._tasks_service = build("tasks", "v1", credentials=creds)
        return self._tasks_service

    def _get_gmail_service(self):
        """Get or lazily initialize Gmail API client."""
        if not self._gmail_service:
            creds = self._get_credentials()
            if not creds:
                return None
            self._gmail_service = build("gmail", "v1", credentials=creds)
        return self._gmail_service

    def _gmail_db_path(self):
        """Return sqlite path used to persist daily Gmail digests."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        metrics_dir = os.path.join(base_dir, STUDY_METRICS_DIR)
        os.makedirs(metrics_dir, exist_ok=True)
        return os.path.join(metrics_dir, "gmail_digest.sqlite3")

    def _ensure_gmail_digest_tables(self):
        """Create Gmail digest persistence tables if missing."""
        db_path = self._gmail_db_path()
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gmail_digests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                digest_date TEXT NOT NULL,
                period TEXT NOT NULL,
                unread_count INTEGER NOT NULL DEFAULT 0,
                unread_titles_json TEXT NOT NULL DEFAULT '[]',
                sent_summary TEXT NOT NULL DEFAULT '',
                sent_applications_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                UNIQUE(digest_date, period)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gmail_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_date TEXT NOT NULL,
                period TEXT NOT NULL,
                important_json TEXT NOT NULL DEFAULT '[]',
                todo_json TEXT NOT NULL DEFAULT '[]',
                unread_json TEXT NOT NULL DEFAULT '[]',
                sent_json TEXT NOT NULL DEFAULT '[]',
                sent_apps_json TEXT NOT NULL DEFAULT '[]',
                read_json TEXT NOT NULL DEFAULT '[]',
                key_info_json TEXT NOT NULL DEFAULT '[]',
                llm_models_json TEXT NOT NULL DEFAULT '{}',
                summary_text TEXT NOT NULL DEFAULT '',
                max_internal_ts INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gmail_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        report_columns = {
            str(row[1]).strip().lower()
            for row in conn.execute("PRAGMA table_info(gmail_reports)").fetchall()
        }
        if "key_info_json" not in report_columns:
            conn.execute(
                "ALTER TABLE gmail_reports ADD COLUMN key_info_json TEXT NOT NULL DEFAULT '[]'"
            )
        if "llm_models_json" not in report_columns:
            conn.execute(
                "ALTER TABLE gmail_reports ADD COLUMN llm_models_json TEXT NOT NULL DEFAULT '{}'"
            )

        conn.commit()
        conn.close()
        return db_path

    def _get_rotated_mail_llm_models(self):
        """Return mail LLM model list ordered by round-robin pointer."""
        base_models = [
            str(x).strip() for x in (MAIL_LLM_MODELS or []) if str(x).strip()
        ]
        if not base_models:
            base_models = ["gemini-3.0-flash", "gemini-2.5-pro"]

        start_index = self._mail_llm_rotate_index % len(base_models)
        ordered = [
            base_models[(start_index + offset) % len(base_models)]
            for offset in range(len(base_models))
        ]
        self._mail_llm_rotate_index = (start_index + 1) % len(base_models)
        return ordered

    async def _call_google_ai_studio_mail(self, system_prompt, user_prompt):
        """Call Google AI Studio for mail-related summarization tasks."""
        api_key = str(GOOGLE_AI_STUDIO_API_KEY or "").strip()
        if not api_key:
            return {
                "ok": False,
                "error": "⚠️ Thiếu GOOGLE_AI_STUDIO_API_KEY trong .env.",
                "model": None,
                "content": None,
            }

        models = self._get_rotated_mail_llm_models()
        errors = []
        timeout = aiohttp.ClientTimeout(total=AI_REQUEST_TIMEOUT_SECONDS)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            for model in models:
                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/models/{quote_plus(model)}:generateContent"
                    f"?key={api_key}"
                )
                payload = {
                    "systemInstruction": {
                        "parts": [{"text": str(system_prompt or "").strip()}]
                    },
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": str(user_prompt or "").strip()}],
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.1,
                    },
                }
                if MAX_OUTPUT_TOKENS:
                    payload["generationConfig"]["maxOutputTokens"] = int(
                        MAX_OUTPUT_TOKENS
                    )

                try:
                    async with session.post(url, json=payload) as response:
                        data = await response.json(content_type=None)
                        if response.status == 200:
                            candidates = data.get("candidates") or []
                            if candidates:
                                parts = (
                                    (candidates[0] or {}).get("content") or {}
                                ).get("parts") or []
                                content = "\n".join(
                                    [
                                        str(p.get("text") or "")
                                        for p in parts
                                        if p.get("text")
                                    ]
                                ).strip()
                                if content:
                                    return {
                                        "ok": True,
                                        "content": content,
                                        "model": model,
                                    }

                            errors.append(f"{model}: empty response")
                        else:
                            err_message = (
                                (data.get("error") or {}).get("message")
                                if isinstance(data, dict)
                                else str(data)
                            )
                            errors.append(
                                f"{model}: HTTP {response.status} - {str(err_message)[:200]}"
                            )
                except asyncio.TimeoutError:
                    errors.append(
                        f"{model}: request timeout sau {AI_REQUEST_TIMEOUT_SECONDS}s"
                    )
                except Exception as e:
                    errors.append(f"{model}: {str(e)}")

        return {
            "ok": False,
            "error": "Google AI Studio lỗi: " + " | ".join(errors[:3]),
            "model": None,
            "content": None,
        }

    def _gmail_get_checkpoint_ts(self, scope="inbox"):
        """Get checkpoint internal timestamp for a logical Gmail scope."""
        db_path = self._ensure_gmail_digest_tables()
        key_map = {
            "inbox": "checkpoint_inbox_internal_ts",
            "sent": "checkpoint_sent_internal_ts",
        }
        key = key_map.get(str(scope or "inbox").lower(), "checkpoint_inbox_internal_ts")
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT value FROM gmail_state WHERE key = ?", (key,)
        ).fetchone()
        conn.close()
        if not row:
            return 0
        try:
            return int(str(row[0]))
        except Exception:
            return 0

    def _gmail_set_checkpoint_ts(self, internal_ts, scope="inbox"):
        """Set checkpoint timestamp for a logical Gmail scope."""
        db_path = self._ensure_gmail_digest_tables()
        key_map = {
            "inbox": "checkpoint_inbox_internal_ts",
            "sent": "checkpoint_sent_internal_ts",
        }
        key = key_map.get(str(scope or "inbox").lower(), "checkpoint_inbox_internal_ts")
        now_iso = datetime.now(self.timezone).isoformat()
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            INSERT INTO gmail_state(key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """,
            (key, str(int(internal_ts or 0)), now_iso),
        )
        conn.commit()
        conn.close()

    def _extract_email_item(self, msg, inbox_kind):
        """Normalize one Gmail message payload into lightweight item dict."""
        headers = msg.get("payload", {}).get("headers", [])
        internal_ts = int(msg.get("internalDate") or 0)
        return {
            "id": str(msg.get("id") or ""),
            "kind": str(inbox_kind),
            "subject": self._extract_header_value(headers, "Subject")
            or "(không có tiêu đề)",
            "from": self._extract_header_value(headers, "From") or "",
            "to": self._extract_header_value(headers, "To") or "",
            "date": self._extract_header_value(headers, "Date") or "",
            "snippet": str(msg.get("snippet", "")).strip(),
            "internal_ts": internal_ts,
        }

    def _collect_gmail_messages(self, query, limit, kind):
        """Fetch Gmail metadata messages by query and normalize the output."""
        service = self._get_gmail_service()
        if not service:
            return None, "⚠️ Cần setup Gmail API (token thiếu scope)."

        try:
            refs = (
                service.users()
                .messages()
                .list(
                    userId="me",
                    q=query,
                    maxResults=max(1, int(limit or 20)),
                    fields="messages/id",
                )
                .execute()
                .get("messages", [])
            )
        except HttpError as e:
            if int(getattr(e.resp, "status", 0) or 0) == 403:
                return (
                    None,
                    "⚠️ Gmail token chưa đủ scope. Xóa token.json và chạy lại `python setup_calendar.py`.",
                )
            return None, f"⚠️ Gmail API lỗi: {str(e)}"
        except Exception as e:
            return None, f"⚠️ Không thể đọc Gmail: {str(e)}"

        message_ids = [
            str(ref.get("id") or "").strip() for ref in refs if ref.get("id")
        ]
        if not message_ids:
            return [], None

        items = []
        chunk_size = 50

        try:
            for offset in range(0, len(message_ids), chunk_size):
                id_chunk = message_ids[offset : offset + chunk_size]
                ordered_map = {}

                def _batch_callback(request_id, response, exception):
                    if exception or not response:
                        return
                    try:
                        index = int(str(request_id))
                    except Exception:
                        return
                    ordered_map[index] = self._extract_email_item(response, kind)

                batch = service.new_batch_http_request(callback=_batch_callback)
                for index, message_id in enumerate(id_chunk):
                    request = (
                        service.users()
                        .messages()
                        .get(
                            userId="me",
                            id=message_id,
                            format="metadata",
                            metadataHeaders=["Subject", "From", "To", "Date"],
                            fields="id,internalDate,snippet,payload/headers",
                        )
                    )
                    batch.add(request, request_id=str(index))

                batch.execute()

                for index in range(len(id_chunk)):
                    if index in ordered_map:
                        items.append(ordered_map[index])

            return items, None
        except Exception:
            for message_id in message_ids:
                try:
                    msg = (
                        service.users()
                        .messages()
                        .get(
                            userId="me",
                            id=message_id,
                            format="metadata",
                            metadataHeaders=["Subject", "From", "To", "Date"],
                            fields="id,internalDate,snippet,payload/headers",
                        )
                        .execute()
                    )
                except Exception:
                    continue
                items.append(self._extract_email_item(msg, kind))
            return items, None

    async def _gmail_llm_json(self, system_prompt, user_prompt):
        """Run Gmail analysis with configured model chain and parse JSON best-effort."""
        ai_result = await self._call_google_ai_studio_mail(system_prompt, user_prompt)
        if not ai_result.get("ok"):
            return None, ai_result.get("error", "LLM error"), ai_result.get("model")
        parsed = self._extract_json_block(ai_result.get("content"))
        return parsed, None, ai_result.get("model")

    def _save_gmail_report(self, report_date, period, payload):
        """Persist advanced Gmail report and return created report id."""
        db_path = self._ensure_gmail_digest_tables()
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO gmail_reports (
                report_date, period, important_json, todo_json, unread_json,
                sent_json, sent_apps_json, read_json, key_info_json, llm_models_json,
                summary_text, max_internal_ts, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report_date.strftime("%Y-%m-%d"),
                str(period or "manual"),
                json.dumps(payload.get("important_mails", []), ensure_ascii=False),
                json.dumps(payload.get("todo_list", []), ensure_ascii=False),
                json.dumps(payload.get("unread_items", []), ensure_ascii=False),
                json.dumps(payload.get("sent_items", []), ensure_ascii=False),
                json.dumps(payload.get("sent_applications", []), ensure_ascii=False),
                json.dumps(payload.get("read_items", []), ensure_ascii=False),
                json.dumps(payload.get("key_info", []), ensure_ascii=False),
                json.dumps(payload.get("llm_models", {}), ensure_ascii=False),
                str(payload.get("summary_text", "")),
                int(payload.get("max_internal_ts") or 0),
                datetime.now(self.timezone).isoformat(),
            ),
        )
        report_id = int(cur.lastrowid)
        conn.commit()
        conn.close()
        return report_id

    def get_gmail_report_row(self, report_id):
        """Load one advanced Gmail report row by id."""
        db_path = self._ensure_gmail_digest_tables()
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM gmail_reports WHERE id = ?", (int(report_id),)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_latest_gmail_report_by_date(self, target_date):
        """Load latest advanced Gmail report for a date."""
        db_path = self._ensure_gmail_digest_tables()
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT * FROM gmail_reports
            WHERE report_date = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (target_date.strftime("%Y-%m-%d"),),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    async def build_advanced_gmail_report(self, period="manual", target_date=None):
        """Build advanced Gmail report with 3 LLM calls and persist full structured payload."""
        target_date = target_date or datetime.now(self.timezone).date()
        lookback_days = max(1, int(GMAIL_LOOKBACK_DAYS or 30))
        checkpoint_ts = self._gmail_get_checkpoint_ts("inbox")
        sent_checkpoint_ts = self._gmail_get_checkpoint_ts("sent")
        sent_limit = max(20, int(GMAIL_SENT_TODAY_LIMIT or 200))

        unread_query = f"in:inbox is:unread newer_than:{lookback_days}d"
        read_query = f"in:inbox -is:unread newer_than:{lookback_days}d"
        after_str, before_str = self._gmail_day_query_bounds(target_date)
        sent_query = f"in:sent after:{after_str} before:{before_str}"

        unread_result, read_result, sent_result = await asyncio.gather(
            asyncio.to_thread(
                self._collect_gmail_messages,
                unread_query,
                GMAIL_UNREAD_LIMIT,
                "unread",
            ),
            asyncio.to_thread(
                self._collect_gmail_messages,
                read_query,
                GMAIL_READ_LIMIT,
                "read",
            ),
            asyncio.to_thread(
                self._collect_gmail_messages,
                sent_query,
                sent_limit,
                "sent",
            ),
        )

        unread_items, unread_err = unread_result
        if unread_err:
            return {"ok": False, "error": unread_err}

        read_items, read_err = read_result
        if read_err:
            return {"ok": False, "error": read_err}

        sent_items, sent_err = sent_result
        if sent_err:
            return {"ok": False, "error": sent_err}

        unread_items = [
            x
            for x in (unread_items or [])
            if int(x.get("internal_ts") or 0) > checkpoint_ts
        ]
        read_items = [
            x
            for x in (read_items or [])
            if int(x.get("internal_ts") or 0) > checkpoint_ts
        ]
        sent_items = [
            x
            for x in (sent_items or [])
            if int(x.get("internal_ts") or 0) > sent_checkpoint_ts
        ]

        max_internal_ts = max(
            [
                int(x.get("internal_ts") or 0)
                for x in (unread_items + read_items + sent_items)
            ]
            or [checkpoint_ts]
        )

        async def _build_important_group(unread_group, read_group):
            merged_items = unread_group + read_group
            mail_brief = []
            for idx, item in enumerate(merged_items[:40], start=1):
                mail_brief.append(
                    f"{idx}. [{item.get('kind')}] {item.get('subject')} | from={item.get('from')} | snippet={item.get('snippet')[:200]}"
                )

            classify_system = (
                "Phân loại email thành 2 nhóm: important và ads/newsletter. "
                "Ưu tiên các mail liên quan apply, interview, missing info, deadline. "
                "Trả JSON: {important_indexes:[...], ad_indexes:[...], notes:'...'}"
            )
            classify_user = (
                "\n".join(mail_brief) if mail_brief else "Không có mail cần phân loại."
            )
            classify_json, classify_err, classify_model = await self._gmail_llm_json(
                classify_system, classify_user
            )

            important_indexes = []
            if isinstance(classify_json, dict):
                important_indexes = [
                    int(x)
                    for x in (classify_json.get("important_indexes") or [])
                    if str(x).isdigit()
                ]

            important_items = []
            for idx in important_indexes:
                if 1 <= idx <= len(merged_items):
                    important_items.append(merged_items[idx - 1])
            if not important_items:
                important_items = merged_items[:8]

            important_brief = []
            for idx, item in enumerate(important_items[:20], start=1):
                important_brief.append(
                    f"{idx}. {item.get('subject')} | from={item.get('from')} | snippet={item.get('snippet')[:220]}"
                )

            summarize_system = (
                "Bạn là trợ lý email. Trích xuất TODO và thông tin quan trọng. "
                "TODO phải ưu tiên việc khẩn/cần bổ sung hồ sơ lên đầu. "
                "Trả JSON: {important_mail_summary:[...], todo_list:[...], key_info:[...]}"
            )
            summarize_user = (
                "\n".join(important_brief)
                if important_brief
                else "Không có mail important."
            )
            summarize_json, summarize_err, summarize_model = await self._gmail_llm_json(
                summarize_system, summarize_user
            )

            important_summary = []
            todo_list = []
            key_info = []
            if isinstance(summarize_json, dict):
                important_summary = [
                    str(x).strip()
                    for x in (summarize_json.get("important_mail_summary") or [])
                    if str(x).strip()
                ][:10]
                todo_list = [
                    str(x).strip()
                    for x in (summarize_json.get("todo_list") or [])
                    if str(x).strip()
                ][:12]
                key_info = [
                    str(x).strip()
                    for x in (summarize_json.get("key_info") or [])
                    if str(x).strip()
                ][:10]

            return {
                "important_items": important_items,
                "important_summary": important_summary,
                "todo_list": todo_list,
                "key_info": key_info,
                "classify_model": classify_model,
                "classify_err": classify_err,
                "summarize_model": summarize_model,
                "summarize_err": summarize_err,
            }

        async def _build_sent_group(sent_group):
            sent_brief = []
            for idx, item in enumerate(sent_group[:250], start=1):
                sent_brief.append(
                    f"{idx}. {item.get('subject')} | to={item.get('to')} | date={item.get('date')} | snippet={item.get('snippet')[:260]}"
                )

            sent_apps = self._parse_sent_application_items(sent_group)
            sent_system = (
                "Phân tích toàn bộ sent mail trong ngày. "
                "Bắt buộc tổng hợp các việc đã thực hiện qua mail và việc cần follow-up. "
                "Trả JSON: {sent_summary:'...', follow_up_tasks:[...], apply_highlights:[...], actions_done:[...]}"
            )
            sent_user = (
                "\n".join(sent_brief)
                if sent_brief
                else "Không có mail sent trong khoảng thời gian yêu cầu."
            )
            sent_json, sent_err, sent_model = await self._gmail_llm_json(
                sent_system, sent_user
            )

            sent_summary = ""
            sent_followups = []
            sent_highlights = []
            sent_actions_done = []
            if isinstance(sent_json, dict):
                sent_summary = str(sent_json.get("sent_summary") or "").strip()
                sent_followups = [
                    str(x).strip()
                    for x in (sent_json.get("follow_up_tasks") or [])
                    if str(x).strip()
                ][:10]
                sent_highlights = [
                    str(x).strip()
                    for x in (sent_json.get("apply_highlights") or [])
                    if str(x).strip()
                ][:10]
                sent_actions_done = [
                    str(x).strip()
                    for x in (sent_json.get("actions_done") or [])
                    if str(x).strip()
                ][:20]

            return {
                "sent_apps": sent_apps,
                "sent_summary": sent_summary,
                "sent_followups": sent_followups,
                "sent_highlights": sent_highlights,
                "sent_actions_done": sent_actions_done,
                "sent_model": sent_model,
                "sent_err": sent_err,
            }

        important_group, sent_group = await asyncio.gather(
            _build_important_group(unread_items, read_items),
            _build_sent_group(sent_items),
        )

        important_items = important_group.get("important_items", [])
        important_summary = important_group.get("important_summary", [])
        todo_list = important_group.get("todo_list", [])
        key_info = important_group.get("key_info", [])
        classify_model = important_group.get("classify_model")
        classify_err = important_group.get("classify_err")
        summarize_model = important_group.get("summarize_model")
        summarize_err = important_group.get("summarize_err")

        sent_apps = sent_group.get("sent_apps", [])
        sent_summary = sent_group.get("sent_summary", "")
        sent_followups = sent_group.get("sent_followups", [])
        sent_highlights = sent_group.get("sent_highlights", [])
        sent_actions_done = sent_group.get("sent_actions_done", [])
        sent_model = sent_group.get("sent_model")
        sent_err = sent_group.get("sent_err")

        missing_info_todos = self._extract_missing_info_todos(important_items)
        todo_list = self._prioritize_todo_list(
            todo_list + missing_info_todos + sent_followups
        )
        key_info = self._normalize_key_info_entries(key_info)

        if not sent_actions_done:
            sent_actions_done = [
                f"{x.get('action')}: {x.get('position')} @ {x.get('company')}"
                for x in [
                    self._extract_sent_action_record(item) for item in sent_items[:30]
                ]
            ][:15]

        if not sent_summary:
            sent_summary = f"Tổng hợp {len(sent_items)} mail đã gửi trong ngày {target_date.strftime('%d/%m/%Y')}."

        summary_text = (
            f"Important={len(important_items)} | Unread={len(unread_items)} | "
            f"SentToday={len(sent_items)} | Todo={len(todo_list)}"
        )

        payload = {
            "important_mails": important_items,
            "important_summary": important_summary,
            "todo_list": todo_list,
            "key_info": key_info,
            "sent_items": sent_items,
            "sent_applications": sent_apps,
            "sent_summary": sent_summary,
            "sent_followups": sent_followups,
            "sent_highlights": sent_highlights,
            "sent_actions_done": sent_actions_done,
            "unread_items": unread_items,
            "read_items": read_items,
            "max_internal_ts": max_internal_ts,
            "summary_text": summary_text,
            "llm_models": {
                "classify": classify_model,
                "important": summarize_model,
                "sent": sent_model,
            },
            "llm_errors": {
                "classify": classify_err,
                "important": summarize_err,
                "sent": sent_err,
            },
        }

        report_id = self._save_gmail_report(target_date, period, payload)
        payload["report_id"] = report_id
        payload["ok"] = True
        return payload

    def mark_report_checkpoint(self, report_id):
        """Backward-compatible alias for inbox checkpoint."""
        return self.mark_report_inbox_checkpoint(report_id)

    def mark_report_inbox_checkpoint(self, report_id):
        """Mark checkpoint for important+unread pipeline (inbox scope)."""
        row = self.get_gmail_report_row(report_id)
        if not row:
            return False, "⚠️ Không tìm thấy report để checkpoint."
        max_ts = 0
        try:
            unread_items = json.loads(row.get("unread_json") or "[]")
            read_items = json.loads(row.get("read_json") or "[]")
            max_ts = max(
                [int(x.get("internal_ts") or 0) for x in (unread_items + read_items)]
                or [0]
            )
        except Exception:
            max_ts = 0
        if max_ts <= 0:
            max_ts = int(row.get("max_internal_ts") or 0)

        self._gmail_set_checkpoint_ts(max_ts, scope="inbox")
        return (
            True,
            "✅ Đã checkpoint cho Important/Unread. Lần sau bỏ qua mail inbox cũ trước mốc này.",
        )

    def mark_report_sent_checkpoint(self, report_id):
        """Mark checkpoint for sent-mail insight pipeline."""
        row = self.get_gmail_report_row(report_id)
        if not row:
            return False, "⚠️ Không tìm thấy report để checkpoint sent."

        max_ts = 0
        try:
            sent_items = json.loads(row.get("sent_json") or "[]")
            max_ts = max([int(x.get("internal_ts") or 0) for x in sent_items] or [0])
        except Exception:
            max_ts = 0

        if max_ts <= 0:
            return False, "ℹ️ Report không có sent mail hợp lệ để checkpoint."

        self._gmail_set_checkpoint_ts(max_ts, scope="sent")
        return (
            True,
            "✅ Đã checkpoint cho Sent Mail Insight. Lần sau bỏ qua sent mail cũ trước mốc này.",
        )

    def mark_unread_as_read_for_report(self, report_id):
        """Mark unread mails from a report as read via Gmail modify scope."""
        row = self.get_gmail_report_row(report_id)
        if not row:
            return False, "⚠️ Không tìm thấy report."
        try:
            unread_items = json.loads(row.get("unread_json") or "[]")
        except Exception:
            unread_items = []
        ids = [str(x.get("id")) for x in unread_items if x.get("id")]
        if not ids:
            return True, "ℹ️ Không có mail unread để đánh dấu đã đọc."

        service = self._get_gmail_service()
        if not service:
            return False, "⚠️ Chưa setup Gmail API."
        try:
            service.users().messages().batchModify(
                userId="me", body={"ids": ids, "removeLabelIds": ["UNREAD"]}
            ).execute()
            return True, f"✅ Đã đánh dấu đã đọc {len(ids)} mail unread."
        except HttpError as e:
            if int(getattr(e.resp, "status", 0) or 0) == 403:
                return (
                    False,
                    "⚠️ Token chưa có `gmail.modify`. Xóa token.json và chạy lại `python setup_calendar.py`.",
                )
            return False, f"⚠️ Gmail API lỗi: {str(e)}"
        except Exception as e:
            return False, f"⚠️ Không thể mark read: {str(e)}"

    async def create_tasks_from_report(self, report_id, task_indexes):
        """Create Google Tasks from selected TODO indexes inside a report."""
        row = self.get_gmail_report_row(report_id)
        if not row:
            return False, "⚠️ Không tìm thấy report."
        try:
            todo_items = json.loads(row.get("todo_json") or "[]")
        except Exception:
            todo_items = []
        if not todo_items:
            return False, "⚠️ Report không có TODO để tạo task."

        created = 0
        for idx in task_indexes or []:
            if 0 <= int(idx) < len(todo_items):
                title = str(todo_items[int(idx)]).strip()
                if not title:
                    continue
                result = await self.add_task(
                    title, due_datetime=None, notes="Auto từ Gmail digest"
                )
                if str(result).startswith("✅"):
                    created += 1

        if created == 0:
            return False, "⚠️ Không tạo được task nào."
        return True, f"✅ Đã tạo {created} task từ Gmail TODO list."

    async def create_calendar_done_from_report(self, report_id):
        """Create completed Google Tasks from sent-mail actions (no calendar events)."""
        row = self.get_gmail_report_row(report_id)
        if not row:
            return False, "⚠️ Không tìm thấy report."
        try:
            sent_apps = json.loads(row.get("sent_apps_json") or "[]")
            sent_items = json.loads(row.get("sent_json") or "[]")
        except Exception:
            sent_apps = []
            sent_items = []

        service = self._get_tasks_service()
        if not service:
            return False, "⚠️ Cần setup Google Tasks."

        try:
            tasklists = service.tasklists().list().execute()
            if not tasklists.get("items"):
                return False, "⚠️ Không tìm thấy tasklist mặc định."
            tasklist_id = tasklists["items"][0]["id"]
        except Exception as e:
            return False, f"⚠️ Không thể lấy tasklist: {str(e)}"

        source_items = sent_items if sent_items else []
        if not source_items and sent_apps:
            source_items = sent_apps
        if not source_items:
            return False, "ℹ️ Không có sent mail để tạo task hoàn thành."

        created = 0
        for raw_item in source_items[:60]:
            if isinstance(raw_item, dict) and raw_item.get("subject"):
                action_item = self._extract_sent_action_record(raw_item)
            else:
                action_item = {
                    "action": "Đã xử lý",
                    "position": str(raw_item.get("position") or "(chưa rõ vị trí)"),
                    "company": str(raw_item.get("company") or "(chưa rõ công ty)"),
                    "subject": str(raw_item.get("subject") or "(không tiêu đề)"),
                    "to": "",
                    "date": "",
                    "snippet": "",
                }

            title = (
                f"[DONE] {action_item['action']}: "
                f"{action_item['position']} @ {action_item['company']}"
            )
            notes = (
                f"Subject: {action_item['subject']}\n"
                f"To: {action_item['to']}\n"
                f"Date: {action_item['date']}\n"
                f"Snippet: {action_item['snippet'][:500]}"
            )

            try:
                inserted = (
                    service.tasks()
                    .insert(
                        tasklist=tasklist_id,
                        body={"title": title[:255], "notes": notes},
                    )
                    .execute()
                )
                inserted["status"] = "completed"
                inserted["completed"] = datetime.now(self.timezone).isoformat()
                service.tasks().update(
                    tasklist=tasklist_id,
                    task=inserted["id"],
                    body=inserted,
                ).execute()
                created += 1
            except Exception:
                continue

        if created == 0:
            return False, "⚠️ Không tạo được task completed nào từ sent mails."
        return (
            True,
            f"✅ Đã tạo và mark completed {created} task từ sent mail trong report.",
        )

    def _extract_header_value(self, headers, name):
        """Extract one header value from Gmail metadata header list."""
        key = str(name or "").lower()
        for item in headers or []:
            if str(item.get("name", "")).lower() == key:
                return str(item.get("value", "")).strip()
        return ""

    def _gmail_day_query_bounds(self, target_date):
        """Build Gmail query date bounds (after/before) for one local day."""
        if isinstance(target_date, datetime):
            target_date = target_date.date()
        after_str = target_date.strftime("%Y/%m/%d")
        before_str = (target_date + timedelta(days=1)).strftime("%Y/%m/%d")
        return after_str, before_str

    def _parse_sent_application_items(self, sent_messages):
        """Infer compact company/position records from sent email subject/snippet."""
        records = []
        for item in sent_messages or []:
            subject = str(item.get("subject", "")).strip()
            snippet = str(item.get("snippet", "")).strip()
            combined = f"{subject} {snippet}".strip()
            combined_lower = combined.lower()

            if not any(
                token in combined_lower
                for token in [
                    "apply",
                    "application",
                    "resume",
                    "cv",
                    "recruit",
                    "interview",
                    "ứng tuyển",
                ]
            ):
                continue

            role_match = re.search(
                r"(?:position|role|vị\s*trí|apply\s*for|ứng\s*tuyển)\s*[:\-]?\s*([A-Za-z0-9\-\+/#\(\) ]{2,60})",
                combined,
                flags=re.IGNORECASE,
            )
            company_match = re.search(
                r"(?:at|to|vào|tại)\s+([A-Za-z0-9&\-\., ]{2,80})",
                combined,
                flags=re.IGNORECASE,
            )

            role = role_match.group(1).strip(" .,-") if role_match else ""
            company = company_match.group(1).strip(" .,-") if company_match else ""

            records.append(
                {
                    "company": company or "(không rõ công ty)",
                    "position": role or "(không rõ vị trí)",
                    "subject": subject[:160],
                }
            )

        deduped = []
        seen = set()
        for rec in records:
            key = (
                rec["company"].lower(),
                rec["position"].lower(),
                rec["subject"].lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(rec)
        return deduped[:20]

    def _normalize_key_info_entries(self, key_info_items):
        """Normalize key-info values into readable bullet lines and priority-sort them."""
        normalized = []

        def _priority_score(text):
            lower = text.lower()
            score = 0
            for token in [
                "required",
                "bổ sung",
                "missing",
                "deadline",
                "urgent",
                "interview",
                "apply",
                "offer",
                "xác nhận",
                "document",
            ]:
                if token in lower:
                    score += 2
            if "expired" in lower or "closed" in lower:
                score += 1
            return score

        for raw in key_info_items or []:
            if isinstance(raw, dict):
                if len(raw) == 1:
                    key, value = list(raw.items())[0]
                    line = f"{str(key).replace('_', ' ').strip().title()}: {str(value).strip()}"
                else:
                    line = " | ".join(
                        [
                            f"{str(k).replace('_', ' ').strip().title()}: {str(v).strip()}"
                            for k, v in raw.items()
                            if str(v).strip()
                        ]
                    )
            else:
                text = str(raw).strip()
                parsed = None
                if text.startswith("{") and text.endswith("}"):
                    try:
                        parsed = ast.literal_eval(text)
                    except Exception:
                        parsed = None
                if isinstance(parsed, dict) and parsed:
                    if len(parsed) == 1:
                        key, value = list(parsed.items())[0]
                        line = f"{str(key).replace('_', ' ').strip().title()}: {str(value).strip()}"
                    else:
                        line = " | ".join(
                            [
                                f"{str(k).replace('_', ' ').strip().title()}: {str(v).strip()}"
                                for k, v in parsed.items()
                                if str(v).strip()
                            ]
                        )
                else:
                    line = text

            line = str(line).strip(" -")
            if line:
                normalized.append(line)

        dedup = []
        seen = set()
        for line in normalized:
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(line)

        dedup.sort(key=lambda x: _priority_score(x), reverse=True)
        return dedup[:12]

    def _extract_missing_info_todos(self, important_items):
        """Extract high-priority TODOs related to missing application information."""
        todos = []
        for item in important_items or []:
            subject = str(item.get("subject") or "").strip()
            snippet = str(item.get("snippet") or "").strip()
            sender = str(item.get("from") or "").strip()
            text = f"{subject} {snippet}".lower()

            has_missing_signal = any(
                token in text
                for token in [
                    "missing",
                    "bổ sung",
                    "additional information",
                    "required document",
                    "complete your profile",
                    "incomplete",
                    "pending",
                ]
            )
            if not has_missing_signal:
                continue

            task = (
                f"Ưu tiên bổ sung thông tin theo mail: {subject[:90]}"
                if subject
                else "Ưu tiên bổ sung thông tin apply còn thiếu"
            )
            if sender:
                task += f" (from: {sender[:60]})"
            todos.append(task)

        dedup = []
        seen = set()
        for task in todos:
            key = task.lower()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(task)
        return dedup[:8]

    def _prioritize_todo_list(self, todo_items):
        """Sort TODO items so critical application/interview tasks appear first."""

        def score(todo_text):
            lower = str(todo_text).lower()
            total = 0
            for token in [
                "ưu tiên",
                "required",
                "missing",
                "bổ sung",
                "deadline",
                "interview",
                "apply",
                "hồ sơ",
                "cv",
                "follow up",
            ]:
                if token in lower:
                    total += 2
            if "urgent" in lower or "gấp" in lower:
                total += 3
            return total

        def normalize_todo_item(todo):
            if isinstance(todo, dict):
                task = str(
                    todo.get("task")
                    or todo.get("title")
                    or todo.get("todo")
                    or todo.get("summary")
                    or ""
                ).strip()
                if not task:
                    return ""
                priority = str(todo.get("priority") or "").strip()
                if priority:
                    return f"{task} (priority: {priority})"
                return task

            if isinstance(todo, (list, tuple)):
                merged = " - ".join([str(x).strip() for x in todo if str(x).strip()])
                return merged.strip()

            return str(todo or "").strip()

        cleaned = []
        for item in todo_items or []:
            normalized = normalize_todo_item(item)
            if normalized:
                cleaned.append(normalized)
        dedup = []
        seen = set()
        for item in cleaned:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(item)
        dedup.sort(key=lambda x: score(x), reverse=True)
        return dedup[:20]

    def _extract_sent_action_record(self, sent_item):
        """Parse one sent mail into a structured action record for completed-task logging."""
        subject = str(sent_item.get("subject") or "").strip()
        snippet = str(sent_item.get("snippet") or "").strip()
        to_value = str(sent_item.get("to") or "").strip()
        date_value = str(sent_item.get("date") or "").strip()
        combined = f"{subject} {snippet}"

        action = "Đã gửi email"
        lowered = combined.lower()
        if any(x in lowered for x in ["apply", "application", "ứng tuyển"]):
            action = "Đã apply"
        elif any(x in lowered for x in ["follow up", "follow-up", "nhắc", "update"]):
            action = "Đã follow-up"
        elif any(x in lowered for x in ["interview", "phỏng vấn"]):
            action = "Đã phản hồi interview"

        position = ""
        role_match = re.search(
            r"(?:position|role|vị\s*trí|apply\s*for|ứng\s*tuyển)\s*[:\-]?\s*([A-Za-z0-9\-\+/#\(\) ]{2,80})",
            combined,
            flags=re.IGNORECASE,
        )
        if role_match:
            position = role_match.group(1).strip(" .,-")

        company = ""
        company_match = re.search(
            r"(?:at|to|vào|tại)\s+([A-Za-z0-9&\-\., ]{2,90})",
            combined,
            flags=re.IGNORECASE,
        )
        if company_match:
            company = company_match.group(1).strip(" .,-")
        if not company and to_value:
            parts = [p.strip() for p in to_value.split(",") if p.strip()]
            company = parts[0][:80] if parts else to_value[:80]

        if not position:
            title_match = re.search(
                r"(Software Engineer|Backend Engineer|Frontend Engineer|Fullstack Developer|Data Researcher|Data Analyst|QA Engineer|DevOps Engineer|Product Manager)",
                combined,
                flags=re.IGNORECASE,
            )
            if title_match:
                position = title_match.group(1)

        return {
            "action": action,
            "position": position or "(chưa rõ vị trí)",
            "company": company or "(chưa rõ công ty)",
            "subject": subject or "(không tiêu đề)",
            "to": to_value,
            "date": date_value,
            "snippet": snippet,
        }

    def _collect_unread_gmail_subjects(self, limit=GMAIL_UNREAD_LIMIT):
        """Fetch unread Gmail messages and return compact metadata list."""
        service = self._get_gmail_service()
        if not service:
            return None, "⚠️ Cần setup Gmail API (cập nhật token với gmail.readonly)."

        q = "is:unread -category:promotions -category:social -category:updates -category:forums"
        try:
            refs = (
                service.users()
                .messages()
                .list(userId="me", q=q, maxResults=max(1, int(limit or 10)))
                .execute()
                .get("messages", [])
            )
        except HttpError as e:
            if int(getattr(e.resp, "status", 0) or 0) == 403:
                return (
                    None,
                    "⚠️ Gmail token chưa có quyền `gmail.readonly` (insufficient scopes). "
                    "Hãy xóa `token.json` rồi chạy lại `python setup_calendar.py`.",
                )
            return None, f"⚠️ Gmail API lỗi: {str(e)}"
        except Exception as e:
            return None, f"⚠️ Không thể đọc Gmail unread: {str(e)}"

        items = []
        for ref in refs:
            message_id = ref.get("id")
            if not message_id:
                continue
            try:
                msg = (
                    service.users()
                    .messages()
                    .get(
                        userId="me",
                        id=message_id,
                        format="metadata",
                        metadataHeaders=["Subject", "From", "Date"],
                    )
                    .execute()
                )
            except Exception:
                continue
            headers = msg.get("payload", {}).get("headers", [])
            items.append(
                {
                    "id": message_id,
                    "subject": self._extract_header_value(headers, "Subject")
                    or "(không có tiêu đề)",
                    "from": self._extract_header_value(headers, "From") or "",
                    "date": self._extract_header_value(headers, "Date") or "",
                    "snippet": str(msg.get("snippet", "")).strip(),
                }
            )

        return items, None

    def _collect_sent_mail_summary(self, target_date, max_results=30):
        """Collect sent-email summary and extracted application hints for one day."""
        service = self._get_gmail_service()
        if not service:
            return (
                None,
                None,
                "⚠️ Cần setup Gmail API (cập nhật token với gmail.readonly).",
            )

        after_str, before_str = self._gmail_day_query_bounds(target_date)
        q = f"in:sent after:{after_str} before:{before_str}"
        try:
            refs = (
                service.users()
                .messages()
                .list(userId="me", q=q, maxResults=max(1, int(max_results or 30)))
                .execute()
                .get("messages", [])
            )
        except HttpError as e:
            if int(getattr(e.resp, "status", 0) or 0) == 403:
                return (
                    None,
                    None,
                    "⚠️ Gmail token chưa có quyền `gmail.readonly` (insufficient scopes). "
                    "Hãy xóa `token.json` rồi chạy lại `python setup_calendar.py`.",
                )
            return None, None, f"⚠️ Gmail API lỗi: {str(e)}"
        except Exception as e:
            return None, None, f"⚠️ Không thể đọc Gmail sent: {str(e)}"

        sent_items = []
        for ref in refs:
            message_id = ref.get("id")
            if not message_id:
                continue
            try:
                msg = (
                    service.users()
                    .messages()
                    .get(
                        userId="me",
                        id=message_id,
                        format="metadata",
                        metadataHeaders=["Subject", "To", "Date"],
                    )
                    .execute()
                )
            except Exception:
                continue
            headers = msg.get("payload", {}).get("headers", [])
            sent_items.append(
                {
                    "id": message_id,
                    "subject": self._extract_header_value(headers, "Subject")
                    or "(không có tiêu đề)",
                    "to": self._extract_header_value(headers, "To") or "",
                    "date": self._extract_header_value(headers, "Date") or "",
                    "snippet": str(msg.get("snippet", "")).strip(),
                }
            )

        applications = self._parse_sent_application_items(sent_items)

        if not sent_items:
            sent_summary = "Không có mail gửi đi trong ngày này."
        else:
            sent_summary = f"Đã gửi {len(sent_items)} mail."
            if applications:
                sent_summary += (
                    f" Phát hiện {len(applications)} mail có dấu hiệu apply/recruit."
                )

        return sent_items, applications, sent_summary

    def _store_gmail_digest(
        self,
        target_date,
        period,
        unread_items,
        sent_summary,
        sent_applications,
    ):
        """Persist one Gmail digest snapshot by date and period."""
        db_path = self._ensure_gmail_digest_tables()
        digest_date = target_date.strftime("%Y-%m-%d")
        created_at = datetime.now(self.timezone).isoformat()
        unread_titles = [str(x.get("subject", "")).strip() for x in unread_items or []]

        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            INSERT OR REPLACE INTO gmail_digests (
                digest_date, period, unread_count, unread_titles_json,
                sent_summary, sent_applications_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                digest_date,
                str(period or "manual"),
                len(unread_titles),
                json.dumps(unread_titles, ensure_ascii=False),
                str(sent_summary or ""),
                json.dumps(sent_applications or [], ensure_ascii=False),
                created_at,
            ),
        )
        conn.commit()
        conn.close()

    def _load_gmail_digest_rows(self, target_date):
        """Load saved Gmail digest rows for a specific date."""
        db_path = self._ensure_gmail_digest_tables()
        digest_date = target_date.strftime("%Y-%m-%d")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT digest_date, period, unread_count, unread_titles_json,
                   sent_summary, sent_applications_json, created_at
            FROM gmail_digests
            WHERE digest_date = ?
            ORDER BY period ASC
            """,
            (digest_date,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    async def build_and_store_gmail_digest(self, period="manual", target_date=None):
        """Build Gmail digest text (unread + sent summary), save snapshot, and return payload."""
        target_date = target_date or datetime.now(self.timezone).date()
        period = str(period or "manual").strip().lower()

        if period in {"morning", "evening"}:
            existing = self._load_gmail_digest_rows(target_date)
            if any(str(r.get("period", "")).lower() == period for r in existing):
                return {
                    "ok": True,
                    "already_exists": True,
                    "text": "",
                    "target_date": target_date,
                    "period": period,
                }

        unread_items, unread_err = self._collect_unread_gmail_subjects(
            GMAIL_UNREAD_LIMIT
        )
        if unread_err:
            return {"ok": False, "error": unread_err}

        sent_items, sent_applications, sent_summary = self._collect_sent_mail_summary(
            target_date
        )
        if sent_summary is None and sent_applications is None:
            return {
                "ok": False,
                "error": "⚠️ Không thể lấy dữ liệu Gmail gửi đi. Hãy kiểm tra token/scope.",
            }

        self._store_gmail_digest(
            target_date=target_date,
            period=period,
            unread_items=unread_items,
            sent_summary=sent_summary,
            sent_applications=sent_applications,
        )

        title = f"📮 **Gmail Digest ({period}) - {target_date.strftime('%d/%m/%Y')}**"
        unread_lines = []
        for idx, item in enumerate(unread_items[:GMAIL_UNREAD_LIMIT], start=1):
            unread_lines.append(f"{idx}. {item.get('subject', '(không tiêu đề)')}")
        if not unread_lines:
            unread_lines = ["Không có mail unread."]

        sent_lines = [f"- {sent_summary}"]
        for rec in (sent_applications or [])[:8]:
            sent_lines.append(
                f"- Apply: {rec.get('company')} | {rec.get('position')} | {rec.get('subject')}"
            )

        body = f"{title}\n\n" f"**📩 Unread (không gồm spam/promo):**\n" + "\n".join(
            unread_lines
        ) + "\n\n**📤 Mail đã gửi (tóm tắt):**\n" + "\n".join(sent_lines)

        return {
            "ok": True,
            "already_exists": False,
            "text": body,
            "target_date": target_date,
            "period": period,
            "unread_count": len(unread_items or []),
            "sent_count": len(sent_items or []),
        }

    def get_gmail_digest_history(self, target_date=None):
        """Return formatted Gmail digest history text for a specific date."""
        target_date = target_date or datetime.now(self.timezone).date()
        rows = self._load_gmail_digest_rows(target_date)
        if not rows:
            return f"📭 Không có Gmail digest lưu cho ngày {target_date.strftime('%d/%m/%Y')}."

        lines = [f"📚 **Gmail digest history {target_date.strftime('%d/%m/%Y')}**"]
        for row in rows:
            period = str(row.get("period", "manual")).lower()
            unread_count = int(row.get("unread_count") or 0)
            sent_summary = str(row.get("sent_summary") or "")
            titles = []
            try:
                titles = json.loads(row.get("unread_titles_json") or "[]")
            except Exception:
                titles = []
            applications = []
            try:
                applications = json.loads(row.get("sent_applications_json") or "[]")
            except Exception:
                applications = []

            lines.append(f"\n**[{period}]** unread={unread_count}")
            for title in titles[:8]:
                lines.append(f"- {title}")
            if sent_summary:
                lines.append(f"- Sent: {sent_summary}")
            for rec in applications[:5]:
                lines.append(
                    f"- Apply: {rec.get('company')} | {rec.get('position')} | {rec.get('subject')}"
                )

        return "\n".join(lines)[:3900]

    async def search_place_natural(self, query):
        """Resolve natural-language place query using OpenStreetMap Nominatim."""
        text = str(query or "").strip()
        if not text:
            return "⚠️ Hãy nhập địa điểm cần tìm."

        base_url = "https://nominatim.openstreetmap.org/search"
        user_agent = "discord-agent-bot/1.0 (contact: local-bot)"

        generic_tokens = {
            "cafe",
            "coffee",
            "quán cafe",
            "restaurant",
            "nhà hàng",
            "hotel",
            "atm",
            "hospital",
            "pharmacy",
            "gym",
        }
        lower_text = text.lower()
        has_location_hint = any(
            token in lower_text
            for token in [
                "đà nẵng",
                "da nang",
                "hà nội",
                "ha noi",
                "hồ chí minh",
                "ho chi minh",
                "việt nam",
                "vietnam",
            ]
        )

        if lower_text in generic_tokens and not has_location_hint:
            local_query = f"{text}, {WEATHER_DEFAULT_LOCATION}, Vietnam"
        else:
            local_query = text

        async def _fetch_nominatim(search_query, countrycodes=""):
            params = {
                "q": search_query,
                "format": "jsonv2",
                "limit": 5,
                "accept-language": "vi",
                "addressdetails": 1,
            }
            if countrycodes:
                params["countrycodes"] = countrycodes

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    base_url,
                    params=params,
                    headers={"User-Agent": user_agent},
                ) as response:
                    if response.status != 200:
                        return (
                            None,
                            f"⚠️ Không thể gọi OpenStreetMap API (HTTP {response.status}).",
                        )
                    payload = await response.json(content_type=None)
                    return payload, None

        try:
            local_data, local_err = await _fetch_nominatim(
                local_query, countrycodes="vn"
            )
            if local_err:
                return local_err

            local_results = local_data if isinstance(local_data, list) else []
            if local_results:
                results = local_results
                scope_label = "(ưu tiên Việt Nam)"
            else:
                global_data, global_err = await _fetch_nominatim(text)
                if global_err:
                    return global_err
                results = global_data if isinstance(global_data, list) else []
                scope_label = "(global fallback)"
        except Exception as e:
            return f"⚠️ Lỗi gọi Maps API: {str(e)}"

        if not results:
            return "📍 Không tìm thấy địa điểm phù hợp."

        top_results = results[:3]
        lines = [
            f"🗺️ **Kết quả tìm địa điểm cho:** {text} {scope_label}",
            "ℹ️ Discord không cung cấp GPS hiện tại của bạn cho bot, nên bot tìm theo text bạn nhập.",
        ]
        for idx, item in enumerate(top_results, start=1):
            formatted = str(item.get("display_name") or "").strip()
            lat = item.get("lat")
            lng = item.get("lon")
            place_id = item.get("place_id", "")
            maps_link = (
                f"https://www.openstreetmap.org/?mlat={lat}&mlon={lng}#map=16/{lat}/{lng}"
                if lat is not None and lng is not None
                else ""
            )
            lines.append(f"\n{idx}. {formatted}")
            if lat is not None and lng is not None:
                lines.append(f"   - Toạ độ: {lat}, {lng}")
            if place_id:
                lines.append(f"   - place_id: {place_id}")
            if maps_link:
                lines.append(f"   - Link: {maps_link}")

        return "\n".join(lines)

    def parse_date(self, date_str):
        """Parse natural-language date input into a concrete date object."""
        if not date_str:
            return None

        date_str = date_str.lower().strip()
        now = datetime.now(self.timezone)

        if date_str in ["today", "hôm nay"]:
            return now.date()
        elif date_str in ["tomorrow", "tmr", "mai"]:
            return (now + timedelta(days=1)).date()
        elif date_str in ["dayafter", "mốt"]:
            return (now + timedelta(days=2)).date()

        weekdays = {
            "monday": 0,
            "mon": 0,
            "tuesday": 1,
            "tue": 1,
            "wednesday": 2,
            "wed": 2,
            "thursday": 3,
            "thu": 3,
            "friday": 4,
            "fri": 4,
            "saturday": 5,
            "sat": 5,
            "sunday": 6,
            "sun": 6,
        }

        for day_name, day_num in weekdays.items():
            if day_name in date_str:
                days_ahead = day_num - now.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                return (now + timedelta(days=days_ahead)).date()

        match = re.search(r"(\d{1,2})[/-](\d{1,2})", date_str)
        if match:
            day = int(match.group(1))
            month = int(match.group(2))
            year = now.year

            try:
                target_date = datetime(year, month, day).date()
                if target_date < now.date():
                    target_date = datetime(year + 1, month, day).date()
                return target_date
            except ValueError:
                return None

        return None

    def parse_time(self, time_str):
        """Parse common time expressions (14:00, 14h, 2pm) into time object."""
        if not time_str:
            return None

        time_str = time_str.lower().strip()

        match = re.search(r"(\d{1,2})[h:](\d{2})", time_str)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return time(hour, minute)

        match = re.search(r"(\d{1,2})h?", time_str)
        if match:
            hour = int(match.group(1))

            if "pm" in time_str and hour < 12:
                hour += 12
            elif "am" in time_str and hour == 12:
                hour = 0

            if 0 <= hour <= 23:
                return time(hour, 0)

        return None

    def parse_history_date(self, date_str):
        """Parse a date for history lookup without forcing it into the future."""
        if not date_str:
            return None

        text = str(date_str).lower().strip()
        now = datetime.now(self.timezone)

        if text in ["today", "hôm nay"]:
            return now.date()
        if text in ["yesterday", "hôm qua"]:
            return (now - timedelta(days=1)).date()

        iso_match = re.search(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", text)
        if iso_match:
            year, month, day = map(int, iso_match.groups())
            try:
                return datetime(year, month, day).date()
            except ValueError:
                return None

        dm_match = re.search(r"^(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?$", text)
        if dm_match:
            day = int(dm_match.group(1))
            month = int(dm_match.group(2))
            year_part = dm_match.group(3)
            year = int(year_part) if year_part else now.year
            if year < 100:
                year += 2000
            try:
                return datetime(year, month, day).date()
            except ValueError:
                return None

        return None

    async def _fetch_weatherapi_forecast(self, target_date=None, min_days=1):
        """Fetch WeatherAPI forecast payload for the given date horizon."""
        now_local = datetime.now(self.timezone)
        date_to_use = target_date or now_local.date()
        day_delta = (date_to_use - now_local.date()).days

        if day_delta < 0:
            return None, "⚠️ Không hỗ trợ xem weather quá khứ."
        if day_delta >= WEATHER_FORECAST_MAX_DAYS:
            return (
                None,
                f"⚠️ Chỉ hỗ trợ dự báo trong {WEATHER_FORECAST_MAX_DAYS} ngày tới.",
            )

        forecast_days = max(int(min_days or 1), day_delta + 1)
        url = (
            "http://api.weatherapi.com/v1/forecast.json"
            f"?key={WEATHER_API_KEY}"
            f"&q={WEATHER_DEFAULT_LOCATION}"
            f"&days={forecast_days}"
            "&lang=vi&aqi=no"
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        return None, "⚠️ Không thể lấy thời tiết"
                    data = await response.json()
            return data, None
        except Exception as e:
            return None, f"⚠️ Lỗi: {str(e)}"

    def _is_bad_weather_hour(self, hour_data):
        """Check whether one forecast hour should be considered weather-risky."""
        condition_text = str(
            (hour_data.get("condition") or {}).get("text") or ""
        ).lower()
        chance_of_rain = int(hour_data.get("chance_of_rain") or 0)
        wind_kph = float(hour_data.get("wind_kph") or 0)

        severe_tokens = [
            "mưa",
            "dông",
            "bão",
            "giông",
            "thunder",
            "storm",
            "heavy rain",
            "moderate rain",
            "rain",
            "sleet",
            "snow",
        ]
        has_severe_condition = any(token in condition_text for token in severe_tokens)

        is_risky = (
            chance_of_rain >= WEATHER_ALERT_RAIN_CHANCE_THRESHOLD
            or wind_kph >= WEATHER_ALERT_WIND_KPH_THRESHOLD
            or has_severe_condition
        )

        reasons = []
        if chance_of_rain >= WEATHER_ALERT_RAIN_CHANCE_THRESHOLD:
            reasons.append(f"mưa {chance_of_rain}%")
        if wind_kph >= WEATHER_ALERT_WIND_KPH_THRESHOLD:
            reasons.append(f"gió {round(wind_kph, 1)} km/h")
        if has_severe_condition:
            reasons.append(f"điều kiện: {condition_text}")

        return is_risky, reasons

    async def get_weather_risk_window(self, hours_ahead=3):
        """Analyze hourly forecast in the next N hours and return risk summary."""
        hours_ahead = max(1, int(hours_ahead or 3))
        now_local = datetime.now(self.timezone)
        target_date = (now_local + timedelta(hours=hours_ahead)).date()
        day_span = (target_date - now_local.date()).days + 1

        data, err = await self._fetch_weatherapi_forecast(
            target_date=target_date,
            min_days=max(1, day_span),
        )
        if err:
            return {"ok": False, "error": err}

        forecast_days_data = data.get("forecast", {}).get("forecastday", [])
        slots = []
        for day_item in forecast_days_data:
            for hour_data in day_item.get("hour", []):
                raw_time = str(hour_data.get("time") or "")
                if not raw_time:
                    continue
                try:
                    slot_time = datetime.strptime(raw_time, "%Y-%m-%d %H:%M")
                    slot_time = self.timezone.localize(slot_time)
                except Exception:
                    continue
                if slot_time < now_local:
                    continue
                if slot_time > now_local + timedelta(hours=hours_ahead):
                    continue

                risky, reasons = self._is_bad_weather_hour(hour_data)
                slots.append(
                    {
                        "time": slot_time,
                        "temp_c": hour_data.get("temp_c"),
                        "condition": str(
                            (hour_data.get("condition") or {}).get("text") or ""
                        ),
                        "humidity": int(hour_data.get("humidity") or 0),
                        "wind_kph": float(hour_data.get("wind_kph") or 0),
                        "chance_of_rain": int(hour_data.get("chance_of_rain") or 0),
                        "is_risky": risky,
                        "reasons": reasons,
                    }
                )

        if not slots:
            return {"ok": False, "error": "⚠️ Không có dữ liệu forecast theo giờ."}

        risky_slots = [slot for slot in slots if slot.get("is_risky")]
        return {
            "ok": True,
            "hours_ahead": hours_ahead,
            "slots": slots,
            "risky_slots": risky_slots,
            "is_risky": bool(risky_slots),
        }

    async def get_weather(self, target_date=None, target_time=None):
        """Fetch current weather or forecast by date/time from WeatherAPI."""
        try:
            now_local = datetime.now(self.timezone)
            date_to_use = target_date or now_local.date()
            data, err = await self._fetch_weatherapi_forecast(
                target_date=date_to_use,
                min_days=1,
            )
            if err:
                return err

            forecast_days_data = data.get("forecast", {}).get("forecastday", [])
            selected_day = next(
                (
                    d
                    for d in forecast_days_data
                    if str(d.get("date", "")) == date_to_use.strftime("%Y-%m-%d")
                ),
                None,
            )
            if not selected_day:
                return "⚠️ Không có dữ liệu dự báo cho ngày được chọn."

            title_date = date_to_use.strftime("%d/%m/%Y")
            if target_date is None and target_time is None:
                current = data["current"]
                result = f"🌤️ **Thời tiết {WEATHER_DEFAULT_LOCATION} (hiện tại)**\n\n"
                result += f"🗓️ {title_date}\n"
                result += (
                    f"🌡️ {current['temp_c']}°C "
                    f"(cảm giác {current['feelslike_c']}°C)\n"
                )
                result += f"☁️ {current['condition']['text']}\n"
                result += f"💧 Độ ẩm: {current['humidity']}%\n"
                result += f"💨 Gió: {current['wind_kph']} km/h\n"
                result += (
                    f"🌧️ Khả năng mưa: {selected_day['day']['daily_chance_of_rain']}%\n"
                )
                return result

            if target_time is not None:
                target_hour = int(target_time.hour)
                hourly_list = selected_day.get("hour", [])
                matched_hour = next(
                    (
                        hour_data
                        for hour_data in hourly_list
                        if str(hour_data.get("time", "")).endswith(
                            f" {target_hour:02d}:00"
                        )
                    ),
                    None,
                )
                if not matched_hour and hourly_list:
                    matched_hour = hourly_list[min(target_hour, len(hourly_list) - 1)]

                if not matched_hour:
                    return "⚠️ Không có dữ liệu theo giờ cho ngày được chọn."

                result = f"🌦️ **Forecast {WEATHER_DEFAULT_LOCATION}**\n\n"
                result += f"🗓️ {title_date} - {target_hour:02d}:00\n"
                result += f"🌡️ {matched_hour.get('temp_c')}°C\n"
                result += f"☁️ {matched_hour.get('condition', {}).get('text', '')}\n"
                result += f"💧 Độ ẩm: {matched_hour.get('humidity')}%\n"
                result += f"💨 Gió: {matched_hour.get('wind_kph')} km/h\n"
                result += f"🌧️ Khả năng mưa: {matched_hour.get('chance_of_rain', 0)}%\n"
                return result

            day = selected_day.get("day", {})
            astro = selected_day.get("astro", {})
            result = (
                f"🌤️ **Forecast ngày {title_date} - {WEATHER_DEFAULT_LOCATION}**\n\n"
            )
            result += f"🌡️ {day.get('mintemp_c')}°C - {day.get('maxtemp_c')}°C\n"
            result += f"☁️ {day.get('condition', {}).get('text', '')}\n"
            result += f"🌧️ Khả năng mưa: {day.get('daily_chance_of_rain', 0)}%\n"
            result += f"💨 Gió tối đa: {day.get('maxwind_kph', 0)} km/h\n"
            result += f"🌅 Mặt trời mọc: {astro.get('sunrise', 'N/A')}\n"
            result += f"🌇 Mặt trời lặn: {astro.get('sunset', 'N/A')}\n"
            return result
        except Exception as e:
            return f"⚠️ Lỗi: {str(e)}"

    async def get_random_cat_image(self):
        """Fetch one random cat image from TheCatAPI."""
        headers = {}
        api_key = str(THECATAPI_KEY or "").strip()
        if api_key:
            headers["x-api-key"] = api_key

        url = "https://api.thecatapi.com/v1/images/search"
        params = {"limit": 1, "size": "med"}

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status != 200:
                        return {
                            "ok": False,
                            "error": f"⚠️ TheCatAPI lỗi HTTP {response.status}.",
                        }
                    data = await response.json(content_type=None)

            if not isinstance(data, list) or not data:
                return {"ok": False, "error": "⚠️ Không nhận được ảnh mèo."}

            item = data[0] if isinstance(data[0], dict) else {}
            image_url = str(item.get("url") or "").strip()
            if not image_url:
                return {"ok": False, "error": "⚠️ Dữ liệu ảnh mèo không hợp lệ."}

            breeds = item.get("breeds") if isinstance(item.get("breeds"), list) else []
            breed_name = ""
            if breeds and isinstance(breeds[0], dict):
                breed_name = str(breeds[0].get("name") or "").strip()

            return {
                "ok": True,
                "url": image_url,
                "breed": breed_name,
            }
        except Exception as e:
            return {"ok": False, "error": f"⚠️ Không thể lấy ảnh mèo: {str(e)}"}

    async def get_events(self, date=None):
        """Fetch calendar events for a date and normalize presentation fields."""
        try:
            service = self._get_calendar_service()
            if not service:
                return "⚠️ Cần setup Google Calendar"

            if date is None:
                date = datetime.now(self.timezone).date()

            start_time = self.timezone.localize(datetime.combine(date, time.min))
            end_time = self.timezone.localize(datetime.combine(date, time.max))

            events_result = (
                service.events()
                .list(
                    calendarId="primary",
                    timeMin=start_time.isoformat(),
                    timeMax=end_time.isoformat(),
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )

            events = events_result.get("items", [])
            events_data = []

            for event in events:
                if event.get("status") == "cancelled":
                    continue

                start = event["start"].get("dateTime", event["start"].get("date"))
                summary = event.get("summary", "Không có tiêu đề")
                description = event.get("description", "")
                event_id = event.get("id")

                if "T" in start or ":" in start:
                    try:
                        event_time = datetime.fromisoformat(
                            start.replace("Z", "+00:00")
                        )
                        if event_time.tzinfo is None:
                            event_time = self.timezone.localize(event_time)
                        else:
                            event_time = event_time.astimezone(self.timezone)

                        end_dt = event["end"].get("dateTime", event["end"].get("date"))
                        end_time_obj = datetime.fromisoformat(
                            end_dt.replace("Z", "+00:00")
                        )
                        if end_time_obj.tzinfo is None:
                            end_time_obj = self.timezone.localize(end_time_obj)
                        else:
                            end_time_obj = end_time_obj.astimezone(self.timezone)

                        events_data.append(
                            {
                                "id": event_id,
                                "time": event_time.strftime("%H:%M"),
                                "end_time": end_time_obj.strftime("%H:%M"),
                                "summary": summary,
                                "description": description,
                                "datetime": event_time,
                                "end_datetime": end_time_obj,
                                "is_important": self._is_important(
                                    summary, description
                                ),
                                "sort_key": event_time,
                            }
                        )
                    except Exception:
                        continue
                else:
                    events_data.append(
                        {
                            "id": event_id,
                            "time": "Cả ngày",
                            "end_time": "",
                            "summary": summary,
                            "description": description,
                            "datetime": None,
                            "end_datetime": None,
                            "is_important": self._is_important(summary, description),
                            "sort_key": start_time,
                        }
                    )

            events_data.sort(key=lambda x: x["sort_key"])
            return events_data if events_data else None

        except Exception as e:
            return f"⚠️ Lỗi: {str(e)}"

    async def add_event(
        self, summary, start_datetime, end_datetime=None, description=""
    ):
        """Create a calendar event with optional description and default duration."""
        try:
            service = self._get_calendar_service()
            if not service:
                return "⚠️ Cần setup Google Calendar"

            if end_datetime is None:
                end_datetime = start_datetime + timedelta(hours=1)

            event = {
                "summary": summary,
                "description": description,
                "start": {
                    "dateTime": start_datetime.isoformat(),
                    "timeZone": "Asia/Ho_Chi_Minh",
                },
                "end": {
                    "dateTime": end_datetime.isoformat(),
                    "timeZone": "Asia/Ho_Chi_Minh",
                },
            }

            service.events().insert(calendarId="primary", body=event).execute()
            time_range = (
                f"{start_datetime.strftime('%H:%M')}-{end_datetime.strftime('%H:%M')}"
            )
            return f"✅ Đã thêm: {summary} ({start_datetime.strftime('%d/%m')} {time_range})"

        except Exception as e:
            return f"⚠️ Lỗi: {str(e)}"

    async def delete_event(self, event_id):
        """Delete event by id from primary calendar."""
        try:
            service = self._get_calendar_service()
            if not service:
                return "⚠️ Cần setup Google Calendar"

            service.events().delete(calendarId="primary", eventId=event_id).execute()
            return "✅ Đã xóa event"
        except Exception as e:
            return f"⚠️ Lỗi: {str(e)}"

    async def update_event(self, event_id, **kwargs):
        """Update mutable fields for an existing event."""
        try:
            service = self._get_calendar_service()
            if not service:
                return "⚠️ Cần setup Google Calendar"

            event = (
                service.events().get(calendarId="primary", eventId=event_id).execute()
            )

            if "summary" in kwargs:
                event["summary"] = kwargs["summary"]
            if "description" in kwargs:
                event["description"] = kwargs["description"]
            if "start" in kwargs:
                event["start"] = kwargs["start"]
            if "end" in kwargs:
                event["end"] = kwargs["end"]

            service.events().update(
                calendarId="primary", eventId=event_id, body=event
            ).execute()
            return "✅ Đã cập nhật event"
        except Exception as e:
            return f"⚠️ Lỗi: {str(e)}"

    async def get_tasks(self, date=None, show_completed=False):
        """Fetch tasks across tasklists and normalize due/overdue metadata."""
        try:
            service = self._get_tasks_service()
            if not service:
                return "⚠️ Cần setup Google Tasks"

            tasklists = service.tasklists().list().execute()
            all_tasks = []

            for tasklist in tasklists.get("items", []):
                tasks_result = (
                    service.tasks()
                    .list(
                        tasklist=tasklist["id"],
                        showCompleted=show_completed,
                        showHidden=False,
                    )
                    .execute()
                )

                for task in tasks_result.get("items", []):
                    due = task.get("due")
                    due_date = None
                    due_time = None
                    overdue = False

                    if due:
                        try:
                            due_dt = datetime.fromisoformat(due.replace("Z", "+00:00"))
                            if due_dt.tzinfo:
                                due_dt = due_dt.astimezone(self.timezone)
                            else:
                                due_dt = self.timezone.localize(due_dt)

                            due_date = due_dt.date()
                            due_time = due_dt.time()

                            now = datetime.now(self.timezone)
                            if due_dt < now:
                                overdue = True
                        except Exception:
                            pass

                    if date is not None and due_date != date:
                        continue

                    all_tasks.append(
                        {
                            "id": task["id"],
                            "tasklist_id": tasklist["id"],
                            "title": task.get("title", "No title"),
                            "notes": task.get("notes", ""),
                            "due": due_date,
                            "due_time": due_time,
                            "status": task.get("status"),
                            "completed": task.get("status") == "completed",
                            "overdue": overdue,
                            "tasklist_name": tasklist.get("title", "Tasks"),
                        }
                    )

            all_tasks.sort(
                key=lambda x: (
                    not x["overdue"],
                    x["due"] if x["due"] else datetime.max.date(),
                    x["due_time"] if x["due_time"] else time.max,
                )
            )

            return all_tasks
        except Exception as e:
            return f"⚠️ Lỗi: {str(e)}"

    async def add_task(self, title, due_datetime=None, notes=""):
        """Create task in default tasklist with optional due datetime and notes."""
        try:
            service = self._get_tasks_service()
            if not service:
                return "⚠️ Cần setup Google Tasks"

            tasklists = service.tasklists().list().execute()
            if not tasklists.get("items"):
                return "⚠️ Không tìm thấy tasklist"

            tasklist_id = tasklists["items"][0]["id"]
            task = {"title": title, "notes": notes}

            if due_datetime:
                task["due"] = due_datetime.isoformat()

            service.tasks().insert(tasklist=tasklist_id, body=task).execute()

            due_str = ""
            if due_datetime:
                due_str = f" (hạn: {due_datetime.strftime('%d/%m %H:%M')})"
            return f"✅ Đã thêm task: {title}{due_str}"

        except Exception as e:
            return f"⚠️ Lỗi: {str(e)}"

    async def complete_task(self, task_id, tasklist_id):
        """Mark task as completed by id and tasklist."""
        try:
            service = self._get_tasks_service()
            if not service:
                return "⚠️ Cần setup Google Tasks"

            task = service.tasks().get(tasklist=tasklist_id, task=task_id).execute()
            task["status"] = "completed"

            service.tasks().update(
                tasklist=tasklist_id, task=task_id, body=task
            ).execute()

            return f"✅ Đã hoàn thành: {task['title']}"
        except Exception as e:
            return f"⚠️ Lỗi: {str(e)}"

    async def delete_task(self, task_id, tasklist_id):
        """Delete task by id from target tasklist."""
        try:
            service = self._get_tasks_service()
            if not service:
                return "⚠️ Cần setup Google Tasks"

            service.tasks().delete(tasklist=tasklist_id, task=task_id).execute()
            return "✅ Đã xóa task"
        except Exception as e:
            return f"⚠️ Lỗi: {str(e)}"

    async def get_calendar(self, date=None):
        """Return combined calendar payload including events and tasks."""
        events = await self.get_events(date)
        tasks = await self.get_tasks(date, show_completed=False)

        return {
            "events": events if isinstance(events, list) else [],
            "tasks": tasks if isinstance(tasks, list) else [],
        }

    def _is_important(self, summary, description):
        """Heuristically detect important items from keywords."""
        keywords = [
            "deadline",
            "exam",
            "thi",
            "nộp",
            "họp",
            "meeting",
            "interview",
            "phỏng vấn",
            "presentation",
            "thuyết trình",
            "important",
            "quan trọng",
            "urgent",
            "gấp",
        ]
        text = f"{summary} {description}".lower()
        return any(kw in text for kw in keywords)

    def check_holiday(self, date=None):
        """Check static holiday map for given date."""
        if date is None:
            date = datetime.now(self.timezone).date()
        elif isinstance(date, datetime):
            date = date.date()
        date_key = date.strftime("%m-%d")
        return VIETNAM_HOLIDAYS.get(date_key)

    def get_next_tet_datetime(self, from_datetime=None):
        """Get next known lunar new year datetime from lookup table."""
        if from_datetime is None:
            from_datetime = datetime.now(self.timezone)

        for year in sorted(LUNAR_TET_DATES.keys()):
            month, day = LUNAR_TET_DATES[year]
            tet_datetime = self.timezone.localize(datetime(year, month, day, 0, 0, 0))
            if tet_datetime > from_datetime:
                return year, tet_datetime

        return None, None

    def add_countdown(self, name, target_datetime, emoji="🎉", label=""):
        """Create or overwrite an active countdown entry."""
        if not isinstance(target_datetime, datetime):
            return False

        if target_datetime.tzinfo is None:
            target_datetime = self.timezone.localize(target_datetime)

        _active_countdowns[name] = {
            "datetime": target_datetime,
            "emoji": emoji,
            "name": name,
            "label": label,
            "milestones": self._default_countdown_milestones(),
            "notified": set(),
            "last_remaining": None,
        }
        return True

    def remove_countdown(self, name):
        """Remove countdown by name, returning whether removal succeeded."""
        if name in _active_countdowns:
            del _active_countdowns[name]
            return True
        return False

    def get_countdowns(self):
        """Return computed countdown status list for display."""
        now = datetime.now(self.timezone)
        result = []

        for name, data in _active_countdowns.items():
            target = data["datetime"]
            remaining = (target - now).total_seconds()

            if remaining < 0:
                status = "ĐÃ QUA"
                time_str = ""
            else:
                days = int(remaining // 86400)
                hours = int((remaining % 86400) // 3600)
                minutes = int((remaining % 3600) // 60)
                seconds = int(remaining % 60)

                if days > 0:
                    time_str = f"{days}d {hours}h {minutes}m"
                elif hours > 0:
                    time_str = f"{hours}h {minutes}m {seconds}s"
                elif minutes > 0:
                    time_str = f"{minutes}m {seconds}s"
                else:
                    time_str = f"{seconds}s"

                status = "ACTIVE"

            result.append(
                {
                    "name": name,
                    "emoji": data["emoji"],
                    "target": target,
                    "remaining_seconds": remaining,
                    "time_str": time_str,
                    "status": status,
                }
            )

        return result

    def format_countdown_message(self, name, remaining_seconds):
        """Build countdown notification message for milestone crossing."""
        if name not in _active_countdowns:
            return None

        data = _active_countdowns[name]
        emoji = data["emoji"]
        countdown_name = data["name"]
        mention = f"<@{YOUR_USER_ID}>" if YOUR_USER_ID else ""
        is_newyear = "newyear" in data.get("label", "").lower()

        if is_newyear:
            if remaining_seconds >= 120:
                minutes = int(remaining_seconds // 60)
                return f"{emoji} **{countdown_name}**\n⏰ Còn **{minutes} phút**!"
            elif remaining_seconds >= 60:
                return (
                    f"{emoji} **{countdown_name}**\n"
                    f"🔥 **COUNTDOWN BẮT ĐẦU!** 🔥\n"
                    f"⏰ Còn **{int(remaining_seconds)} GIÂY**! 🎊"
                )
            elif remaining_seconds > 0:
                seconds = int(remaining_seconds)
                return f"🎇 **{seconds}** 🎇"
            elif abs(remaining_seconds) < 3:
                year = data["datetime"].year
                return (
                    f"{mention}\n"
                    f"🎆🎆🎆🎆🎆🎆🎆🎆🎆🎆\n"
                    f"🎊 **CHÚC MỪNG NĂM MỚI {year}!** 🎊\n"
                    f"🎉 HAPPY NEW YEAR {year}! 🎉\n"
                    f"✨ Chúc mọi người năm mới an khang thịnh vượng! ✨\n"
                    f"🎆🎆🎆🎆🎆🎆🎆🎆🎆🎆"
                )
        else:
            if remaining_seconds >= 120:
                minutes = int(remaining_seconds // 60)
                return f"{emoji} **{countdown_name}**\n⏰ Còn **{minutes} phút**"
            elif remaining_seconds >= 60:
                return f"{emoji} **{countdown_name}**\n⏰ Còn **{int(remaining_seconds)} giây**"
            elif remaining_seconds > 0:
                seconds = int(remaining_seconds)
                return f"{emoji} **{countdown_name}**\n⏰ **{seconds} giây**"
            elif abs(remaining_seconds) < 3:
                return (
                    f"{mention}\n"
                    f"{emoji * 5}\n"
                    f"🎊 **{countdown_name}** 🎊\n"
                    f"🎉 ĐÃ ĐẾN! 🎉\n"
                    f"{emoji * 5}"
                )

        return None

    async def summarize_daily_knowledge(
        self, messages, channel_name="", offset=0, batch_size=50
    ):
        """Summarize batched daily messages into key points and review questions."""
        if not messages:
            return None, False

        safe_batch_size = max(1, int(batch_size or 1))
        total = len(messages)
        start_idx = offset
        end_idx = min(offset + safe_batch_size, total)

        batch_messages = messages[start_idx:end_idx]
        has_more = end_idx < total

        compact_messages = []
        max_chars_per_message = 650
        for msg in batch_messages:
            row = str(msg or "").strip()
            if not row:
                continue
            if len(row) > max_chars_per_message:
                row = row[: max_chars_per_message - 1].rstrip() + "…"
            compact_messages.append(f"- {row}")

        message_text = "\n".join(compact_messages)
        max_prompt_chars = 14000
        if len(message_text) > max_prompt_chars:
            message_text = (
                message_text[: max_prompt_chars - 50].rstrip()
                + "\n- ...(đã rút gọn để tránh quá dài trong 1 lượt summary)"
            )

        progress_info = f"Tổng hợp {start_idx + 1}-{end_idx}/{total} tin nhắn"
        if channel_name:
            progress_info += f" từ #{channel_name}"

        ai_result = await self._call_ai_with_fallback(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là trợ lý học tập. Trả về JSON hợp lệ với format:\n"
                        '{"summary_points": ["..."], "detailed_summary": "...", "review_questions": ["..."]}\n'
                        "- summary_points: 6-10 ý chính, rõ ý\n"
                        "- detailed_summary: phân tích sâu, có cấu trúc, giải thích đủ dài nhưng <= 2500 ký tự\n"
                        "- review_questions: 3-5 câu hỏi kiểm tra hiểu bài, mỗi câu <= 180 ký tự"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{progress_info}\n\n{message_text}\n\n"
                        "Yêu cầu đầu ra chi tiết:"
                        "\n1) Tóm tắt đầy đủ kiến thức chính, tránh quá ngắn."
                        "\n2) Phần detailed_summary phải có tiêu đề nhỏ theo chủ đề,"
                        " nêu khái niệm, quy trình, ví dụ, lỗi thường gặp nếu có."
                        "\n3) Viết bằng tiếng Việt rõ ràng, dễ học lại."
                    ),
                },
            ],
            primary_model=SUMMARY_MODEL_PRIMARY,
            fallback_models=SUMMARY_MODEL_FALLBACKS,
            temperature=0.1,
            max_tokens=SUMMARY_MAX_OUTPUT_TOKENS,
        )

        if not ai_result["ok"]:
            return {"error": f"⚠️ Lỗi API: {ai_result['error']}"}, False

        parsed = self._extract_json_block(ai_result["content"])
        if not parsed:
            return {
                "error": "⚠️ Không parse được kết quả summary từ model",
                "raw": ai_result["content"],
                "model": ai_result["model"],
            }, False

        summary_points = parsed.get("summary_points", [])
        detailed_summary = str(parsed.get("detailed_summary", "")).strip()
        review_questions = parsed.get("review_questions", [])

        if not isinstance(summary_points, list):
            summary_points = []
        if not isinstance(review_questions, list):
            review_questions = []

        summary_points = [str(x).strip() for x in summary_points if str(x).strip()][:10]
        review_questions = [str(x).strip() for x in review_questions if str(x).strip()][
            :5
        ]
        if len(detailed_summary) > 6000:
            detailed_summary = detailed_summary[:5999].rstrip() + "…"

        return {
            "summary_points": summary_points,
            "detailed_summary": detailed_summary,
            "review_questions": review_questions,
            "model": ai_result["model"],
            "processed_count": len(batch_messages),
        }, has_more

    async def expand_summary_analysis(
        self,
        channel_name,
        summary_points,
        detailed_summary,
        review_questions,
    ):
        """Ask model to produce deeper structured analysis from summary artifacts."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý học tập. Mục tiêu: mở rộng phần tóm tắt hiện có thành"
                    " phiên bản sâu hơn, có hệ thống, dễ ôn tập."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Channel: #{channel_name}\n"
                    f"Ý chính hiện có:\n{chr(10).join([f'- {p}' for p in (summary_points or [])])}\n\n"
                    f"Phân tích hiện có:\n{detailed_summary}\n\n"
                    f"Câu hỏi ôn tập:\n{chr(10).join([f'- {q}' for q in (review_questions or [])])}\n\n"
                    "Hãy viết phần phân tích sâu hơn, dài hơn, có cấu trúc rõ ràng"
                    " (khái niệm, mối liên hệ, ví dụ, checklist ôn tập nhanh)."
                ),
            },
        ]

        ai_result = await self._call_ai_with_fallback(
            messages=messages,
            primary_model=SUMMARY_MODEL_PRIMARY,
            fallback_models=SUMMARY_MODEL_FALLBACKS,
            temperature=0.2,
            max_tokens=SUMMARY_MAX_OUTPUT_TOKENS,
        )

        if not ai_result.get("ok"):
            return {
                "ok": False,
                "error": ai_result.get("error", "Unknown error"),
                "model": ai_result.get("model"),
            }

        return {
            "ok": True,
            "content": (ai_result.get("content") or "").strip(),
            "model": ai_result.get("model"),
        }

    async def review_study_answer(self, question, user_answer, summary_points=None):
        """Review user answer and return scored feedback payload."""
        summary_context = "\n".join([f"- {p}" for p in (summary_points or [])])
        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là gia sư chấm bài ngắn gọn."
                    'Hãy trả JSON: {"score": <0-10>, "feedback": "...", "suggestion": "..."}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Câu hỏi: {question}\n"
                    f"Câu trả lời của học viên: {user_answer}\n"
                    f"Ngữ cảnh tóm tắt (nếu có):\n{summary_context}"
                ),
            },
        ]

        ai_result = await self._call_ai_with_fallback(
            messages,
            ANSWER_MODEL_PRIMARY,
            ANSWER_MODEL_FALLBACKS,
            temperature=0.1,
            max_tokens=MAX_OUTPUT_TOKENS,
        )

        if not ai_result["ok"]:
            return {
                "ok": False,
                "error": f"⚠️ Lỗi API: {ai_result['error']}",
                "model": None,
            }

        parsed = self._extract_json_block(ai_result["content"]) or {}
        score = parsed.get("score", "?")
        feedback = parsed.get("feedback", ai_result["content"][:700])
        suggestion = parsed.get("suggestion", "")

        return {
            "ok": True,
            "score": score,
            "feedback": feedback,
            "suggestion": suggestion,
            "model": ai_result["model"],
        }
