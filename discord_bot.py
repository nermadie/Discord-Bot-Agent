import discord
from discord import app_commands
from discord.ext import commands, tasks
import os
import asyncio
from datetime import datetime, time, timedelta, timezone
import pytz
import aiohttp
import json
import sqlite3
import random
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from dotenv import load_dotenv
import re

from tools.constants import (
    SUMMARY_BATCH_SIZE,
    SUMMARY_FETCH_MAX_MESSAGES,
    WEATHER_DEFAULT_LOCATION,
    WEATHER_FORECAST_MAX_DAYS,
)
from tools.embed_builders import (
    build_calendar_embed,
    build_events_embed,
    build_tasks_embed,
)
from tools import study_memory


load_dotenv()


def _parse_model_fallbacks(env_key, default_csv):
    raw = os.getenv(env_key, default_csv)
    return [item.strip() for item in str(raw).split(",") if item.strip()]


# ==============================
# CONFIG
# ==============================
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_MODEL = os.getenv("GITHUB_MODEL", "openai/gpt-4o-mini")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
WEATHER_PROVIDER = os.getenv("WEATHER_PROVIDER", "weatherapi")
YOUR_USER_ID = int(os.getenv("YOUR_USER_ID", "0"))
MAIN_CHANNEL_ID = int(os.getenv("CHANNEL_MAIN", "0"))
APP_GUILD_ID = int(os.getenv("APP_GUILD_ID", "0"))

CHAT_MODEL_PRIMARY = os.getenv("CHAT_MODEL_PRIMARY", "openai/gpt-5")
CHAT_MODEL_FALLBACKS = _parse_model_fallbacks(
    "CHAT_MODEL_FALLBACKS", "openai/gpt-5-mini,openai/gpt-5-nano,openai/gpt-4o"
)

VISION_MODEL_PRIMARY = os.getenv(
    "VISION_MODEL_PRIMARY", "meta/Llama-4-Maverick-17B-128E-Instruct-FP8"
)
VISION_MODEL_FALLBACKS = _parse_model_fallbacks(
    "VISION_MODEL_FALLBACKS", "openai/gpt-4.1-nano,openai/gpt-4o-mini"
)

SUMMARY_MODEL_PRIMARY = os.getenv("SUMMARY_MODEL_PRIMARY", "openai/gpt-5-chat")
SUMMARY_MODEL_FALLBACKS = _parse_model_fallbacks(
    "SUMMARY_MODEL_FALLBACKS", "openai/gpt-5-mini,openai/gpt-5-nano,openai/gpt-4o"
)

ANSWER_MODEL_PRIMARY = os.getenv("ANSWER_MODEL_PRIMARY", "openai/gpt-5-chat")
ANSWER_MODEL_FALLBACKS = _parse_model_fallbacks(
    "ANSWER_MODEL_FALLBACKS", "openai/gpt-5-mini,openai/gpt-5-nano,openai/gpt-4o"
)

REASONING_MODEL_PRIMARY = os.getenv(
    "REASONING_MODEL_PRIMARY", "deepseek/DeepSeek-R1-0528"
)
REASONING_MODEL_FALLBACKS = _parse_model_fallbacks(
    "REASONING_MODEL_FALLBACKS",
    "microsoft/Phi-4-reasoning,microsoft/Phi-4-mini-reasoning",
)

MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "0"))
SUMMARY_MAX_OUTPUT_TOKENS = int(os.getenv("SUMMARY_MAX_OUTPUT_TOKENS", "16000"))
AI_REQUEST_TIMEOUT_SECONDS = int(os.getenv("AI_REQUEST_TIMEOUT_SECONDS", "45"))
REASONING_REQUEST_TIMEOUT_SECONDS = int(
    os.getenv("REASONING_REQUEST_TIMEOUT_SECONDS", "90")
)
STUDY_POINTS_PASS = int(os.getenv("STUDY_POINTS_PASS", "10"))
STUDY_POINTS_MISS = int(os.getenv("STUDY_POINTS_MISS", "3"))
STUDY_PASS_THRESHOLD = float(os.getenv("STUDY_PASS_THRESHOLD", "7"))
STUDY_METRICS_DIR = os.getenv("STUDY_METRICS_DIR", "study_metrics")
SLOGAN_IDLE_MINUTES = int(os.getenv("SLOGAN_IDLE_MINUTES", "180"))
SLOGAN_CHECK_INTERVAL_MINUTES = int(os.getenv("SLOGAN_CHECK_INTERVAL_MINUTES", "30"))

CHANNELS_TO_MONITOR_STR = os.getenv("CHANNELS_TO_MONITOR", "")
CHANNELS_TO_MONITOR = [
    int(ch.strip()) for ch in CHANNELS_TO_MONITOR_STR.split(",") if ch.strip()
]

VIETNAM_TZ = timezone(timedelta(hours=7))


# ==============================
# DISCORD SETUP
# ==============================
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
bot = commands.Bot(command_prefix=commands.when_mentioned, intents=intents)
bot.remove_command("help")


# Runtime state
daily_messages = {}
summary_state = {}
_last_tasks = {}
_last_events = {}
_active_countdowns = {}
_sent_upcoming_reminders = set()
_study_questions = {}
_chat_sessions = {}
_pending_chat_context = {}
_daily_messages_date = None
_last_interaction_at = {}
_last_slogan_sent_at = {}
_last_summary_fetch_message_ids = {}


def _get_summary_channel_option_items():
    items = [("All monitored channels", "all")]
    for channel_id in CHANNELS_TO_MONITOR:
        if channel_id == MAIN_CHANNEL_ID:
            continue
        channel = bot.get_channel(channel_id)
        label = f"#{channel.name}" if channel else f"channel-{channel_id}"
        items.append((label, str(channel_id)))
    return items


async def summary_channel_autocomplete(interaction: discord.Interaction, current: str):
    current_text = (current or "").lower().strip()
    choices = []
    for name, value in _get_summary_channel_option_items():
        if not current_text or current_text in name.lower() or current_text in value:
            choices.append(app_commands.Choice(name=name, value=value))
    return choices[:25]


def _today_vn_date():
    return datetime.now(VIETNAM_TZ).date()


def _ensure_daily_window_rollover():
    global _daily_messages_date
    today = _today_vn_date()
    if _daily_messages_date is None:
        _daily_messages_date = today
        return
    if _daily_messages_date != today:
        daily_messages.clear()
        summary_state.clear()
        _sent_upcoming_reminders.clear()
        _daily_messages_date = today


def _mark_user_interaction(user_id):
    _last_interaction_at[int(user_id)] = datetime.now(VIETNAM_TZ)


def _build_study_status_text(user_id):
    profile = _get_or_create_study_profile(user_id)
    return (
        "📈 **Tình hình học tập hôm nay**\n"
        f"• ⭐ Điểm: **{profile.get('total_points', 0)}**\n"
        f"• 🔥 Streak: **{profile.get('streak_days', 0)} ngày**\n"
        f"• ✅ Trả lời đạt: **{profile.get('passed_count', 0)}**\n"
        f"• ❌ Bỏ lỡ: **{profile.get('missed_count', 0)}**"
    )


async def _fetch_motivational_slogan():
    fallback = [
        "Hôm nay khó một chút, ngày mai bạn mạnh hơn rất nhiều.",
        "Mỗi lần học thêm 1%, bạn đang vượt qua phiên bản cũ của mình.",
        "Không cần hoàn hảo, chỉ cần đều đặn.",
        "Đi chậm vẫn hơn đứng yên.",
        "Kỷ luật hôm nay là tự do ngày mai.",
    ]

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=8)
        ) as session:
            async with session.get("https://zenquotes.io/api/random") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, list) and data:
                        quote = str(data[0].get("q", "")).strip()
                        author = str(data[0].get("a", "")).strip()
                        if quote:
                            return f"{quote} — {author}" if author else quote
    except Exception:
        pass

    return random.choice(fallback)


async def _collect_channel_messages_for_summary(channel, latest_messages=50):
    limit = max(1, min(int(latest_messages or 50), 500))
    collected = []
    async for msg in channel.history(limit=limit):
        if msg.author.bot:
            continue
        created_local = msg.created_at.astimezone(VIETNAM_TZ)
        timestamp = created_local.strftime("%H:%M")
        attachment_context = _attachment_context_for_summary(msg)
        collected.append(
            f"[{timestamp}] {msg.author.name}: {msg.content}{attachment_context}"
        )
    collected.reverse()
    return collected


async def _collect_new_messages_since(
    channel,
    after_message_id=None,
    latest_messages=SUMMARY_FETCH_MAX_MESSAGES,
    only_today=False,
):
    limit = max(
        1,
        min(
            int(latest_messages or SUMMARY_FETCH_MAX_MESSAGES),
            SUMMARY_FETCH_MAX_MESSAGES,
        ),
    )
    kwargs = {"limit": 500}
    if after_message_id:
        kwargs["after"] = discord.Object(id=int(after_message_id))

    rows = []
    today_vn = _today_vn_date()
    async for msg in channel.history(**kwargs):
        if msg.author.bot:
            continue

        created_local = msg.created_at.astimezone(VIETNAM_TZ)
        if only_today and created_local.date() != today_vn:
            if created_local.date() < today_vn:
                break
            continue

        timestamp = created_local.strftime("%H:%M")
        attachment_context = _attachment_context_for_summary(msg)
        rows.append(
            {
                "id": int(msg.id),
                "text": f"[{timestamp}] {msg.author.name}: {msg.content}{attachment_context}",
            }
        )

    rows.sort(key=lambda x: x["id"])
    if len(rows) > limit:
        rows = rows[-limit:]

    messages = [item["text"] for item in rows]
    newest_id = rows[-1]["id"] if rows else None
    return messages, newest_id


def _metrics_dir_path():
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, STUDY_METRICS_DIR)


def _metrics_db_path(target_date=None):
    target_date = target_date or datetime.now(VIETNAM_TZ)
    month_key = target_date.strftime("%Y-%m")
    return os.path.join(_metrics_dir_path(), f"study_metrics_{month_key}.sqlite3")


def _ensure_metrics_db(target_date=None):
    os.makedirs(_metrics_dir_path(), exist_ok=True)
    db_path = _metrics_db_path(target_date)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS study_profile (
            user_id INTEGER PRIMARY KEY,
            total_points INTEGER NOT NULL DEFAULT 0,
            streak_days INTEGER NOT NULL DEFAULT 0,
            last_streak_date TEXT,
            answered_count INTEGER NOT NULL DEFAULT 0,
            passed_count INTEGER NOT NULL DEFAULT 0,
            missed_count INTEGER NOT NULL DEFAULT 0,
            summaries_count INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS study_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            points_delta INTEGER NOT NULL DEFAULT 0,
            question_index INTEGER,
            channel_name TEXT,
            score REAL,
            note TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()
    return db_path


def _get_or_create_study_profile(user_id, target_date=None):
    db_path = _ensure_metrics_db(target_date)
    now_iso = datetime.now(VIETNAM_TZ).isoformat()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM study_profile WHERE user_id = ?", (int(user_id),))
    row = cur.fetchone()
    if row is None:
        cur.execute(
            """
            INSERT INTO study_profile (
                user_id, total_points, streak_days, last_streak_date,
                answered_count, passed_count, missed_count, summaries_count, updated_at
            ) VALUES (?, 0, 0, NULL, 0, 0, 0, 0, ?)
            """,
            (int(user_id), now_iso),
        )
        conn.commit()
        cur.execute("SELECT * FROM study_profile WHERE user_id = ?", (int(user_id),))
        row = cur.fetchone()
    conn.close()
    return dict(row) if row else {}


def _update_streak(profile, event_dt):
    profile = dict(profile or {})
    event_date = event_dt.date()
    last_date_str = profile.get("last_streak_date")
    current_streak = int(profile.get("streak_days") or 0)

    if not last_date_str:
        return 1, event_date.isoformat()

    try:
        last_date = datetime.fromisoformat(str(last_date_str)).date()
    except Exception:
        return 1, event_date.isoformat()

    delta_days = (event_date - last_date).days
    if delta_days == 0:
        return current_streak, last_date_str
    if delta_days == 1:
        return max(0, current_streak) + 1, event_date.isoformat()
    return 1, event_date.isoformat()


def _append_study_event(
    user_id,
    event_type,
    points_delta=0,
    question_index=None,
    channel_name="",
    score=None,
    note="",
    target_date=None,
):
    db_path = _ensure_metrics_db(target_date)
    now_dt = datetime.now(VIETNAM_TZ)
    now_iso = now_dt.isoformat()
    profile = _get_or_create_study_profile(user_id, target_date)

    total_points = int(profile.get("total_points") or 0) + int(points_delta)
    answered_count = int(profile.get("answered_count") or 0)
    passed_count = int(profile.get("passed_count") or 0)
    missed_count = int(profile.get("missed_count") or 0)
    summaries_count = int(profile.get("summaries_count") or 0)

    if event_type == "answer":
        answered_count += 1
    elif event_type == "pass":
        answered_count += 1
        passed_count += 1
    elif event_type == "missed":
        missed_count += 1
    elif event_type == "summary":
        summaries_count += 1

    streak_days, last_streak_date = _update_streak(profile, now_dt)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO study_events (
            user_id, event_type, points_delta, question_index,
            channel_name, score, note, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(user_id),
            str(event_type),
            int(points_delta),
            question_index,
            (channel_name or "")[:200],
            score,
            (note or "")[:1000],
            now_iso,
        ),
    )
    cur.execute(
        """
        UPDATE study_profile
        SET total_points = ?, streak_days = ?, last_streak_date = ?,
            answered_count = ?, passed_count = ?, missed_count = ?,
            summaries_count = ?, updated_at = ?
        WHERE user_id = ?
        """,
        (
            total_points,
            int(streak_days),
            last_streak_date,
            answered_count,
            passed_count,
            missed_count,
            summaries_count,
            now_iso,
            int(user_id),
        ),
    )
    conn.commit()
    conn.close()

    return {
        "total_points": total_points,
        "streak_days": int(streak_days),
        "answered_count": answered_count,
        "passed_count": passed_count,
        "missed_count": missed_count,
        "summaries_count": summaries_count,
        "db_path": db_path,
    }


def _normalize_score_value(score):
    if score is None:
        return None
    try:
        return float(str(score).replace(",", ".").strip())
    except Exception:
        return None


def _mark_question_answered(user_id, question_index):
    question_bank = _study_questions.get(user_id, [])
    target = next((q for q in question_bank if q.get("index") == question_index), None)
    if target:
        target["answered"] = True
    return target


def _apply_unanswered_penalty(user_id):
    question_bank = list(_study_questions.get(user_id, []))
    unanswered = [q for q in question_bank if not q.get("answered")]
    if not unanswered:
        return {"applied": False, "count": 0, "points_delta": 0}

    _mark_spaced_unanswered(user_id, unanswered)

    penalty_each = abs(int(STUDY_POINTS_MISS))
    points_delta = -(penalty_each * len(unanswered))
    stats = _append_study_event(
        user_id=user_id,
        event_type="missed",
        points_delta=points_delta,
        note=f"Bỏ lỡ {len(unanswered)} câu hỏi ôn tập chưa trả lời.",
    )
    return {
        "applied": True,
        "count": len(unanswered),
        "points_delta": points_delta,
        "stats": stats,
    }


def _build_study_metrics_embed(user_id, username):
    profile = _get_or_create_study_profile(user_id)
    month_label = datetime.now(VIETNAM_TZ).strftime("%m/%Y")

    embed = discord.Embed(
        title="🔥 Study Streak",
        color=discord.Color.gold(),
        timestamp=datetime.now(VIETNAM_TZ),
    )
    embed.add_field(name="👤 User", value=str(username), inline=True)
    embed.add_field(name="🗓️ Tháng", value=month_label, inline=True)
    embed.add_field(
        name="⭐ Điểm", value=str(profile.get("total_points", 0)), inline=True
    )
    embed.add_field(
        name="🔥 Streak", value=f"{profile.get('streak_days', 0)} ngày", inline=True
    )
    embed.add_field(
        name="✅ Trả lời đạt", value=str(profile.get("passed_count", 0)), inline=True
    )
    embed.add_field(
        name="🧪 Tổng trả lời", value=str(profile.get("answered_count", 0)), inline=True
    )
    embed.add_field(
        name="❌ Bỏ lỡ", value=str(profile.get("missed_count", 0)), inline=True
    )
    embed.add_field(
        name="📚 Lần summary", value=str(profile.get("summaries_count", 0)), inline=True
    )
    embed.set_footer(text=f"DB: {_metrics_db_path()}")
    return embed


def _ensure_study_memory_tables():
    db_path = _metrics_db_path()
    study_memory.ensure_study_tables(db_path)
    return db_path


def _build_question_theory_text(summary_points, detailed_summary):
    points = [str(item).strip() for item in (summary_points or []) if str(item).strip()]
    theory = str(detailed_summary or "").strip()
    lines = []
    if points:
        lines.append("Ý chính:")
        lines.extend([f"- {item}" for item in points[:10]])
    if theory:
        lines.append("Phân tích:")
        lines.append(theory[:3000])
    return "\n".join(lines).strip()


def _persist_questions_for_spaced_repetition(
    user_id, channel_name, summary_data, numbered_questions
):
    if not numbered_questions:
        return

    db_path = _ensure_study_memory_tables()
    theory_text = _build_question_theory_text(
        summary_data.get("summary_points", []),
        summary_data.get("detailed_summary", ""),
    )
    topic = (channel_name or "").strip().lower()

    for item in numbered_questions:
        question_text = str(item.get("question", "")).strip()
        if not question_text:
            continue
        study_memory.upsert_card(
            db_path=db_path,
            user_id=int(user_id),
            channel_name=channel_name,
            question=question_text,
            theory=theory_text,
            topic=topic,
        )


def _record_spaced_review(
    user_id, target_question, score_value, answered=True, note=""
):
    db_path = _ensure_study_memory_tables()
    return study_memory.record_review(
        db_path=db_path,
        user_id=int(user_id),
        channel_name=str(target_question.get("channel_name", "")),
        question=str(target_question.get("question", "")),
        theory=str(target_question.get("theory", "")),
        score=score_value,
        answered=answered,
        note=note,
    )


def _mark_spaced_unanswered(user_id, unanswered_items):
    db_path = _ensure_study_memory_tables()
    study_memory.mark_unanswered_cards(
        db_path=db_path,
        user_id=int(user_id),
        pending_items=unanswered_items,
    )


def _split_text_chunks(text, chunk_size=1800):
    content = str(text or "")
    return [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]


def _is_table_like_line(line):
    s = str(line or "").strip()
    if not s:
        return False
    if s.count("|") >= 2:
        return True
    return bool(re.match(r"^[\s\-|:]{6,}$", s))


def _split_table_cells(line):
    s = str(line or "").strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [cell.strip() for cell in s.split("|")]


def _is_separator_row(cells):
    if not cells:
        return False
    normalized = [re.sub(r"\s+", "", str(c)) for c in cells]
    if not any(normalized):
        return False
    return all(bool(re.fullmatch(r"[-:]+", c or "")) for c in normalized if c != "")


def _render_table_block(lines):
    rows = [_split_table_cells(line) for line in (lines or []) if str(line).strip()]
    rows = [row for row in rows if any(str(c).strip() for c in row)]
    if not rows:
        return "\n".join(lines)

    rows = [row for row in rows if not _is_separator_row(row)]
    if len(rows) < 2:
        return "\n".join(lines)

    col_count = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (col_count - len(row)) for row in rows]
    widths = []
    for idx in range(col_count):
        widths.append(max(len(str(row[idx])) for row in normalized_rows))

    def format_row(row):
        return (
            "| "
            + " | ".join(str(row[idx]).ljust(widths[idx]) for idx in range(col_count))
            + " |"
        )

    header = format_row(normalized_rows[0])
    sep = (
        "| " + " | ".join("-" * max(3, widths[idx]) for idx in range(col_count)) + " |"
    )
    body = [format_row(row) for row in normalized_rows[1:]]

    return "```text\n" + "\n".join([header, sep] + body) + "\n```"


def _stylize_line(line):
    raw = str(line or "")
    stripped = raw.strip()
    if not stripped:
        return raw

    heading_match = re.match(r"^#{1,6}\s+(.+)$", stripped)
    if heading_match:
        title = heading_match.group(1).strip()
        return f"**{title}**"

    if (
        re.match(r"^\d+\.\s+.+$", stripped)
        and "|" not in stripped
        and len(stripped) <= 120
    ):
        return f"**{stripped}**"

    if stripped.endswith(":") and len(stripped) <= 100 and "|" not in stripped:
        return f"***{stripped}***"

    if re.search(r"\b(ví dụ|example)\b", stripped, flags=re.IGNORECASE):
        return f"*{stripped}*"

    return raw


def _format_rich_text_for_discord(text):
    content = str(text or "").strip()
    if not content:
        return ""

    segments = re.split(r"(```[\s\S]*?```)", content)
    output_segments = []

    for segment in segments:
        if not segment:
            continue
        if segment.startswith("```") and segment.endswith("```"):
            output_segments.append(segment)
            continue

        lines = segment.splitlines()
        out_lines = []
        idx = 0
        while idx < len(lines):
            if _is_table_like_line(lines[idx]):
                start = idx
                while idx < len(lines) and _is_table_like_line(lines[idx]):
                    idx += 1
                block = lines[start:idx]
                if len(block) >= 2:
                    out_lines.append(_render_table_block(block))
                else:
                    out_lines.extend([_stylize_line(item) for item in block])
                continue

            out_lines.append(_stylize_line(lines[idx]))
            idx += 1

        output_segments.append("\n".join(out_lines).strip())

    return "\n\n".join([item for item in output_segments if item]).strip()


def _build_reason_single_message(prompt, answer_text, model_used):
    lines = [
        "🧩 **Reasoning Assistant**",
        f"📝 **Bài toán:** **{prompt}**",
        "",
        "**Phân tích:**",
        str(answer_text or "").strip(),
        "",
        f"**Model:** {model_used}",
    ]

    return "\n".join(lines).strip()


async def _safe_followup_send(
    interaction: discord.Interaction,
    content: str = None,
    embed: discord.Embed = None,
    view: discord.ui.View = None,
    ephemeral: bool = False,
):
    kwargs = {"ephemeral": ephemeral}
    if content is not None:
        kwargs["content"] = content
    if embed is not None:
        kwargs["embed"] = embed
    if view is not None:
        kwargs["view"] = view

    async def _fallback_channel_send():
        channel = interaction.channel
        if channel is None:
            raise

        if ephemeral:
            fallback_content = content or "⚠️ Không thể gửi ephemeral response."
            return await channel.send(f"{interaction.user.mention} {fallback_content}")

        channel_kwargs = {}
        if content is not None:
            channel_kwargs["content"] = content
        if embed is not None:
            channel_kwargs["embed"] = embed
        if view is not None:
            channel_kwargs["view"] = view
        return await channel.send(**channel_kwargs)

    for attempt in range(3):
        try:
            return await interaction.followup.send(**kwargs)
        except discord.NotFound:
            return await _fallback_channel_send()
        except discord.HTTPException as http_err:
            if http_err.status == 429 and attempt < 2:
                retry_after = getattr(http_err, "retry_after", None)
                wait_seconds = (
                    float(retry_after)
                    if retry_after is not None
                    else 1.5 * (attempt + 1)
                )
                await asyncio.sleep(max(0.5, min(wait_seconds, 8.0)))
                continue

            if http_err.status == 429:
                try:
                    return await _fallback_channel_send()
                except Exception:
                    return None
            raise

    return None


def _extract_image_urls_from_attachments(attachments):
    image_urls = []
    for attachment in attachments or []:
        content_type = (attachment.content_type or "").lower()
        filename = (attachment.filename or "").lower()
        if content_type.startswith("image/") or filename.endswith(
            (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
        ):
            image_urls.append(attachment.url)
    return image_urls


def _attachment_context_for_summary(message):
    attachments = list(message.attachments or [])
    if not attachments:
        return ""

    image_urls = _extract_image_urls_from_attachments(attachments)
    file_names = [a.filename for a in attachments if a.filename]

    parts = []
    if image_urls:
        parts.append("Ảnh: " + ", ".join(image_urls[:3]))
    if file_names:
        parts.append("File: " + ", ".join(file_names[:3]))

    return " | " + " | ".join(parts) if parts else ""


def _build_summary_embed(
    channel_name, total_messages, summary_data, question_start_index=1
):
    summary_data = summary_data or {}
    summary_points = list(summary_data.get("summary_points") or [])
    questions = list(
        summary_data.get("review_questions")
        or summary_data.get("study_questions")
        or []
    )
    detailed_summary = str(summary_data.get("detailed_summary") or "").strip()
    model_used = summary_data.get("model") or "unknown"

    embed = discord.Embed(
        title=f"📚 Tổng kết #{channel_name}",
        color=0x2ECC71,
        timestamp=datetime.now(VIETNAM_TZ),
    )
    embed.add_field(name="🧾 Số tin nhắn", value=str(total_messages), inline=True)
    embed.add_field(name="🤖 Model", value=str(model_used), inline=True)

    if summary_points:
        summary_lines = [f"• {str(item)}" for item in summary_points[:10]]
        embed.add_field(
            name="✨ Ý chính",
            value="\n".join(summary_lines)[:1024],
            inline=False,
        )

    if detailed_summary:
        embed.add_field(
            name="📖 Phân tích sâu (rút gọn)",
            value=detailed_summary[:1024],
            inline=False,
        )

    numbered_questions = []
    if questions:
        question_lines = []
        for idx, question in enumerate(questions[:5], start=question_start_index):
            question_text = str(question).strip()
            if not question_text:
                continue
            numbered_questions.append({"index": idx, "question": question_text})
            question_lines.append(f"{idx}. {question_text}")

        if question_lines:
            embed.add_field(
                name="📝 Câu hỏi ôn tập",
                value="\n".join(question_lines)[:1024],
                inline=False,
            )

    embed.set_footer(text=f"Model: {model_used}")
    return embed, numbered_questions


def _build_study_context_text(channel_name, summary_data, numbered_questions=None):
    numbered_questions = numbered_questions or []
    summary_points = list(summary_data.get("summary_points") or [])
    detailed_summary = str(summary_data.get("detailed_summary") or "").strip()

    lines = [f"Ngữ cảnh học tập từ #{channel_name}:"]
    if summary_points:
        lines.append("Ý chính:")
        lines.extend([f"- {item}" for item in summary_points])

    if detailed_summary:
        lines.append("Phân tích sâu:")
        lines.append(detailed_summary)

    if numbered_questions:
        lines.append("Câu hỏi ôn tập:")
        for item in numbered_questions:
            lines.append(f"- Câu {item['index']}: {item['question']}")

    lines.append(
        "Hãy trả lời câu hỏi tiếp theo của người dùng dựa trên ngữ cảnh này, ưu tiên giải thích rõ và có ví dụ."
    )
    return "\n".join(lines)


def _build_detailed_summary_text(channel_name, summary_data):
    detailed_summary = str(summary_data.get("detailed_summary") or "").strip()
    if not detailed_summary:
        return ""
    return f"📖 **Phân tích sâu #{channel_name}:**\n{detailed_summary}"


SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/tasks",
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

# Predefined special countdowns
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


# ==============================
# KNOWLEDGE BOT
# ==============================
class KnowledgeBot:
    def __init__(self):
        self.timezone = pytz.timezone("Asia/Ho_Chi_Minh")
        self._calendar_service = None
        self._tasks_service = None

    def _default_countdown_milestones(self):
        """Mốc nhắc countdown mặc định"""
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
        """Gọi model với fallback tự động khi lỗi/rate limit"""
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
                                result = {
                                    "ok": True,
                                    "content": content,
                                    "model": model,
                                }
                                return result

                            if isinstance(data, list):
                                result = {
                                    "ok": True,
                                    "content": data,
                                    "model": model,
                                }
                                return result

                            if isinstance(data, dict):
                                generic_content = data.get("content") or data.get(
                                    "message"
                                )
                                if generic_content:
                                    result = {
                                        "ok": True,
                                        "content": self._normalize_model_content(
                                            generic_content
                                        ),
                                        "model": model,
                                    }
                                    return result

                            if isinstance(data, str) and data.strip():
                                result = {
                                    "ok": True,
                                    "content": data.strip(),
                                    "model": model,
                                }
                                return result

                        err_text = (
                            f"{model}: HTTP {response.status} - {str(data)[:800]}"
                        )
                        errors.append(err_text)
                except asyncio.TimeoutError:
                    err_text = f"{model}: request timeout sau {effective_timeout}s"
                    errors.append(err_text)
                except Exception as e:
                    err_text = f"{model}: {str(e)}"
                    errors.append(err_text)

        result = {
            "ok": False,
            "error": (
                "Hệ thống AI phản hồi quá chậm hoặc lỗi endpoint. "
                + " | ".join(errors[:3])
            ),
            "model": None,
            "content": None,
        }
        return result

    def _normalize_model_content(self, content):
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
        if not text:
            return ""
        cleaned = re.sub(
            r"<think>.*?</think>", "", str(text), flags=re.DOTALL | re.IGNORECASE
        )
        return cleaned.strip()

    def _extract_visible_reasoning_message(self, content):
        """Parse output từ reasoning model và chỉ lấy phần hiển thị được"""
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

        # Case 1: API trả về history array như mẫu user đưa
        if isinstance(parsed, list):
            assistant_messages = [
                str(item.get("message", "")).strip()
                for item in parsed
                if isinstance(item, dict) and item.get("role") == "assistant"
            ]
            if assistant_messages:
                return self._strip_think_block(assistant_messages[-1])

        # Case 2: plain text content
        return self._strip_think_block(content)

    def _extract_reasoning_message_raw(self, content):
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
                return assistant_messages[-1]

        return str(content)

    async def _extract_single_image_information(
        self,
        image_url,
        user_prompt="",
        username="User",
        image_index=1,
        total_images=1,
    ):
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
        """Lấy credentials"""
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
        if not self._calendar_service:
            creds = self._get_credentials()
            if not creds:
                return None
            self._calendar_service = build("calendar", "v3", credentials=creds)
        return self._calendar_service

    def _get_tasks_service(self):
        if not self._tasks_service:
            creds = self._get_credentials()
            if not creds:
                return None
            self._tasks_service = build("tasks", "v1", credentials=creds)
        return self._tasks_service

    # --------------------------
    # DATE/TIME PARSING
    # --------------------------
    def parse_date(self, date_str):
        """
        Parse ngày: today, tomorrow, dayafter, monday, 18/2, 18-2
        """
        if not date_str:
            return None

        date_str = date_str.lower().strip()
        now = datetime.now(self.timezone)

        # Keywords
        if date_str in ["today", "hôm nay"]:
            return now.date()
        elif date_str in ["tomorrow", "tmr", "mai"]:
            return (now + timedelta(days=1)).date()
        elif date_str in ["dayafter", "mốt"]:
            return (now + timedelta(days=2)).date()

        # Weekdays
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

        # DD/MM hoặc DD-MM
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
        """
        Parse giờ: 14:30, 14h30, 14h, 2pm
        """
        if not time_str:
            return None

        time_str = time_str.lower().strip()

        # 14:30 hoặc 14h30
        match = re.search(r"(\d{1,2})[h:](\d{2})", time_str)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return time(hour, minute)

        # 14h hoặc 14
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

    # --------------------------
    # WEATHER (simplified)
    # --------------------------
    async def get_weather(self, target_date=None, target_time=None):
        """Thời tiết hiện tại hoặc forecast theo ngày/giờ"""
        try:
            now_local = datetime.now(self.timezone)
            date_to_use = target_date or now_local.date()
            day_delta = (date_to_use - now_local.date()).days

            if day_delta < 0:
                return "⚠️ Không hỗ trợ xem weather quá khứ."
            if day_delta >= WEATHER_FORECAST_MAX_DAYS:
                return (
                    f"⚠️ Chỉ hỗ trợ dự báo trong {WEATHER_FORECAST_MAX_DAYS} ngày tới."
                )

            forecast_days = max(1, day_delta + 1)
            url = (
                "http://api.weatherapi.com/v1/forecast.json"
                f"?key={WEATHER_API_KEY}"
                f"&q={WEATHER_DEFAULT_LOCATION}"
                f"&days={forecast_days}"
                "&lang=vi&aqi=no"
            )
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        return "⚠️ Không thể lấy thời tiết"
                    data = await response.json()

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

    # --------------------------
    # CALENDAR - EVENTS
    # --------------------------
    async def get_events(self, date=None):
        """Lấy EVENTS trong ngày (không có tasks)"""
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
                    except:
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
        """Thêm event với thời gian bắt đầu và kết thúc"""
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
        """Xóa event"""
        try:
            service = self._get_calendar_service()
            if not service:
                return "⚠️ Cần setup Google Calendar"

            service.events().delete(calendarId="primary", eventId=event_id).execute()
            return "✅ Đã xóa event"
        except Exception as e:
            return f"⚠️ Lỗi: {str(e)}"

    async def update_event(self, event_id, **kwargs):
        """Cập nhật event"""
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

    # --------------------------
    # TASKS
    # --------------------------
    async def get_tasks(self, date=None, show_completed=False):
        """
        Lấy tasks
        date=None: tất cả tasks
        date=specific: tasks có due date = ngày đó
        """
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
                            # Due có thể là date hoặc datetime
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
                        except:
                            pass

                    # Filter by date if specified
                    if date is not None:
                        if due_date != date:
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

            # Sort
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
        """Thêm task với due date và time"""
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
                # Google Tasks API accepts RFC 3339 timestamp
                task["due"] = due_datetime.isoformat()

            service.tasks().insert(tasklist=tasklist_id, body=task).execute()

            due_str = ""
            if due_datetime:
                due_str = f" (hạn: {due_datetime.strftime('%d/%m %H:%M')})"
            return f"✅ Đã thêm task: {title}{due_str}"

        except Exception as e:
            return f"⚠️ Lỗi: {str(e)}"

    async def complete_task(self, task_id, tasklist_id):
        """Đánh dấu hoàn thành"""
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
        """Xóa task"""
        try:
            service = self._get_tasks_service()
            if not service:
                return "⚠️ Cần setup Google Tasks"

            service.tasks().delete(tasklist=tasklist_id, task=task_id).execute()
            return "✅ Đã xóa task"
        except Exception as e:
            return f"⚠️ Lỗi: {str(e)}"

    # --------------------------
    # CALENDAR (FULL - Events + Tasks)
    # --------------------------
    async def get_calendar(self, date=None):
        """Lấy TOÀN BỘ lịch: events + tasks"""
        events = await self.get_events(date)
        tasks = await self.get_tasks(date, show_completed=False)

        return {
            "events": events if isinstance(events, list) else [],
            "tasks": tasks if isinstance(tasks, list) else [],
        }

    # --------------------------
    # UTILITIES
    # --------------------------
    def _is_important(self, summary, description):
        """Kiểm tra quan trọng"""
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
        """Kiểm tra ngày lễ"""
        if date is None:
            date = datetime.now(self.timezone).date()
        elif isinstance(date, datetime):
            date = date.date()
        date_key = date.strftime("%m-%d")
        return VIETNAM_HOLIDAYS.get(date_key)

    def get_next_tet_datetime(self, from_datetime=None):
        """Lấy thời điểm Tết Âm lịch gần nhất trong tương lai"""
        if from_datetime is None:
            from_datetime = datetime.now(self.timezone)

        for year in sorted(LUNAR_TET_DATES.keys()):
            month, day = LUNAR_TET_DATES[year]
            tet_datetime = self.timezone.localize(datetime(year, month, day, 0, 0, 0))
            if tet_datetime > from_datetime:
                return year, tet_datetime

        return None, None

    # --------------------------
    # COUNTDOWN
    # --------------------------
    def add_countdown(self, name, target_datetime, emoji="🎉", label=""):
        """
        Thêm countdown mới
        label: "newyear" cho format đặc biệt năm mới, "" cho format thông thường
        """
        if not isinstance(target_datetime, datetime):
            return False

        # Ensure timezone aware
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
        """Xóa countdown"""
        if name in _active_countdowns:
            del _active_countdowns[name]
            return True
        return False

    def get_countdowns(self):
        """Lấy danh sách countdowns"""
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
        """Format thông báo countdown"""
        if name not in _active_countdowns:
            return None

        data = _active_countdowns[name]
        emoji = data["emoji"]
        countdown_name = data["name"]
        mention = f"<@{YOUR_USER_ID}>" if YOUR_USER_ID else ""

        # Check if this is New Year countdown (special format)
        is_newyear = "newyear" in data.get("label", "").lower()

        if is_newyear:
            # SPECIAL FORMAT FOR NEW YEAR
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
                # NEW YEAR CELEBRATION
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
            # STANDARD FORMAT FOR OTHER EVENTS
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

    # --------------------------
    # AI SUMMARY
    # --------------------------
    async def summarize_daily_knowledge(
        self, messages, channel_name="", offset=0, batch_size=50
    ):
        if not messages:
            return None, False

        total = len(messages)
        start_idx = offset
        end_idx = min(offset + batch_size, total)

        batch_messages = messages[start_idx:end_idx]
        has_more = end_idx < total

        message_text = "\n".join([f"- {msg}" for msg in batch_messages])

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
                        "- detailed_summary: phân tích sâu, có cấu trúc, giải thích đủ dài\n"
                        "- review_questions: 3-5 câu hỏi kiểm tra hiểu bài"
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

        return {
            "summary_points": summary_points,
            "detailed_summary": detailed_summary,
            "review_questions": review_questions,
            "model": ai_result["model"],
        }, has_more

    async def expand_summary_analysis(
        self,
        channel_name,
        summary_points,
        detailed_summary,
        review_questions,
    ):
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


knowledge_bot = KnowledgeBot()


def _build_chat_context_text(session):
    prompt = session.get("prompt", "")
    answer = session.get("answer", "")
    return (
        "Ngữ cảnh phiên trước:\n"
        f"- User hỏi: {prompt}\n"
        f"- Assistant trả lời: {answer}\n"
        "Hãy tiếp tục dựa trên ngữ cảnh này."
    )


def _create_chat_session(
    user_id,
    username,
    prompt,
    answer,
    model_used,
    image_urls=None,
    image_extractions=None,
    vision_models=None,
):
    image_urls = image_urls or []
    image_extractions = image_extractions or []
    vision_models = vision_models or []

    session_id = f"chat-{user_id}-{int(datetime.now(VIETNAM_TZ).timestamp() * 1000)}"
    _chat_sessions[session_id] = {
        "user_id": user_id,
        "username": username,
        "prompt": prompt,
        "answer": answer,
        "model": model_used,
        "image_urls": image_urls,
        "image_extractions": image_extractions,
        "vision_models": vision_models,
        "created_ts": datetime.now(VIETNAM_TZ).timestamp(),
    }

    if len(_chat_sessions) > 200:
        oldest = sorted(
            _chat_sessions.items(), key=lambda x: x[1].get("created_ts", 0)
        )[:50]
        for key, _ in oldest:
            _chat_sessions.pop(key, None)

    return session_id


async def _continue_summary_for_user(user_id):
    _ensure_daily_window_rollover()

    if user_id != YOUR_USER_ID:
        return {"ok": False, "message": "⛔ Bạn không có quyền dùng lệnh này."}

    if not summary_state:
        return {"ok": False, "message": "📚 Không có phần dở"}

    channel_id = list(summary_state.keys())[0]
    state = summary_state[channel_id]

    summary_data, has_more = await knowledge_bot.summarize_daily_knowledge(
        state["messages"], state["channel_name"], state["offset"], 50
    )
    if summary_data.get("error"):
        return {"ok": False, "message": summary_data["error"]}

    current_questions = _study_questions.get(user_id, [])
    next_question_index = max([q["index"] for q in current_questions], default=0) + 1

    embed, numbered_questions = _build_summary_embed(
        state["channel_name"],
        len(state["messages"]),
        summary_data,
        question_start_index=next_question_index,
    )

    _persist_questions_for_spaced_repetition(
        user_id=user_id,
        channel_name=state["channel_name"],
        summary_data=summary_data,
        numbered_questions=numbered_questions,
    )
    theory_text = _build_question_theory_text(
        summary_data.get("summary_points", []),
        summary_data.get("detailed_summary", ""),
    )

    for item in numbered_questions:
        _study_questions.setdefault(user_id, []).append(
            {
                "index": item["index"],
                "channel_name": state["channel_name"],
                "question": item["question"],
                "summary_points": summary_data.get("summary_points", []),
                "theory": theory_text,
            }
        )

    if has_more:
        summary_state[channel_id]["offset"] += 50
        remaining = len(state["messages"]) - summary_state[channel_id]["offset"]
        return {
            "ok": True,
            "embed": embed,
            "summary_data": summary_data,
            "numbered_questions": numbered_questions,
            "channel_name": state["channel_name"],
            "has_more": True,
            "remaining": max(0, remaining),
            "channel_id": channel_id,
        }

    del summary_state[channel_id]

    return {
        "ok": True,
        "embed": embed,
        "summary_data": summary_data,
        "numbered_questions": numbered_questions,
        "channel_name": state["channel_name"],
        "has_more": False,
        "remaining": 0,
        "channel_id": channel_id,
    }


class ChatSessionView(discord.ui.View):
    def __init__(self, session_id):
        super().__init__(timeout=1800)
        self.session_id = session_id

    async def _check_owner(self, interaction: discord.Interaction):
        session = _chat_sessions.get(self.session_id)
        if not session:
            await interaction.response.send_message(
                "⚠️ Session chat không còn khả dụng.", ephemeral=True
            )
            return None
        if interaction.user.id != session.get("user_id"):
            await interaction.response.send_message(
                "⛔ Bạn không thể thao tác trên session của người khác.",
                ephemeral=True,
            )
            return None
        return session

    @discord.ui.button(label="Dùng làm context tiếp", style=discord.ButtonStyle.primary)
    async def use_context_next(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        session = await self._check_owner(interaction)
        if not session:
            return

        _pending_chat_context[interaction.user.id] = _build_chat_context_text(session)
        await interaction.response.send_message(
            "✅ Đã lưu context. Tin nhắn chat kế tiếp của bạn sẽ tự dùng context này.",
            ephemeral=True,
        )

    @discord.ui.button(label="Continue", style=discord.ButtonStyle.success)
    async def continue_chat(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        session = await self._check_owner(interaction)
        if not session:
            return

        await interaction.response.defer(thinking=True)
        continue_prompt = (
            "Hãy tiếp tục trả lời phần còn thiếu của nội dung trước đó. "
            "Không lặp lại phần đã trả lời, đi thẳng vào phần tiếp theo."
        )
        prior_context = _build_chat_context_text(session)
        ai_result = await knowledge_bot.chat(
            continue_prompt,
            session.get("username") or interaction.user.display_name,
            image_urls=[],
            prior_context=prior_context,
        )
        if not ai_result.get("ok"):
            await interaction.followup.send(
                f"⚠️ Continue thất bại: {ai_result.get('error')}", ephemeral=True
            )
            return

        answer = (ai_result.get("content") or "").strip()
        display_answer = _format_rich_text_for_discord(answer)
        model_used = ai_result.get("model")
        new_session_id = _create_chat_session(
            user_id=interaction.user.id,
            username=session.get("username") or interaction.user.display_name,
            prompt=continue_prompt,
            answer=answer,
            model_used=model_used,
            image_urls=[],
            image_extractions=ai_result.get("image_extractions", []),
            vision_models=ai_result.get("vision_models", []),
        )

        embed = discord.Embed(
            title="💬 Chatbot (Continue)",
            description=display_answer[:3900],
            color=discord.Color.blurple(),
            timestamp=datetime.now(VIETNAM_TZ),
        )
        embed.add_field(name="🔁 Yêu cầu", value=continue_prompt[:1024], inline=False)
        embed.set_footer(text=f"Đang trả lời bằng: {model_used}")
        await interaction.followup.send(
            embed=embed, view=ChatSessionView(new_session_id)
        )

        remaining = display_answer[3900:]
        for chunk in _split_text_chunks(remaining, 1900):
            await interaction.followup.send(f"📎 Phần tiếp theo:\n{chunk}")


class StudyAnswerModal(discord.ui.Modal):
    def __init__(
        self,
        owner_id,
        question_index,
        question_text,
        theory_text="",
        summary_points=None,
        channel_name="",
    ):
        super().__init__(title=f"Trả lời câu {question_index}")
        self.owner_id = owner_id
        self.question_index = question_index
        self.question_text = question_text
        self.theory_text = str(theory_text or "").strip()
        self.summary_points = summary_points or []
        self.channel_name = channel_name

        theory_preview = (
            self.theory_text[:3800]
            if self.theory_text
            else "Không có lý thuyết đính kèm."
        )

        self.theory_display = discord.ui.TextInput(
            label="Lý thuyết ôn nhanh (đọc trước)",
            style=discord.TextStyle.paragraph,
            default=theory_preview,
            required=False,
            max_length=4000,
        )
        self.add_item(self.theory_display)

        self.question_display = discord.ui.TextInput(
            label="Câu hỏi",
            style=discord.TextStyle.paragraph,
            default=str(question_text)[:1000],
            required=False,
            max_length=1000,
        )
        self.add_item(self.question_display)

        self.user_answer = discord.ui.TextInput(
            label="Câu trả lời của bạn",
            style=discord.TextStyle.paragraph,
            placeholder="Nhập câu trả lời của bạn tại đây...",
            required=True,
            max_length=2000,
        )
        self.add_item(self.user_answer)

    async def on_submit(self, interaction: discord.Interaction):
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message(
                "⛔ Bạn không thể trả lời cho session của người khác.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(thinking=True, ephemeral=True)
        review = await knowledge_bot.review_study_answer(
            self.question_text,
            str(self.user_answer.value).strip(),
            self.summary_points,
        )

        if not review.get("ok"):
            await interaction.followup.send(
                review.get("error", "⚠️ Có lỗi khi chấm câu trả lời."),
                ephemeral=True,
            )
            return

        score_value = _normalize_score_value(review.get("score"))
        passed = score_value is not None and score_value >= STUDY_PASS_THRESHOLD
        points_delta = int(STUDY_POINTS_PASS) if passed else 0

        updated_stats = _append_study_event(
            user_id=interaction.user.id,
            event_type="pass" if passed else "answer",
            points_delta=points_delta,
            question_index=self.question_index,
            score=score_value,
            note=("Đạt ngưỡng" if passed else "Chưa đạt ngưỡng"),
        )
        sm2_result = _record_spaced_review(
            user_id=interaction.user.id,
            target_question={
                "channel_name": self.channel_name,
                "question": self.question_text,
                "theory": self.theory_text,
            },
            score_value=score_value,
            answered=True,
            note=("Đạt ngưỡng" if passed else "Chưa đạt ngưỡng"),
        )
        _mark_question_answered(interaction.user.id, self.question_index)

        embed = discord.Embed(
            title=f"🧪 Nhận xét câu {self.question_index}",
            color=discord.Color.green(),
            timestamp=datetime.now(VIETNAM_TZ),
        )
        embed.add_field(
            name="❓ Câu hỏi", value=self.question_text[:1024], inline=False
        )
        embed.add_field(
            name="📝 Câu trả lời của bạn",
            value=str(self.user_answer.value)[:1024],
            inline=False,
        )
        embed.add_field(
            name="📊 Điểm", value=str(review.get("score", "?")), inline=True
        )
        embed.add_field(
            name="💬 Nhận xét",
            value=str(review.get("feedback", ""))[:1024],
            inline=False,
        )
        if review.get("suggestion"):
            embed.add_field(
                name="✅ Gợi ý cải thiện",
                value=str(review.get("suggestion"))[:1024],
                inline=False,
            )
        embed.add_field(
            name="🔥 Study points",
            value=(
                f"{'+%d' % points_delta if points_delta else '+0'} điểm | "
                f"Tổng: {updated_stats.get('total_points', 0)} | "
                f"Streak: {updated_stats.get('streak_days', 0)} ngày"
            )[:1024],
            inline=False,
        )
        if sm2_result:
            embed.add_field(
                name="🧠 Spaced Repetition",
                value=(
                    f"Quality: {sm2_result.get('quality')} | "
                    f"Interval: {sm2_result.get('interval_days')} ngày | "
                    f"Due: {sm2_result.get('due_date')}"
                )[:1024],
                inline=False,
            )
        embed.set_footer(text=f"Đang trả lời bằng: {review.get('model')}")
        await interaction.followup.send(embed=embed, ephemeral=True)


class SummaryAnswerButton(discord.ui.Button):
    def __init__(self, owner_id, question_item):
        self.owner_id = owner_id
        self.question_item = question_item
        index = int(question_item.get("index", 0))
        row = 0 if index <= 5 else 1
        super().__init__(
            label=f"Trả lời câu {index}",
            style=discord.ButtonStyle.primary,
            row=row,
        )

    async def callback(self, interaction: discord.Interaction):
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message(
                "⛔ Bạn không thể thao tác trên summary của người khác.",
                ephemeral=True,
            )
            return

        await interaction.response.send_modal(
            StudyAnswerModal(
                owner_id=self.owner_id,
                question_index=self.question_item["index"],
                question_text=self.question_item["question"],
                theory_text=self.question_item.get("theory", ""),
                summary_points=self.question_item.get("summary_points", []),
                channel_name=self.question_item.get("channel_name", ""),
            )
        )


class SummaryInteractiveView(discord.ui.View):
    def __init__(
        self,
        owner_id,
        channel_name,
        summary_data,
        numbered_questions,
        has_more=False,
    ):
        super().__init__(timeout=1800)
        self.owner_id = owner_id
        self.channel_name = channel_name
        self.summary_data = summary_data or {}
        self.numbered_questions = numbered_questions or []
        self.has_more = bool(has_more)

        for item in self.numbered_questions[:5]:
            payload = {
                "index": item["index"],
                "question": item["question"],
                "summary_points": self.summary_data.get("summary_points", []),
                "theory": _build_question_theory_text(
                    self.summary_data.get("summary_points", []),
                    self.summary_data.get("detailed_summary", ""),
                ),
                "channel_name": self.channel_name,
            }
            self.add_item(SummaryAnswerButton(self.owner_id, payload))

        if not self.has_more:
            for child in list(self.children):
                if (
                    isinstance(child, discord.ui.Button)
                    and str(child.label or "") == "Continue Summary"
                ):
                    self.remove_item(child)

    async def _ensure_owner(self, interaction: discord.Interaction):
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message(
                "⛔ Bạn không thể thao tác trên summary của người khác.",
                ephemeral=True,
            )
            return False
        return True

    @discord.ui.button(
        label="Dùng summary làm context", style=discord.ButtonStyle.secondary, row=2
    )
    async def use_summary_context(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        if not await self._ensure_owner(interaction):
            return

        _pending_chat_context[interaction.user.id] = _build_study_context_text(
            self.channel_name,
            self.summary_data,
            self.numbered_questions,
        )
        await interaction.response.send_message(
            "✅ Đã lưu context từ summary. Tin nhắn `!chat` hoặc `/chat` kế tiếp sẽ dùng context này.",
            ephemeral=True,
        )

    @discord.ui.button(
        label="Phân tích sâu hơn", style=discord.ButtonStyle.success, row=2
    )
    async def deepen_summary(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        if not await self._ensure_owner(interaction):
            return

        await interaction.response.defer(thinking=True)
        result = await knowledge_bot.expand_summary_analysis(
            self.channel_name,
            self.summary_data.get("summary_points", []),
            self.summary_data.get("detailed_summary", ""),
            self.summary_data.get("review_questions", []),
        )

        if not result.get("ok"):
            await interaction.followup.send(
                f"⚠️ Không thể phân tích sâu hơn: {result.get('error', 'Unknown error')}",
                ephemeral=True,
            )
            return

        content = (result.get("content") or "").strip()
        if not content:
            await interaction.followup.send(
                "⚠️ Model không trả về nội dung.", ephemeral=True
            )
            return

        display_content = _format_rich_text_for_discord(content)

        embed = discord.Embed(
            title=f"🔍 Phân tích sâu hơn #{self.channel_name}",
            description=display_content[:3900],
            color=discord.Color.green(),
            timestamp=datetime.now(VIETNAM_TZ),
        )
        embed.set_footer(text=f"Model: {result.get('model')}")
        await interaction.followup.send(embed=embed)

        for chunk in _split_text_chunks(display_content[3900:], 1900):
            await interaction.followup.send(f"📎 Phần tiếp theo:\n{chunk}")

    @discord.ui.button(
        label="Continue Summary", style=discord.ButtonStyle.primary, row=2
    )
    async def continue_summary(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        if not await self._ensure_owner(interaction):
            return

        if not self.has_more:
            await interaction.response.send_message(
                "✅ Summary đã hết phần còn lại cho phiên hiện tại.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(thinking=True)
        result = await _continue_summary_for_user(interaction.user.id)

        if not result.get("ok"):
            await interaction.followup.send(
                result.get("message", "⚠️ Lỗi không xác định")
            )
            return

        if result.get("embed"):
            next_view = (
                SummaryInteractiveView(
                    interaction.user.id,
                    result.get("channel_name", "unknown"),
                    result.get("summary_data", {}),
                    result.get("numbered_questions", []),
                    has_more=result.get("has_more"),
                )
                if result.get("has_more")
                else SummaryInteractiveView(
                    interaction.user.id,
                    result.get("channel_name", "unknown"),
                    result.get("summary_data", {}),
                    result.get("numbered_questions", []),
                    has_more=False,
                )
            )
            await interaction.followup.send(embed=result["embed"], view=next_view)

        if result.get("has_more"):
            await interaction.followup.send(
                f"💡 Còn {result.get('remaining', 0)} tin nhắn chưa summary. Bấm `Continue Summary`."
            )
        else:
            await interaction.followup.send("✅ Đã summary xong toàn bộ phần còn lại.")


# ==============================
# EVENTS
# ==============================
@bot.event
async def on_ready():
    print(f"✅ Bot: {bot.user}")
    _ensure_daily_window_rollover()

    try:
        if APP_GUILD_ID:
            guild = discord.Object(id=APP_GUILD_ID)
            synced = await bot.tree.sync(guild=guild)
            print(f"✅ Synced {len(synced)} slash command(s) to guild {APP_GUILD_ID}")
        else:
            synced = await bot.tree.sync()
            print(f"✅ Synced {len(synced)} global slash command(s)")
    except Exception as e:
        print(f"⚠️ Slash sync lỗi: {e}")

    morning_greeting.start()
    calendar_reminder.start()
    evening_summary.start()
    end_of_day_review.start()
    countdown_checker.start()
    daily_rollover.start()
    idle_motivation.start()

    # Auto-activate New Year countdown if today is Dec 31
    now = datetime.now(knowledge_bot.timezone)
    if now.month == 12 and now.day == 31:
        # New Year countdown with special format
        ny_datetime = knowledge_bot.timezone.localize(
            datetime(now.year + 1, 1, 1, 0, 0, 0)
        )
        knowledge_bot.add_countdown(
            f"Năm Mới {now.year + 1}",
            ny_datetime,
            "🎆",
            label="newyear",  # Special format
        )
        print(f"🎆 Auto-activated New Year {now.year + 1} countdown!")

    # Auto-activate Tet countdown if within 7 days
    tet_year, tet_datetime = knowledge_bot.get_next_tet_datetime(now)
    if tet_datetime:
        days_to_tet = (tet_datetime - now).total_seconds() / 86400
    else:
        days_to_tet = -1

    if 0 <= days_to_tet <= 7 and tet_year:
        knowledge_bot.add_countdown(
            f"Tết Nguyên Đán {tet_year}",
            tet_datetime,
            "🧧",
            label="",  # Standard format
        )
        print(f"🧧 Auto-activated Tet {tet_year} countdown!")


@bot.event
async def on_message(message):
    if message.author.bot:
        return

    _ensure_daily_window_rollover()

    if message.author.id == YOUR_USER_ID:
        _mark_user_interaction(message.author.id)

    if message.channel.id in CHANNELS_TO_MONITOR:
        channel_id = message.channel.id
        if channel_id not in daily_messages:
            daily_messages[channel_id] = []
        timestamp = datetime.now(knowledge_bot.timezone).strftime("%H:%M")
        attachment_context = _attachment_context_for_summary(message)
        daily_messages[channel_id].append(
            f"[{timestamp}] {message.author.name}: {message.content}{attachment_context}"
        )

    # Slash-first: không xử lý prefix command (ví dụ !...)
    return


@bot.event
async def on_command_completion(ctx):
    _mark_user_interaction(ctx.author.id)


@bot.event
async def on_app_command_completion(interaction: discord.Interaction, command):
    _mark_user_interaction(interaction.user.id)


# ==============================
# TASKS
# ==============================
@tasks.loop(time=time(hour=0, minute=0, tzinfo=VIETNAM_TZ))
async def daily_rollover():
    _ensure_daily_window_rollover()


@daily_rollover.before_loop
async def before_daily_rollover():
    await bot.wait_until_ready()


@tasks.loop(minutes=SLOGAN_CHECK_INTERVAL_MINUTES)
async def idle_motivation():
    if MAIN_CHANNEL_ID == 0 or YOUR_USER_ID == 0:
        return

    channel = bot.get_channel(MAIN_CHANNEL_ID)
    if not channel:
        return

    now = datetime.now(VIETNAM_TZ)
    last_interaction = _last_interaction_at.get(YOUR_USER_ID)
    if last_interaction is None:
        return

    idle_minutes = (now - last_interaction).total_seconds() / 60
    if idle_minutes < SLOGAN_IDLE_MINUTES:
        return

    last_sent = _last_slogan_sent_at.get(YOUR_USER_ID)
    if last_sent:
        sent_gap = (now - last_sent).total_seconds() / 60
        if sent_gap < SLOGAN_IDLE_MINUTES:
            return

    slogan = await _fetch_motivational_slogan()
    await channel.send(
        f"💡 **Nhắc nhẹ học tập**\n"
        f"Bạn đã im lặng khoảng **{int(idle_minutes)} phút**.\n"
        f"*{slogan}*"
    )
    _last_slogan_sent_at[YOUR_USER_ID] = now


@idle_motivation.before_loop
async def before_idle_motivation():
    await bot.wait_until_ready()


@tasks.loop(time=time(hour=6, minute=30, tzinfo=VIETNAM_TZ))
async def morning_greeting():
    if MAIN_CHANNEL_ID == 0:
        return

    channel = bot.get_channel(MAIN_CHANNEL_ID)
    if not channel:
        return

    weather = await knowledge_bot.get_weather()
    calendar_data = await knowledge_bot.get_calendar()
    holiday = knowledge_bot.check_holiday()

    message = f"🌅 **Chào buổi sáng!**\n\n"

    if holiday:
        message += f"🎉 **{holiday}**\n\n"

    message += f"{weather}\n"
    if YOUR_USER_ID:
        message += "\n" + _build_study_status_text(YOUR_USER_ID) + "\n"

    # Events
    events = calendar_data["events"]
    if events:
        message += "\n📅 **Events hôm nay:**\n"
        has_important = any(e["is_important"] for e in events)
        if has_important:
            message += "⚠️ **CÓ SỰ KIỆN QUAN TRỌNG!**\n"
        for e in events[:10]:
            icon = "🔴" if e["is_important"] else "•"
            time_str = f"{e['time']}-{e['end_time']}" if e["end_time"] else e["time"]
            message += f"{icon} {time_str} {e['summary']}\n"

    # Tasks
    tasks_list = calendar_data["tasks"]
    if tasks_list:
        overdue = [t for t in tasks_list if t["overdue"]]
        today_tasks = [t for t in tasks_list if not t["overdue"]]

        if overdue:
            message += f"\n🔴 **TASKS QUÁ HẠN ({len(overdue)}):**\n"
            for t in overdue[:5]:
                message += f"• {t['title']}\n"

        if today_tasks:
            message += f"\n📋 **Tasks ({len(today_tasks)}):**\n"
            for t in today_tasks[:5]:
                message += f"• {t['title']}\n"

    await channel.send(message)


@morning_greeting.before_loop
async def before_morning_greeting():
    await bot.wait_until_ready()


@tasks.loop(minutes=1)
async def calendar_reminder():
    if MAIN_CHANNEL_ID == 0:
        return

    channel = bot.get_channel(MAIN_CHANNEL_ID)
    if not channel:
        return

    now = datetime.now(knowledge_bot.timezone)

    # Events reminder (trước 30 phút)
    events = await knowledge_bot.get_events()
    if events and isinstance(events, list):
        for event in events:
            if event["datetime"] is None:
                continue

            event_time = event["datetime"]
            minutes_until = (event_time - now).total_seconds() / 60
            reminder_key = (
                "event",
                event.get("id"),
                event_time.strftime("%Y-%m-%d %H:%M:%S"),
            )

            if 0 < minutes_until <= 30 and reminder_key not in _sent_upcoming_reminders:
                icon = "🔔" if event["is_important"] else "⏰"
                await channel.send(
                    f"{icon} **30 phút nữa:**\n📌 {event['summary']} ({event['time']})"
                )
                _sent_upcoming_reminders.add(reminder_key)

    # Tasks reminder (trước 30 phút) - chỉ task có due time
    tasks_list = await knowledge_bot.get_tasks(date=now.date(), show_completed=False)
    if isinstance(tasks_list, list):
        for task in tasks_list:
            if task.get("overdue") or not task.get("due") or not task.get("due_time"):
                continue

            task_due_dt = datetime.combine(task["due"], task["due_time"])
            if task_due_dt.tzinfo is None:
                task_due_dt = knowledge_bot.timezone.localize(task_due_dt)
            else:
                task_due_dt = task_due_dt.astimezone(knowledge_bot.timezone)

            minutes_until = (task_due_dt - now).total_seconds() / 60
            reminder_key = (
                "task",
                task.get("tasklist_id"),
                task.get("id"),
                task_due_dt.strftime("%Y-%m-%d %H:%M:%S"),
            )

            if 0 < minutes_until <= 30 and reminder_key not in _sent_upcoming_reminders:
                due_time = task_due_dt.strftime("%H:%M")
                await channel.send(
                    f"📋 **30 phút nữa đến hạn task:**\n📝 {task['title']} ({due_time})"
                )
                _sent_upcoming_reminders.add(reminder_key)


@calendar_reminder.before_loop
async def before_calendar_reminder():
    await bot.wait_until_ready()


@tasks.loop(time=time(hour=20, minute=0, tzinfo=VIETNAM_TZ))
async def end_of_day_review():
    if MAIN_CHANNEL_ID == 0:
        return

    channel = bot.get_channel(MAIN_CHANNEL_ID)
    if not channel:
        return

    now = datetime.now(knowledge_bot.timezone)
    today_tasks = await knowledge_bot.get_tasks(date=now.date(), show_completed=False)
    all_tasks = await knowledge_bot.get_tasks(show_completed=False)

    if isinstance(today_tasks, str) or isinstance(all_tasks, str):
        return

    overdue = [t for t in all_tasks if t["overdue"]]

    if not today_tasks and not overdue:
        await channel.send("✅ **Tuyệt vời!** Tất cả tasks đã xong!")
        return

    message = "📊 **Review cuối ngày:**\n\n"

    if today_tasks:
        message += f"⚠️ **Tasks hôm nay chưa xong ({len(today_tasks)}):**\n"
        for task in today_tasks:
            time_str = task["due_time"].strftime("%H:%M") if task["due_time"] else ""
            message += f"• {task['title']} {time_str}\n"
        message += "\n💡 Nhớ hoàn thành trước khi ngủ!\n"

    if overdue:
        message += f"\n🔴 **Tasks quá hạn ({len(overdue)}):**\n"
        for task in overdue[:10]:
            due_str = task["due"].strftime("%d/%m") if task["due"] else "N/A"
            message += f"• {task['title']} (hạn: {due_str})\n"
        message += "\n⚡ Ưu tiên xử lý ngay!\n"

    if YOUR_USER_ID:
        message += "\n" + _build_study_status_text(YOUR_USER_ID)

    await channel.send(message)


@end_of_day_review.before_loop
async def before_end_of_day_review():
    await bot.wait_until_ready()


@tasks.loop(time=time(hour=21, minute=0, tzinfo=VIETNAM_TZ))
async def evening_summary():
    _ensure_daily_window_rollover()

    if MAIN_CHANNEL_ID == 0:
        return

    channel = bot.get_channel(MAIN_CHANNEL_ID)
    if not channel:
        return

    if not daily_messages:
        await channel.send("📚 Không có tin nhắn hôm nay")
        return

    for channel_id, messages in daily_messages.items():
        discord_channel = bot.get_channel(channel_id)
        channel_name = discord_channel.name if discord_channel else str(channel_id)

        summary_data, has_more = await knowledge_bot.summarize_daily_knowledge(
            messages, channel_name, 0, SUMMARY_BATCH_SIZE
        )

        if summary_data:
            if summary_data.get("error"):
                await channel.send(summary_data["error"])
            else:
                embed, _ = _build_summary_embed(
                    channel_name, len(messages), summary_data
                )
                await channel.send(embed=embed)

            if has_more:
                summary_state[channel_id] = {
                    "messages": messages,
                    "channel_name": channel_name,
                    "offset": SUMMARY_BATCH_SIZE,
                }
                await channel.send(
                    f"💡 Còn {len(messages) - SUMMARY_BATCH_SIZE} tin nhắn. Dùng `/continue_summary`"
                )

    if not summary_state:
        await channel.send("🧠 Dữ liệu học tập hôm nay vẫn được giữ đến hết ngày.")


@evening_summary.before_loop
async def before_evening_summary():
    await bot.wait_until_ready()


# ==============================
# COUNTDOWN TASK
# ==============================
@tasks.loop(seconds=1)
async def countdown_checker():
    """Check countdowns every second"""
    if MAIN_CHANNEL_ID == 0:
        return

    channel = bot.get_channel(MAIN_CHANNEL_ID)
    if not channel:
        return

    if not _active_countdowns:
        return

    now = datetime.now(knowledge_bot.timezone)

    for name, data in list(_active_countdowns.items()):
        target = data["datetime"]
        remaining = (target - now).total_seconds()
        last_remaining = data.get("last_remaining")
        if last_remaining is None:
            last_remaining = remaining + 1

        # Remove expired countdowns (after 5 seconds past target)
        if remaining < -5:
            del _active_countdowns[name]
            continue

        # Check milestones
        milestones = data["milestones"]
        notified = data["notified"]

        for milestone in milestones:
            # Check if we should notify for this milestone
            if milestone not in notified:
                # Send when crossing from above -> below milestone
                if last_remaining > milestone >= remaining:
                    display_remaining = milestone
                    message = knowledge_bot.format_countdown_message(
                        name, display_remaining
                    )
                    if message:
                        await channel.send(message)
                        notified.add(milestone)
                    break

        data["last_remaining"] = remaining


@countdown_checker.before_loop
async def before_countdown_checker():
    await bot.wait_until_ready()


# ==============================
# COMMANDS - HELP
# ==============================
@bot.command(name="help")
async def show_help(ctx, category=""):
    """Hiển thị trợ giúp"""

    if not category:
        embed = discord.Embed(
            title="🤖 Bot Agent - Trợ Lý Toàn Diện",
            description="Gõ `!help <category>` để xem chi tiết",
            color=discord.Color.blue(),
        )

        embed.add_field(
            name="📅 Calendar", value="`!help calendar` - Quản lý lịch", inline=True
        )
        embed.add_field(
            name="📋 Tasks", value="`!help tasks` - Quản lý công việc", inline=True
        )
        embed.add_field(
            name="⏰ Countdown", value="`!help countdown` - Đếm ngược", inline=True
        )
        embed.add_field(
            name="🌤️ Weather", value="`!help weather` - Thời tiết", inline=True
        )
        embed.add_field(name="📚 Study", value="`!help study` - Học tập", inline=True)
        embed.add_field(
            name="💬 Chatbot", value="`!help chatbot` - Chat AI", inline=True
        )
        embed.add_field(
            name="🤖 Automation", value="`!help automation` - Tự động hóa", inline=True
        )

        embed.add_field(
            name="🎯 Quick Start",
            value=(
                "`!calendar` - Xem lịch hôm nay\n"
                "`!tasks` - Xem tasks\n"
                "`!countdown` - Xem countdowns\n"
                "`!weather` - Thời tiết\n"
                "`!slogan` - Câu động lực học\n"
                "`!summary` - Tổng hợp học tập\n"
                "`!chat` - Chat trực tiếp với AI"
            ),
            inline=False,
        )

        embed.add_field(
            name="🎆 Quick Countdowns",
            value=("`!newyear` - Countdown năm mới\n" "`!tet` - Countdown Tết"),
            inline=False,
        )

        embed.add_field(
            name="📍 Thông Báo Tự Động",
            value=f"Gửi vào channel ID: **{MAIN_CHANNEL_ID}**\nCấu hình trong `.env`",
            inline=False,
        )

        await ctx.send(embed=embed)
        return

    category = category.lower()

    if category == "calendar":
        embed = discord.Embed(title="📅 Lệnh Calendar", color=discord.Color.green())
        embed.add_field(
            name="📍 Xem Lịch",
            value=(
                "`!calendar` - Toàn bộ (events+tasks) hôm nay\n"
                "`!calendar tomorrow` - Ngày mai\n"
                "`!calendar monday` - Thứ 2\n"
                "`!calendar 18/2` - Ngày 18/2"
            ),
            inline=False,
        )
        embed.add_field(
            name="📌 Xem Events",
            value=(
                "`!events` - Events hôm nay\n"
                "`!events tomorrow` - Events ngày mai\n"
                "`!events 18/2` - Events ngày 18/2"
            ),
            inline=False,
        )
        embed.add_field(
            name="➕ Thêm Event",
            value=(
                "`!add_event <title> | <date start-end> | <desc>`\n\n"
                "**Ví dụ:**\n"
                "`!add_event Họp | 18/2 14:00-16:00 | Sprint`\n"
                "`!add_event Deadline | 20/2 23:59 | Nộp báo cáo`\n"
                "`!add_event Học | tomorrow 19:00-21:00`"
            ),
            inline=False,
        )
        embed.add_field(
            name="🗑️ Xóa Event",
            value=(
                "1. `!events` - Hiện danh sách có số\n"
                "2. `!del_event 2` - Xóa event số 2"
            ),
            inline=False,
        )
        embed.add_field(
            name="🔄 Đổi Giờ Event",
            value=(
                "1. `!events` - Hiện danh sách có số\n"
                "2. `!move_event 1 | 19/2 15:00` - Đổi event 1 sang 19/2 lúc 15h"
            ),
            inline=False,
        )
        embed.add_field(
            name="📅 Date Formats",
            value="`today`, `tomorrow`, `dayafter`, `monday`, `tuesday`, `18/2`, `18-2`",
            inline=False,
        )
        embed.add_field(
            name="🕐 Time Formats",
            value="`14:00`, `14h30`, `14h`, `2pm`, `14:00-16:00`",
            inline=False,
        )

    elif category == "tasks":
        embed = discord.Embed(title="📋 Lệnh Tasks", color=discord.Color.orange())
        embed.add_field(
            name="📍 Xem Tasks",
            value=(
                "`!tasks` - Tất cả tasks chưa xong\n"
                "`!tasks today` - Tasks hôm nay\n"
                "`!tasks tomorrow` - Tasks ngày mai\n"
                "`!tasks 18/2` - Tasks ngày 18/2\n"
                "`!overdue` - Tasks quá hạn 🔴"
            ),
            inline=False,
        )
        embed.add_field(
            name="➕ Thêm Task",
            value=(
                "`!add_task <title> | <date time> | <notes>`\n\n"
                "**Ví dụ:**\n"
                "`!add_task Học Python | 20/2 18:00 | Bài 5`\n"
                "`!add_task Đi gym | tomorrow 17:00`\n"
                "`!add_task Nộp bài | friday 23:59`"
            ),
            inline=False,
        )
        embed.add_field(
            name="✅ Đánh Dấu Xong",
            value=(
                "1. `!tasks` - Hiện danh sách có số\n"
                "2. `!done 3` - Hoàn thành task số 3"
            ),
            inline=False,
        )
        embed.add_field(
            name="🗑️ Xóa Task",
            value=(
                "1. `!tasks` - Hiện danh sách có số\n"
                "2. `!del_task 5` - Xóa task số 5"
            ),
            inline=False,
        )

    elif category == "weather":
        embed = discord.Embed(title="🌤️ Lệnh Weather", color=discord.Color.blue())
        embed.add_field(name="!weather", value="Thời tiết hiện tại", inline=False)

    elif category == "countdown":
        embed = discord.Embed(title="⏰ Lệnh Countdown", color=discord.Color.red())
        embed.add_field(
            name="📍 Xem Countdowns",
            value="`!countdown` - Xem tất cả countdowns đang chạy",
            inline=False,
        )
        embed.add_field(
            name="➕ Thêm Countdown",
            value=(
                "`!add_countdown <tên> | <date time> | <emoji>`\n\n"
                "**Ví dụ:**\n"
                "`!add_countdown Sinh nhật | 20/2 00:00 | 🎂`\n"
                "`!add_countdown Deadline | tomorrow 23:59 | ⏰`\n"
                "`!add_countdown Concert | friday 20:00 | 🎸`"
            ),
            inline=False,
        )
        embed.add_field(
            name="🗑️ Xóa Countdown",
            value="`!del_countdown <tên>` - Xóa countdown",
            inline=False,
        )
        embed.add_field(
            name="🎆 New Year Countdown (Đặc Biệt)",
            value=(
                "`!newyear` - Năm mới tự động\n"
                "`!newyear 2026` - Năm mới 2026\n"
                "`!newyear 2026 1 1 23 59` - Custom chính xác\n\n"
                "✨ Format đặc biệt với đếm ngược hoành tráng!"
            ),
            inline=False,
        )
        embed.add_field(
            name="🧧 Tết Countdown",
            value="`!tet` - Tự động countdown Tết Âm lịch gần nhất",
            inline=False,
        )
        embed.add_field(
            name="🔔 Milestones Tự Động",
            value=(
                "**New Year (format đặc biệt):**\n"
                "• Còn 5', 4', 3', 2'\n"
                "• Đếm chi tiết 60s → 0s\n"
                "• Chúc mừng hoành tráng! 🎆\n\n"
                "**Các sự kiện khác:**\n"
                "• Còn 5', 4', 3', 2'\n"
                "• Đếm chi tiết 60s → 0s"
            ),
            inline=False,
        )
        embed.add_field(
            name="📍 Gửi Đến",
            value=f"Tất cả countdown → CHANNEL_MAIN (ID: {MAIN_CHANNEL_ID})",
            inline=False,
        )

    elif category == "study":
        embed = discord.Embed(title="📚 Lệnh Study", color=discord.Color.purple())
        embed.add_field(
            name="📝 Tổng Hợp",
            value=(
                "`!summary` - Tổng hợp tin nhắn hôm nay\n"
                "`/summary channel:<kênh> latest_messages:<N>` - Tổng hợp N tin gần nhất của 1 kênh\n"
                "`/continue_summary` - Tiếp tục phần còn lại\n"
                "`!stats` - Thống kê theo channel\n"
                "`!study_stats` - Xem streak/điểm học tập tháng\n"
                "`!answer <số> | <trả lời>` - Trả lời câu hỏi ôn tập"
            ),
            inline=False,
        )
        embed.add_field(
            name="ℹ️ Lưu Ý",
            value=(
                "• Bot theo dõi tin nhắn trong CHANNELS_TO_MONITOR\n"
                "• Nếu tin nhắn có ảnh/file, bot sẽ kèm URL/tên file vào dữ liệu summary\n"
                "• Tự động tổng hợp lúc 21:00 hàng ngày\n"
                "• Mỗi lần xử lý 50 tin nhắn\n"
                "• `!summary` dùng model chính: `openai/gpt-5-chat`"
            ),
            inline=False,
        )

    elif category == "chatbot":
        embed = discord.Embed(title="💬 Lệnh Chatbot", color=discord.Color.blurple())
        embed.add_field(
            name="💡 Chat trực tiếp",
            value=(
                "`!chat <nội dung>` - Hỏi đáp trực tiếp với AI (hỗ trợ kèm ảnh)\n"
                "`!reason <nội dung>` - Reasoning mode trả lời dễ đọc, không LaTeX\n"
                "Ví dụ: `!reason Tích phân của x^2`"
            ),
            inline=False,
        )
        embed.add_field(
            name="🧠 Model",
            value=(
                "• Chat dùng model chính: `openai/gpt-5`\n"
                "• Ảnh trong chat: `meta/Llama-4-Maverick-17B-128E-Instruct-FP8` → fallback vision\n"
                "• Reasoning ưu tiên: `deepseek/DeepSeek-R1-0528`\n"
                "• Tự fallback nếu lỗi/limit theo biến `.env`"
            ),
            inline=False,
        )
        embed.add_field(
            name="⚙️ Cấu hình `.env`",
            value=(
                "`CHAT_MODEL_PRIMARY`\n"
                "`CHAT_MODEL_FALLBACKS`\n"
                "`VISION_MODEL_PRIMARY`\n"
                "`VISION_MODEL_FALLBACKS`\n"
                "`REASONING_MODEL_PRIMARY`\n"
                "`REASONING_MODEL_FALLBACKS`\n"
                "`SUMMARY_MODEL_PRIMARY`\n"
                "`SUMMARY_MODEL_FALLBACKS`\n"
                "`ANSWER_MODEL_PRIMARY`\n"
                "`ANSWER_MODEL_FALLBACKS`"
            ),
            inline=False,
        )

    elif category == "automation":
        embed = discord.Embed(title="🤖 Automation", color=discord.Color.gold())
        embed.add_field(
            name="⏰ Lịch Tự Động",
            value=(
                "**06:30** - Chào sáng\n"
                "• Thời tiết\n"
                "• Events hôm nay\n"
                "• Tasks hôm nay + quá hạn\n\n"
                "**Mỗi 15 phút** - Nhắc nhở\n"
                "• Events sắp tới (trước 15 phút)\n\n"
                "**20:00** - Review cuối ngày\n"
                "• Tasks chưa xong\n"
                "• Tasks quá hạn (HỐI!)\n\n"
                "**21:00** - Tổng hợp học tập\n"
                "• Tóm tắt tin nhắn\n"
                "• Câu hỏi ôn tập"
            ),
            inline=False,
        )
        embed.add_field(
            name="📍 Gửi Đến",
            value=(
                "Tất cả thông báo tự động gửi vào:\n"
                f"**CHANNEL_MAIN** (ID: {MAIN_CHANNEL_ID})\n\n"
                "Cấu hình trong file `.env`"
            ),
            inline=False,
        )

    else:
        await ctx.send(
            "⚠️ Category: `calendar`, `tasks`, `countdown`, `weather`, `study`, `chatbot`, `automation`"
        )
        return

    await ctx.send(embed=embed)


# ==============================
# COMMANDS - CALENDAR
# ==============================
@bot.command()
async def calendar(ctx, *, date_str=""):
    """Xem toàn bộ lịch: events + tasks"""
    target_date = knowledge_bot.parse_date(date_str) if date_str else None
    calendar_data = await knowledge_bot.get_calendar(target_date)

    date_display = target_date.strftime("%d/%m") if target_date else "hôm nay"
    message = f"📅 **Lịch {date_display}:**\n\n"

    # Events
    events = calendar_data["events"]
    if events:
        message += "**📌 EVENTS:**\n"
        for e in events:
            icon = "🔴" if e["is_important"] else "•"
            time_str = f"{e['time']}-{e['end_time']}" if e["end_time"] else e["time"]
            message += f"{icon} {time_str} {e['summary']}\n"
        message += "\n"

    # Tasks
    tasks = calendar_data["tasks"]
    if tasks:
        message += "**📋 TASKS:**\n"
        for t in tasks:
            icon = "🔴" if t["overdue"] else "•"
            time_str = t["due_time"].strftime("%H:%M") if t["due_time"] else ""
            message += f"{icon} {time_str} {t['title']}\n"

    if not events and not tasks:
        message += "Không có gì cả"

    await ctx.send(message)


@bot.command()
async def events(ctx, *, date_str=""):
    """Xem chỉ events"""
    target_date = knowledge_bot.parse_date(date_str) if date_str else None
    events = await knowledge_bot.get_events(target_date)

    if isinstance(events, str):
        await ctx.send(events)
        return

    if not events:
        date_display = target_date.strftime("%d/%m") if target_date else "hôm nay"
        await ctx.send(f"📅 Không có events {date_display}")
        return

    # Lưu để dùng cho !del_event, !move_event
    _last_events[ctx.author.id] = events

    date_display = target_date.strftime("%d/%m") if target_date else "hôm nay"
    message = f"📅 **Events {date_display}:**\n\n"

    for i, e in enumerate(events, 1):
        icon = "🔴" if e["is_important"] else ""
        time_str = f"{e['time']}-{e['end_time']}" if e["end_time"] else e["time"]
        message += f"{i}. {icon} {time_str} **{e['summary']}**\n"
        if e["description"]:
            message += f"   ↳ {e['description'][:100]}\n"

    await ctx.send(message)


@bot.command()
async def add_event(ctx, *, args):
    """
    Thêm event
    Format: !add_event <title> | <date start-end> | <description>
    Ví dụ: !add_event Họp team | 18/2 14:00-16:00 | Sprint planning
          !add_event Deadline | 20/2 23:59 | Nộp báo cáo
    """
    parts = [p.strip() for p in args.split("|")]
    if len(parts) < 2:
        await ctx.send("⚠️ Format: `!add_event <title> | <date time-endtime> | <desc>`")
        return

    title = parts[0]
    datetime_str = parts[1]
    description = parts[2] if len(parts) > 2 else ""

    # Parse: "18/2 14:00-16:00" hoặc "18/2 14:00"
    # Extract date
    date_match = re.search(
        r"(\d{1,2}[/-]\d{1,2}|today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        datetime_str,
        re.I,
    )
    if not date_match:
        await ctx.send("⚠️ Không tìm thấy ngày. VD: `18/2 14:00-16:00`")
        return

    date_part = date_match.group(1)
    target_date = knowledge_bot.parse_date(date_part)
    if not target_date:
        await ctx.send("⚠️ Ngày không hợp lệ")
        return

    # Extract times
    time_match = re.search(
        r"(\d{1,2}[h:]\d{2}|\d{1,2}h?)\s*-\s*(\d{1,2}[h:]\d{2}|\d{1,2}h?)", datetime_str
    )

    if time_match:
        # Có start-end
        start_time = knowledge_bot.parse_time(time_match.group(1))
        end_time = knowledge_bot.parse_time(time_match.group(2))

        if not start_time or not end_time:
            await ctx.send("⚠️ Giờ không hợp lệ")
            return

        start_dt = knowledge_bot.timezone.localize(
            datetime.combine(target_date, start_time)
        )
        end_dt = knowledge_bot.timezone.localize(
            datetime.combine(target_date, end_time)
        )
    else:
        # Chỉ có start time
        single_time_match = re.search(r"(\d{1,2}[h:]\d{2}|\d{1,2}h?)", datetime_str)
        if not single_time_match:
            await ctx.send("⚠️ Không tìm thấy giờ. VD: `14:00` hoặc `14:00-16:00`")
            return

        start_time = knowledge_bot.parse_time(single_time_match.group(1))
        if not start_time:
            await ctx.send("⚠️ Giờ không hợp lệ")
            return

        start_dt = knowledge_bot.timezone.localize(
            datetime.combine(target_date, start_time)
        )
        end_dt = start_dt + timedelta(hours=1)

    result = await knowledge_bot.add_event(title, start_dt, end_dt, description)
    await ctx.send(result)


@bot.command()
async def del_event(ctx, index: int):
    """Xóa event"""
    if ctx.author.id not in _last_events:
        await ctx.send("⚠️ Gọi `!events` trước")
        return

    events = _last_events[ctx.author.id]
    if index < 1 or index > len(events):
        await ctx.send(f"⚠️ Chọn từ 1-{len(events)}")
        return

    event = events[index - 1]
    result = await knowledge_bot.delete_event(event["id"])
    await ctx.send(result)
    del _last_events[ctx.author.id]


@bot.command()
async def move_event(ctx, *, args):
    """
    Đổi giờ event
    Format: !move_event <số> | <date time>
    VD: !move_event 1 | 19/2 15:00
    """
    parts = [p.strip() for p in args.split("|")]
    if len(parts) < 2:
        await ctx.send("⚠️ Format: `!move_event <số> | <date time>`")
        return

    try:
        index = int(parts[0])
    except:
        await ctx.send("⚠️ Số không hợp lệ")
        return

    if ctx.author.id not in _last_events:
        await ctx.send("⚠️ Gọi `!events` trước")
        return

    events = _last_events[ctx.author.id]
    if index < 1 or index > len(events):
        await ctx.send(f"⚠️ Chọn từ 1-{len(events)}")
        return

    event = events[index - 1]
    datetime_str = parts[1]

    # Parse new datetime
    date_match = re.search(r"(\d{1,2}[/-]\d{1,2}|today|tomorrow)", datetime_str, re.I)
    if not date_match:
        await ctx.send("⚠️ Không tìm thấy ngày")
        return

    target_date = knowledge_bot.parse_date(date_match.group(1))
    time_match = re.search(r"(\d{1,2}[h:]\d{2}|\d{1,2}h?)", datetime_str)

    if not time_match:
        await ctx.send("⚠️ Không tìm thấy giờ")
        return

    new_time = knowledge_bot.parse_time(time_match.group(1))
    new_start = knowledge_bot.timezone.localize(datetime.combine(target_date, new_time))

    # Calculate duration from original
    if event["datetime"] and event["end_datetime"]:
        duration = event["end_datetime"] - event["datetime"]
        new_end = new_start + duration
    else:
        new_end = new_start + timedelta(hours=1)

    result = await knowledge_bot.update_event(
        event["id"],
        start={"dateTime": new_start.isoformat(), "timeZone": "Asia/Ho_Chi_Minh"},
        end={"dateTime": new_end.isoformat(), "timeZone": "Asia/Ho_Chi_Minh"},
    )
    await ctx.send(result)
    del _last_events[ctx.author.id]


# ==============================
# COMMANDS - TASKS
# ==============================
@bot.command()
async def tasks(ctx, *, date_str=""):
    """Xem tasks"""
    target_date = knowledge_bot.parse_date(date_str) if date_str else None
    tasks_list = await knowledge_bot.get_tasks(date=target_date, show_completed=False)

    if isinstance(tasks_list, str):
        await ctx.send(tasks_list)
        return

    if not tasks_list:
        date_display = target_date.strftime("%d/%m") if target_date else ""
        await ctx.send(f"📋 Không có tasks {date_display}")
        return

    _last_tasks[ctx.author.id] = tasks_list

    date_display = target_date.strftime("%d/%m") if target_date else ""
    message = f"📋 **Tasks {date_display}:**\n\n"

    for i, task in enumerate(tasks_list, 1):
        icon = "🔴" if task["overdue"] else "•"
        time_str = task["due_time"].strftime("%H:%M") if task["due_time"] else ""
        due_str = task["due"].strftime("%d/%m") if task["due"] else "Không hạn"
        message += f"{i}. {icon} **{task['title']}** ({due_str} {time_str})\n"
        if task["notes"]:
            message += f"   ↳ {task['notes'][:100]}\n"

    message += f"\n💡 `!done <số>` để hoàn thành"
    await ctx.send(message)


@bot.command()
async def overdue(ctx):
    """Tasks quá hạn"""
    all_tasks = await knowledge_bot.get_tasks(show_completed=False)

    if isinstance(all_tasks, str):
        await ctx.send(all_tasks)
        return

    overdue_tasks = [t for t in all_tasks if t["overdue"]]

    if not overdue_tasks:
        await ctx.send("✅ Không có tasks quá hạn!")
        return

    _last_tasks[ctx.author.id] = overdue_tasks

    message = f"🔴 **Tasks quá hạn ({len(overdue_tasks)}):**\n\n"
    for i, task in enumerate(overdue_tasks, 1):
        due_str = task["due"].strftime("%d/%m") if task["due"] else "N/A"
        message += f"{i}. **{task['title']}** (hạn: {due_str})\n"

    message += f"\n💡 `!done <số>` để hoàn thành"
    await ctx.send(message)


@bot.command()
async def add_task(ctx, *, args):
    """
    Thêm task
    Format: !add_task <title> | <date time> | <notes>
    VD: !add_task Học Python | 20/2 18:00 | Hoàn thành bài 5
        !add_task Đi gym | tomorrow 17:00
    """
    parts = [p.strip() for p in args.split("|")]
    if len(parts) < 1:
        await ctx.send("⚠️ Format: `!add_task <title> | <date time> | <notes>`")
        return

    title = parts[0]
    due_datetime = None
    notes = ""

    if len(parts) >= 2:
        datetime_str = parts[1]

        # Parse date
        date_match = re.search(
            r"(\d{1,2}[/-]\d{1,2}|today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            datetime_str,
            re.I,
        )
        if date_match:
            target_date = knowledge_bot.parse_date(date_match.group(1))

            if target_date:
                # Parse time
                time_match = re.search(r"(\d{1,2}[h:]\d{2}|\d{1,2}h?)", datetime_str)
                if time_match:
                    target_time = knowledge_bot.parse_time(time_match.group(1))
                    if target_time:
                        due_datetime = knowledge_bot.timezone.localize(
                            datetime.combine(target_date, target_time)
                        )
                else:
                    # No time, set to 23:59
                    due_datetime = knowledge_bot.timezone.localize(
                        datetime.combine(target_date, time(23, 59))
                    )

    if len(parts) >= 3:
        notes = parts[2]

    result = await knowledge_bot.add_task(title, due_datetime, notes)
    await ctx.send(result)


@bot.command()
async def done(ctx, index: int):
    """Đánh dấu hoàn thành"""
    if ctx.author.id not in _last_tasks:
        await ctx.send("⚠️ Gọi `!tasks` trước")
        return

    tasks_list = _last_tasks[ctx.author.id]
    if index < 1 or index > len(tasks_list):
        await ctx.send(f"⚠️ Chọn từ 1-{len(tasks_list)}")
        return

    task = tasks_list[index - 1]
    result = await knowledge_bot.complete_task(task["id"], task["tasklist_id"])
    await ctx.send(result)
    del _last_tasks[ctx.author.id]


@bot.command()
async def del_task(ctx, index: int):
    """Xóa task"""
    if ctx.author.id not in _last_tasks:
        await ctx.send("⚠️ Gọi `!tasks` trước")
        return

    tasks_list = _last_tasks[ctx.author.id]
    if index < 1 or index > len(tasks_list):
        await ctx.send(f"⚠️ Chọn từ 1-{len(tasks_list)}")
        return

    task = tasks_list[index - 1]
    result = await knowledge_bot.delete_task(task["id"], task["tasklist_id"])
    await ctx.send(result)
    del _last_tasks[ctx.author.id]


# ==============================
# COMMANDS - WEATHER
# ==============================
@bot.command()
async def weather(ctx):
    """Thời tiết hiện tại"""
    result = await knowledge_bot.get_weather()
    await ctx.send(result)


@bot.command(name="slogan")
async def slogan(ctx):
    if ctx.author.id != YOUR_USER_ID:
        return
    _mark_user_interaction(ctx.author.id)
    text = await _fetch_motivational_slogan()
    await ctx.send(f"💪 **Slogan học tập:**\n*{text}*")


@bot.command()
async def chat(ctx, *, prompt=""):
    """Chat trực tiếp với AI"""
    image_urls = _extract_image_urls_from_attachments(ctx.message.attachments)
    if not prompt.strip() and not image_urls:
        await ctx.send("⚠️ Dùng: `!chat <nội dung cần hỏi>` hoặc đính kèm ảnh")
        return

    prior_context = _pending_chat_context.pop(ctx.author.id, "")

    async with ctx.typing():
        ai_result = await knowledge_bot.chat(
            prompt.strip(),
            ctx.author.display_name,
            image_urls=image_urls,
            prior_context=prior_context,
        )

    if not ai_result["ok"]:
        await ctx.send(f"⚠️ Không thể gọi AI: {ai_result['error']}")
        return

    answer = ai_result["content"].strip()
    display_answer = _format_rich_text_for_discord(answer)
    model_used = ai_result["model"]
    vision_models = ai_result.get("vision_models", [])
    image_extractions = ai_result.get("image_extractions", [])

    session_id = _create_chat_session(
        user_id=ctx.author.id,
        username=ctx.author.display_name,
        prompt=prompt.strip() or "(phân tích ảnh)",
        answer=answer,
        model_used=model_used,
        image_urls=image_urls,
        image_extractions=image_extractions,
        vision_models=vision_models,
    )

    embed = discord.Embed(
        title="💬 Chatbot",
        description=display_answer[:3900],
        color=discord.Color.blurple(),
        timestamp=datetime.now(VIETNAM_TZ),
    )
    embed.add_field(
        name="🙋 Bạn hỏi",
        value=(prompt[:1000] or "(phân tích ảnh đính kèm)"),
        inline=False,
    )
    if prior_context:
        embed.add_field(
            name="🧷 Context đã dùng",
            value="Đã tự động chèn context từ chat trước (do bạn chọn bằng nút).",
            inline=False,
        )
    if image_urls:
        embed.add_field(
            name="🖼️ Ảnh gửi kèm",
            value="\n".join([f"- {url}" for url in image_urls[:3]])[:1024],
            inline=False,
        )
        embed.set_image(url=image_urls[0])
    if vision_models:
        embed.add_field(
            name="🧠 Vision model dùng",
            value=", ".join(vision_models)[:1024],
            inline=False,
        )
    extracted_ok = [x for x in image_extractions if x.get("ok") and x.get("text")]
    if extracted_ok:
        lines = [f"Ảnh {x['index']}: {str(x['text'])[:220]}" for x in extracted_ok[:2]]
        embed.add_field(
            name="🔎 Trích xuất ảnh",
            value="\n".join(lines)[:1024],
            inline=False,
        )
    embed.set_footer(text=f"Đang trả lời bằng: {model_used}")
    await ctx.send(embed=embed, view=ChatSessionView(session_id))

    if len(image_urls) > 1:
        for idx, image_url in enumerate(image_urls[1:4], start=2):
            image_embed = discord.Embed(
                title=f"🖼️ Ảnh đính kèm {idx}",
                color=discord.Color.blurple(),
                timestamp=datetime.now(VIETNAM_TZ),
            )
            image_embed.set_image(url=image_url)
            await ctx.send(embed=image_embed)

    remaining = display_answer[3900:]
    for chunk in _split_text_chunks(remaining, 1900):
        await ctx.send(f"📎 Phần tiếp theo:\n{chunk}")


@bot.command()
async def reason(ctx, *, prompt=""):
    """Reasoning mode hiển thị kết quả rõ ràng, dễ đọc"""
    prompt_clean = prompt.strip()

    if not prompt_clean:
        await ctx.send("⚠️ Dùng: `!reason <nội dung cần phân tích>`")
        return

    async with ctx.typing():
        ai_result = await knowledge_bot.reasoning(
            prompt_clean,
            ctx.author.display_name,
        )

    if not ai_result["ok"]:
        await ctx.send(f"⚠️ Không thể gọi Reasoning AI: {ai_result['error']}")
        return

    answer = (ai_result.get("content") or "").strip()
    if not answer:
        answer = "⚠️ Không có nội dung hiển thị được."
    display_answer = _format_rich_text_for_discord(answer)

    model_used = ai_result["model"]
    combined_message = _build_reason_single_message(
        prompt_clean, display_answer, model_used
    )

    if len(combined_message) <= 2000:
        await ctx.send(combined_message)
    else:
        chunks = _split_text_chunks(combined_message, 1900)
        await ctx.send(chunks[0])
        for chunk in chunks[1:]:
            await ctx.send(chunk)


# ==============================
# COMMANDS - STUDY
# ==============================
@bot.command()
async def summary(ctx):
    """Tổng hợp"""
    if ctx.author.id != YOUR_USER_ID:
        return

    _ensure_daily_window_rollover()
    _mark_user_interaction(ctx.author.id)

    if not daily_messages:
        await ctx.send("📚 Không có tin nhắn")
        return

    penalty = _apply_unanswered_penalty(ctx.author.id)
    if penalty.get("applied"):
        await ctx.send(
            f"⚠️ Bạn còn {penalty.get('count', 0)} câu hỏi chưa trả lời từ phiên trước."
            f" Trừ {abs(int(penalty.get('points_delta', 0)))} điểm."
        )

    _study_questions[ctx.author.id] = []
    question_index = 1

    for channel_id, messages in daily_messages.items():
        discord_channel = bot.get_channel(channel_id)
        channel_name = discord_channel.name if discord_channel else str(channel_id)

        summary_data, has_more = await knowledge_bot.summarize_daily_knowledge(
            messages, channel_name, 0, 50
        )

        if summary_data:
            if summary_data.get("error"):
                await ctx.send(summary_data["error"])
            else:
                embed, numbered_questions = _build_summary_embed(
                    channel_name,
                    len(messages),
                    summary_data,
                    question_start_index=question_index,
                )
                view = SummaryInteractiveView(
                    ctx.author.id,
                    channel_name,
                    summary_data,
                    numbered_questions,
                    has_more=has_more,
                )
                await ctx.send(embed=embed, view=view)

                _persist_questions_for_spaced_repetition(
                    user_id=ctx.author.id,
                    channel_name=channel_name,
                    summary_data=summary_data,
                    numbered_questions=numbered_questions,
                )
                theory_text = _build_question_theory_text(
                    summary_data.get("summary_points", []),
                    summary_data.get("detailed_summary", ""),
                )

                _append_study_event(
                    user_id=ctx.author.id,
                    event_type="summary",
                    points_delta=0,
                    channel_name=channel_name,
                    note=f"Tạo summary với {len(messages)} tin nhắn",
                )

                for item in numbered_questions:
                    _study_questions[ctx.author.id].append(
                        {
                            "index": item["index"],
                            "channel_name": channel_name,
                            "question": item["question"],
                            "summary_points": summary_data.get("summary_points", []),
                            "theory": theory_text,
                        }
                    )
                question_index += len(numbered_questions)

            if has_more:
                summary_state[channel_id] = {
                    "messages": messages,
                    "channel_name": channel_name,
                    "offset": 50,
                }
                await ctx.send(
                    f"💡 Còn {len(messages) - 50} tin nhắn chưa summary. Bấm `Continue Summary` ngay dưới embed vừa gửi hoặc dùng `/continue_summary`.",
                )


@bot.command()
async def answer(ctx, *, args=""):
    """Trả lời câu hỏi ôn tập: !answer <số> | <câu trả lời>"""
    if ctx.author.id != YOUR_USER_ID:
        return

    if "|" not in args:
        await ctx.send("⚠️ Dùng: `!answer <số câu> | <câu trả lời của bạn>`")
        return

    left, user_answer = [x.strip() for x in args.split("|", 1)]
    if not left.isdigit() or not user_answer:
        await ctx.send("⚠️ Dùng: `!answer <số câu> | <câu trả lời của bạn>`")
        return

    question_index = int(left)
    question_bank = _study_questions.get(ctx.author.id, [])
    target_question = next(
        (q for q in question_bank if q["index"] == question_index), None
    )

    if not target_question:
        await ctx.send("⚠️ Không tìm thấy câu hỏi đó. Hãy chạy `!summary` trước.")
        return

    async with ctx.typing():
        review = await knowledge_bot.review_study_answer(
            target_question["question"],
            user_answer,
            target_question.get("summary_points", []),
        )

    if not review["ok"]:
        await ctx.send(review["error"])
        return

    score_value = _normalize_score_value(review.get("score"))
    passed = score_value is not None and score_value >= STUDY_PASS_THRESHOLD
    points_delta = int(STUDY_POINTS_PASS) if passed else 0
    stats = _append_study_event(
        user_id=ctx.author.id,
        event_type="pass" if passed else "answer",
        points_delta=points_delta,
        question_index=question_index,
        channel_name=target_question.get("channel_name", ""),
        score=score_value,
        note=("Đạt ngưỡng" if passed else "Chưa đạt ngưỡng"),
    )
    sm2_result = _record_spaced_review(
        user_id=ctx.author.id,
        target_question=target_question,
        score_value=score_value,
        answered=True,
        note=("Đạt ngưỡng" if passed else "Chưa đạt ngưỡng"),
    )
    _mark_question_answered(ctx.author.id, question_index)

    embed = discord.Embed(
        title=f"🧪 Nhận xét câu {question_index}",
        color=discord.Color.green(),
        timestamp=datetime.now(VIETNAM_TZ),
    )
    embed.add_field(
        name="❓ Câu hỏi", value=target_question["question"][:1024], inline=False
    )
    embed.add_field(
        name="📝 Câu trả lời của bạn", value=user_answer[:1024], inline=False
    )
    embed.add_field(name="📊 Điểm", value=str(review["score"]), inline=True)
    embed.add_field(
        name="💬 Nhận xét", value=str(review["feedback"])[:1024], inline=False
    )
    if review.get("suggestion"):
        embed.add_field(
            name="✅ Gợi ý cải thiện",
            value=str(review["suggestion"])[:1024],
            inline=False,
        )
    embed.add_field(
        name="🔥 Study points",
        value=(
            f"{'+%d' % points_delta if points_delta else '+0'} điểm | "
            f"Tổng: {stats.get('total_points', 0)} | "
            f"Streak: {stats.get('streak_days', 0)} ngày"
        )[:1024],
        inline=False,
    )
    if sm2_result:
        embed.add_field(
            name="🧠 Spaced Repetition",
            value=(
                f"Quality: {sm2_result.get('quality')} | "
                f"Interval: {sm2_result.get('interval_days')} ngày | "
                f"Due: {sm2_result.get('due_date')}"
            )[:1024],
            inline=False,
        )
    embed.set_footer(text=f"Đang trả lời bằng: {review['model']}")
    await ctx.send(embed=embed)


@bot.command()
async def stats(ctx):
    """Thống kê"""
    if not daily_messages:
        await ctx.send("📊 Chưa có tin nhắn")
        return

    message = "📊 **Thống kê:**\n\n"
    total = 0

    for channel_id, messages in daily_messages.items():
        discord_channel = bot.get_channel(channel_id)
        channel_name = discord_channel.name if discord_channel else str(channel_id)
        count = len(messages)
        total += count
        message += f"• #{channel_name}: {count}\n"

    message += f"\n**Tổng:** {total}"
    await ctx.send(message)


@bot.command(name="study_stats")
async def study_stats(ctx):
    if ctx.author.id != YOUR_USER_ID:
        return
    await ctx.send(
        embed=_build_study_metrics_embed(ctx.author.id, ctx.author.display_name)
    )


@bot.command()
async def ping(ctx):
    await ctx.send(f"🏓 Pong! {round(bot.latency * 1000)}ms")


# ==============================
# SLASH COMMANDS (UI MATCHING)
# ==============================
@bot.tree.command(name="help", description="Xem lệnh nhanh của bot")
@app_commands.describe(category="Nhóm lệnh muốn xem")
async def slash_help(
    interaction: discord.Interaction,
    category: str = "overview",
):
    category = (category or "overview").lower().strip()

    embed = discord.Embed(
        title="🤖 Bot Agent - Slash Commands",
        color=discord.Color.blurple(),
        timestamp=datetime.now(VIETNAM_TZ),
    )

    if category in ["overview", "all"]:
        embed.description = (
            "Gõ `/` để Discord tự gợi ý command matching.\n"
            "Lệnh nổi bật: `/calendar`, `/tasks`, `/countdown`, `/chat`, `/summary`."
        )
        embed.add_field(
            name="📅 Calendar",
            value="`/calendar`, `/events`, `/add_event`, `/del_event`, `/move_event`",
            inline=False,
        )
        embed.add_field(
            name="📋 Tasks",
            value="`/tasks`, `/overdue`, `/add_task`, `/done`, `/del_task`",
            inline=False,
        )
        embed.add_field(
            name="⏰ Countdown",
            value="`/countdown`, `/add_countdown`, `/del_countdown`, `/newyear`, `/tet`",
            inline=False,
        )
        embed.add_field(
            name="📚 Study",
            value="`/summary`, `/continue_summary`, `/answer`, `/study_stats`",
            inline=False,
        )
        embed.add_field(name="💬 Chat", value="`/chat`, `/reason`", inline=False)
        embed.add_field(
            name="🛠️ Utility", value="`/weather`, `/ping`, `/stats`", inline=False
        )
        embed.add_field(name="💪 Motivation", value="`/slogan`", inline=False)
    elif category == "calendar":
        embed.description = "Lệnh calendar"
        embed.add_field(name="`/calendar`", value="Xem lịch tổng", inline=False)
        embed.add_field(name="`/events`", value="Xem events", inline=False)
        embed.add_field(name="`/add_event`", value="Thêm event", inline=False)
        embed.add_field(name="`/del_event`", value="Xóa event theo số", inline=False)
        embed.add_field(name="`/move_event`", value="Đổi giờ event", inline=False)
    elif category == "tasks":
        embed.description = "Lệnh tasks"
        embed.add_field(name="`/tasks`", value="Xem tasks", inline=False)
        embed.add_field(name="`/overdue`", value="Xem tasks quá hạn", inline=False)
        embed.add_field(name="`/add_task`", value="Thêm task", inline=False)
        embed.add_field(name="`/done`", value="Đánh dấu hoàn thành", inline=False)
        embed.add_field(name="`/del_task`", value="Xóa task", inline=False)
    elif category == "countdown":
        embed.description = "Lệnh countdown"
        embed.add_field(
            name="`/countdown`", value="Xem countdown đang chạy", inline=False
        )
        embed.add_field(name="`/add_countdown`", value="Thêm countdown", inline=False)
        embed.add_field(name="`/del_countdown`", value="Xóa countdown", inline=False)
        embed.add_field(name="`/newyear`", value="Bật countdown năm mới", inline=False)
        embed.add_field(name="`/tet`", value="Bật countdown tết", inline=False)
    elif category == "study":
        embed.description = "Lệnh học tập"
        embed.add_field(
            name="`/summary`",
            value=(
                "Tổng hợp + tạo câu hỏi\n"
                "• mode `cache`: dùng dữ liệu lưu trong ngày\n"
                "• mode `channel`: fetch 1 kênh + `latest_messages` (tối đa 20), chọn kênh qua autocomplete\n"
                "• mode `all`: quét kênh theo dõi, chỉ summary kênh có tin mới"
            ),
            inline=False,
        )
        embed.add_field(
            name="`/continue_summary`", value="Tiếp tục phần còn lại", inline=False
        )
        embed.add_field(
            name="`/answer`", value="Trả lời câu hỏi và nhận xét", inline=False
        )
        embed.add_field(
            name="`/study_stats`", value="Xem streak/điểm học tập tháng", inline=False
        )
    elif category == "chat":
        embed.description = "Lệnh chatbot"
        embed.add_field(
            name="`/chat`",
            value="Chat trực tiếp với AI (hỗ trợ tối đa 4 ảnh: image_1..image_4)",
            inline=False,
        )
        embed.add_field(
            name="`/reason`",
            value="Reasoning mode trả lời rõ ràng, dễ đọc, không dùng LaTeX",
            inline=False,
        )
    elif category == "utility":
        embed.description = "Lệnh tiện ích"
        embed.add_field(
            name="`/weather`",
            value="Xem thời tiết hiện tại hoặc forecast theo ngày/giờ",
            inline=False,
        )
        embed.add_field(name="`/ping`", value="Kiểm tra độ trễ bot", inline=False)
        embed.add_field(
            name="`/stats`", value="Thống kê tin nhắn theo dõi", inline=False
        )
        embed.add_field(
            name="`/slogan`", value="In slogan tạo động lực học", inline=False
        )
    else:
        embed.description = (
            "Category hợp lệ: `overview`, `calendar`, `tasks`, `countdown`, "
            "`study`, `chat`, `utility`"
        )

    await interaction.response.send_message(embed=embed, ephemeral=True)


@bot.tree.command(name="ping", description="Kiểm tra độ trễ của bot")
async def slash_ping(interaction: discord.Interaction):
    await interaction.response.send_message(
        f"🏓 Pong! {round(bot.latency * 1000)}ms", ephemeral=True
    )


@bot.tree.command(
    name="weather", description="Xem thời tiết hiện tại hoặc forecast theo ngày/giờ"
)
@app_commands.describe(
    date="Ngày cần xem: today, tomorrow, 18/2...",
    hour="Giờ cần xem: 14:00, 9h, 18h30...",
)
async def slash_weather(
    interaction: discord.Interaction,
    date: str = "",
    hour: str = "",
):
    target_date = knowledge_bot.parse_date(date) if date else None
    target_time = knowledge_bot.parse_time(hour) if hour else None

    if date and not target_date:
        await interaction.response.send_message(
            "⚠️ Ngày không hợp lệ. VD: `today`, `tomorrow`, `18/2`.",
            ephemeral=True,
        )
        return

    if hour and not target_time:
        await interaction.response.send_message(
            "⚠️ Giờ không hợp lệ. VD: `14:00`, `9h`, `18h30`.",
            ephemeral=True,
        )
        return

    await interaction.response.defer(thinking=True)
    result = await knowledge_bot.get_weather(
        target_date=target_date, target_time=target_time
    )
    await interaction.followup.send(result)


@bot.tree.command(name="slogan", description="Nhận 1 slogan tạo động lực học tập")
async def slash_slogan(interaction: discord.Interaction):
    if interaction.user.id != YOUR_USER_ID:
        await interaction.response.send_message(
            "⛔ Bạn không có quyền dùng lệnh này.", ephemeral=True
        )
        return
    _mark_user_interaction(interaction.user.id)
    text = await _fetch_motivational_slogan()
    await interaction.response.send_message(f"💪 **Slogan học tập:**\n*{text}*")


@bot.tree.command(name="chat", description="Chat trực tiếp với AI")
@app_commands.describe(
    prompt="Nội dung bạn muốn hỏi",
    image_1="Ảnh 1 (tuỳ chọn)",
    image_2="Ảnh 2 (tuỳ chọn)",
    image_3="Ảnh 3 (tuỳ chọn)",
    image_4="Ảnh 4 (tuỳ chọn)",
)
async def slash_chat(
    interaction: discord.Interaction,
    prompt: str = "",
    image_1: discord.Attachment = None,
    image_2: discord.Attachment = None,
    image_3: discord.Attachment = None,
    image_4: discord.Attachment = None,
):
    provided_images = [image_1, image_2, image_3, image_4]
    image_urls = _extract_image_urls_from_attachments([x for x in provided_images if x])

    if not prompt.strip() and not image_urls:
        await interaction.response.send_message(
            "⚠️ Nhập prompt hoặc đính kèm ảnh.", ephemeral=True
        )
        return

    prior_context = _pending_chat_context.pop(interaction.user.id, "")

    await interaction.response.defer(thinking=True)

    ai_result = await knowledge_bot.chat(
        prompt.strip(),
        interaction.user.display_name,
        image_urls=image_urls,
        prior_context=prior_context,
    )
    if not ai_result["ok"]:
        await interaction.followup.send(f"⚠️ Không thể gọi AI: {ai_result['error']}")
        return

    answer = ai_result["content"].strip()
    display_answer = _format_rich_text_for_discord(answer)
    model_used = ai_result["model"]
    vision_models = ai_result.get("vision_models", [])
    image_extractions = ai_result.get("image_extractions", [])

    session_id = _create_chat_session(
        user_id=interaction.user.id,
        username=interaction.user.display_name,
        prompt=prompt.strip() or "(phân tích ảnh)",
        answer=answer,
        model_used=model_used,
        image_urls=image_urls,
        image_extractions=image_extractions,
        vision_models=vision_models,
    )

    embed = discord.Embed(
        title="💬 Chatbot",
        description=display_answer[:3900],
        color=discord.Color.blurple(),
        timestamp=datetime.now(VIETNAM_TZ),
    )
    embed.add_field(
        name="🙋 Bạn hỏi",
        value=(prompt[:1000] or "(phân tích ảnh đính kèm)"),
        inline=False,
    )
    if prior_context:
        embed.add_field(
            name="🧷 Context đã dùng",
            value="Đã tự động chèn context từ chat trước (do bạn chọn bằng nút).",
            inline=False,
        )
    if image_urls:
        embed.add_field(
            name="🖼️ Ảnh gửi kèm",
            value="\n".join([f"- {url}" for url in image_urls[:3]])[:1024],
            inline=False,
        )
        embed.set_image(url=image_urls[0])
    if vision_models:
        embed.add_field(
            name="🧠 Vision model dùng",
            value=", ".join(vision_models)[:1024],
            inline=False,
        )
    extracted_ok = [x for x in image_extractions if x.get("ok") and x.get("text")]
    if extracted_ok:
        lines = [f"Ảnh {x['index']}: {str(x['text'])[:220]}" for x in extracted_ok[:2]]
        embed.add_field(
            name="🔎 Trích xuất ảnh",
            value="\n".join(lines)[:1024],
            inline=False,
        )
    embed.set_footer(text=f"Đang trả lời bằng: {model_used}")
    await interaction.followup.send(embed=embed, view=ChatSessionView(session_id))

    if len(image_urls) > 1:
        for idx, image_url in enumerate(image_urls[1:4], start=2):
            image_embed = discord.Embed(
                title=f"🖼️ Ảnh đính kèm {idx}",
                color=discord.Color.blurple(),
                timestamp=datetime.now(VIETNAM_TZ),
            )
            image_embed.set_image(url=image_url)
            await interaction.followup.send(embed=image_embed)

    remaining = display_answer[3900:]
    for chunk in _split_text_chunks(remaining, 1900):
        await interaction.followup.send(f"📎 Phần tiếp theo:\n{chunk}")


@bot.tree.command(name="reason", description="Reasoning mode trả lời rõ ràng, dễ đọc")
@app_commands.describe(prompt="Nội dung cần reasoning")
async def slash_reason(
    interaction: discord.Interaction,
    prompt: str,
):
    await interaction.response.defer(thinking=True)

    ai_result = await knowledge_bot.reasoning(
        prompt.strip(),
        interaction.user.display_name,
    )
    if not ai_result["ok"]:
        await _safe_followup_send(
            interaction, f"⚠️ Không thể gọi Reasoning AI: {ai_result['error']}"
        )
        return

    answer = (ai_result.get("content") or "").strip()
    if not answer:
        answer = "⚠️ Không có nội dung hiển thị được."
    display_answer = _format_rich_text_for_discord(answer)

    model_used = ai_result["model"]
    combined_message = _build_reason_single_message(prompt, display_answer, model_used)

    if len(combined_message) <= 2000:
        await _safe_followup_send(interaction, combined_message)
    else:
        chunks = _split_text_chunks(combined_message, 1900)
        await _safe_followup_send(interaction, chunks[0])
        for chunk in chunks[1:]:
            await _safe_followup_send(interaction, chunk)


@bot.tree.command(name="summary", description="Tổng hợp học tập và tạo câu hỏi ôn tập")
@app_commands.describe(
    mode="cache: dữ liệu lưu trong ngày | channel: fetch 1 kênh | all: quét các kênh có tin mới",
    channel_option="Chọn channel từ danh sách gợi ý (hoặc all)",
    latest_messages="Số tin gần nhất khi fetch (tối đa 20). Bỏ trống ở mode=all để chỉ lấy tin mới trong hôm nay",
)
@app_commands.choices(
    mode=[
        app_commands.Choice(name="cache", value="cache"),
        app_commands.Choice(name="channel", value="channel"),
        app_commands.Choice(name="all", value="all"),
    ]
)
async def slash_summary(
    interaction: discord.Interaction,
    mode: app_commands.Choice[str] = None,
    channel_option: str = "",
    latest_messages: int = None,
):
    if interaction.user.id != YOUR_USER_ID:
        await interaction.response.send_message(
            "⛔ Bạn không có quyền dùng lệnh này.", ephemeral=True
        )
        return

    _ensure_daily_window_rollover()
    _mark_user_interaction(interaction.user.id)

    if latest_messages is not None and not (
        1 <= int(latest_messages) <= SUMMARY_FETCH_MAX_MESSAGES
    ):
        await interaction.response.send_message(
            f"⚠️ `latest_messages` chỉ nhận giá trị từ 1 đến {SUMMARY_FETCH_MAX_MESSAGES}.",
            ephemeral=True,
        )
        return

    fetch_limit = (
        int(latest_messages)
        if latest_messages is not None
        else SUMMARY_FETCH_MAX_MESSAGES
    )

    source_batches = []
    fetch_checkpoints = {}
    selected_mode = (mode.value if mode else "cache").lower().strip()
    selected_channel_option = (channel_option or "").strip().lower()

    def _resolve_channel_by_option(option_text):
        if not option_text:
            return None
        if option_text == "all":
            return "all"
        if option_text.isdigit():
            return bot.get_channel(int(option_text))

        for channel_id in CHANNELS_TO_MONITOR:
            channel_obj = bot.get_channel(channel_id)
            if channel_obj and channel_obj.name.lower() == option_text.lower():
                return channel_obj
            if (
                f"#{channel_obj.name}".lower() == option_text.lower()
                if channel_obj
                else False
            ):
                return channel_obj
        return None

    if selected_mode == "cache":
        if not daily_messages:
            await interaction.response.send_message(
                "📚 Không có tin nhắn", ephemeral=True
            )
            return
        for channel_id, messages in daily_messages.items():
            discord_channel = bot.get_channel(channel_id)
            channel_name = discord_channel.name if discord_channel else str(channel_id)
            source_batches.append((channel_id, channel_name, messages))
    elif selected_mode == "channel":
        if not selected_channel_option:
            await interaction.response.send_message(
                "⚠️ Chọn `channel_option` khi dùng mode `channel`.", ephemeral=True
            )
            return

        resolved = _resolve_channel_by_option(selected_channel_option)
        if resolved == "all":
            await interaction.response.send_message(
                "⚠️ mode `channel` không dùng `all`. Hãy chọn 1 channel cụ thể.",
                ephemeral=True,
            )
            return

        channel = resolved
        if channel is None:
            await interaction.response.send_message(
                "⚠️ Không tìm thấy channel theo lựa chọn.", ephemeral=True
            )
            return

        fetched_messages, newest_id = await _collect_new_messages_since(
            channel,
            after_message_id=None,
            latest_messages=fetch_limit,
        )
        if not fetched_messages:
            await interaction.response.send_message(
                f"📚 Không có tin nhắn phù hợp trong #{channel.name}.",
                ephemeral=True,
            )
            return
        source_batches.append((channel.id, channel.name, fetched_messages))
        if newest_id:
            fetch_checkpoints[channel.id] = newest_id
    elif selected_mode == "all":
        if selected_channel_option and selected_channel_option != "all":
            await interaction.response.send_message(
                "⚠️ mode `all` chỉ nhận `channel_option=all` hoặc để trống.",
                ephemeral=True,
            )
            return

        for channel_id in CHANNELS_TO_MONITOR:
            discord_channel = bot.get_channel(channel_id)
            if discord_channel is None:
                continue

            if latest_messages is None:
                last_checkpoint = _last_summary_fetch_message_ids.get(channel_id)
                fetched_messages, newest_id = await _collect_new_messages_since(
                    discord_channel,
                    after_message_id=last_checkpoint,
                    latest_messages=SUMMARY_FETCH_MAX_MESSAGES,
                    only_today=True,
                )
            else:
                fetched_messages, newest_id = await _collect_new_messages_since(
                    discord_channel,
                    after_message_id=None,
                    latest_messages=fetch_limit,
                    only_today=False,
                )

            if not fetched_messages:
                continue

            source_batches.append((channel_id, discord_channel.name, fetched_messages))
            if newest_id:
                fetch_checkpoints[channel_id] = newest_id

        if not source_batches:
            await interaction.response.send_message(
                "📚 Không có channel nào có tin nhắn mới để summary.",
                ephemeral=True,
            )
            return
    else:
        await interaction.response.send_message(
            "⚠️ mode không hợp lệ. Dùng: cache, channel, all.",
            ephemeral=True,
        )
        return

    await interaction.response.defer(thinking=True)

    penalty = _apply_unanswered_penalty(interaction.user.id)
    if penalty.get("applied"):
        await interaction.followup.send(
            f"⚠️ Bạn còn {penalty.get('count', 0)} câu hỏi chưa trả lời từ phiên trước."
            f" Trừ {abs(int(penalty.get('points_delta', 0)))} điểm."
        )

    _study_questions[interaction.user.id] = []
    question_index = 1

    for channel_id, channel_name, messages in source_batches:

        summary_data, has_more = await knowledge_bot.summarize_daily_knowledge(
            messages, channel_name, 0, SUMMARY_BATCH_SIZE
        )

        if summary_data.get("error"):
            await interaction.followup.send(summary_data["error"])
            continue

        embed, numbered_questions = _build_summary_embed(
            channel_name,
            len(messages),
            summary_data,
            question_start_index=question_index,
        )
        view = SummaryInteractiveView(
            interaction.user.id,
            channel_name,
            summary_data,
            numbered_questions,
            has_more=has_more,
        )
        await interaction.followup.send(embed=embed, view=view)

        _persist_questions_for_spaced_repetition(
            user_id=interaction.user.id,
            channel_name=channel_name,
            summary_data=summary_data,
            numbered_questions=numbered_questions,
        )
        theory_text = _build_question_theory_text(
            summary_data.get("summary_points", []),
            summary_data.get("detailed_summary", ""),
        )

        _append_study_event(
            user_id=interaction.user.id,
            event_type="summary",
            points_delta=0,
            channel_name=channel_name,
            note=f"Tạo summary với {len(messages)} tin nhắn",
        )

        for item in numbered_questions:
            _study_questions[interaction.user.id].append(
                {
                    "index": item["index"],
                    "channel_name": channel_name,
                    "question": item["question"],
                    "summary_points": summary_data.get("summary_points", []),
                    "theory": theory_text,
                }
            )
        question_index += len(numbered_questions)

        if has_more:
            summary_state[channel_id] = {
                "messages": messages,
                "channel_name": channel_name,
                "offset": SUMMARY_BATCH_SIZE,
            }
            await interaction.followup.send(
                f"💡 Còn {len(messages) - SUMMARY_BATCH_SIZE} tin nhắn chưa summary trong #{channel_name}. Bấm `Continue Summary` ngay dưới embed vừa gửi hoặc dùng `/continue_summary`.",
            )

    for channel_id, newest_id in fetch_checkpoints.items():
        _last_summary_fetch_message_ids[channel_id] = int(newest_id)


@slash_summary.autocomplete("channel_option")
async def slash_summary_channel_option_autocomplete(
    interaction: discord.Interaction, current: str
):
    return await summary_channel_autocomplete(interaction, current)


@bot.tree.command(name="continue_summary", description="Tiếp tục summary phần còn lại")
async def slash_continue_summary(interaction: discord.Interaction):
    _ensure_daily_window_rollover()
    _mark_user_interaction(interaction.user.id)
    await interaction.response.defer(thinking=True)
    result = await _continue_summary_for_user(interaction.user.id)
    if not result.get("ok"):
        await interaction.followup.send(
            result.get("message", "⚠️ Có lỗi khi continue summary")
        )
        return

    await interaction.followup.send(
        embed=result.get("embed"),
        view=SummaryInteractiveView(
            interaction.user.id,
            result.get("channel_name", "unknown"),
            result.get("summary_data", {}),
            result.get("numbered_questions", []),
            has_more=result.get("has_more"),
        ),
    )

    if result.get("has_more"):
        await interaction.followup.send(
            f"💡 Còn {result.get('remaining', 0)} tin nhắn chưa summary. Bấm `Continue Summary` hoặc dùng `/continue_summary`."
        )
    else:
        await interaction.followup.send("✅ Đã summary xong toàn bộ phần còn lại.")


@bot.tree.command(name="answer", description="Trả lời câu hỏi ôn tập và nhận xét")
@app_commands.describe(
    question_number="Số thứ tự câu hỏi từ summary",
    user_answer="Câu trả lời của bạn",
)
async def slash_answer(
    interaction: discord.Interaction,
    question_number: int,
    user_answer: str,
):
    if interaction.user.id != YOUR_USER_ID:
        await interaction.response.send_message(
            "⛔ Bạn không có quyền dùng lệnh này.", ephemeral=True
        )
        return

    question_bank = _study_questions.get(interaction.user.id, [])
    target_question = next(
        (q for q in question_bank if q["index"] == question_number), None
    )

    if not target_question:
        await interaction.response.send_message(
            "⚠️ Không tìm thấy câu hỏi đó. Hãy chạy `/summary` trước.",
            ephemeral=True,
        )
        return

    await interaction.response.defer(thinking=True)
    review = await knowledge_bot.review_study_answer(
        target_question["question"],
        user_answer,
        target_question.get("summary_points", []),
    )
    if not review["ok"]:
        await interaction.followup.send(review["error"])
        return

    score_value = _normalize_score_value(review.get("score"))
    passed = score_value is not None and score_value >= STUDY_PASS_THRESHOLD
    points_delta = int(STUDY_POINTS_PASS) if passed else 0
    stats = _append_study_event(
        user_id=interaction.user.id,
        event_type="pass" if passed else "answer",
        points_delta=points_delta,
        question_index=question_number,
        channel_name=target_question.get("channel_name", ""),
        score=score_value,
        note=("Đạt ngưỡng" if passed else "Chưa đạt ngưỡng"),
    )
    sm2_result = _record_spaced_review(
        user_id=interaction.user.id,
        target_question=target_question,
        score_value=score_value,
        answered=True,
        note=("Đạt ngưỡng" if passed else "Chưa đạt ngưỡng"),
    )
    _mark_question_answered(interaction.user.id, question_number)

    embed = discord.Embed(
        title=f"🧪 Nhận xét câu {question_number}",
        color=discord.Color.green(),
        timestamp=datetime.now(VIETNAM_TZ),
    )
    embed.add_field(
        name="❓ Câu hỏi", value=target_question["question"][:1024], inline=False
    )
    embed.add_field(
        name="📝 Câu trả lời của bạn", value=user_answer[:1024], inline=False
    )
    embed.add_field(name="📊 Điểm", value=str(review["score"]), inline=True)
    embed.add_field(
        name="💬 Nhận xét", value=str(review["feedback"])[:1024], inline=False
    )
    if review.get("suggestion"):
        embed.add_field(
            name="✅ Gợi ý cải thiện",
            value=str(review["suggestion"])[:1024],
            inline=False,
        )
    embed.add_field(
        name="🔥 Study points",
        value=(
            f"{'+%d' % points_delta if points_delta else '+0'} điểm | "
            f"Tổng: {stats.get('total_points', 0)} | "
            f"Streak: {stats.get('streak_days', 0)} ngày"
        )[:1024],
        inline=False,
    )
    if sm2_result:
        embed.add_field(
            name="🧠 Spaced Repetition",
            value=(
                f"Quality: {sm2_result.get('quality')} | "
                f"Interval: {sm2_result.get('interval_days')} ngày | "
                f"Due: {sm2_result.get('due_date')}"
            )[:1024],
            inline=False,
        )
    embed.set_footer(text=f"Đang trả lời bằng: {review['model']}")
    await interaction.followup.send(embed=embed)


@bot.tree.command(
    name="study_stats", description="Xem streak và điểm học tập tháng hiện tại"
)
async def slash_study_stats(interaction: discord.Interaction):
    if interaction.user.id != YOUR_USER_ID:
        await interaction.response.send_message(
            "⛔ Bạn không có quyền dùng lệnh này.", ephemeral=True
        )
        return
    await interaction.response.send_message(
        embed=_build_study_metrics_embed(
            interaction.user.id, interaction.user.display_name
        ),
        ephemeral=True,
    )


@bot.tree.command(
    name="knowledge_history",
    description="Xem các thẻ kiến thức đã học trong N ngày gần đây",
)
@app_commands.describe(days="Số ngày cần xem (mặc định 7)")
async def slash_knowledge_history(interaction: discord.Interaction, days: int = 7):
    if interaction.user.id != YOUR_USER_ID:
        await interaction.response.send_message(
            "⛔ Bạn không có quyền dùng lệnh này.", ephemeral=True
        )
        return

    days = max(1, min(int(days or 7), 60))
    db_path = _ensure_study_memory_tables()
    rows = study_memory.get_knowledge_by_days(
        db_path=db_path,
        user_id=interaction.user.id,
        days=days,
    )

    if not rows:
        await interaction.response.send_message(
            f"📚 Chưa có dữ liệu học tập trong {days} ngày gần đây.",
            ephemeral=True,
        )
        return

    embed = discord.Embed(
        title=f"📘 Knowledge History ({days} ngày)",
        color=discord.Color.blurple(),
        timestamp=datetime.now(VIETNAM_TZ),
    )
    preview_lines = []
    for item in rows[:10]:
        channel_name = (item.get("channel_name") or "unknown").strip()
        question = str(item.get("question") or "").strip()[:90]
        score = item.get("last_score")
        score_text = "N/A" if score is None else str(score)
        due = item.get("due_date") or "N/A"
        weak = "⚠️" if int(item.get("weak_flag") or 0) == 1 else "✅"
        preview_lines.append(
            f"{weak} #{channel_name}: {question} (score: {score_text}, due: {due})"
        )

    embed.description = "\n".join(preview_lines)[:4000]
    embed.set_footer(text=f"Hiển thị {min(len(rows), 10)}/{len(rows)} thẻ gần nhất")
    await interaction.response.send_message(embed=embed, ephemeral=True)


@bot.tree.command(
    name="adaptive_path",
    description="Gợi ý lộ trình ôn tập tuần dựa trên các điểm yếu",
)
@app_commands.describe(days="Số ngày dùng để phân tích dữ liệu (mặc định 7)")
async def slash_adaptive_path(interaction: discord.Interaction, days: int = 7):
    if interaction.user.id != YOUR_USER_ID:
        await interaction.response.send_message(
            "⛔ Bạn không có quyền dùng lệnh này.", ephemeral=True
        )
        return

    days = max(1, min(int(days or 7), 60))
    db_path = _ensure_study_memory_tables()
    plan = study_memory.build_adaptive_path(
        db_path=db_path,
        user_id=interaction.user.id,
        days=days,
    )

    embed = discord.Embed(
        title=f"🧠 Adaptive Learning Path ({days} ngày)",
        color=discord.Color.purple(),
        timestamp=datetime.now(VIETNAM_TZ),
    )

    focus_topics = plan.get("focus_topics", [])
    if focus_topics:
        topic_lines = []
        for topic in focus_topics[:5]:
            topic_lines.append(
                f"• {topic.get('topic', 'general')} | TB: {topic.get('avg_score', 'N/A')} | yếu: {topic.get('weak_count', 0)}"
            )
        embed.add_field(
            name="🎯 Chủ đề ưu tiên",
            value="\n".join(topic_lines)[:1024],
            inline=False,
        )

    weak_items = plan.get("weak_items", [])
    if weak_items:
        weak_lines = []
        for item in weak_items[:5]:
            channel_name = (item.get("channel_name") or "unknown").strip()
            question = str(item.get("question") or "").strip()[:80]
            weak_lines.append(f"• #{channel_name}: {question}")
        embed.add_field(
            name="⚠️ Thẻ yếu/chưa trả lời",
            value="\n".join(weak_lines)[:1024],
            inline=False,
        )

    actions = plan.get("next_actions", []) or ["Tiếp tục duy trì lịch học đều."]
    embed.add_field(
        name="🗺️ Gợi ý hành động",
        value="\n".join([f"• {a}" for a in actions[:5]])[:1024],
        inline=False,
    )
    await interaction.response.send_message(embed=embed, ephemeral=True)


# ==============================
# SLASH COMMANDS - CALENDAR
# ==============================
@bot.tree.command(name="calendar", description="Xem lịch (events + tasks)")
@app_commands.describe(date="Ngày cần xem: today, tomorrow, 18/2...")
async def slash_calendar(interaction: discord.Interaction, date: str = ""):
    target_date = knowledge_bot.parse_date(date) if date else None
    calendar_data = await knowledge_bot.get_calendar(target_date)

    date_display = target_date.strftime("%d/%m") if target_date else "hôm nay"
    events = calendar_data["events"]
    tasks_list = calendar_data["tasks"]
    embed = build_calendar_embed(
        date_display=date_display,
        events=events,
        tasks=tasks_list,
        timestamp=datetime.now(VIETNAM_TZ),
    )
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="events", description="Xem danh sách events")
@app_commands.describe(date="Ngày cần xem: today, tomorrow, 18/2...")
async def slash_events(interaction: discord.Interaction, date: str = ""):
    target_date = knowledge_bot.parse_date(date) if date else None
    events = await knowledge_bot.get_events(target_date)

    if isinstance(events, str):
        await interaction.response.send_message(events)
        return

    if not events:
        date_display = target_date.strftime("%d/%m") if target_date else "hôm nay"
        await interaction.response.send_message(f"📅 Không có events {date_display}")
        return

    _last_events[interaction.user.id] = events

    date_display = target_date.strftime("%d/%m") if target_date else "hôm nay"
    embed = build_events_embed(
        date_display=date_display,
        events=events,
        timestamp=datetime.now(VIETNAM_TZ),
    )
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="add_event", description="Thêm event vào Google Calendar")
@app_commands.describe(
    title="Tiêu đề event",
    datetime_input="Ví dụ: 18/2 14:00-16:00 hoặc tomorrow 19:00",
    description="Mô tả (tuỳ chọn)",
)
async def slash_add_event(
    interaction: discord.Interaction,
    title: str,
    datetime_input: str,
    description: str = "",
):
    date_match = re.search(
        r"(\d{1,2}[/-]\d{1,2}|today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        datetime_input,
        re.I,
    )
    if not date_match:
        await interaction.response.send_message(
            "⚠️ Không tìm thấy ngày. VD: `18/2 14:00-16:00`", ephemeral=True
        )
        return

    date_part = date_match.group(1)
    target_date = knowledge_bot.parse_date(date_part)
    if not target_date:
        await interaction.response.send_message("⚠️ Ngày không hợp lệ", ephemeral=True)
        return

    time_match = re.search(
        r"(\d{1,2}[h:]\d{2}|\d{1,2}h?)\s*-\s*(\d{1,2}[h:]\d{2}|\d{1,2}h?)",
        datetime_input,
    )

    if time_match:
        start_time = knowledge_bot.parse_time(time_match.group(1))
        end_time = knowledge_bot.parse_time(time_match.group(2))

        if not start_time or not end_time:
            await interaction.response.send_message(
                "⚠️ Giờ không hợp lệ", ephemeral=True
            )
            return

        start_dt = knowledge_bot.timezone.localize(
            datetime.combine(target_date, start_time)
        )
        end_dt = knowledge_bot.timezone.localize(
            datetime.combine(target_date, end_time)
        )
    else:
        single_time_match = re.search(r"(\d{1,2}[h:]\d{2}|\d{1,2}h?)", datetime_input)
        if not single_time_match:
            await interaction.response.send_message(
                "⚠️ Không tìm thấy giờ. VD: `14:00` hoặc `14:00-16:00`",
                ephemeral=True,
            )
            return

        start_time = knowledge_bot.parse_time(single_time_match.group(1))
        if not start_time:
            await interaction.response.send_message(
                "⚠️ Giờ không hợp lệ", ephemeral=True
            )
            return

        start_dt = knowledge_bot.timezone.localize(
            datetime.combine(target_date, start_time)
        )
        end_dt = start_dt + timedelta(hours=1)

    await interaction.response.defer(thinking=True)
    result = await knowledge_bot.add_event(title, start_dt, end_dt, description)
    await interaction.followup.send(result)


@bot.tree.command(name="del_event", description="Xoá event theo số thứ tự từ /events")
@app_commands.describe(index="Số thứ tự event")
async def slash_del_event(interaction: discord.Interaction, index: int):
    if interaction.user.id not in _last_events:
        await interaction.response.send_message("⚠️ Gọi `/events` trước", ephemeral=True)
        return

    events = _last_events[interaction.user.id]
    if index < 1 or index > len(events):
        await interaction.response.send_message(
            f"⚠️ Chọn từ 1-{len(events)}", ephemeral=True
        )
        return

    await interaction.response.defer(thinking=True)
    event = events[index - 1]
    result = await knowledge_bot.delete_event(event["id"])
    await interaction.followup.send(result)
    del _last_events[interaction.user.id]


@bot.tree.command(name="move_event", description="Đổi giờ event theo số từ /events")
@app_commands.describe(
    index="Số thứ tự event",
    datetime_input="Ngày giờ mới. VD: 19/2 15:00",
)
async def slash_move_event(
    interaction: discord.Interaction, index: int, datetime_input: str
):
    if interaction.user.id not in _last_events:
        await interaction.response.send_message("⚠️ Gọi `/events` trước", ephemeral=True)
        return

    events = _last_events[interaction.user.id]
    if index < 1 or index > len(events):
        await interaction.response.send_message(
            f"⚠️ Chọn từ 1-{len(events)}", ephemeral=True
        )
        return

    date_match = re.search(r"(\d{1,2}[/-]\d{1,2}|today|tomorrow)", datetime_input, re.I)
    if not date_match:
        await interaction.response.send_message("⚠️ Không tìm thấy ngày", ephemeral=True)
        return

    target_date = knowledge_bot.parse_date(date_match.group(1))
    time_match = re.search(r"(\d{1,2}[h:]\d{2}|\d{1,2}h?)", datetime_input)
    if not time_match:
        await interaction.response.send_message("⚠️ Không tìm thấy giờ", ephemeral=True)
        return

    new_time = knowledge_bot.parse_time(time_match.group(1))
    new_start = knowledge_bot.timezone.localize(datetime.combine(target_date, new_time))

    event = events[index - 1]
    if event["datetime"] and event["end_datetime"]:
        duration = event["end_datetime"] - event["datetime"]
        new_end = new_start + duration
    else:
        new_end = new_start + timedelta(hours=1)

    await interaction.response.defer(thinking=True)
    result = await knowledge_bot.update_event(
        event["id"],
        start={"dateTime": new_start.isoformat(), "timeZone": "Asia/Ho_Chi_Minh"},
        end={"dateTime": new_end.isoformat(), "timeZone": "Asia/Ho_Chi_Minh"},
    )
    await interaction.followup.send(result)
    del _last_events[interaction.user.id]


# ==============================
# SLASH COMMANDS - TASKS
# ==============================
@bot.tree.command(name="tasks", description="Xem danh sách tasks")
@app_commands.describe(date="Ngày cần xem: today, tomorrow, 18/2...")
async def slash_tasks(interaction: discord.Interaction, date: str = ""):
    target_date = knowledge_bot.parse_date(date) if date else None
    tasks_list = await knowledge_bot.get_tasks(date=target_date, show_completed=False)

    if isinstance(tasks_list, str):
        await interaction.response.send_message(tasks_list)
        return

    if not tasks_list:
        date_display = target_date.strftime("%d/%m") if target_date else ""
        await interaction.response.send_message(f"📋 Không có tasks {date_display}")
        return

    _last_tasks[interaction.user.id] = tasks_list

    date_display = target_date.strftime("%d/%m") if target_date else ""
    embed = build_tasks_embed(
        date_display=date_display,
        tasks=tasks_list,
        timestamp=datetime.now(VIETNAM_TZ),
        overdue_only=False,
    )
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="overdue", description="Xem tasks quá hạn")
async def slash_overdue(interaction: discord.Interaction):
    all_tasks = await knowledge_bot.get_tasks(show_completed=False)

    if isinstance(all_tasks, str):
        await interaction.response.send_message(all_tasks)
        return

    overdue_tasks = [t for t in all_tasks if t["overdue"]]
    if not overdue_tasks:
        await interaction.response.send_message("✅ Không có tasks quá hạn!")
        return

    _last_tasks[interaction.user.id] = overdue_tasks

    embed = build_tasks_embed(
        date_display="",
        tasks=overdue_tasks,
        timestamp=datetime.now(VIETNAM_TZ),
        overdue_only=True,
    )
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="add_task", description="Thêm task mới")
@app_commands.describe(
    title="Tiêu đề task",
    due="Ngày giờ. VD: 20/2 18:00, tomorrow 17:00",
    notes="Ghi chú (tuỳ chọn)",
)
async def slash_add_task(
    interaction: discord.Interaction,
    title: str,
    due: str = "",
    notes: str = "",
):
    due_datetime = None

    if due:
        date_match = re.search(
            r"(\d{1,2}[/-]\d{1,2}|today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            due,
            re.I,
        )
        if date_match:
            target_date = knowledge_bot.parse_date(date_match.group(1))
            if target_date:
                time_match = re.search(r"(\d{1,2}[h:]\d{2}|\d{1,2}h?)", due)
                if time_match:
                    target_time = knowledge_bot.parse_time(time_match.group(1))
                    if target_time:
                        due_datetime = knowledge_bot.timezone.localize(
                            datetime.combine(target_date, target_time)
                        )
                else:
                    due_datetime = knowledge_bot.timezone.localize(
                        datetime.combine(target_date, time(23, 59))
                    )

    await interaction.response.defer(thinking=True)
    result = await knowledge_bot.add_task(title, due_datetime, notes)
    await interaction.followup.send(result)


@bot.tree.command(name="done", description="Đánh dấu task hoàn thành")
@app_commands.describe(index="Số thứ tự task từ /tasks hoặc /overdue")
async def slash_done(interaction: discord.Interaction, index: int):
    if interaction.user.id not in _last_tasks:
        await interaction.response.send_message("⚠️ Gọi `/tasks` trước", ephemeral=True)
        return

    tasks_list = _last_tasks[interaction.user.id]
    if index < 1 or index > len(tasks_list):
        await interaction.response.send_message(
            f"⚠️ Chọn từ 1-{len(tasks_list)}", ephemeral=True
        )
        return

    await interaction.response.defer(thinking=True)
    task = tasks_list[index - 1]
    result = await knowledge_bot.complete_task(task["id"], task["tasklist_id"])
    await interaction.followup.send(result)
    del _last_tasks[interaction.user.id]


@bot.tree.command(name="del_task", description="Xoá task theo số thứ tự")
@app_commands.describe(index="Số thứ tự task từ /tasks hoặc /overdue")
async def slash_del_task(interaction: discord.Interaction, index: int):
    if interaction.user.id not in _last_tasks:
        await interaction.response.send_message("⚠️ Gọi `/tasks` trước", ephemeral=True)
        return

    tasks_list = _last_tasks[interaction.user.id]
    if index < 1 or index > len(tasks_list):
        await interaction.response.send_message(
            f"⚠️ Chọn từ 1-{len(tasks_list)}", ephemeral=True
        )
        return

    await interaction.response.defer(thinking=True)
    task = tasks_list[index - 1]
    result = await knowledge_bot.delete_task(task["id"], task["tasklist_id"])
    await interaction.followup.send(result)
    del _last_tasks[interaction.user.id]


@bot.tree.command(name="stats", description="Thống kê tin nhắn đã theo dõi")
async def slash_stats(interaction: discord.Interaction):
    if not daily_messages:
        await interaction.response.send_message("📊 Chưa có tin nhắn")
        return

    message = "📊 **Thống kê:**\n\n"
    total = 0

    for channel_id, messages in daily_messages.items():
        discord_channel = bot.get_channel(channel_id)
        channel_name = discord_channel.name if discord_channel else str(channel_id)
        count = len(messages)
        total += count
        message += f"• #{channel_name}: {count}\n"

    message += f"\n**Tổng:** {total}"
    await interaction.response.send_message(message)


# ==============================
# SLASH COMMANDS - COUNTDOWN
# ==============================
@bot.tree.command(name="countdown", description="Xem tất cả countdown đang chạy")
@app_commands.describe(name="Lọc theo tên (tuỳ chọn)")
async def slash_countdown(interaction: discord.Interaction, name: str = ""):
    countdowns = knowledge_bot.get_countdowns()

    if name.strip():
        countdowns = [
            cd for cd in countdowns if name.strip().lower() in cd["name"].lower()
        ]

    if not countdowns:
        await interaction.response.send_message(
            "⏰ Không có countdown nào đang chạy\nDùng `/add_countdown` để thêm mới"
        )
        return

    message = "⏰ **COUNTDOWNS ĐANG CHẠY:**\n\n"
    for cd in countdowns:
        icon = "🔴" if cd["status"] == "ĐÃ QUA" else "🟢"
        message += f"{icon} {cd['emoji']} **{cd['name']}**\n"
        message += f"   📅 {cd['target'].strftime('%d/%m/%Y %H:%M:%S')}\n"
        if cd["status"] == "ACTIVE":
            message += f"   ⏳ Còn: **{cd['time_str']}**\n"
        else:
            message += f"   ⏳ {cd['status']}\n"
        message += "\n"

    await interaction.response.send_message(message)


@bot.tree.command(name="add_countdown", description="Thêm countdown mới")
@app_commands.describe(
    name="Tên countdown",
    datetime_input="Ví dụ: 20/2 00:00 hoặc tomorrow 23:59",
    emoji="Emoji hiển thị",
)
async def slash_add_countdown(
    interaction: discord.Interaction,
    name: str,
    datetime_input: str,
    emoji: str = "🎉",
):
    date_match = re.search(
        r"(\d{1,2}[/-]\d{1,2}|today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        datetime_input,
        re.I,
    )
    if not date_match:
        await interaction.response.send_message("⚠️ Không tìm thấy ngày", ephemeral=True)
        return

    date_part = date_match.group(1)
    target_date = knowledge_bot.parse_date(date_part)
    if not target_date:
        await interaction.response.send_message("⚠️ Ngày không hợp lệ", ephemeral=True)
        return

    remaining_str = datetime_input[date_match.end() :].strip()
    time_match = re.search(r"(\d{1,2})[h:](\d{2})", remaining_str)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        target_time = time(hour, minute)
    else:
        hour_only_match = re.search(r"(\d{1,2})h\b", remaining_str)
        if hour_only_match:
            hour = int(hour_only_match.group(1))
            target_time = time(hour, 0)
        else:
            target_time = time(0, 0, 0)

    target_datetime = knowledge_bot.timezone.localize(
        datetime.combine(target_date, target_time)
    )

    now = datetime.now(knowledge_bot.timezone)
    time_diff_seconds = (target_datetime - now).total_seconds()

    if time_diff_seconds < -60:
        hours_past = int(abs(time_diff_seconds) // 3600)
        minutes_past = int((abs(time_diff_seconds) % 3600) // 60)
        await interaction.response.send_message(
            f"⚠️ **Thời gian phải trong tương lai**\n\n"
            f"📅 Bạn nhập: `{target_datetime.strftime('%d/%m/%Y %H:%M:%S')}`\n"
            f"🕐 Hiện tại VN: `{now.strftime('%d/%m/%Y %H:%M:%S')}`\n"
            f"⏰ Đã qua: **{hours_past} giờ {minutes_past} phút**",
            ephemeral=True,
        )
        return

    if knowledge_bot.add_countdown(name, target_datetime, emoji):
        remaining = max(0, (target_datetime - now).total_seconds())
        if remaining < 3600:
            minutes = int(remaining // 60)
            seconds = int(remaining % 60)
            await interaction.response.send_message(
                f"✅ Đã thêm countdown!\n\n"
                f"{emoji} **{name}**\n"
                f"📅 {target_datetime.strftime('%d/%m/%Y %H:%M:%S')}\n"
                f"⏳ Còn: **{minutes}m {seconds}s**\n\n"
                f"🔔 Bot sẽ tự động nhắc!"
            )
        else:
            days = int(remaining // 86400)
            hours = int((remaining % 86400) // 3600)
            await interaction.response.send_message(
                f"✅ Đã thêm countdown!\n\n"
                f"{emoji} **{name}**\n"
                f"📅 {target_datetime.strftime('%d/%m/%Y %H:%M:%S')}\n"
                f"⏳ Còn: {days}d {hours}h\n\n"
                f"Bot sẽ nhắc:\n"
                f"• Còn 5 phút\n"
                f"• Còn 4 phút\n"
                f"• Còn 3 phút\n"
                f"• Còn 2 phút\n"
                f"• Đếm ngược 60s cuối!"
            )
    else:
        await interaction.response.send_message(
            "⚠️ Không thể thêm countdown", ephemeral=True
        )


@bot.tree.command(name="del_countdown", description="Xoá countdown theo tên")
@app_commands.describe(name="Tên countdown cần xoá")
async def slash_del_countdown(interaction: discord.Interaction, name: str):
    if knowledge_bot.remove_countdown(name):
        await interaction.response.send_message(f"✅ Đã xóa countdown: {name}")
    else:
        await interaction.response.send_message(
            f"⚠️ Không tìm thấy countdown: {name}", ephemeral=True
        )


@bot.tree.command(name="newyear", description="Bật countdown năm mới")
@app_commands.describe(
    year="Năm mục tiêu, bỏ trống để tự động",
    month="Tháng (mặc định 1)",
    day="Ngày (mặc định 1)",
    hour="Giờ (mặc định 0)",
    minute="Phút (mặc định 0)",
)
async def slash_newyear(
    interaction: discord.Interaction,
    year: int = None,
    month: int = 1,
    day: int = 1,
    hour: int = 0,
    minute: int = 0,
):
    now = datetime.now(knowledge_bot.timezone)

    if year is None:
        if now.month == 12 and now.day == 31:
            year = now.year + 1
        elif now.month == 1 and now.day == 1:
            await interaction.response.send_message("🎆 Hôm nay là năm mới rồi!")
            return
        else:
            year = now.year + 1

    try:
        ny_datetime = knowledge_bot.timezone.localize(
            datetime(year, month, day, hour, minute, 0)
        )
    except ValueError:
        await interaction.response.send_message(
            "⚠️ Ngày giờ không hợp lệ", ephemeral=True
        )
        return

    if ny_datetime <= now:
        await interaction.response.send_message(
            "⚠️ Thời gian phải trong tương lai", ephemeral=True
        )
        return

    countdown_name = f"Năm Mới {year}"
    if knowledge_bot.add_countdown(countdown_name, ny_datetime, "🎆", label="newyear"):
        remaining = (ny_datetime - now).total_seconds()
        days = int(remaining // 86400)
        hours = int((remaining % 86400) // 3600)
        minutes = int((remaining % 3600) // 60)
        await interaction.response.send_message(
            f"🎆 **ĐÃ BẬT COUNTDOWN NĂM MỚI {year}!**\n\n"
            f"📅 {ny_datetime.strftime('%d/%m/%Y %H:%M:%S')}\n"
            f"⏳ Còn: **{days} ngày {hours} giờ {minutes} phút**\n\n"
            f"✨ Format đặc biệt cho năm mới!\n"
            f"🎊 Bot sẽ tự động đếm ngược và chúc mừng! 🎉"
        )
    else:
        await interaction.response.send_message(
            "⚠️ Countdown đã tồn tại hoặc lỗi", ephemeral=True
        )


@bot.tree.command(name="tet", description="Bật countdown Tết Âm lịch gần nhất")
async def slash_tet(interaction: discord.Interaction):
    now = datetime.now(knowledge_bot.timezone)

    year, tet_datetime = knowledge_bot.get_next_tet_datetime(now)
    if not tet_datetime:
        await interaction.response.send_message(
            "⚠️ Chưa có dữ liệu ngày Tết Âm cho năm tiếp theo", ephemeral=True
        )
        return

    countdown_name = f"Tết Nguyên Đán {year}"
    if knowledge_bot.add_countdown(countdown_name, tet_datetime, "🧧", label=""):
        remaining = max(0, (tet_datetime - now).total_seconds())
        days = int(remaining // 86400)
        hours = int((remaining % 86400) // 3600)
        minutes = int((remaining % 3600) // 60)
        await interaction.response.send_message(
            f"🧧 **ĐÃ BẬT COUNTDOWN TẾT {year}!**\n\n"
            f"📅 {tet_datetime.strftime('%d/%m/%Y %H:%M:%S')}\n"
            f"⏳ Còn: **{days} ngày {hours} giờ {minutes} phút**\n\n"
            f"Chúc mừng năm mới! 🎊"
        )
    else:
        await interaction.response.send_message(
            "⚠️ Countdown đã tồn tại hoặc lỗi", ephemeral=True
        )


# ==============================
# COMMANDS - COUNTDOWN
# ==============================
@bot.command()
async def countdown(ctx, *, name=""):
    """Xem tất cả countdowns đang active"""
    countdowns = knowledge_bot.get_countdowns()

    if not countdowns:
        await ctx.send(
            "⏰ Không có countdown nào đang chạy\nDùng `!add_countdown` để thêm mới"
        )
        return

    message = "⏰ **COUNTDOWNS ĐANG CHẠY:**\n\n"

    for cd in countdowns:
        icon = "🔴" if cd["status"] == "ĐÃ QUA" else "🟢"
        message += f"{icon} {cd['emoji']} **{cd['name']}**\n"
        message += f"   📅 {cd['target'].strftime('%d/%m/%Y %H:%M:%S')}\n"
        if cd["status"] == "ACTIVE":
            message += f"   ⏳ Còn: **{cd['time_str']}**\n"
        else:
            message += f"   ⏳ {cd['status']}\n"
        message += "\n"

    await ctx.send(message)


@bot.command()
async def add_countdown(ctx, *, args):
    parts = [p.strip() for p in args.split("|")]
    if len(parts) < 2:
        await ctx.send(
            "⚠️ Format: `!add_countdown <n> | <date time> | <emoji>`\n"
            "VD: `!add_countdown Sinh nhật | 20/2 00:00 | 🎂`"
        )
        return

    name = parts[0]
    datetime_str = parts[1]
    emoji = parts[2] if len(parts) > 2 else "🎉"

    # Parse date TRƯỚC
    date_match = re.search(
        r"(\d{1,2}[/-]\d{1,2}|today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        datetime_str,
        re.I,
    )
    if not date_match:
        await ctx.send("⚠️ Không tìm thấy ngày")
        return

    date_part = date_match.group(1)
    target_date = knowledge_bot.parse_date(date_part)
    if not target_date:
        await ctx.send("⚠️ Ngày không hợp lệ")
        return

    # Parse time - BỎ QUA phần date đã match
    # Lấy phần còn lại sau date
    remaining_str = datetime_str[date_match.end() :].strip()

    # FIX: Regex chặt chẽ hơn - BẮT BUỘC có : hoặc h
    time_match = re.search(r"(\d{1,2})[h:](\d{2})", remaining_str)

    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        target_time = time(hour, minute)
    else:
        # Nếu không tìm thấy time với format HH:MM hoặc HHhMM
        # Thử tìm chỉ giờ: "14h" hoặc "14"
        hour_only_match = re.search(r"(\d{1,2})h\b", remaining_str)
        if hour_only_match:
            hour = int(hour_only_match.group(1))
            target_time = time(hour, 0)
        else:
            # Không có time, dùng 00:00
            target_time = time(0, 0, 0)

    target_datetime = knowledge_bot.timezone.localize(
        datetime.combine(target_date, target_time)
    )

    # Check if in future
    now = datetime.now(knowledge_bot.timezone)
    time_diff_seconds = (target_datetime - now).total_seconds()

    if time_diff_seconds < -60:
        hours_past = int(abs(time_diff_seconds) // 3600)
        minutes_past = int((abs(time_diff_seconds) % 3600) // 60)

        await ctx.send(
            f"⚠️ **Thời gian phải trong tương lai**\n\n"
            f"📅 Bạn nhập: `{target_datetime.strftime('%d/%m/%Y %H:%M:%S')}`\n"
            f"🕐 Hiện tại VN: `{now.strftime('%d/%m/%Y %H:%M:%S')}`\n"
            f"⏰ Đã qua: **{hours_past} giờ {minutes_past} phút**"
        )
        return

    # Add countdown
    if knowledge_bot.add_countdown(name, target_datetime, emoji):
        remaining = max(0, (target_datetime - now).total_seconds())

        if remaining < 3600:
            minutes = int(remaining // 60)
            seconds = int(remaining % 60)
            await ctx.send(
                f"✅ Đã thêm countdown!\n\n"
                f"{emoji} **{name}**\n"
                f"📅 {target_datetime.strftime('%d/%m/%Y %H:%M:%S')}\n"
                f"⏳ Còn: **{minutes}m {seconds}s**\n\n"
                f"🔔 Bot sẽ tự động nhắc!"
            )
        else:
            days = int(remaining // 86400)
            hours = int((remaining % 86400) // 3600)

            await ctx.send(
                f"✅ Đã thêm countdown!\n\n"
                f"{emoji} **{name}**\n"
                f"📅 {target_datetime.strftime('%d/%m/%Y %H:%M:%S')}\n"
                f"⏳ Còn: {days}d {hours}h\n\n"
                f"Bot sẽ nhắc:\n"
                f"• Còn 5 phút\n"
                f"• Còn 4 phút\n"
                f"• Còn 3 phút\n"
                f"• Còn 2 phút\n"
                f"• Đếm ngược 60s cuối!"
            )
    else:
        await ctx.send("⚠️ Không thể thêm countdown")


@bot.command()
async def del_countdown(ctx, *, name):
    """Xóa countdown"""
    if knowledge_bot.remove_countdown(name):
        await ctx.send(f"✅ Đã xóa countdown: {name}")
    else:
        await ctx.send(f"⚠️ Không tìm thấy countdown: {name}")


@bot.command()
async def newyear(
    ctx, year: int = None, month: int = 1, day: int = 1, hour: int = 0, minute: int = 0
):
    """
    Bật countdown năm mới với format đặc biệt
    !newyear - Năm mới tự động (1/1 năm sau)
    !newyear 2026 - Năm mới 2026 (1/1/2026 00:00)
    !newyear 2026 1 1 23 59 - Custom ngày giờ chính xác
    """
    now = datetime.now(knowledge_bot.timezone)

    if year is None:
        # Auto determine next new year
        if now.month == 12 and now.day == 31:
            year = now.year + 1
        elif now.month == 1 and now.day == 1:
            await ctx.send("🎆 Hôm nay là năm mới rồi!")
            return
        else:
            year = now.year + 1

    # Create datetime
    try:
        ny_datetime = knowledge_bot.timezone.localize(
            datetime(year, month, day, hour, minute, 0)
        )
    except ValueError:
        await ctx.send("⚠️ Ngày giờ không hợp lệ")
        return

    if ny_datetime <= now:
        await ctx.send("⚠️ Thời gian phải trong tương lai")
        return

    countdown_name = f"Năm Mới {year}"

    # Add with "newyear" label for special format
    if knowledge_bot.add_countdown(countdown_name, ny_datetime, "🎆", label="newyear"):
        remaining = (ny_datetime - now).total_seconds()
        days = int(remaining // 86400)
        hours = int((remaining % 86400) // 3600)
        minutes = int((remaining % 3600) // 60)

        await ctx.send(
            f"🎆 **ĐÃ BẬT COUNTDOWN NĂM MỚI {year}!**\n\n"
            f"📅 {ny_datetime.strftime('%d/%m/%Y %H:%M:%S')}\n"
            f"⏳ Còn: **{days} ngày {hours} giờ {minutes} phút**\n\n"
            f"✨ Format đặc biệt cho năm mới!\n"
            f"🎊 Bot sẽ tự động đếm ngược và chúc mừng! 🎉"
        )
    else:
        await ctx.send("⚠️ Countdown đã tồn tại hoặc lỗi")


@bot.command()
async def tet(ctx):
    """Quick activate Tet countdown (Tết Âm lịch gần nhất)"""
    now = datetime.now(knowledge_bot.timezone)

    year, tet_datetime = knowledge_bot.get_next_tet_datetime(now)
    if not tet_datetime:
        await ctx.send("⚠️ Chưa có dữ liệu ngày Tết Âm cho năm tiếp theo")
        return

    countdown_name = f"Tết Nguyên Đán {year}"

    if knowledge_bot.add_countdown(countdown_name, tet_datetime, "🧧", label=""):
        remaining = max(0, (tet_datetime - now).total_seconds())
        days = int(remaining // 86400)
        hours = int((remaining % 86400) // 3600)
        minutes = int((remaining % 3600) // 60)

        await ctx.send(
            f"🧧 **ĐÃ BẬT COUNTDOWN TẾT {year}!**\n\n"
            f"📅 {tet_datetime.strftime('%d/%m/%Y %H:%M:%S')}\n"
            f"⏳ Còn: **{days} ngày {hours} giờ {minutes} phút**\n\n"
            f"Chúc mừng năm mới! 🎊"
        )
    else:
        await ctx.send("⚠️ Countdown đã tồn tại hoặc lỗi")


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("❌ Thiếu DISCORD_TOKEN")
        exit()
    if not GITHUB_TOKEN:
        print("❌ Thiếu GITHUB_TOKEN")
        exit()
    if YOUR_USER_ID == 0:
        print("❌ Thiếu YOUR_USER_ID")
        exit()

    print("🚀 Bot khởi động...")
    bot.run(DISCORD_TOKEN)
