import os
import sqlite3
from datetime import datetime

import discord

from bot.config.settings import STUDY_METRICS_DIR, STUDY_POINTS_MISS, VIETNAM_TZ
from bot.state.runtime import _study_questions
from tools import study_memory


def _metrics_dir_path():
    """Return absolute directory path used to store monthly study metric databases."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, STUDY_METRICS_DIR)


def _metrics_db_path(target_date=None):
    """Build month-scoped SQLite file path for study metrics."""
    target_date = target_date or datetime.now(VIETNAM_TZ)
    month_key = target_date.strftime("%Y-%m")
    return os.path.join(_metrics_dir_path(), f"study_metrics_{month_key}.sqlite3")


def _ensure_metrics_db(target_date=None):
    """Create metrics database and required tables when missing."""
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
    """Fetch existing study profile for user or initialize a default row."""
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
    """Update streak counters from event date using last recorded streak date."""
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
    """Persist a study event and update profile aggregates atomically."""
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
    """Normalize model score output into float or None when invalid."""
    if score is None:
        return None
    try:
        return float(str(score).replace(",", ".").strip())
    except Exception:
        return None


def _mark_question_answered(user_id, question_index):
    """Mark a pending in-memory review question as answered."""
    question_bank = _study_questions.get(user_id, [])
    target = next((q for q in question_bank if q.get("index") == question_index), None)
    if target:
        target["answered"] = True
    return target


def _mark_spaced_unanswered(user_id, unanswered_items):
    """Flag unanswered questions in spaced-repetition storage."""
    db_path = _ensure_study_memory_tables()
    study_memory.mark_unanswered_cards(
        db_path=db_path,
        user_id=int(user_id),
        pending_items=unanswered_items,
    )


def _apply_unanswered_penalty(user_id):
    """Apply missed-question penalties and persist missed study events."""
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
    """Build Discord embed with monthly study metrics for a user."""
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
    """Ensure spaced-repetition tables exist and return database path."""
    db_path = _metrics_db_path()
    study_memory.ensure_study_tables(db_path)
    return db_path


def _build_question_theory_text(summary_points, detailed_summary):
    """Compose compact theory text from summary points and detailed explanation."""
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
    """Upsert review questions into spaced-repetition card storage."""
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
    """Record a spaced-repetition review attempt for one question."""
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


__all__ = [
    "_get_or_create_study_profile",
    "_append_study_event",
    "_normalize_score_value",
    "_mark_question_answered",
    "_apply_unanswered_penalty",
    "_build_study_metrics_embed",
    "_ensure_study_memory_tables",
    "_build_question_theory_text",
    "_persist_questions_for_spaced_repetition",
    "_record_spaced_review",
]
