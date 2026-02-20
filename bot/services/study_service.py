import os
import random
import sqlite3
from datetime import datetime

import discord

from bot.config.settings import STUDY_METRICS_DIR, STUDY_POINTS_MISS, VIETNAM_TZ
from bot.state.runtime import _study_questions
from tools import study_memory


DAILY_MISSION_POOL = [
    {
        "mission_key": "ans_2",
        "title": "Trả lời 2 câu hỏi ôn tập",
        "metric_key": "answer_count",
        "target_value": 2,
        "reward_points": 2,
    },
    {
        "mission_key": "ans_3",
        "title": "Trả lời 3 câu hỏi ôn tập",
        "metric_key": "answer_count",
        "target_value": 3,
        "reward_points": 3,
    },
    {
        "mission_key": "ans_4",
        "title": "Trả lời 4 câu hỏi ôn tập",
        "metric_key": "answer_count",
        "target_value": 4,
        "reward_points": 4,
    },
    {
        "mission_key": "ans_5",
        "title": "Trả lời 5 câu hỏi ôn tập",
        "metric_key": "answer_count",
        "target_value": 5,
        "reward_points": 5,
    },
    {
        "mission_key": "ans_6",
        "title": "Trả lời 6 câu hỏi ôn tập",
        "metric_key": "answer_count",
        "target_value": 6,
        "reward_points": 6,
    },
    {
        "mission_key": "pass_1",
        "title": "Đạt ngưỡng ở 1 câu",
        "metric_key": "pass_count",
        "target_value": 1,
        "reward_points": 2,
    },
    {
        "mission_key": "pass_2",
        "title": "Đạt ngưỡng ở 2 câu",
        "metric_key": "pass_count",
        "target_value": 2,
        "reward_points": 4,
    },
    {
        "mission_key": "pass_3",
        "title": "Đạt ngưỡng ở 3 câu",
        "metric_key": "pass_count",
        "target_value": 3,
        "reward_points": 6,
    },
    {
        "mission_key": "pass_4",
        "title": "Đạt ngưỡng ở 4 câu",
        "metric_key": "pass_count",
        "target_value": 4,
        "reward_points": 8,
    },
    {
        "mission_key": "sum_1",
        "title": "Hoàn thành 1 lượt summary",
        "metric_key": "summary_count",
        "target_value": 1,
        "reward_points": 3,
    },
    {
        "mission_key": "sum_2",
        "title": "Hoàn thành 2 lượt summary",
        "metric_key": "summary_count",
        "target_value": 2,
        "reward_points": 6,
    },
    {
        "mission_key": "sum_3",
        "title": "Hoàn thành 3 lượt summary",
        "metric_key": "summary_count",
        "target_value": 3,
        "reward_points": 9,
    },
    {
        "mission_key": "pt_5",
        "title": "Kiếm 5 study points trong ngày",
        "metric_key": "points_earned",
        "target_value": 5,
        "reward_points": 3,
    },
    {
        "mission_key": "pt_8",
        "title": "Kiếm 8 study points trong ngày",
        "metric_key": "points_earned",
        "target_value": 8,
        "reward_points": 4,
    },
    {
        "mission_key": "pt_10",
        "title": "Kiếm 10 study points trong ngày",
        "metric_key": "points_earned",
        "target_value": 10,
        "reward_points": 5,
    },
    {
        "mission_key": "pt_12",
        "title": "Kiếm 12 study points trong ngày",
        "metric_key": "points_earned",
        "target_value": 12,
        "reward_points": 6,
    },
    {
        "mission_key": "mix_a",
        "title": "Đà học tốt: trả lời 3 câu",
        "metric_key": "answer_count",
        "target_value": 3,
        "reward_points": 4,
    },
    {
        "mission_key": "mix_b",
        "title": "Tăng chất lượng: đạt ngưỡng 2 câu",
        "metric_key": "pass_count",
        "target_value": 2,
        "reward_points": 5,
    },
    {
        "mission_key": "mix_c",
        "title": "Duy trì nhịp: thêm 1 summary",
        "metric_key": "summary_count",
        "target_value": 1,
        "reward_points": 4,
    },
    {
        "mission_key": "mix_d",
        "title": "Bứt tốc: kiếm 15 điểm trong ngày",
        "metric_key": "points_earned",
        "target_value": 15,
        "reward_points": 8,
    },
]


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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_study_missions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            mission_date TEXT NOT NULL,
            mission_key TEXT NOT NULL,
            title TEXT NOT NULL,
            metric_key TEXT NOT NULL,
            target_value INTEGER NOT NULL,
            progress_value INTEGER NOT NULL DEFAULT 0,
            reward_points INTEGER NOT NULL DEFAULT 0,
            is_completed INTEGER NOT NULL DEFAULT 0,
            completed_at TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(user_id, mission_date, mission_key)
        )
        """
    )
    conn.commit()
    conn.close()
    return db_path


def _today_key(target_date=None):
    """Return YYYY-MM-DD key for mission assignment in Vietnam timezone."""
    target = target_date or datetime.now(VIETNAM_TZ)
    return target.strftime("%Y-%m-%d")


def _fetch_daily_missions(user_id, target_date=None):
    """Fetch persisted daily missions for one user/date."""
    db_path = _ensure_metrics_db(target_date)
    date_key = _today_key(target_date)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM daily_study_missions
        WHERE user_id = ? AND mission_date = ?
        ORDER BY id ASC
        """,
        (int(user_id), date_key),
    )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def _ensure_daily_missions(user_id, target_date=None):
    """Create 2-3 deterministic random daily missions for user if missing."""
    existing = _fetch_daily_missions(user_id, target_date)
    if len(existing) >= 2:
        return existing

    db_path = _ensure_metrics_db(target_date)
    date_key = _today_key(target_date)
    now_iso = datetime.now(VIETNAM_TZ).isoformat()
    seed = f"{int(user_id)}::{date_key}"
    rng = random.Random(seed)
    mission_count = 2 if rng.random() < 0.5 else 3
    mission_count = max(2, min(3, mission_count))
    picked = rng.sample(DAILY_MISSION_POOL, k=mission_count)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for mission in picked:
        cur.execute(
            """
            INSERT OR IGNORE INTO daily_study_missions (
                user_id, mission_date, mission_key, title, metric_key,
                target_value, progress_value, reward_points, is_completed,
                completed_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, 0, NULL, ?)
            """,
            (
                int(user_id),
                date_key,
                str(mission["mission_key"]),
                str(mission["title"]),
                str(mission["metric_key"]),
                int(mission["target_value"]),
                int(mission["reward_points"]),
                now_iso,
            ),
        )
    conn.commit()
    conn.close()
    return _fetch_daily_missions(user_id, target_date)


def _build_daily_mission_status(user_id, target_date=None):
    """Build mission list with per-item progress percent for display."""
    rows = _ensure_daily_missions(user_id, target_date)
    missions = []
    for row in rows:
        target_value = max(1, int(row.get("target_value") or 1))
        progress_value = max(0, int(row.get("progress_value") or 0))
        progress_pct = int(min(100, round((progress_value / target_value) * 100)))
        missions.append(
            {
                "mission_key": str(row.get("mission_key") or ""),
                "title": str(row.get("title") or "Nhiệm vụ học tập"),
                "metric_key": str(row.get("metric_key") or ""),
                "target_value": target_value,
                "progress_value": progress_value,
                "progress_pct": progress_pct,
                "reward_points": int(row.get("reward_points") or 0),
                "is_completed": int(row.get("is_completed") or 0) == 1,
            }
        )

    done = sum(1 for item in missions if item["is_completed"])
    total = len(missions)
    summary = f"{done}/{total} nhiệm vụ hoàn thành" if total else "Chưa có nhiệm vụ"
    lines = []
    for item in missions:
        icon = "✅" if item["is_completed"] else "🟡"
        lines.append(
            f"{icon} {item['title']}: {item['progress_value']}/{item['target_value']} ({item['progress_pct']}%) • +{item['reward_points']}đ"
        )

    return {
        "summary": summary,
        "lines": lines,
        "missions": missions,
        "completed": done,
        "total": total,
    }


def _apply_daily_mission_progress(
    user_id,
    event_type,
    points_delta=0,
    target_date=None,
):
    """Advance missions by matching event metrics and return newly completed tasks."""
    rows = _ensure_daily_missions(user_id, target_date)
    increments = {}

    if event_type in {"answer", "pass"}:
        increments["answer_count"] = increments.get("answer_count", 0) + 1
    if event_type == "pass":
        increments["pass_count"] = increments.get("pass_count", 0) + 1
    if event_type == "summary":
        increments["summary_count"] = increments.get("summary_count", 0) + 1
    if int(points_delta or 0) > 0:
        increments["points_earned"] = increments.get("points_earned", 0) + int(
            points_delta
        )

    if not increments:
        return {
            "completed_missions": [],
            "status": _build_daily_mission_status(user_id, target_date),
        }

    db_path = _ensure_metrics_db(target_date)
    now_iso = datetime.now(VIETNAM_TZ).isoformat()
    completed = []

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for row in rows:
        metric_key = str(row.get("metric_key") or "")
        inc = int(increments.get(metric_key, 0))
        if inc <= 0:
            continue

        current_progress = int(row.get("progress_value") or 0)
        target_value = max(1, int(row.get("target_value") or 1))
        is_completed = int(row.get("is_completed") or 0) == 1

        next_progress = min(target_value, current_progress + inc)
        next_completed = is_completed or next_progress >= target_value

        if next_progress != current_progress or next_completed != is_completed:
            cur.execute(
                """
                UPDATE daily_study_missions
                SET progress_value = ?, is_completed = ?, completed_at = ?
                WHERE user_id = ? AND mission_date = ? AND mission_key = ?
                """,
                (
                    int(next_progress),
                    1 if next_completed else 0,
                    now_iso if next_completed else None,
                    int(user_id),
                    _today_key(target_date),
                    str(row.get("mission_key") or ""),
                ),
            )

        if next_completed and not is_completed:
            completed.append(
                {
                    "mission_key": str(row.get("mission_key") or ""),
                    "title": str(row.get("title") or "Nhiệm vụ học tập"),
                    "reward_points": int(row.get("reward_points") or 0),
                }
            )

    conn.commit()
    conn.close()
    return {
        "completed_missions": completed,
        "status": _build_daily_mission_status(user_id, target_date),
    }


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
    apply_daily_mission=True,
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

    result = {
        "total_points": total_points,
        "streak_days": int(streak_days),
        "answered_count": answered_count,
        "passed_count": passed_count,
        "missed_count": missed_count,
        "summaries_count": summaries_count,
        "db_path": db_path,
    }

    mission_status = _build_daily_mission_status(user_id, target_date)
    completed_missions = []

    if apply_daily_mission and event_type != "mission_complete":
        mission_progress = _apply_daily_mission_progress(
            user_id=user_id,
            event_type=event_type,
            points_delta=points_delta,
            target_date=target_date,
        )
        mission_status = mission_progress.get("status", mission_status)
        completed_missions = mission_progress.get("completed_missions", [])

        if completed_missions:
            for mission in completed_missions:
                mission_reward = int(mission.get("reward_points") or 0)
                if mission_reward <= 0:
                    continue
                reward_stats = _append_study_event(
                    user_id=user_id,
                    event_type="mission_complete",
                    points_delta=mission_reward,
                    note=f"Hoàn thành nhiệm vụ ngày: {mission.get('title', '')}",
                    target_date=target_date,
                    apply_daily_mission=False,
                )
                result.update(
                    {
                        "total_points": reward_stats.get(
                            "total_points", result["total_points"]
                        ),
                        "streak_days": reward_stats.get(
                            "streak_days", result["streak_days"]
                        ),
                        "answered_count": reward_stats.get(
                            "answered_count", result["answered_count"]
                        ),
                        "passed_count": reward_stats.get(
                            "passed_count", result["passed_count"]
                        ),
                        "missed_count": reward_stats.get(
                            "missed_count", result["missed_count"]
                        ),
                        "summaries_count": reward_stats.get(
                            "summaries_count", result["summaries_count"]
                        ),
                    }
                )

            mission_status = _build_daily_mission_status(user_id, target_date)

    result["daily_mission_status"] = mission_status
    result["daily_mission_lines"] = list(mission_status.get("lines") or [])
    result["daily_mission_summary"] = str(mission_status.get("summary") or "")
    result["completed_missions"] = completed_missions
    return result


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
    mission_status = _build_daily_mission_status(user_id)
    mission_lines = mission_status.get("lines") or ["Chưa có nhiệm vụ hôm nay."]
    mission_text = "\n".join(mission_lines[:3])
    mission_text = f"{mission_status.get('summary', '')}\n{mission_text}".strip()
    embed.add_field(
        name="🎯 Nhiệm vụ tự học hôm nay", value=mission_text[:1024], inline=False
    )
    embed.set_footer(text=f"DB: {_metrics_db_path()}")
    return embed


def _get_daily_mission_status(user_id, target_date=None):
    """Public helper returning current daily mission progress payload."""
    return _build_daily_mission_status(user_id, target_date)


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
    "_get_daily_mission_status",
]
