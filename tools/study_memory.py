import hashlib
import sqlite3
from datetime import datetime, timedelta

from .constants import (
    DEFAULT_SM2_EASINESS,
    DEFAULT_SM2_INTERVAL,
    DEFAULT_SM2_REPETITION,
    MIN_SM2_EASINESS,
)


def ensure_study_tables(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS study_cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            card_hash TEXT NOT NULL,
            channel_name TEXT,
            question TEXT NOT NULL,
            theory TEXT,
            topic TEXT,
            easiness REAL NOT NULL DEFAULT 2.5,
            interval_days INTEGER NOT NULL DEFAULT 1,
            repetition INTEGER NOT NULL DEFAULT 0,
            due_date TEXT,
            last_reviewed_at TEXT,
            last_score REAL,
            weak_flag INTEGER NOT NULL DEFAULT 0,
            unanswered_flag INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(user_id, card_hash)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS study_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            card_hash TEXT NOT NULL,
            score REAL,
            quality INTEGER,
            answered INTEGER NOT NULL DEFAULT 1,
            note TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def _card_hash(question: str, channel_name: str = ""):
    key = f"{channel_name}|{question}".strip().lower()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _score_to_quality(score):
    if score is None:
        return 0
    s = float(score)
    if s >= 9:
        return 5
    if s >= 8:
        return 4
    if s >= 7:
        return 3
    if s >= 5:
        return 2
    if s >= 3:
        return 1
    return 0


def _apply_sm2(interval_days, repetition, easiness, quality):
    quality = int(max(0, min(5, quality)))
    interval_days = int(interval_days or DEFAULT_SM2_INTERVAL)
    repetition = int(repetition or DEFAULT_SM2_REPETITION)
    easiness = float(easiness or DEFAULT_SM2_EASINESS)

    if quality < 3:
        repetition = 0
        interval_days = 1
    else:
        repetition += 1
        if repetition == 1:
            interval_days = 1
        elif repetition == 2:
            interval_days = 6
        else:
            interval_days = max(1, round(interval_days * easiness))

    easiness = easiness + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    easiness = max(MIN_SM2_EASINESS, easiness)

    return {
        "interval_days": interval_days,
        "repetition": repetition,
        "easiness": round(easiness, 3),
        "weak_flag": 1 if quality < 3 else 0,
    }


def upsert_card(
    db_path: str,
    user_id: int,
    channel_name: str,
    question: str,
    theory: str,
    topic: str = "",
):
    ensure_study_tables(db_path)
    now_iso = datetime.now().isoformat()
    card_hash = _card_hash(question, channel_name)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM study_cards WHERE user_id = ? AND card_hash = ?",
        (int(user_id), card_hash),
    )
    row = cur.fetchone()

    if row:
        cur.execute(
            """
            UPDATE study_cards
            SET theory = ?, topic = ?, channel_name = ?, updated_at = ?
            WHERE user_id = ? AND card_hash = ?
            """,
            (
                (theory or "")[:6000],
                (topic or "")[:200],
                (channel_name or "")[:200],
                now_iso,
                int(user_id),
                card_hash,
            ),
        )
    else:
        cur.execute(
            """
            INSERT INTO study_cards (
                user_id, card_hash, channel_name, question, theory, topic,
                easiness, interval_days, repetition, due_date,
                weak_flag, unanswered_flag, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?)
            """,
            (
                int(user_id),
                card_hash,
                (channel_name or "")[:200],
                (question or "")[:1000],
                (theory or "")[:6000],
                (topic or "")[:200],
                DEFAULT_SM2_EASINESS,
                DEFAULT_SM2_INTERVAL,
                DEFAULT_SM2_REPETITION,
                datetime.now().date().isoformat(),
                now_iso,
                now_iso,
            ),
        )

    conn.commit()
    conn.close()
    return card_hash


def record_review(
    db_path: str,
    user_id: int,
    channel_name: str,
    question: str,
    theory: str,
    score=None,
    answered=True,
    note="",
):
    ensure_study_tables(db_path)
    now = datetime.now()
    now_iso = now.isoformat()
    card_hash = upsert_card(db_path, user_id, channel_name, question, theory)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM study_cards WHERE user_id = ? AND card_hash = ?",
        (int(user_id), card_hash),
    )
    row = cur.fetchone()
    if row is None:
        conn.close()
        return None

    quality = 0 if not answered else _score_to_quality(score)
    sm2 = _apply_sm2(
        interval_days=row["interval_days"],
        repetition=row["repetition"],
        easiness=row["easiness"],
        quality=quality,
    )
    due_date = (now.date() + timedelta(days=sm2["interval_days"])).isoformat()

    cur.execute(
        """
        UPDATE study_cards
        SET easiness = ?, interval_days = ?, repetition = ?, due_date = ?,
            last_reviewed_at = ?, last_score = ?, weak_flag = ?,
            unanswered_flag = ?, updated_at = ?
        WHERE user_id = ? AND card_hash = ?
        """,
        (
            sm2["easiness"],
            sm2["interval_days"],
            sm2["repetition"],
            due_date,
            now_iso,
            score,
            sm2["weak_flag"],
            0 if answered else 1,
            now_iso,
            int(user_id),
            card_hash,
        ),
    )

    cur.execute(
        """
        INSERT INTO study_reviews (
            user_id, card_hash, score, quality, answered, note, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(user_id),
            card_hash,
            score,
            quality,
            1 if answered else 0,
            (note or "")[:1000],
            now_iso,
        ),
    )

    conn.commit()
    conn.close()

    return {
        "card_hash": card_hash,
        "quality": quality,
        "due_date": due_date,
        "interval_days": sm2["interval_days"],
        "easiness": sm2["easiness"],
        "weak_flag": sm2["weak_flag"],
    }


def mark_unanswered_cards(db_path: str, user_id: int, pending_items: list):
    ensure_study_tables(db_path)
    now_iso = datetime.now().isoformat()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    for item in pending_items or []:
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        channel_name = str(item.get("channel_name", "")).strip()
        theory = str(item.get("theory", "")).strip()
        card_hash = upsert_card(db_path, user_id, channel_name, question, theory)

        cur.execute(
            """
            UPDATE study_cards
            SET weak_flag = 1, unanswered_flag = 1, updated_at = ?
            WHERE user_id = ? AND card_hash = ?
            """,
            (now_iso, int(user_id), card_hash),
        )

        cur.execute(
            """
            INSERT INTO study_reviews (
                user_id, card_hash, score, quality, answered, note, created_at
            ) VALUES (?, ?, NULL, 0, 0, ?, ?)
            """,
            (int(user_id), card_hash, "Unanswered in previous session", now_iso),
        )

    conn.commit()
    conn.close()


def get_knowledge_by_days(db_path: str, user_id: int, days: int = 7):
    ensure_study_tables(db_path)
    from_dt = datetime.now() - timedelta(days=max(1, int(days)))

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT channel_name, topic, question, theory, due_date, last_score,
               weak_flag, unanswered_flag, updated_at
        FROM study_cards
        WHERE user_id = ? AND datetime(updated_at) >= datetime(?)
        ORDER BY datetime(updated_at) DESC
        """,
        (int(user_id), from_dt.isoformat()),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def build_adaptive_path(db_path: str, user_id: int, days: int = 7):
    rows = get_knowledge_by_days(db_path, user_id, days)
    if not rows:
        return {
            "weak_items": [],
            "focus_topics": [],
            "next_actions": ["Tiếp tục học đều, chưa có dữ liệu yếu trong kỳ này."],
        }

    weak_items = [
        row
        for row in rows
        if int(row.get("weak_flag") or 0) == 1
        or int(row.get("unanswered_flag") or 0) == 1
    ]

    topic_scores = {}
    for row in rows:
        topic = (
            (row.get("topic") or row.get("channel_name") or "general").strip().lower()
        )
        score = row.get("last_score")
        score_val = float(score) if score is not None else 0.0
        entry = topic_scores.setdefault(topic, {"sum": 0.0, "count": 0, "weak": 0})
        entry["sum"] += score_val
        entry["count"] += 1
        if int(row.get("weak_flag") or 0) == 1:
            entry["weak"] += 1

    ranked_topics = []
    for topic, info in topic_scores.items():
        avg = (info["sum"] / info["count"]) if info["count"] else 0.0
        ranked_topics.append(
            {
                "topic": topic,
                "avg_score": round(avg, 2),
                "weak_count": info["weak"],
                "total": info["count"],
            }
        )

    ranked_topics.sort(key=lambda x: (x["avg_score"], -x["weak_count"], -x["total"]))
    focus_topics = ranked_topics[:3]

    actions = []
    if weak_items:
        actions.append(
            f"Ưu tiên ôn lại {min(len(weak_items), 10)} thẻ yếu/chưa trả lời."
        )
    for topic in focus_topics:
        actions.append(
            f"Dành 2 phiên tuần này cho chủ đề '{topic['topic']}' (điểm TB {topic['avg_score']})."
        )
    actions.append("Sau mỗi phiên ôn, trả lời lại câu yếu để cập nhật lịch SM-2.")

    return {
        "weak_items": weak_items[:15],
        "focus_topics": focus_topics,
        "next_actions": actions,
    }
