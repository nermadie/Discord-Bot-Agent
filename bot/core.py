import asyncio
import random
from datetime import datetime

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands

from bot.config.settings import *
from bot.state.runtime import *
from bot.services.knowledge_service import KnowledgeBot
from bot.services.study_service import (
    _append_study_event,
    _apply_unanswered_penalty,
    _get_daily_mission_status,
    _build_question_theory_text,
    _build_study_metrics_embed,
    _ensure_study_memory_tables,
    _get_or_create_study_profile,
    _mark_question_answered,
    _normalize_score_value,
    _persist_questions_for_spaced_repetition,
    _record_spaced_review,
)
from bot.utils import (
    _attachment_context_for_summary,
    _build_reason_single_message,
    _extract_image_urls_from_attachments,
    _format_rich_text_for_discord,
    _split_text_chunks,
)
from bot.views import ChatSessionView, SummaryInteractiveView, configure_views
from bot.registrars import (
    register_events_and_tasks,
    register_prefix_commands,
    register_slash_commands,
)
from tools.constants import SUMMARY_BATCH_SIZE, SUMMARY_FETCH_MAX_MESSAGES


intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
bot = commands.Bot(command_prefix=commands.when_mentioned, intents=intents)
bot.remove_command("help")


knowledge_bot = KnowledgeBot()


def _get_summary_channel_option_items():
    """Build selectable channel options for summary command autocomplete."""
    items = [("All monitored channels", "all")]
    for channel_id in CHANNELS_TO_MONITOR:
        if channel_id == MAIN_CHANNEL_ID:
            continue

        safe_channel_id = str(channel_id).strip()
        channel = (
            bot.get_channel(int(channel_id)) if str(channel_id).isdigit() else None
        )
        label = f"#{channel.name}" if channel else f"channel-{safe_channel_id}"

        label = label[:100]
        safe_value = safe_channel_id[:100]
        if not safe_value:
            continue
        items.append((label, safe_value))
    return items


async def summary_channel_autocomplete(interaction: discord.Interaction, current: str):
    """Return filtered summary channel choices for slash autocomplete."""
    try:
        current_text = (current or "").lower().strip()
        choices = []
        for name, value in _get_summary_channel_option_items():
            if (
                not current_text
                or current_text in name.lower()
                or current_text in value
            ):
                choices.append(app_commands.Choice(name=name[:100], value=value[:100]))
        return choices[:25]
    except Exception:
        fallback = [app_commands.Choice(name="All monitored channels", value="all")]
        for channel_id in CHANNELS_TO_MONITOR:
            if channel_id == MAIN_CHANNEL_ID:
                continue
            value = str(channel_id).strip()[:100]
            if not value:
                continue
            fallback.append(
                app_commands.Choice(name=f"channel-{value}"[:100], value=value)
            )
        return fallback[:25]


def _today_vn_date():
    """Get current date in Vietnam timezone."""
    return datetime.now(VIETNAM_TZ).date()


def _ensure_daily_window_rollover():
    """Reset daily in-memory state when date changes in Vietnam timezone."""
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
    """Track last interaction timestamp for idle motivation logic."""
    _last_interaction_at[int(user_id)] = datetime.now(VIETNAM_TZ)


def _build_study_status_text(user_id):
    """Build compact study status text from current user profile metrics."""
    profile = _get_or_create_study_profile(user_id)
    return (
        "📈 **Tình hình học tập hôm nay**\n"
        f"• ⭐ Điểm: **{profile.get('total_points', 0)}**\n"
        f"• 🔥 Streak: **{profile.get('streak_days', 0)} ngày**\n"
        f"• ✅ Trả lời đạt: **{profile.get('passed_count', 0)}**\n"
        f"• ❌ Bỏ lỡ: **{profile.get('missed_count', 0)}**"
    )


async def _fetch_motivational_slogan():
    """Fetch a motivational quote from API with local fallback pool."""
    fallback = [
        "Học mỗi ngày một chút, tương lai đổi rất nhiều.",
        "Kỷ luật học tập hôm nay là tự do ngày mai.",
        "Không cần giỏi ngay, chỉ cần không bỏ cuộc.",
        "1% tiến bộ mỗi ngày vẫn là tiến bộ.",
        "Bạn không cần chạy nhanh, chỉ cần học đều.",
        "Mỗi lần ngồi vào bàn học là một lần thắng sự trì hoãn.",
    ]

    study_keywords = {
        "study",
        "learn",
        "learning",
        "education",
        "student",
        "knowledge",
        "discipline",
        "practice",
        "school",
        "book",
    }

    def _is_study_oriented(text):
        lowered = str(text or "").lower()
        return any(keyword in lowered for keyword in study_keywords)

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=8)
        ) as session:
            async with session.get(
                "https://api.quotable.io/random",
                params={"tags": "education|inspirational|success"},
            ) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    if isinstance(data, dict):
                        quote = str(data.get("content", "")).strip()
                        author = str(data.get("author", "")).strip()
                        if quote:
                            return f"{quote} — {author}" if author else quote

            async with session.get("https://zenquotes.io/api/random") as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    if isinstance(data, list) and data:
                        quote = str(data[0].get("q", "")).strip()
                        author = str(data[0].get("a", "")).strip()
                        if quote and _is_study_oriented(quote):
                            return f"{quote} — {author}" if author else quote
    except Exception:
        pass

    return random.choice(fallback)


async def _collect_new_messages_since(
    channel,
    after_message_id=None,
    latest_messages=SUMMARY_FETCH_MAX_MESSAGES,
    only_today=False,
):
    """Collect normalized message rows from a channel for summary processing."""
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

    rows.sort(key=lambda item: item["id"])
    rows = rows[-limit:]
    texts = [item["text"] for item in rows]
    newest_id = rows[-1]["id"] if rows else None
    return texts, newest_id


async def _safe_followup_send(
    interaction: discord.Interaction,
    content: str = None,
    embed: discord.Embed = None,
    view: discord.ui.View = None,
    ephemeral: bool = False,
):
    """Send follow-up response safely with retry and channel fallback on failures."""
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


def _build_summary_embed(
    channel_name, total_messages, summary_data, question_start_index=1
):
    """Build study summary embed and indexed review-question payload."""

    def _clip_field_text(text, max_chars=1024):
        value = str(text or "").strip()
        if len(value) <= max_chars:
            return value
        clipped = value[: max_chars - 1].rstrip()
        return clipped + "…"

    def _chunk_field_text(text, chunk_chars=950, max_chunks=3):
        rows = [str(line).rstrip() for line in str(text or "").splitlines()]
        rows = [line for line in rows if line.strip()]
        if not rows:
            return []

        chunks = []
        current = ""
        for line in rows:
            line_value = line
            if len(line_value) > chunk_chars:
                line_value = line_value[: chunk_chars - 1].rstrip() + "…"

            candidate = line_value if not current else f"{current}\n{line_value}"
            if len(candidate) <= chunk_chars:
                current = candidate
                continue

            if current:
                chunks.append(current)
            current = line_value

            if len(chunks) >= max_chunks:
                break

        if current and len(chunks) < max_chunks:
            chunks.append(current)
        return chunks

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
            value=_clip_field_text("\n".join(summary_lines)),
            inline=False,
        )

    if detailed_summary:
        detail_chunks = _chunk_field_text(
            detailed_summary, chunk_chars=950, max_chunks=3
        )
        if not detail_chunks:
            detail_chunks = [_clip_field_text(detailed_summary, max_chars=950)]

        if len(detail_chunks) == 1:
            embed.add_field(
                name="📖 Phân tích sâu (rút gọn)",
                value=detail_chunks[0],
                inline=False,
            )
        else:
            for idx, chunk in enumerate(detail_chunks, start=1):
                embed.add_field(
                    name=f"📖 Phân tích sâu ({idx}/{len(detail_chunks)})",
                    value=chunk,
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
                value=_clip_field_text("\n".join(question_lines)),
                inline=False,
            )

    embed.set_footer(text=f"Model: {model_used}")
    return embed, numbered_questions


def _build_study_context_text(channel_name, summary_data, numbered_questions=None):
    """Create reusable context text from summary output for later chat prompts."""
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


def _build_chat_context_text(session):
    """Build compact prior chat context from a stored chat session."""
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
    """Persist chat session metadata in memory and return new session id."""
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
    """Continue paginated summary generation for authorized user."""
    _ensure_daily_window_rollover()

    if user_id != YOUR_USER_ID:
        return {"ok": False, "message": "⛔ Bạn không có quyền dùng lệnh này."}

    if not summary_state:
        return {"ok": False, "message": "📚 Không có phần dở"}

    channel_id = list(summary_state.keys())[0]
    state = summary_state[channel_id]

    summary_data, has_more = await knowledge_bot.summarize_daily_knowledge(
        state["messages"], state["channel_name"], state["offset"], SUMMARY_BATCH_SIZE
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
        processed_count = max(
            1, int(summary_data.get("processed_count") or SUMMARY_BATCH_SIZE)
        )
        summary_state[channel_id]["offset"] = min(
            len(state["messages"]),
            int(summary_state[channel_id].get("offset", 0)) + processed_count,
        )
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


configure_views(
    chat_sessions=_chat_sessions,
    pending_chat_context=_pending_chat_context,
    knowledge_bot=knowledge_bot,
    build_chat_context_text=_build_chat_context_text,
    create_chat_session=_create_chat_session,
    format_rich_text_for_discord=_format_rich_text_for_discord,
    split_text_chunks=_split_text_chunks,
    normalize_score_value=_normalize_score_value,
    append_study_event=_append_study_event,
    record_spaced_review=_record_spaced_review,
    mark_question_answered=_mark_question_answered,
    build_question_theory_text=_build_question_theory_text,
    build_study_context_text=_build_study_context_text,
    continue_summary_for_user=_continue_summary_for_user,
    get_daily_mission_status=_get_daily_mission_status,
    fetch_motivational_slogan=_fetch_motivational_slogan,
)

register_events_and_tasks(
    bot,
    {
        "knowledge_bot": knowledge_bot,
        "APP_GUILD_ID": APP_GUILD_ID,
        "MAIN_CHANNEL_ID": MAIN_CHANNEL_ID,
        "YOUR_USER_ID": YOUR_USER_ID,
        "CHANNELS_TO_MONITOR": CHANNELS_TO_MONITOR,
        "VIETNAM_TZ": VIETNAM_TZ,
        "SLOGAN_CHECK_INTERVAL_MINUTES": SLOGAN_CHECK_INTERVAL_MINUTES,
        "SLOGAN_IDLE_MINUTES": SLOGAN_IDLE_MINUTES,
        "SUMMARY_BATCH_SIZE": SUMMARY_BATCH_SIZE,
        "daily_messages": daily_messages,
        "summary_state": summary_state,
        "_sent_upcoming_reminders": _sent_upcoming_reminders,
        "_active_countdowns": _active_countdowns,
        "_last_interaction_at": _last_interaction_at,
        "_last_slogan_sent_at": _last_slogan_sent_at,
        "_ensure_daily_window_rollover": _ensure_daily_window_rollover,
        "_mark_user_interaction": _mark_user_interaction,
        "_attachment_context_for_summary": _attachment_context_for_summary,
        "_fetch_motivational_slogan": _fetch_motivational_slogan,
        "_build_study_status_text": _build_study_status_text,
        "_build_summary_embed": _build_summary_embed,
    },
)

register_prefix_commands(
    bot,
    {
        "knowledge_bot": knowledge_bot,
        "MAIN_CHANNEL_ID": MAIN_CHANNEL_ID,
        "YOUR_USER_ID": YOUR_USER_ID,
        "VIETNAM_TZ": VIETNAM_TZ,
        "STUDY_PASS_THRESHOLD": STUDY_PASS_THRESHOLD,
        "STUDY_POINTS_PASS": STUDY_POINTS_PASS,
        "SUMMARY_BATCH_SIZE": SUMMARY_BATCH_SIZE,
        "daily_messages": daily_messages,
        "summary_state": summary_state,
        "_last_events": _last_events,
        "_last_tasks": _last_tasks,
        "_pending_chat_context": _pending_chat_context,
        "_study_questions": _study_questions,
        "_create_chat_session": _create_chat_session,
        "_mark_user_interaction": _mark_user_interaction,
        "_fetch_motivational_slogan": _fetch_motivational_slogan,
        "_ensure_daily_window_rollover": _ensure_daily_window_rollover,
        "_apply_unanswered_penalty": _apply_unanswered_penalty,
        "_build_summary_embed": _build_summary_embed,
        "_persist_questions_for_spaced_repetition": _persist_questions_for_spaced_repetition,
        "_build_question_theory_text": _build_question_theory_text,
        "_append_study_event": _append_study_event,
        "_normalize_score_value": _normalize_score_value,
        "_record_spaced_review": _record_spaced_review,
        "_build_study_metrics_embed": _build_study_metrics_embed,
        "_get_daily_mission_status": _get_daily_mission_status,
        "_mark_question_answered": _mark_question_answered,
        "_split_text_chunks": _split_text_chunks,
        "_format_rich_text_for_discord": _format_rich_text_for_discord,
        "_extract_image_urls_from_attachments": _extract_image_urls_from_attachments,
        "_build_reason_single_message": _build_reason_single_message,
        "ChatSessionView": ChatSessionView,
        "SummaryInteractiveView": SummaryInteractiveView,
    },
)

register_slash_commands(
    bot,
    {
        "knowledge_bot": knowledge_bot,
        "YOUR_USER_ID": YOUR_USER_ID,
        "CHANNELS_TO_MONITOR": CHANNELS_TO_MONITOR,
        "VIETNAM_TZ": VIETNAM_TZ,
        "STUDY_PASS_THRESHOLD": STUDY_PASS_THRESHOLD,
        "STUDY_POINTS_PASS": STUDY_POINTS_PASS,
        "SUMMARY_BATCH_SIZE": SUMMARY_BATCH_SIZE,
        "SUMMARY_FETCH_MAX_MESSAGES": SUMMARY_FETCH_MAX_MESSAGES,
        "daily_messages": daily_messages,
        "summary_state": summary_state,
        "_last_events": _last_events,
        "_last_tasks": _last_tasks,
        "_pending_chat_context": _pending_chat_context,
        "_study_questions": _study_questions,
        "_last_summary_fetch_message_ids": _last_summary_fetch_message_ids,
        "_create_chat_session": _create_chat_session,
        "_mark_user_interaction": _mark_user_interaction,
        "_fetch_motivational_slogan": _fetch_motivational_slogan,
        "_ensure_daily_window_rollover": _ensure_daily_window_rollover,
        "_collect_new_messages_since": _collect_new_messages_since,
        "_continue_summary_for_user": _continue_summary_for_user,
        "_safe_followup_send": _safe_followup_send,
        "summary_channel_autocomplete": summary_channel_autocomplete,
        "_apply_unanswered_penalty": _apply_unanswered_penalty,
        "_build_summary_embed": _build_summary_embed,
        "_persist_questions_for_spaced_repetition": _persist_questions_for_spaced_repetition,
        "_build_question_theory_text": _build_question_theory_text,
        "_append_study_event": _append_study_event,
        "_normalize_score_value": _normalize_score_value,
        "_record_spaced_review": _record_spaced_review,
        "_build_study_metrics_embed": _build_study_metrics_embed,
        "_get_daily_mission_status": _get_daily_mission_status,
        "_mark_question_answered": _mark_question_answered,
        "_ensure_study_memory_tables": _ensure_study_memory_tables,
        "_split_text_chunks": _split_text_chunks,
        "_format_rich_text_for_discord": _format_rich_text_for_discord,
        "_extract_image_urls_from_attachments": _extract_image_urls_from_attachments,
        "_build_reason_single_message": _build_reason_single_message,
        "ChatSessionView": ChatSessionView,
        "SummaryInteractiveView": SummaryInteractiveView,
    },
)


def run():
    """Validate required config and start Discord bot runtime."""
    if not DISCORD_TOKEN:
        print("❌ Thiếu DISCORD_TOKEN")
        return
    if not GITHUB_TOKEN:
        print("❌ Thiếu GITHUB_TOKEN")
        return
    if YOUR_USER_ID == 0:
        print("❌ Thiếu YOUR_USER_ID")
        return

    print("🚀 Bot khởi động...")
    bot.run(DISCORD_TOKEN)
