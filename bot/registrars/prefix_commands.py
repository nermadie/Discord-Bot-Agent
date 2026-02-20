import re
import json
from datetime import datetime, time, timedelta

import discord


def register_prefix_commands(bot, deps):
    """Register all prefix commands with dependency injection from core."""
    knowledge_bot = deps["knowledge_bot"]

    MAIN_CHANNEL_ID = deps["MAIN_CHANNEL_ID"]
    YOUR_USER_ID = deps["YOUR_USER_ID"]
    VIETNAM_TZ = deps["VIETNAM_TZ"]
    STUDY_PASS_THRESHOLD = deps["STUDY_PASS_THRESHOLD"]
    STUDY_POINTS_PASS = deps["STUDY_POINTS_PASS"]
    SUMMARY_BATCH_SIZE = deps.get("SUMMARY_BATCH_SIZE", 20)

    daily_messages = deps["daily_messages"]
    summary_state = deps["summary_state"]
    _last_events = deps["_last_events"]
    _last_tasks = deps["_last_tasks"]
    _pending_chat_context = deps["_pending_chat_context"]
    _study_questions = deps["_study_questions"]

    _create_chat_session = deps["_create_chat_session"]
    _mark_user_interaction = deps["_mark_user_interaction"]
    _fetch_motivational_slogan = deps["_fetch_motivational_slogan"]
    _ensure_daily_window_rollover = deps["_ensure_daily_window_rollover"]
    _apply_unanswered_penalty = deps["_apply_unanswered_penalty"]
    _build_summary_embed = deps["_build_summary_embed"]
    _persist_questions_for_spaced_repetition = deps[
        "_persist_questions_for_spaced_repetition"
    ]
    _build_question_theory_text = deps["_build_question_theory_text"]
    _append_study_event = deps["_append_study_event"]
    _normalize_score_value = deps["_normalize_score_value"]
    _record_spaced_review = deps["_record_spaced_review"]
    _build_study_metrics_embed = deps["_build_study_metrics_embed"]
    _get_daily_mission_status = deps["_get_daily_mission_status"]
    _mark_question_answered = deps["_mark_question_answered"]
    _split_text_chunks = deps["_split_text_chunks"]
    _format_rich_text_for_discord = deps["_format_rich_text_for_discord"]
    _extract_image_urls_from_attachments = deps["_extract_image_urls_from_attachments"]
    _build_reason_single_message = deps["_build_reason_single_message"]

    ChatSessionView = deps["ChatSessionView"]
    SummaryInteractiveView = deps["SummaryInteractiveView"]

    def _build_mission_progress_text(user_id, stats_payload=None):
        payload = stats_payload or {}
        lines = list(payload.get("daily_mission_lines") or [])
        summary = str(payload.get("daily_mission_summary") or "").strip()
        if not lines and not summary:
            mission_status = _get_daily_mission_status(user_id)
            lines = list(mission_status.get("lines") or [])
            summary = str(mission_status.get("summary") or "").strip()
        preview = "\n".join(lines[:3])
        return f"{summary}\n{preview}".strip()[:1024]

    def _build_slogan_embed(slogan_text, title="💪 Slogan học tập"):
        raw_lines = [str(line).strip() for line in str(slogan_text or "").splitlines()]
        raw_lines = [line for line in raw_lines if line]

        source_line = next(
            (line for line in raw_lines if line.lower().startswith("nguồn:")),
            "",
        )
        quote_lines = [line for line in raw_lines if line != source_line]
        quote_text = "\n".join(quote_lines).strip()

        embed = discord.Embed(
            title=title,
            description=(f"*{quote_text}*" if quote_text else "(không có nội dung)"),
            color=discord.Color.gold(),
            timestamp=datetime.now(VIETNAM_TZ),
        )
        if source_line:
            embed.add_field(
                name="Nguồn",
                value=source_line.replace("Nguồn:", "").strip(),
                inline=False,
            )
        return embed

    async def _send_mission_completion_reward(ctx, stats_payload):
        completed = list((stats_payload or {}).get("completed_missions") or [])
        if not completed:
            return

        total_reward = sum(int(item.get("reward_points") or 0) for item in completed)
        lines = [
            f"✅ {str(item.get('title') or 'Nhiệm vụ')} (+{int(item.get('reward_points') or 0)}đ)"
            for item in completed
        ]
        await ctx.send(
            "🎉 Hoàn thành nhiệm vụ tự học trong ngày!\n"
            + "\n".join(lines)
            + f"\n⭐ Thưởng nhiệm vụ: +{total_reward} điểm"
        )

        try:
            slogan_text = str(await _fetch_motivational_slogan()).strip()
        except Exception:
            slogan_text = ""

        try:
            cat_result = await knowledge_bot.get_random_cat_image()
        except Exception:
            cat_result = {}

        reward_embed = _build_slogan_embed(
            (
                slogan_text
                if slogan_text
                else "Giữ nhịp học đều mỗi ngày, bạn đang đi đúng hướng!"
            ),
            title="🐱 Phần thưởng nhiệm vụ",
        )
        if cat_result.get("ok") and cat_result.get("url"):
            reward_embed.set_image(url=str(cat_result.get("url")))
            reward_embed.set_footer(text="Source: TheCatAPI")
        await ctx.send(embed=reward_embed)

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
            embed.add_field(
                name="📚 Study", value="`!help study` - Học tập", inline=True
            )
            embed.add_field(
                name="💬 Chatbot", value="`!help chatbot` - Chat AI", inline=True
            )
            embed.add_field(
                name="🤖 Automation",
                value="`!help automation` - Tự động hóa",
                inline=True,
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
                    f"• Mỗi lần xử lý {SUMMARY_BATCH_SIZE} tin nhắn\n"
                    "• `!summary` dùng model chính: `openai/gpt-5-chat`"
                ),
                inline=False,
            )

        elif category == "chatbot":
            embed = discord.Embed(
                title="💬 Lệnh Chatbot", color=discord.Color.blurple()
            )
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

    @bot.command()
    async def calendar(ctx, *, date_str=""):
        """Show combined events and tasks for a parsed target date."""
        target_date = knowledge_bot.parse_date(date_str) if date_str else None
        calendar_data = await knowledge_bot.get_calendar(target_date)

        date_display = target_date.strftime("%d/%m") if target_date else "hôm nay"
        message = f"📅 **Lịch {date_display}:**\n\n"

        events = calendar_data["events"]
        if events:
            message += "**📌 EVENTS:**\n"
            for e in events:
                icon = "🔴" if e["is_important"] else "•"
                time_str = (
                    f"{e['time']}-{e['end_time']}" if e["end_time"] else e["time"]
                )
                message += f"{icon} {time_str} {e['summary']}\n"
            message += "\n"

        tasks_list = calendar_data["tasks"]
        if tasks_list:
            message += "**📋 TASKS:**\n"
            for t in tasks_list:
                icon = "🔴" if t["overdue"] else "•"
                time_str = t["due_time"].strftime("%H:%M") if t["due_time"] else ""
                message += f"{icon} {time_str} {t['title']}\n"

        if not events and not tasks_list:
            message += "Không có gì cả"

        await ctx.send(message)

    @bot.command()
    async def events(ctx, *, date_str=""):
        """List events for a target date and cache result for index-based actions."""
        target_date = knowledge_bot.parse_date(date_str) if date_str else None
        events_list = await knowledge_bot.get_events(target_date)

        if isinstance(events_list, str):
            await ctx.send(events_list)
            return

        if not events_list:
            date_display = target_date.strftime("%d/%m") if target_date else "hôm nay"
            await ctx.send(f"📅 Không có events {date_display}")
            return

        _last_events[ctx.author.id] = events_list

        date_display = target_date.strftime("%d/%m") if target_date else "hôm nay"
        message = f"📅 **Events {date_display}:**\n\n"

        for i, e in enumerate(events_list, 1):
            icon = "🔴" if e["is_important"] else ""
            time_str = f"{e['time']}-{e['end_time']}" if e["end_time"] else e["time"]
            message += f"{i}. {icon} {time_str} **{e['summary']}**\n"
            if e["description"]:
                message += f"   ↳ {e['description'][:100]}\n"

        await ctx.send(message)

    @bot.command()
    async def add_event(ctx, *, args):
        """Create a new calendar event from pipe-delimited input."""
        parts = [p.strip() for p in args.split("|")]
        if len(parts) < 2:
            await ctx.send(
                "⚠️ Format: `!add_event <title> | <date time-endtime> | <desc>`"
            )
            return

        title = parts[0]
        datetime_str = parts[1]
        description = parts[2] if len(parts) > 2 else ""

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

        time_match = re.search(
            r"(\d{1,2}[h:]\d{2}|\d{1,2}h?)\s*-\s*(\d{1,2}[h:]\d{2}|\d{1,2}h?)",
            datetime_str,
        )

        if time_match:
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
        """Delete cached event by 1-based index from last `!events` output."""
        if ctx.author.id not in _last_events:
            await ctx.send("⚠️ Gọi `!events` trước")
            return

        events_list = _last_events[ctx.author.id]
        if index < 1 or index > len(events_list):
            await ctx.send(f"⚠️ Chọn từ 1-{len(events_list)}")
            return

        event = events_list[index - 1]
        result = await knowledge_bot.delete_event(event["id"])
        await ctx.send(result)
        del _last_events[ctx.author.id]

    @bot.command()
    async def move_event(ctx, *, args):
        """Move event time by index using parsed date/time input."""
        parts = [p.strip() for p in args.split("|")]
        if len(parts) < 2:
            await ctx.send("⚠️ Format: `!move_event <số> | <date time>`")
            return

        try:
            index = int(parts[0])
        except Exception:
            await ctx.send("⚠️ Số không hợp lệ")
            return

        if ctx.author.id not in _last_events:
            await ctx.send("⚠️ Gọi `!events` trước")
            return

        events_list = _last_events[ctx.author.id]
        if index < 1 or index > len(events_list):
            await ctx.send(f"⚠️ Chọn từ 1-{len(events_list)}")
            return

        event = events_list[index - 1]
        datetime_str = parts[1]

        date_match = re.search(
            r"(\d{1,2}[/-]\d{1,2}|today|tomorrow)", datetime_str, re.I
        )
        if not date_match:
            await ctx.send("⚠️ Không tìm thấy ngày")
            return

        target_date = knowledge_bot.parse_date(date_match.group(1))
        time_match = re.search(r"(\d{1,2}[h:]\d{2}|\d{1,2}h?)", datetime_str)

        if not time_match:
            await ctx.send("⚠️ Không tìm thấy giờ")
            return

        new_time = knowledge_bot.parse_time(time_match.group(1))
        new_start = knowledge_bot.timezone.localize(
            datetime.combine(target_date, new_time)
        )

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

    @bot.command()
    async def tasks(ctx, *, date_str=""):
        """List tasks by date and cache list for complete/delete commands."""
        target_date = knowledge_bot.parse_date(date_str) if date_str else None
        tasks_list = await knowledge_bot.get_tasks(
            date=target_date, show_completed=False
        )

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

        message += "\n💡 `!done <số>` để hoàn thành"
        await ctx.send(message)

    @bot.command()
    async def overdue(ctx):
        """List overdue tasks only and cache result for follow-up actions."""
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

        message += "\n💡 `!done <số>` để hoàn thành"
        await ctx.send(message)

    @bot.command()
    async def add_task(ctx, *, args):
        """Create a task from pipe-delimited title/date/notes input."""
        parts = [p.strip() for p in args.split("|")]
        if len(parts) < 1:
            await ctx.send("⚠️ Format: `!add_task <title> | <date time> | <notes>`")
            return

        title = parts[0]
        due_datetime = None
        notes = ""

        if len(parts) >= 2:
            datetime_str = parts[1]
            date_match = re.search(
                r"(\d{1,2}[/-]\d{1,2}|today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
                datetime_str,
                re.I,
            )
            if date_match:
                target_date = knowledge_bot.parse_date(date_match.group(1))
                if target_date:
                    time_match = re.search(
                        r"(\d{1,2}[h:]\d{2}|\d{1,2}h?)", datetime_str
                    )
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

        if len(parts) >= 3:
            notes = parts[2]

        result = await knowledge_bot.add_task(title, due_datetime, notes)
        await ctx.send(result)

    @bot.command()
    async def done(ctx, index: int):
        """Mark cached task as completed using 1-based index."""
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
        """Delete cached task by 1-based index."""
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

    @bot.command()
    async def weather(ctx):
        """Show current weather summary."""
        result = await knowledge_bot.get_weather()
        await ctx.send(result)

    @bot.command(name="slogan")
    async def slogan(ctx):
        """Send one motivational slogan for authorized owner account."""
        if ctx.author.id != YOUR_USER_ID:
            return
        _mark_user_interaction(ctx.author.id)
        text = await _fetch_motivational_slogan()
        await ctx.send(embed=_build_slogan_embed(text))

    @bot.command()
    async def chat(ctx, *, prompt=""):
        """Run chat flow with optional image attachments and continuation session."""
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
            lines = [
                f"Ảnh {x['index']}: {str(x['text'])[:220]}" for x in extracted_ok[:2]
            ]
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
        """Run reasoning-focused answer flow and format output for Discord."""
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

    @bot.command()
    async def summary(ctx):
        """Generate study summary from cached daily messages and create review questions."""
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
                messages, channel_name, 0, SUMMARY_BATCH_SIZE
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

                    summary_stats = _append_study_event(
                        user_id=ctx.author.id,
                        event_type="summary",
                        points_delta=0,
                        channel_name=channel_name,
                        note=f"Tạo summary với {len(messages)} tin nhắn",
                    )
                    mission_text = _build_mission_progress_text(
                        ctx.author.id, summary_stats
                    )
                    if mission_text:
                        await ctx.send(f"🎯 **Nhiệm vụ hôm nay**\n{mission_text}")
                    await _send_mission_completion_reward(ctx, summary_stats)

                    for item in numbered_questions:
                        _study_questions[ctx.author.id].append(
                            {
                                "index": item["index"],
                                "channel_name": channel_name,
                                "question": item["question"],
                                "summary_points": summary_data.get(
                                    "summary_points", []
                                ),
                                "theory": theory_text,
                            }
                        )
                    question_index += len(numbered_questions)

                if has_more:
                    processed_count = max(
                        1,
                        int(summary_data.get("processed_count") or SUMMARY_BATCH_SIZE),
                    )
                    summary_state[channel_id] = {
                        "messages": messages,
                        "channel_name": channel_name,
                        "offset": processed_count,
                    }
                    await ctx.send(
                        f"💡 Còn {max(0, len(messages) - processed_count)} tin nhắn chưa summary. Bấm `Continue Summary` ngay dưới embed vừa gửi hoặc dùng `/continue_summary`.",
                    )

    @bot.command()
    async def answer(ctx, *, args=""):
        """Grade answer for a specific review question and update study metrics."""
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
        mission_text = _build_mission_progress_text(ctx.author.id, stats)
        if mission_text:
            embed.add_field(
                name="🎯 Nhiệm vụ hôm nay", value=mission_text, inline=False
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
        await _send_mission_completion_reward(ctx, stats)

    @bot.command()
    async def stats(ctx):
        """Show per-channel and total monitored message counts in memory."""
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
        """Display current monthly study metrics embed for authorized owner."""
        if ctx.author.id != YOUR_USER_ID:
            return
        await ctx.send(
            embed=_build_study_metrics_embed(ctx.author.id, ctx.author.display_name)
        )

    @bot.command()
    async def ping(ctx):
        """Return current bot latency."""
        await ctx.send(f"🏓 Pong! {round(bot.latency * 1000)}ms")

    @bot.command()
    async def countdown(ctx, *, name=""):
        """List active countdowns, optionally filtered by name."""
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
        """Create countdown from free-form date/time expression and optional emoji."""
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

        remaining_str = datetime_str[date_match.end() :].strip()
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

            await ctx.send(
                f"⚠️ **Thời gian phải trong tương lai**\n\n"
                f"📅 Bạn nhập: `{target_datetime.strftime('%d/%m/%Y %H:%M:%S')}`\n"
                f"🕐 Hiện tại VN: `{now.strftime('%d/%m/%Y %H:%M:%S')}`\n"
                f"⏰ Đã qua: **{hours_past} giờ {minutes_past} phút**"
            )
            return

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
        """Remove countdown by exact name."""
        if knowledge_bot.remove_countdown(name):
            await ctx.send(f"✅ Đã xóa countdown: {name}")
        else:
            await ctx.send(f"⚠️ Không tìm thấy countdown: {name}")

    @bot.command()
    async def newyear(
        ctx,
        year: int = None,
        month: int = 1,
        day: int = 1,
        hour: int = 0,
        minute: int = 0,
    ):
        """Create special New Year countdown with celebratory output style."""
        now = datetime.now(knowledge_bot.timezone)

        if year is None:
            if now.month == 12 and now.day == 31:
                year = now.year + 1
            elif now.month == 1 and now.day == 1:
                await ctx.send("🎆 Hôm nay là năm mới rồi!")
                return
            else:
                year = now.year + 1

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
        if knowledge_bot.add_countdown(
            countdown_name, ny_datetime, "🎆", label="newyear"
        ):
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
        """Create lunar new year countdown using built-in Tet date lookup."""
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

    @bot.command(name="gmail_digest")
    async def gmail_digest(ctx, *, date_str=""):
        """Show Gmail digest for a date or generate/store one immediately when no date provided."""
        if ctx.author.id != YOUR_USER_ID:
            return

        target_date = knowledge_bot.parse_history_date(date_str) if date_str else None
        if date_str and not target_date:
            await ctx.send("⚠️ Date không hợp lệ. VD: `today`, `tomorrow`, `18/2`.")
            return

        if target_date:
            row = knowledge_bot.get_latest_gmail_report_by_date(target_date)
            if not row:
                history = knowledge_bot.get_gmail_digest_history(target_date)
                await ctx.send(history)
                return

            important = json.loads(row.get("important_json") or "[]")
            todo_list = json.loads(row.get("todo_json") or "[]")
            unread = json.loads(row.get("unread_json") or "[]")
            sent_items = json.loads(row.get("sent_json") or "[]")
            lines = [f"📮 Gmail report {row.get('report_date')}"]
            lines.append("\nImportant Mail:")
            lines.extend(
                [
                    f"- {x.get('subject') if isinstance(x, dict) else x}"
                    for x in important[:8]
                ]
            )
            lines.append("\nTo do list:")
            lines.extend([f"- {x}" for x in todo_list[:10]])
            lines.append("\nMail Unread:")
            lines.extend(
                [
                    f"- {x.get('subject') if isinstance(x, dict) else x}"
                    for x in unread[:20]
                ]
            )
            lines.append("\nSent Mail Insight:")
            lines.append(f"- Tổng sent mail trong ngày: {len(sent_items)}")
            for item in sent_items[:10]:
                rec = knowledge_bot._extract_sent_action_record(item)
                lines.append(
                    f"- {rec.get('action')}: {rec.get('position')} @ {rec.get('company')}"
                )
            await ctx.send("\n".join(lines)[:3900])
            return

        result = await knowledge_bot.build_advanced_gmail_report(period="manual")
        if not result.get("ok"):
            await ctx.send(result.get("error", "⚠️ Không thể tạo Gmail digest."))
            return

        lines = ["📮 Gmail report"]
        lines.append("\nImportant Mail:")
        lines.extend(
            [f"- {x.get('subject')}" for x in result.get("important_mails", [])[:8]]
        )
        lines.append("\nTo do list:")
        lines.extend([f"- {x}" for x in result.get("todo_list", [])[:10]])
        lines.append("\nMail Unread:")
        lines.extend(
            [f"- {x.get('subject')}" for x in result.get("unread_items", [])[:20]]
        )
        lines.append("\nThông tin quan trọng:")
        lines.extend([f"- {x}" for x in result.get("key_info", [])[:8]])
        lines.append("\nSent Mail Insight:")
        if result.get("sent_summary"):
            lines.append(f"- {result.get('sent_summary')}")
        for item in result.get("sent_actions_done", [])[:10]:
            lines.append(f"- {item}")
        await ctx.send("\n".join(lines)[:3900])

    @bot.command(name="map")
    async def map_search(ctx, *, query=""):
        """Search places using natural language via OpenStreetMap Nominatim."""
        if not query.strip():
            await ctx.send("⚠️ Dùng: `!map <địa điểm cần tìm>`")
            return
        result = await knowledge_bot.search_place_natural(query)
        await ctx.send(result)
