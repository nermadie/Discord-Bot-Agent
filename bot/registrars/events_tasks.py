from datetime import datetime, time, timedelta

import discord
from discord.ext import tasks

from bot.config.settings import (
    WEATHER_ALERT_INTERVAL_HOURS as CFG_WEATHER_ALERT_INTERVAL_HOURS,
    WEATHER_ALERT_LOOKAHEAD_HOURS as CFG_WEATHER_ALERT_LOOKAHEAD_HOURS,
)


def register_events_and_tasks(bot, deps):
    """Register Discord events and background tasks using injected dependencies."""
    knowledge_bot = deps["knowledge_bot"]

    APP_GUILD_ID = deps["APP_GUILD_ID"]
    MAIN_CHANNEL_ID = deps["MAIN_CHANNEL_ID"]
    YOUR_USER_ID = deps["YOUR_USER_ID"]
    CHANNELS_TO_MONITOR = deps["CHANNELS_TO_MONITOR"]
    VIETNAM_TZ = deps["VIETNAM_TZ"]
    SLOGAN_CHECK_INTERVAL_MINUTES = deps["SLOGAN_CHECK_INTERVAL_MINUTES"]
    SLOGAN_IDLE_MINUTES = deps["SLOGAN_IDLE_MINUTES"]
    SUMMARY_BATCH_SIZE = deps["SUMMARY_BATCH_SIZE"]

    daily_messages = deps["daily_messages"]
    summary_state = deps["summary_state"]
    _sent_upcoming_reminders = deps["_sent_upcoming_reminders"]
    _active_countdowns = deps["_active_countdowns"]
    _last_interaction_at = deps["_last_interaction_at"]
    _last_slogan_sent_at = deps["_last_slogan_sent_at"]

    _ensure_daily_window_rollover = deps["_ensure_daily_window_rollover"]
    _mark_user_interaction = deps["_mark_user_interaction"]
    _attachment_context_for_summary = deps["_attachment_context_for_summary"]
    _fetch_motivational_slogan = deps["_fetch_motivational_slogan"]
    _build_study_status_text = deps["_build_study_status_text"]
    _build_summary_embed = deps["_build_summary_embed"]

    WEATHER_ALERT_INTERVAL_HOURS = int(
        deps.get("WEATHER_ALERT_INTERVAL_HOURS", CFG_WEATHER_ALERT_INTERVAL_HOURS)
    )
    WEATHER_ALERT_LOOKAHEAD_HOURS = int(
        deps.get("WEATHER_ALERT_LOOKAHEAD_HOURS", CFG_WEATHER_ALERT_LOOKAHEAD_HOURS)
    )

    @bot.event
    async def on_ready():
        """Handle bot startup: sync slash commands, start loops, auto-enable seasonal countdowns."""
        print(f"✅ Bot: {bot.user}")
        _ensure_daily_window_rollover()

        try:
            # Ưu tiên guild sync khi có APP_GUILD_ID để cập nhật command gần như ngay lập tức.
            if APP_GUILD_ID:
                guild = discord.Object(id=APP_GUILD_ID)
                synced = await bot.tree.sync(guild=guild)
                print(
                    f"✅ Synced {len(synced)} slash command(s) to guild {APP_GUILD_ID}"
                )
            else:
                synced = await bot.tree.sync()
                print(f"✅ Synced {len(synced)} global slash command(s)")
        except Exception as e:
            print(f"⚠️ Slash sync lỗi: {e}")

        morning_greeting.start()
        calendar_reminder.start()
        evening_summary.start()
        nightly_gmail_digest.start()
        weather_risk_alert.start()
        end_of_day_review.start()
        countdown_checker.start()
        daily_rollover.start()
        idle_motivation.start()

        now = datetime.now(knowledge_bot.timezone)
        if now.month == 12 and now.day == 31:
            ny_datetime = knowledge_bot.timezone.localize(
                datetime(now.year + 1, 1, 1, 0, 0, 0)
            )
            knowledge_bot.add_countdown(
                f"Năm Mới {now.year + 1}",
                ny_datetime,
                "🎆",
                label="newyear",
            )
            print(f"🎆 Auto-activated New Year {now.year + 1} countdown!")

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
                label="",
            )
            print(f"🧧 Auto-activated Tet {tet_year} countdown!")

    @bot.event
    async def on_message(message):
        """Track user messages for daily summary cache and interaction timestamp."""
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

        return

    @bot.event
    async def on_command_completion(ctx):
        """Update user interaction time after prefix command completion."""
        _mark_user_interaction(ctx.author.id)

    @bot.event
    async def on_app_command_completion(interaction: discord.Interaction, command):
        """Update user interaction time after slash command completion."""
        _mark_user_interaction(interaction.user.id)

    @tasks.loop(time=time(hour=0, minute=0, tzinfo=VIETNAM_TZ))
    async def daily_rollover():
        """Daily midnight reset guard for in-memory daily state."""
        _ensure_daily_window_rollover()

    @daily_rollover.before_loop
    async def before_daily_rollover():
        """Wait until bot is ready before starting daily rollover loop."""
        await bot.wait_until_ready()

    @tasks.loop(minutes=SLOGAN_CHECK_INTERVAL_MINUTES)
    async def idle_motivation():
        """Send motivation ping when owner stays inactive longer than configured threshold."""
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
        """Wait until bot is ready before starting idle-motivation loop."""
        await bot.wait_until_ready()

    @tasks.loop(time=time(hour=6, minute=30, tzinfo=VIETNAM_TZ))
    async def morning_greeting():
        """Send daily morning digest: weather, calendar, tasks, and study status."""
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

        events = calendar_data["events"]
        if events:
            message += "\n📅 **Events hôm nay:**\n"
            has_important = any(e["is_important"] for e in events)
            if has_important:
                message += "⚠️ **CÓ SỰ KIỆN QUAN TRỌNG!**\n"
            for e in events[:10]:
                icon = "🔴" if e["is_important"] else "•"
                time_str = (
                    f"{e['time']}-{e['end_time']}" if e["end_time"] else e["time"]
                )
                message += f"{icon} {time_str} {e['summary']}\n"

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
        """Wait until bot is ready before starting morning greeting loop."""
        await bot.wait_until_ready()

    @tasks.loop(minutes=1)
    async def calendar_reminder():
        """Check upcoming events/tasks and push one-time 30-minute reminders."""
        if MAIN_CHANNEL_ID == 0:
            return

        channel = bot.get_channel(MAIN_CHANNEL_ID)
        if not channel:
            return

        now = datetime.now(knowledge_bot.timezone)

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

                # reminder_key giúp chống gửi trùng khi loop chạy mỗi phút.

                if (
                    0 < minutes_until <= 30
                    and reminder_key not in _sent_upcoming_reminders
                ):
                    icon = "🔔" if event["is_important"] else "⏰"
                    await channel.send(
                        f"{icon} **30 phút nữa:**\n📌 {event['summary']} ({event['time']})"
                    )
                    _sent_upcoming_reminders.add(reminder_key)

        tasks_list = await knowledge_bot.get_tasks(
            date=now.date(), show_completed=False
        )
        if isinstance(tasks_list, list):
            for task in tasks_list:
                if (
                    task.get("overdue")
                    or not task.get("due")
                    or not task.get("due_time")
                ):
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

                if (
                    0 < minutes_until <= 30
                    and reminder_key not in _sent_upcoming_reminders
                ):
                    due_time = task_due_dt.strftime("%H:%M")
                    await channel.send(
                        f"📋 **30 phút nữa đến hạn task:**\n📝 {task['title']} ({due_time})"
                    )
                    _sent_upcoming_reminders.add(reminder_key)

    @calendar_reminder.before_loop
    async def before_calendar_reminder():
        """Wait until bot is ready before starting calendar reminder loop."""
        await bot.wait_until_ready()

    @tasks.loop(time=time(hour=20, minute=0, tzinfo=VIETNAM_TZ))
    async def end_of_day_review():
        """Send end-of-day review for unfinished and overdue tasks."""
        if MAIN_CHANNEL_ID == 0:
            return

        channel = bot.get_channel(MAIN_CHANNEL_ID)
        if not channel:
            return

        now = datetime.now(knowledge_bot.timezone)
        today_tasks = await knowledge_bot.get_tasks(
            date=now.date(), show_completed=False
        )
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
                time_str = (
                    task["due_time"].strftime("%H:%M") if task["due_time"] else ""
                )
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
        """Wait until bot is ready before starting end-of-day review loop."""
        await bot.wait_until_ready()

    @tasks.loop(time=time(hour=21, minute=0, tzinfo=VIETNAM_TZ))
    async def evening_summary():
        """Run scheduled nightly summary for monitored channel message caches."""
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
                    processed_count = max(
                        1,
                        int(summary_data.get("processed_count") or SUMMARY_BATCH_SIZE),
                    )
                    summary_state[channel_id] = {
                        "messages": messages,
                        "channel_name": channel_name,
                        "offset": processed_count,
                    }
                    await channel.send(
                        f"💡 Còn {max(0, len(messages) - processed_count)} tin nhắn. Dùng `/continue_summary`"
                    )

        if not summary_state:
            await channel.send("🧠 Dữ liệu học tập hôm nay vẫn được giữ đến hết ngày.")

    @evening_summary.before_loop
    async def before_evening_summary():
        """Wait until bot is ready before starting evening summary loop."""
        await bot.wait_until_ready()

    def _clip_lines(lines, max_chars=1024):
        text = "\n".join(
            [str(x).strip() for x in (lines or []) if str(x).strip()]
        ).strip()
        if not text:
            return "Không có"
        return text[: max_chars - 1].rstrip() + "…" if len(text) > max_chars else text

    def _build_nightly_gmail_embed(result):
        date_label = datetime.now(VIETNAM_TZ).strftime("%d/%m/%Y")
        important = result.get("important_mails", [])
        unread_items = result.get("unread_items", [])
        sent_items = result.get("sent_items", [])

        embed = discord.Embed(
            title=f"📮 Gmail nightly digest - {date_label} (23:30)",
            description="Tổng hợp nhanh mail quan trọng, TODO ưu tiên và insight từ sent mail.",
            color=discord.Color.blue(),
            timestamp=datetime.now(VIETNAM_TZ),
        )
        embed.add_field(name="🔥 Important", value=str(len(important)), inline=True)
        embed.add_field(name="📥 Unread", value=str(len(unread_items)), inline=True)
        embed.add_field(name="📤 Sent hôm nay", value=str(len(sent_items)), inline=True)

        todo_list = result.get("todo_list", [])
        todo_lines = []
        for idx, item in enumerate(todo_list[:8], start=1):
            todo_lines.append(f"{idx}. {str(item)}")
        if not todo_lines:
            todo_lines = ["Không có TODO ưu tiên mới."]
        embed.add_field(
            name="✅ Top TODO ưu tiên",
            value=_clip_lines(todo_lines),
            inline=False,
        )

        key_info = result.get("key_info", [])
        key_info_lines = [f"- {str(x)}" for x in key_info[:8]]
        if not key_info_lines:
            key_info_lines = ["Không có thông tin quan trọng nổi bật."]
        embed.add_field(
            name="📌 Thông tin quan trọng",
            value=_clip_lines(key_info_lines),
            inline=False,
        )

        sent_actions = result.get("sent_actions_done", [])
        sent_lines = []
        sent_summary = str(result.get("sent_summary") or "").strip()
        if sent_summary:
            sent_lines.append(f"🧾 {sent_summary}")
        if sent_actions:
            sent_lines.extend([f"- {str(x)}" for x in sent_actions[:6]])
        if not sent_lines:
            sent_lines = ["Chưa có hành động sent mail đáng chú ý."]
        embed.add_field(
            name="🧠 Sent Mail Insight",
            value=_clip_lines(sent_lines),
            inline=False,
        )
        return embed

    @tasks.loop(time=time(hour=23, minute=30, tzinfo=VIETNAM_TZ))
    async def nightly_gmail_digest():
        """Generate and send Gmail digest once per day at 23:30 local time."""
        if MAIN_CHANNEL_ID == 0:
            return

        channel = bot.get_channel(MAIN_CHANNEL_ID)
        if not channel:
            return

        result = await knowledge_bot.build_advanced_gmail_report(period="nightly")
        if not result.get("ok"):
            await channel.send(result.get("error", "⚠️ Không thể tạo Gmail report."))
            return

        await channel.send(embed=_build_nightly_gmail_embed(result))

    @nightly_gmail_digest.before_loop
    async def before_nightly_gmail_digest():
        """Wait until bot is ready before starting nightly Gmail digest loop."""
        await bot.wait_until_ready()

    @tasks.loop(hours=max(1, WEATHER_ALERT_INTERVAL_HOURS))
    async def weather_risk_alert():
        """Check weather risk window and warn when bad weather may affect upcoming plans."""
        if MAIN_CHANNEL_ID == 0:
            return

        channel = bot.get_channel(MAIN_CHANNEL_ID)
        if not channel:
            return

        lookahead_hours = max(1, int(WEATHER_ALERT_LOOKAHEAD_HOURS or 3))
        risk = await knowledge_bot.get_weather_risk_window(hours_ahead=lookahead_hours)
        if not risk.get("ok"):
            return
        if not risk.get("is_risky"):
            return

        now = datetime.now(knowledge_bot.timezone)
        end_time = now + timedelta(hours=lookahead_hours)

        upcoming = []
        date_candidates = sorted({now.date(), end_time.date()})
        for date_item in date_candidates:
            events = await knowledge_bot.get_events(date_item)
            if isinstance(events, list):
                for event in events:
                    event_dt = event.get("datetime")
                    if event_dt and now <= event_dt <= end_time:
                        upcoming.append(
                            f"📅 {event_dt.strftime('%H:%M')} - {event.get('summary', '(không tiêu đề)')}"
                        )

            tasks_list = await knowledge_bot.get_tasks(
                date=date_item, show_completed=False
            )
            if isinstance(tasks_list, list):
                for task in tasks_list:
                    if task.get("overdue"):
                        continue
                    if not task.get("due") or not task.get("due_time"):
                        continue
                    task_dt = datetime.combine(task["due"], task["due_time"])
                    if task_dt.tzinfo is None:
                        task_dt = knowledge_bot.timezone.localize(task_dt)
                    else:
                        task_dt = task_dt.astimezone(knowledge_bot.timezone)
                    if now <= task_dt <= end_time:
                        upcoming.append(
                            f"📝 {task_dt.strftime('%H:%M')} - {task.get('title', '(không tiêu đề)')}"
                        )

        risky_slots = risk.get("risky_slots", [])
        weather_lines = []
        for slot in risky_slots[:4]:
            reason = ", ".join(slot.get("reasons") or [])
            weather_lines.append(
                f"• {slot['time'].strftime('%H:%M')} | {slot.get('condition')} | {reason}"
            )

        message = [
            f"⚠️ **Cảnh báo thời tiết xấu trong {lookahead_hours} giờ tới**",
            "Khu vực: **Da Nang**",
            "",
            "**Khung giờ rủi ro:**",
            *weather_lines,
        ]

        if upcoming:
            message.append("")
            message.append("**Bạn sắp có việc trong khung giờ thời tiết xấu:**")
            message.extend([f"- {item}" for item in upcoming[:8]])

        await channel.send("\n".join(message)[:3900])

    @weather_risk_alert.before_loop
    async def before_weather_risk_alert():
        """Wait until bot is ready before starting weather risk alert loop."""
        await bot.wait_until_ready()

    @tasks.loop(seconds=1)
    async def countdown_checker():
        """Emit countdown notifications when a milestone threshold is crossed."""
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

            if remaining < -5:
                del _active_countdowns[name]
                continue

            milestones = data["milestones"]
            notified = data["notified"]

            # Kiểm tra theo kiểu "crossing threshold": chỉ bắn khi vừa đi qua mốc.

            for milestone in milestones:
                if milestone not in notified:
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
        """Wait until bot is ready before starting countdown checker loop."""
        await bot.wait_until_ready()
