import re
import json
from datetime import datetime, time, timedelta

import discord
from discord import app_commands

from tools import study_memory
from tools.embed_builders import (
    build_calendar_embed,
    build_events_embed,
    build_tasks_embed,
)


def register_slash_commands(bot, deps):
    """Register all slash commands and autocomplete handlers."""
    knowledge_bot = deps["knowledge_bot"]

    YOUR_USER_ID = deps["YOUR_USER_ID"]
    CHANNELS_TO_MONITOR = deps["CHANNELS_TO_MONITOR"]
    VIETNAM_TZ = deps["VIETNAM_TZ"]
    STUDY_PASS_THRESHOLD = deps["STUDY_PASS_THRESHOLD"]
    STUDY_POINTS_PASS = deps["STUDY_POINTS_PASS"]
    SUMMARY_BATCH_SIZE = deps["SUMMARY_BATCH_SIZE"]
    SUMMARY_FETCH_MAX_MESSAGES = deps["SUMMARY_FETCH_MAX_MESSAGES"]

    daily_messages = deps["daily_messages"]
    summary_state = deps["summary_state"]
    _last_events = deps["_last_events"]
    _last_tasks = deps["_last_tasks"]
    _pending_chat_context = deps["_pending_chat_context"]
    _study_questions = deps["_study_questions"]
    _last_summary_fetch_message_ids = deps["_last_summary_fetch_message_ids"]

    _create_chat_session = deps["_create_chat_session"]
    _mark_user_interaction = deps["_mark_user_interaction"]
    _fetch_motivational_slogan = deps["_fetch_motivational_slogan"]
    _ensure_daily_window_rollover = deps["_ensure_daily_window_rollover"]
    _collect_new_messages_since = deps["_collect_new_messages_since"]
    _continue_summary_for_user = deps["_continue_summary_for_user"]
    _safe_followup_send = deps["_safe_followup_send"]
    summary_channel_autocomplete = deps["summary_channel_autocomplete"]
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
    _mark_question_answered = deps["_mark_question_answered"]
    _ensure_study_memory_tables = deps["_ensure_study_memory_tables"]
    _split_text_chunks = deps["_split_text_chunks"]
    _format_rich_text_for_discord = deps["_format_rich_text_for_discord"]
    _extract_image_urls_from_attachments = deps["_extract_image_urls_from_attachments"]
    _build_reason_single_message = deps["_build_reason_single_message"]

    ChatSessionView = deps["ChatSessionView"]
    SummaryInteractiveView = deps["SummaryInteractiveView"]

    @bot.tree.command(name="help", description="Xem lệnh nhanh của bot")
    @app_commands.describe(category="Nhóm lệnh muốn xem")
    async def slash_help(
        interaction: discord.Interaction,
        category: str = "overview",
    ):
        """Display slash command guide grouped by category."""
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
            embed.add_field(name="🐱 Fun", value="`/animal`", inline=False)
        elif category == "calendar":
            embed.description = "Lệnh calendar"
            embed.add_field(name="`/calendar`", value="Xem lịch tổng", inline=False)
            embed.add_field(name="`/events`", value="Xem events", inline=False)
            embed.add_field(name="`/add_event`", value="Thêm event", inline=False)
            embed.add_field(
                name="`/del_event`", value="Xóa event theo số", inline=False
            )
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
            embed.add_field(
                name="`/add_countdown`", value="Thêm countdown", inline=False
            )
            embed.add_field(
                name="`/del_countdown`", value="Xóa countdown", inline=False
            )
            embed.add_field(
                name="`/newyear`", value="Bật countdown năm mới", inline=False
            )
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
                name="`/study_stats`",
                value="Xem streak/điểm học tập tháng",
                inline=False,
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
            embed.add_field(
                name="`/animal`",
                value="Lấy ảnh động vật (hiện hỗ trợ cat)",
                inline=False,
            )
        else:
            embed.description = (
                "Category hợp lệ: `overview`, `calendar`, `tasks`, `countdown`, "
                "`study`, `chat`, `utility`"
            )

        await interaction.response.send_message(embed=embed, ephemeral=True)

    @bot.tree.command(name="ping", description="Kiểm tra độ trễ của bot")
    async def slash_ping(interaction: discord.Interaction):
        """Return bot heartbeat latency for quick health checks."""
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
        """Show current weather or forecast by optional date/hour."""
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
        """Send one motivational slogan for authorized owner user."""
        if interaction.user.id != YOUR_USER_ID:
            await interaction.response.send_message(
                "⛔ Bạn không có quyền dùng lệnh này.", ephemeral=True
            )
            return
        _mark_user_interaction(interaction.user.id)
        text = await _fetch_motivational_slogan()
        await interaction.response.send_message(f"💪 **Slogan học tập:**\n*{text}*")

    @bot.tree.command(
        name="animal",
        description="Lấy ảnh động vật ngẫu nhiên (hiện hỗ trợ cat)",
    )
    @app_commands.describe(kind="Loài động vật")
    @app_commands.choices(
        kind=[
            app_commands.Choice(name="cat", value="cat"),
        ]
    )
    async def slash_animal(
        interaction: discord.Interaction,
        kind: app_commands.Choice[str],
    ):
        """Fetch animal image (currently only cat) from TheCatAPI."""
        if (kind.value or "").lower() != "cat":
            await interaction.response.send_message(
                "⚠️ Hiện tại bot chỉ hỗ trợ `cat`.", ephemeral=True
            )
            return

        await interaction.response.defer(thinking=True)
        result = await knowledge_bot.get_random_cat_image()
        if not result.get("ok"):
            await interaction.followup.send(
                result.get("error", "⚠️ Không lấy được ảnh.")
            )
            return

        embed = discord.Embed(
            title="🐱 Random Cat",
            color=discord.Color.blurple(),
            timestamp=datetime.now(VIETNAM_TZ),
        )
        embed.set_image(url=result.get("url"))
        breed = str(result.get("breed") or "").strip()
        if breed:
            embed.add_field(name="Breed", value=breed, inline=True)
        embed.set_footer(text="Source: TheCatAPI")
        await interaction.followup.send(embed=embed)

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
        """Handle chat request with optional images and session continuation metadata."""
        provided_images = [image_1, image_2, image_3, image_4]
        image_urls = _extract_image_urls_from_attachments(
            [x for x in provided_images if x]
        )

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
            lines = [
                f"Ảnh {x['index']}: {str(x['text'])[:220]}" for x in extracted_ok[:2]
            ]
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

    @bot.tree.command(
        name="reason", description="Reasoning mode trả lời rõ ràng, dễ đọc"
    )
    @app_commands.describe(prompt="Nội dung cần reasoning")
    async def slash_reason(
        interaction: discord.Interaction,
        prompt: str,
    ):
        """Run reasoning-mode answer flow with robust multi-message fallback."""
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
        combined_message = _build_reason_single_message(
            prompt, display_answer, model_used
        )

        if len(combined_message) <= 2000:
            await _safe_followup_send(interaction, combined_message)
        else:
            chunks = _split_text_chunks(combined_message, 1900)
            await _safe_followup_send(interaction, chunks[0])
            for chunk in chunks[1:]:
                await _safe_followup_send(interaction, chunk)

    @bot.tree.command(
        name="summary", description="Tổng hợp học tập và tạo câu hỏi ôn tập"
    )
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
        """Generate study summaries in cache/channel/all modes and persist question state."""
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
            """Resolve channel option text/value from autocomplete into a channel object."""
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
            # mode=cache: dùng dữ liệu message đang được giữ trong RAM theo ngày.
            if not daily_messages:
                await interaction.response.send_message(
                    "📚 Không có tin nhắn", ephemeral=True
                )
                return
            for channel_id, messages in daily_messages.items():
                discord_channel = bot.get_channel(channel_id)
                channel_name = (
                    discord_channel.name if discord_channel else str(channel_id)
                )
                source_batches.append((channel_id, channel_name, messages))
        elif selected_mode == "channel":
            # mode=channel: fetch trực tiếp 1 channel cụ thể theo lựa chọn người dùng.
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
            # mode=all: quét tất cả channel monitor; có checkpoint để chỉ lấy tin mới.
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

                source_batches.append(
                    (channel_id, discord_channel.name, fetched_messages)
                )
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
                processed_count = max(
                    1,
                    int(summary_data.get("processed_count") or SUMMARY_BATCH_SIZE),
                )
                summary_state[channel_id] = {
                    "messages": messages,
                    "channel_name": channel_name,
                    "offset": processed_count,
                }
                await interaction.followup.send(
                    f"💡 Còn {max(0, len(messages) - processed_count)} tin nhắn chưa summary trong #{channel_name}. Bấm `Continue Summary` ngay dưới embed vừa gửi hoặc dùng `/continue_summary`.",
                )

        for channel_id, newest_id in fetch_checkpoints.items():
            _last_summary_fetch_message_ids[channel_id] = int(newest_id)

    @slash_summary.autocomplete("channel_option")
    async def slash_summary_channel_option_autocomplete(
        interaction: discord.Interaction, current: str
    ):
        """Return dynamic autocomplete choices for summary channel selection."""
        return await summary_channel_autocomplete(interaction, current)

    @bot.tree.command(
        name="continue_summary", description="Tiếp tục summary phần còn lại"
    )
    async def slash_continue_summary(interaction: discord.Interaction):
        """Continue pending paginated summary for the current owner."""
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
        """Grade one review question answer and update spaced-repetition metrics."""
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
        """Show monthly study KPI embed for authorized owner user."""
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
        """Show recent knowledge cards reviewed within a time window."""
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
        """Build adaptive learning plan from recent weak topics and cards."""
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

    @bot.tree.command(name="calendar", description="Xem lịch (events + tasks)")
    @app_commands.describe(date="Ngày cần xem: today, tomorrow, 18/2...")
    async def slash_calendar(interaction: discord.Interaction, date: str = ""):
        """Show merged calendar/tasks embed for a target date."""
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
        """List events for target date and cache the result for index-based commands."""
        target_date = knowledge_bot.parse_date(date) if date else None
        events = await knowledge_bot.get_events(target_date)

        if isinstance(events, str):
            await interaction.response.send_message(events)
            return

        if not events:
            date_display = target_date.strftime("%d/%m") if target_date else "hôm nay"
            await interaction.response.send_message(
                f"📅 Không có events {date_display}"
            )
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
        """Create calendar event from slash inputs with parsed date/time."""
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
            await interaction.response.send_message(
                "⚠️ Ngày không hợp lệ", ephemeral=True
            )
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
            single_time_match = re.search(
                r"(\d{1,2}[h:]\d{2}|\d{1,2}h?)", datetime_input
            )
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

    @bot.tree.command(
        name="del_event", description="Xoá event theo số thứ tự từ /events"
    )
    @app_commands.describe(index="Số thứ tự event")
    async def slash_del_event(interaction: discord.Interaction, index: int):
        """Delete event by 1-based index from last cached `/events` call."""
        if interaction.user.id not in _last_events:
            await interaction.response.send_message(
                "⚠️ Gọi `/events` trước", ephemeral=True
            )
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
        """Move selected event to a new date/time while preserving duration when possible."""
        if interaction.user.id not in _last_events:
            await interaction.response.send_message(
                "⚠️ Gọi `/events` trước", ephemeral=True
            )
            return

        events = _last_events[interaction.user.id]
        if index < 1 or index > len(events):
            await interaction.response.send_message(
                f"⚠️ Chọn từ 1-{len(events)}", ephemeral=True
            )
            return

        date_match = re.search(
            r"(\d{1,2}[/-]\d{1,2}|today|tomorrow)", datetime_input, re.I
        )
        if not date_match:
            await interaction.response.send_message(
                "⚠️ Không tìm thấy ngày", ephemeral=True
            )
            return

        target_date = knowledge_bot.parse_date(date_match.group(1))
        time_match = re.search(r"(\d{1,2}[h:]\d{2}|\d{1,2}h?)", datetime_input)
        if not time_match:
            await interaction.response.send_message(
                "⚠️ Không tìm thấy giờ", ephemeral=True
            )
            return

        new_time = knowledge_bot.parse_time(time_match.group(1))
        new_start = knowledge_bot.timezone.localize(
            datetime.combine(target_date, new_time)
        )

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

    @bot.tree.command(name="tasks", description="Xem danh sách tasks")
    @app_commands.describe(date="Ngày cần xem: today, tomorrow, 18/2...")
    async def slash_tasks(interaction: discord.Interaction, date: str = ""):
        """List tasks by date and cache list for done/delete operations."""
        target_date = knowledge_bot.parse_date(date) if date else None
        tasks_list = await knowledge_bot.get_tasks(
            date=target_date, show_completed=False
        )

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
        """Show overdue tasks only and cache the set for follow-up actions."""
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
        """Create a task with optional due expression and free-form notes."""
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
        """Mark cached task as completed via 1-based index."""
        if interaction.user.id not in _last_tasks:
            await interaction.response.send_message(
                "⚠️ Gọi `/tasks` trước", ephemeral=True
            )
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
        """Delete cached task via 1-based index."""
        if interaction.user.id not in _last_tasks:
            await interaction.response.send_message(
                "⚠️ Gọi `/tasks` trước", ephemeral=True
            )
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
        """Show in-memory monitored message statistics grouped by channel."""
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

    @bot.tree.command(
        name="gmail_digest",
        description="Xem Gmail digest theo ngày hoặc tạo digest ngay",
    )
    @app_commands.describe(
        date="Ngày cần xem: today, 18/2... (để trống = tạo digest mới)"
    )
    async def slash_gmail_digest(interaction: discord.Interaction, date: str = ""):
        """Build/show advanced Gmail report as two styled messages with separate checkpoints."""

        def _clip_lines(lines, max_chars=1024):
            text = "\n".join(lines).strip()
            return text[:max_chars] if text else "Không có"

        def _mail_llm_label(report):
            llm_models = report.get("llm_models") or {}
            if not isinstance(llm_models, dict):
                return "N/A"
            classify = str(llm_models.get("classify") or "-")
            important = str(llm_models.get("important") or "-")
            sent = str(llm_models.get("sent") or "-")
            return f"C:{classify} | I:{important} | S:{sent}"

        def _build_inbox_embed(report):
            report_date = report.get("report_date")
            date_label = (
                report_date
                if isinstance(report_date, str)
                else datetime.now(VIETNAM_TZ).strftime("%Y-%m-%d")
            )

            important = report.get("important_mails", [])
            todo_list = report.get("todo_list", [])
            unread = report.get("unread_items", [])

            embed = discord.Embed(
                title=f"📬 Gmail Inbox Focus • {date_label}",
                description=(
                    "Ưu tiên đọc mail quan trọng + xử lý TODO trước, sau đó dọn unread."
                ),
                color=discord.Color.blue(),
                timestamp=datetime.now(VIETNAM_TZ),
            )

            important_lines = []
            for item in important[:8]:
                sender = str(item.get("from") or "").strip()
                sender_suffix = f" • {sender[:36]}" if sender else ""
                important_lines.append(
                    f"🔸 {str(item.get('subject') or '(không tiêu đề)')[:85]}{sender_suffix}"
                )
            embed.add_field(
                name="🔥 Important Mail",
                value=_clip_lines(important_lines),
                inline=False,
            )

            todo_lines = [f"{idx+1}. {str(t)}" for idx, t in enumerate(todo_list[:10])]
            embed.add_field(
                name="✅ TODO Ưu Tiên",
                value=_clip_lines(todo_lines),
                inline=False,
            )

            unread_lines = [
                f"📩 {str(item.get('subject') or '(không tiêu đề)')[:110]}"
                for item in unread[:20]
            ]
            embed.add_field(
                name="📥 Mail Unread",
                value=_clip_lines(unread_lines),
                inline=False,
            )

            embed.set_footer(
                text=(
                    f"report_id={report.get('report_id')} | "
                    f"important={len(important)} | unread={len(unread)} | "
                    f"LLM={_mail_llm_label(report)}"
                )
            )
            return embed

        def _build_insight_embed(report):
            report_date = report.get("report_date")
            date_label = (
                report_date
                if isinstance(report_date, str)
                else datetime.now(VIETNAM_TZ).strftime("%Y-%m-%d")
            )

            key_info = report.get("key_info", [])
            sent_summary = report.get("sent_summary", "")
            sent_actions_done = report.get("sent_actions_done", [])
            sent_highlights = report.get("sent_highlights", [])
            sent_items = report.get("sent_items", [])

            embed = discord.Embed(
                title=f"📤 Gmail Insight • {date_label}",
                description="Thông tin quan trọng + tổng hợp sent mail trong ngày.",
                color=discord.Color.gold(),
                timestamp=datetime.now(VIETNAM_TZ),
            )

            key_info_lines = [f"⚠️ {str(x)}" for x in key_info[:10]]
            if not key_info_lines:
                key_info_lines = [
                    "Không có thông tin quan trọng nổi bật trong report này."
                ]
            embed.add_field(
                name="📌 Thông Tin Quan Trọng",
                value=_clip_lines(key_info_lines),
                inline=False,
            )

            sent_lines = []
            if sent_summary:
                sent_lines.append(f"🧾 {str(sent_summary)}")
            sent_lines.append(f"📨 Tổng sent mail trong ngày: {len(sent_items)}")
            if sent_actions_done:
                sent_lines.append("")
                sent_lines.append("Việc đã thực hiện qua sent mail:")
                sent_lines.extend([f"- {str(x)}" for x in sent_actions_done[:8]])
            elif sent_highlights:
                sent_lines.append("")
                sent_lines.append("Điểm nổi bật:")
                sent_lines.extend([f"- {str(x)}" for x in sent_highlights[:8]])

            embed.add_field(
                name="🧠 Sent Mail Insight",
                value=_clip_lines(sent_lines),
                inline=False,
            )
            embed.set_footer(
                text=(
                    f"report_id={report.get('report_id')} | sent={len(sent_items)} | "
                    f"LLM={_mail_llm_label(report)}"
                )
            )
            return embed

        class TaskSelectionModal(discord.ui.Modal, title="Chọn TODO để tạo task"):
            todo_indexes = discord.ui.TextInput(
                label="Nhập số thứ tự TODO",
                placeholder="Ví dụ: 1,3,5 (để trống = lấy 3 việc ưu tiên đầu)",
                required=False,
                max_length=120,
            )

            def __init__(self, owner_id, todo_items, view_ref):
                super().__init__(timeout=300)
                self.owner_id = owner_id
                self.todo_items = todo_items
                self.view_ref = view_ref

            async def on_submit(self, interaction: discord.Interaction):
                if interaction.user.id != self.owner_id:
                    await interaction.response.send_message(
                        "⛔ Bạn không có quyền thao tác báo cáo này.", ephemeral=True
                    )
                    return

                raw = str(self.todo_indexes.value or "").strip()
                if not raw:
                    self.view_ref.selected_indexes = []
                    await interaction.response.send_message(
                        "✅ Không nhập số, bot sẽ tạo theo top ưu tiên.",
                        ephemeral=True,
                    )
                    return

                picked = []
                max_idx = len(self.todo_items)
                for token in re.split(r"[,;\s]+", raw):
                    if not token.isdigit():
                        continue
                    idx = int(token)
                    if 1 <= idx <= max_idx:
                        picked.append(idx - 1)

                dedup = []
                seen = set()
                for idx in picked:
                    if idx in seen:
                        continue
                    seen.add(idx)
                    dedup.append(idx)

                self.view_ref.selected_indexes = dedup
                await interaction.response.send_message(
                    f"✅ Đã chọn {len(dedup)} task để tạo.", ephemeral=True
                )

        class GmailInboxActionView(discord.ui.View):
            def __init__(self, owner_id, report_id, todo_items):
                super().__init__(timeout=1800)
                self.owner_id = owner_id
                self.report_id = int(report_id)
                self.selected_indexes = []
                self.todo_items = list(todo_items or [])

            async def _ensure_owner(self, interaction: discord.Interaction):
                if interaction.user.id != self.owner_id:
                    await interaction.response.send_message(
                        "⛔ Bạn không có quyền thao tác báo cáo này.", ephemeral=True
                    )
                    return False
                return True

            @discord.ui.button(
                label="Checkpoint Inbox", style=discord.ButtonStyle.secondary, row=1
            )
            async def checkpoint_inbox(
                self, interaction: discord.Interaction, button: discord.ui.Button
            ):
                if not await self._ensure_owner(interaction):
                    return
                ok, msg = knowledge_bot.mark_report_inbox_checkpoint(self.report_id)
                await interaction.response.send_message(msg, ephemeral=True)

            @discord.ui.button(
                label="Mark All Read", style=discord.ButtonStyle.success, row=1
            )
            async def mark_read(
                self, interaction: discord.Interaction, button: discord.ui.Button
            ):
                if not await self._ensure_owner(interaction):
                    return
                ok, msg = knowledge_bot.mark_unread_as_read_for_report(self.report_id)
                await interaction.response.send_message(msg, ephemeral=True)

            @discord.ui.button(
                label="Select TODO", style=discord.ButtonStyle.secondary, row=2
            )
            async def select_todo(
                self, interaction: discord.Interaction, button: discord.ui.Button
            ):
                if not await self._ensure_owner(interaction):
                    return
                if not self.todo_items:
                    await interaction.response.send_message(
                        "ℹ️ Report không có TODO để chọn.", ephemeral=True
                    )
                    return
                modal = TaskSelectionModal(
                    owner_id=self.owner_id,
                    todo_items=self.todo_items,
                    view_ref=self,
                )
                await interaction.response.send_modal(modal)

            @discord.ui.button(
                label="Create Task", style=discord.ButtonStyle.primary, row=2
            )
            async def create_tasks(
                self, interaction: discord.Interaction, button: discord.ui.Button
            ):
                if not await self._ensure_owner(interaction):
                    return
                await interaction.response.defer(thinking=True, ephemeral=True)
                selected = list(self.selected_indexes)
                if not selected and self.todo_items:
                    selected = list(range(min(3, len(self.todo_items))))
                ok, msg = await knowledge_bot.create_tasks_from_report(
                    self.report_id, selected
                )
                await interaction.followup.send(msg, ephemeral=True)

        class GmailInsightActionView(discord.ui.View):
            def __init__(self, owner_id, report_id):
                super().__init__(timeout=1800)
                self.owner_id = owner_id
                self.report_id = int(report_id)

            async def _ensure_owner(self, interaction: discord.Interaction):
                if interaction.user.id != self.owner_id:
                    await interaction.response.send_message(
                        "⛔ Bạn không có quyền thao tác báo cáo này.", ephemeral=True
                    )
                    return False
                return True

            @discord.ui.button(
                label="Checkpoint Sent", style=discord.ButtonStyle.secondary, row=1
            )
            async def checkpoint_sent(
                self, interaction: discord.Interaction, button: discord.ui.Button
            ):
                if not await self._ensure_owner(interaction):
                    return
                ok, msg = knowledge_bot.mark_report_sent_checkpoint(self.report_id)
                await interaction.response.send_message(msg, ephemeral=True)

            @discord.ui.button(
                label="Mark Sent Done", style=discord.ButtonStyle.primary, row=1
            )
            async def mark_sent_done(
                self, interaction: discord.Interaction, button: discord.ui.Button
            ):
                if not await self._ensure_owner(interaction):
                    return
                await interaction.response.defer(thinking=True, ephemeral=True)
                ok, msg = await knowledge_bot.create_calendar_done_from_report(
                    self.report_id
                )
                await interaction.followup.send(msg, ephemeral=True)

        if interaction.user.id != YOUR_USER_ID:
            await interaction.response.send_message(
                "⛔ Bạn không có quyền dùng lệnh này.", ephemeral=True
            )
            return

        target_date = knowledge_bot.parse_history_date(date) if date else None
        if date and not target_date:
            await interaction.response.send_message(
                "⚠️ Ngày không hợp lệ. VD: `today`, `18/2`.", ephemeral=True
            )
            return

        if target_date:
            row = knowledge_bot.get_latest_gmail_report_by_date(target_date)
            if not row:
                history = knowledge_bot.get_gmail_digest_history(target_date)
                await interaction.response.send_message(history, ephemeral=True)
                return

            report = {
                "report_id": int(row.get("id") or 0),
                "report_date": row.get("report_date"),
                "important_mails": json.loads(row.get("important_json") or "[]"),
                "todo_list": json.loads(row.get("todo_json") or "[]"),
                "unread_items": json.loads(row.get("unread_json") or "[]"),
                "sent_items": json.loads(row.get("sent_json") or "[]"),
                "sent_highlights": [
                    f"{str(x.get('position') or '(chưa rõ vị trí)')} @ {str(x.get('company') or '(chưa rõ công ty)')}"
                    for x in json.loads(row.get("sent_apps_json") or "[]")[:10]
                ],
                "key_info": (
                    json.loads(row.get("key_info_json") or "[]")
                    if row.get("key_info_json")
                    else []
                ),
                "llm_models": (
                    json.loads(row.get("llm_models_json") or "{}")
                    if row.get("llm_models_json")
                    else {}
                ),
                "sent_summary": str(row.get("summary_text") or "").strip(),
            }
            report["sent_actions_done"] = [
                (f"{rec.get('action')}: {rec.get('position')} @ {rec.get('company')}")
                for rec in [
                    knowledge_bot._extract_sent_action_record(item)
                    for item in report.get("sent_items", [])[:12]
                ]
            ]
            inbox_embed = _build_inbox_embed(report)
            inbox_view = GmailInboxActionView(
                interaction.user.id,
                report["report_id"],
                report.get("todo_list", []),
            )
            await interaction.response.send_message(
                embed=inbox_embed, view=inbox_view, ephemeral=True
            )
            insight_embed = _build_insight_embed(report)
            insight_view = GmailInsightActionView(
                interaction.user.id,
                report["report_id"],
            )
            await interaction.followup.send(
                embed=insight_embed,
                view=insight_view,
                ephemeral=True,
            )
            return

        await interaction.response.defer(thinking=True, ephemeral=True)
        result = await knowledge_bot.build_advanced_gmail_report(period="manual")
        if not result.get("ok"):
            await interaction.followup.send(
                result.get("error", "⚠️ Không thể tạo Gmail digest."),
                ephemeral=True,
            )
            return
        inbox_embed = _build_inbox_embed(result)
        inbox_view = GmailInboxActionView(
            interaction.user.id,
            int(result.get("report_id") or 0),
            result.get("todo_list", []),
        )
        await interaction.followup.send(
            embed=inbox_embed, view=inbox_view, ephemeral=True
        )
        insight_embed = _build_insight_embed(result)
        insight_view = GmailInsightActionView(
            interaction.user.id,
            int(result.get("report_id") or 0),
        )
        await interaction.followup.send(
            embed=insight_embed, view=insight_view, ephemeral=True
        )

    @bot.tree.command(name="map", description="Tìm địa điểm bằng ngôn ngữ tự nhiên")
    @app_commands.describe(query="Ví dụ: quán cafe yên tĩnh gần Landmark 81")
    async def slash_map(interaction: discord.Interaction, query: str):
        """Search places using natural language with OpenStreetMap Nominatim."""
        await interaction.response.defer(thinking=True)
        result = await knowledge_bot.search_place_natural(query)
        await interaction.followup.send(result)

    @bot.tree.command(name="countdown", description="Xem tất cả countdown đang chạy")
    @app_commands.describe(name="Lọc theo tên (tuỳ chọn)")
    async def slash_countdown(interaction: discord.Interaction, name: str = ""):
        """List active countdowns and optionally filter by partial name."""
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
        """Create countdown using parsed date/time expression and optional emoji."""
        date_match = re.search(
            r"(\d{1,2}[/-]\d{1,2}|today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            datetime_input,
            re.I,
        )
        if not date_match:
            await interaction.response.send_message(
                "⚠️ Không tìm thấy ngày", ephemeral=True
            )
            return

        date_part = date_match.group(1)
        target_date = knowledge_bot.parse_date(date_part)
        if not target_date:
            await interaction.response.send_message(
                "⚠️ Ngày không hợp lệ", ephemeral=True
            )
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
        """Delete countdown by exact name."""
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
        """Create New Year countdown with special celebratory milestone style."""
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
        if knowledge_bot.add_countdown(
            countdown_name, ny_datetime, "🎆", label="newyear"
        ):
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
        """Create Tet countdown using pre-defined lunar date mapping."""
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
