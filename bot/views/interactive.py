from datetime import datetime

import discord

from bot.config.settings import STUDY_PASS_THRESHOLD, STUDY_POINTS_PASS, VIETNAM_TZ


_chat_sessions = None
_pending_chat_context = None
_knowledge_bot = None
_build_chat_context_text = None
_create_chat_session = None
_format_rich_text_for_discord = None
_split_text_chunks = None
_normalize_score_value = None
_append_study_event = None
_record_spaced_review = None
_mark_question_answered = None
_build_question_theory_text = None
_build_study_context_text = None
_continue_summary_for_user = None


def configure_views(
    *,
    chat_sessions,
    pending_chat_context,
    knowledge_bot,
    build_chat_context_text,
    create_chat_session,
    format_rich_text_for_discord,
    split_text_chunks,
    normalize_score_value,
    append_study_event,
    record_spaced_review,
    mark_question_answered,
    build_question_theory_text,
    build_study_context_text,
    continue_summary_for_user,
):
    """Inject runtime dependencies used by interactive Discord views/modals."""
    global _chat_sessions
    global _pending_chat_context
    global _knowledge_bot
    global _build_chat_context_text
    global _create_chat_session
    global _format_rich_text_for_discord
    global _split_text_chunks
    global _normalize_score_value
    global _append_study_event
    global _record_spaced_review
    global _mark_question_answered
    global _build_question_theory_text
    global _build_study_context_text
    global _continue_summary_for_user

    _chat_sessions = chat_sessions
    _pending_chat_context = pending_chat_context
    _knowledge_bot = knowledge_bot
    _build_chat_context_text = build_chat_context_text
    _create_chat_session = create_chat_session
    _format_rich_text_for_discord = format_rich_text_for_discord
    _split_text_chunks = split_text_chunks
    _normalize_score_value = normalize_score_value
    _append_study_event = append_study_event
    _record_spaced_review = record_spaced_review
    _mark_question_answered = mark_question_answered
    _build_question_theory_text = build_question_theory_text
    _build_study_context_text = build_study_context_text
    _continue_summary_for_user = continue_summary_for_user


class ChatSessionView(discord.ui.View):
    """View for chat session follow-up actions (reuse context / continue)."""

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
        ai_result = await _knowledge_bot.chat(
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
    """Modal allowing user to submit an answer for a summary question."""

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
        review = await _knowledge_bot.review_study_answer(
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
    """Button opening answer modal for one indexed review question."""

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
    """View wrapping summary actions: answer, deepen, reuse context, continue."""

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
        result = await _knowledge_bot.expand_summary_analysis(
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
