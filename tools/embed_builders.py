from datetime import datetime
import discord

from .constants import (
    EMBED_COLOR_PRIMARY,
    EMBED_COLOR_SUCCESS,
    EMBED_COLOR_WARNING,
    EMBED_COLOR_DANGER,
)


def build_calendar_embed(
    date_display: str, events: list, tasks: list, timestamp: datetime
):
    embed = discord.Embed(
        title=f"📅 Lịch {date_display}",
        color=EMBED_COLOR_PRIMARY,
        timestamp=timestamp,
    )

    if events:
        lines = []
        for event in events[:12]:
            icon = "🔴" if event.get("is_important") else "•"
            time_str = (
                f"{event.get('time')}-{event.get('end_time')}"
                if event.get("end_time")
                else event.get("time", "")
            )
            lines.append(f"{icon} {time_str} {event.get('summary', '')}")
        embed.add_field(name="📌 Events", value="\n".join(lines)[:1024], inline=False)

    if tasks:
        lines = []
        for task in tasks[:12]:
            icon = "🔴" if task.get("overdue") else "•"
            due_time = task.get("due_time")
            time_str = due_time.strftime("%H:%M") if due_time else ""
            lines.append(f"{icon} {time_str} {task.get('title', '')}")
        embed.add_field(name="📋 Tasks", value="\n".join(lines)[:1024], inline=False)

    if not events and not tasks:
        embed.description = "Không có dữ liệu cho ngày này."

    return embed


def build_events_embed(date_display: str, events: list, timestamp: datetime):
    embed = discord.Embed(
        title=f"📌 Events {date_display}",
        color=EMBED_COLOR_SUCCESS,
        timestamp=timestamp,
    )

    if not events:
        embed.description = "Không có event."
        return embed

    lines = []
    for index, event in enumerate(events[:20], start=1):
        icon = "🔴" if event.get("is_important") else ""
        time_str = (
            f"{event.get('time')}-{event.get('end_time')}"
            if event.get("end_time")
            else event.get("time", "")
        )
        line = f"{index}. {icon} {time_str} **{event.get('summary', '')}**"
        if event.get("description"):
            line += f"\n↳ {str(event.get('description'))[:120]}"
        lines.append(line)

    embed.description = "\n".join(lines)[:3900]
    return embed


def build_tasks_embed(
    date_display: str, tasks: list, timestamp: datetime, overdue_only: bool = False
):
    title = (
        f"🔴 Tasks quá hạn ({len(tasks)})"
        if overdue_only
        else f"📋 Tasks {date_display}"
    )
    color = EMBED_COLOR_DANGER if overdue_only else EMBED_COLOR_WARNING
    embed = discord.Embed(title=title, color=color, timestamp=timestamp)

    if not tasks:
        embed.description = "Không có task phù hợp."
        return embed

    lines = []
    for index, task in enumerate(tasks[:20], start=1):
        icon = "🔴" if task.get("overdue") else "•"
        due_date = task.get("due")
        due_time = task.get("due_time")
        due_date_str = due_date.strftime("%d/%m") if due_date else "Không hạn"
        due_time_str = due_time.strftime("%H:%M") if due_time else ""
        lines.append(
            f"{index}. {icon} **{task.get('title', '')}** ({due_date_str} {due_time_str})"
        )
        if task.get("notes"):
            lines.append(f"↳ {str(task.get('notes'))[:100]}")

    embed.description = "\n".join(lines)[:3900]
    return embed
