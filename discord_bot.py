import discord
from discord.ext import commands, tasks
import os
from datetime import datetime, time, timedelta, timezone
import pytz
import aiohttp
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# ==============================
# CONFIG
# ==============================
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_MODEL = os.getenv("GITHUB_MODEL", "gpt-4o-mini")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
WEATHER_PROVIDER = os.getenv("WEATHER_PROVIDER", "weatherapi")
YOUR_USER_ID = int(os.getenv("YOUR_USER_ID", "0"))
MAIN_CHANNEL_ID = int(os.getenv("CHANNEL_MAIN", "0"))

CHANNELS_TO_MONITOR_STR = os.getenv("CHANNELS_TO_MONITOR", "")
CHANNELS_TO_MONITOR = [
    int(ch.strip()) for ch in CHANNELS_TO_MONITOR_STR.split(",") if ch.strip()
]

print(f"📺 Theo dõi {len(CHANNELS_TO_MONITOR)} channel(s)")
print(f"📢 Main channel: {MAIN_CHANNEL_ID}")
print(f"🌤️ Weather provider: {WEATHER_PROVIDER}")

# Vietnam timezone constant - USE THIS instead of pytz for scheduled tasks
VIETNAM_TZ = timezone(timedelta(hours=7))

# ==============================
# DISCORD SETUP
# ==============================
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
bot = commands.Bot(command_prefix="!", intents=intents)
bot.remove_command("help")

# State
daily_messages = {}
summary_state = {}
_last_tasks = {}
_last_events = {}
_active_countdowns = {}  # {name: datetime}
_sent_upcoming_reminders = set()

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
    async def get_weather(self):
        """Thời tiết hiện tại"""
        try:
            url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q=Da Nang&days=1&lang=vi&aqi=no"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        return "⚠️ Không thể lấy thời tiết"
                    data = await response.json()

            current = data["current"]
            forecast_day = data["forecast"]["forecastday"][0]

            result = f"🌤️ **Thời tiết Đà Nẵng**\n\n"
            result += f"🌡️ {current['temp_c']}°C (cảm giác {current['feelslike_c']}°C)\n"
            result += f"☁️ {current['condition']['text']}\n"
            result += f"💧 Độ ẩm: {current['humidity']}%\n"
            result += f"💨 Gió: {current['wind_kph']} km/h\n"
            result += (
                f"🌧️ Khả năng mưa: {forecast_day['day']['daily_chance_of_rain']}%\n"
            )
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

        url = "https://models.github.ai/inference/chat/completions"
        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Content-Type": "application/json",
        }

        progress_info = f"Tổng hợp {start_idx + 1}-{end_idx}/{total} tin nhắn"
        if channel_name:
            progress_info += f" từ #{channel_name}"

        payload = {
            "model": GITHUB_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Tóm tắt kiến thức và tạo câu hỏi ôn tập.",
                },
                {
                    "role": "user",
                    "content": f"{progress_info}\n\n{message_text}\n\n1. Tóm tắt 3-5 ý chính\n2. Tạo 3-5 câu hỏi ôn tập",
                },
            ],
            "temperature": 0.7,
            "max_tokens": 1500,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    data = await response.json()
                    if response.status == 200:
                        return data["choices"][0]["message"]["content"], has_more
                    else:
                        return f"⚠️ Lỗi API: {data}", False
        except Exception as e:
            return f"⚠️ Lỗi: {str(e)}", False


knowledge_bot = KnowledgeBot()


# ==============================
# EVENTS
# ==============================
@bot.event
async def on_ready():
    print(f"✅ Bot: {bot.user}")
    morning_greeting.start()
    calendar_reminder.start()
    evening_summary.start()
    end_of_day_review.start()
    countdown_checker.start()

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

    if message.channel.id in CHANNELS_TO_MONITOR:
        channel_id = message.channel.id
        if channel_id not in daily_messages:
            daily_messages[channel_id] = []
        timestamp = datetime.now(knowledge_bot.timezone).strftime("%H:%M")
        daily_messages[channel_id].append(
            f"[{timestamp}] {message.author.name}: {message.content}"
        )

    await bot.process_commands(message)


# ==============================
# TASKS
# ==============================
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

    await channel.send(message)


@end_of_day_review.before_loop
async def before_end_of_day_review():
    await bot.wait_until_ready()


@tasks.loop(time=time(hour=21, minute=0, tzinfo=VIETNAM_TZ))
async def evening_summary():
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

        summary, has_more = await knowledge_bot.summarize_daily_knowledge(
            messages, channel_name, 0, 50
        )

        if summary:
            header = f"📚 **Tổng hợp #{channel_name}** ({len(messages)} tin nhắn)\n\n"
            await channel.send(header + summary)

            if has_more:
                summary_state[channel_id] = {
                    "messages": messages,
                    "channel_name": channel_name,
                    "offset": 50,
                }
                await channel.send(f"💡 Còn {len(messages) - 50} tin nhắn. `!continue`")

    if not summary_state:
        daily_messages.clear()


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
            name="🤖 Automation", value="`!help automation` - Tự động hóa", inline=True
        )

        embed.add_field(
            name="🎯 Quick Start",
            value=(
                "`!calendar` - Xem lịch hôm nay\n"
                "`!tasks` - Xem tasks\n"
                "`!countdown` - Xem countdowns\n"
                "`!weather` - Thời tiết\n"
                "`!summary` - Tổng hợp học tập"
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
                "`!continue` - Tiếp tục phần còn lại\n"
                "`!stats` - Thống kê theo channel"
            ),
            inline=False,
        )
        embed.add_field(
            name="ℹ️ Lưu Ý",
            value=(
                "• Bot theo dõi tin nhắn trong CHANNELS_TO_MONITOR\n"
                "• Tự động tổng hợp lúc 21:00 hàng ngày\n"
                "• Mỗi lần xử lý 50 tin nhắn"
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
            "⚠️ Category: `calendar`, `tasks`, `countdown`, `weather`, `study`, `automation`"
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


# ==============================
# COMMANDS - STUDY
# ==============================
@bot.command()
async def summary(ctx):
    """Tổng hợp"""
    if ctx.author.id != YOUR_USER_ID:
        return

    if not daily_messages:
        await ctx.send("📚 Không có tin nhắn")
        return

    for channel_id, messages in daily_messages.items():
        discord_channel = bot.get_channel(channel_id)
        channel_name = discord_channel.name if discord_channel else str(channel_id)

        summary_text, has_more = await knowledge_bot.summarize_daily_knowledge(
            messages, channel_name, 0, 50
        )

        if summary_text:
            await ctx.send(
                f"📚 **#{channel_name}** ({len(messages)} tin nhắn)\n\n{summary_text}"
            )

            if has_more:
                summary_state[channel_id] = {
                    "messages": messages,
                    "channel_name": channel_name,
                    "offset": 50,
                }
                await ctx.send(f"💡 Còn {len(messages) - 50} tin nhắn. `!continue`")


@bot.command(name="continue")
async def continue_summary(ctx):
    if ctx.author.id != YOUR_USER_ID:
        return

    if not summary_state:
        await ctx.send("📚 Không có phần dở")
        return

    channel_id = list(summary_state.keys())[0]
    state = summary_state[channel_id]

    summary_text, has_more = await knowledge_bot.summarize_daily_knowledge(
        state["messages"], state["channel_name"], state["offset"], 50
    )

    if summary_text:
        await ctx.send(summary_text)

        if has_more:
            summary_state[channel_id]["offset"] += 50
            remaining = len(state["messages"]) - summary_state[channel_id]["offset"]
            await ctx.send(f"💡 Còn {remaining} tin nhắn. `!continue`")
        else:
            del summary_state[channel_id]
            await ctx.send("✅ Xong!")
            if channel_id in daily_messages:
                del daily_messages[channel_id]


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


@bot.command()
async def ping(ctx):
    await ctx.send(f"🏓 Pong! {round(bot.latency * 1000)}ms")


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
