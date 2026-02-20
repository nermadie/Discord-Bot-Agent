# Discord Agent Bot

Bot Discord phục vụ học tập và quản lý công việc với trọng tâm slash commands.

## Tính năng chính

- Calendar + Events (Google Calendar)
- Tasks (Google Tasks)
- Study Summary (AI) + câu hỏi ôn tập + chấm điểm + streak
- Chat / Reasoning + xử lý ảnh
- Weather hiện tại + forecast theo ngày/giờ
- Gmail digest 23:30 hằng ngày (unread + sent mail trong ngày), lưu DB và tra cứu theo ngày
- OpenStreetMap natural search theo ngôn ngữ tự nhiên
- Slogan động lực tự động khi idle + lệnh thủ công

## Cấu trúc dự án (đã tách module)

- app.py: entrypoint chạy bot
- bot/core.py: orchestration + wiring dependencies (không chứa block command decorators)
- bot/config/settings.py: toàn bộ env/config runtime
- bot/state/runtime.py: state runtime trong phiên
- bot/registrars/events_tasks.py: đăng ký events + background tasks
- bot/registrars/prefix_commands.py: đăng ký prefix commands
- bot/registrars/slash_commands.py: đăng ký slash commands
- bot/services/knowledge_service.py: service AI + Calendar/Tasks + Countdown + Weather
- bot/services/study_service.py: service study metrics + spaced repetition persistence
- bot/views/interactive.py: Discord View/Modal classes cho chat + summary workflow
- bot/utils/formatting.py: format output Discord (table/chunk/rich text)
- bot/utils/attachments.py: parse ảnh/file attachments cho chat + summary
- tools/constants.py: hằng số dùng chung
- tools/embed_builders.py: UI embed cho calendar/events/tasks
- tools/study_memory.py: SM-2 + adaptive learning persistence
- setup_calendar.py: setup OAuth
- requirements.txt

## Cài đặt

1. Tạo môi trường ảo và cài package

   - Windows:
     - python -m venv .venv
     - .\.venv\Scripts\activate
     - pip install -r requirements.txt

1. Tạo file .env

```env
DISCORD_TOKEN=...
GITHUB_TOKEN=...
WEATHER_API_KEY=...
YOUR_USER_ID=...
CHANNEL_MAIN=...
APP_GUILD_ID=...

# Thứ tự trong danh sách này được dùng cho autocomplete /summary channel_option
# Đặt theo thứ tự mong muốn từ trên xuống, KHÔNG bao gồm CHANNEL_MAIN
CHANNELS_TO_MONITOR=111111111111111111,222222222222222222,333333333333333333

CHAT_MODEL_PRIMARY=openai/gpt-5
CHAT_MODEL_FALLBACKS=openai/gpt-5-mini,openai/gpt-5-nano,openai/gpt-4o

VISION_MODEL_PRIMARY=meta/Llama-4-Maverick-17B-128E-Instruct-FP8
VISION_MODEL_FALLBACKS=openai/gpt-4.1-nano,openai/gpt-4o-mini

SUMMARY_MODEL_PRIMARY=openai/gpt-5-chat
SUMMARY_MODEL_FALLBACKS=openai/gpt-5-mini,openai/gpt-5-nano,openai/gpt-4o
SUMMARY_MAX_OUTPUT_TOKENS=16000

ANSWER_MODEL_PRIMARY=openai/gpt-5-chat
ANSWER_MODEL_FALLBACKS=openai/gpt-5-mini,openai/gpt-5-nano,openai/gpt-4o

REASONING_MODEL_PRIMARY=deepseek/DeepSeek-R1-0528
REASONING_MODEL_FALLBACKS=microsoft/Phi-4-reasoning,microsoft/Phi-4-mini-reasoning

STUDY_POINTS_PASS=10
STUDY_POINTS_MISS=3
STUDY_PASS_THRESHOLD=7
STUDY_METRICS_DIR=study_metrics

SLOGAN_IDLE_MINUTES=180
SLOGAN_CHECK_INTERVAL_MINUTES=30
GMAIL_UNREAD_LIMIT=10
GMAIL_SENT_TODAY_LIMIT=200
```

1. Chạy bot

- python app.py

## Slash commands (chuẩn hoá)

- /help
- /calendar, /events, /add_event, /del_event, /move_event
- /tasks, /overdue, /add_task, /done, /del_task
- /summary, /continue_summary, /answer, /study_stats
- /chat, /reason
- /weather
- /gmail_digest
- /map
- /slogan
- /countdown, /add_countdown, /del_countdown, /newyear, /tet

## Summary modes

- /summary mode=cache
  - Dùng dữ liệu tin nhắn tạm trong ngày
- /summary mode=channel channel_option=<id-kênh-từ-autocomplete> latest_messages=<1..20>
  - Fetch trực tiếp N tin gần nhất của channel đó để summary
- /summary mode=all
  - Nếu bỏ latest_messages: chỉ lấy tin mới trong hôm nay
  - Nếu có latest_messages: có thể summary lịch sử gần nhất của mỗi channel monitor

## Weather

- /weather (hiện tại)
- /weather date:tomorrow
- /weather date:18/2 hour:14:00

## Gmail + Maps

- Gmail digest tự gửi vào channel chính 1 lần lúc 23:30
- `/gmail_digest` để tạo digest thủ công
- `/gmail_digest date:2026-02-20` hoặc `date:18/2` để xem dữ liệu đã lưu theo ngày
- `/map query:<mô tả địa điểm>` để tìm địa điểm bằng ngôn ngữ tự nhiên
- Prefix tương đương: `!gmail_digest [date]`, `!map <query>`

## Lưu ý quan trọng

- Dữ liệu summary trong ngày không bị xoá khi chạy /summary; chỉ reset sang ngày mới.
- Nếu muốn slash command cập nhật nhanh trong server test, set APP_GUILD_ID.
- Sau khi chỉnh command code trong registrar, nên restart bot để sync tree ổn định.
- Nếu mới bật Gmail, chạy lại `python setup_calendar.py` để cấp thêm scope `gmail.readonly` và `gmail.modify`.
- Nếu IDE báo thiếu import, kiểm tra đúng .venv và pip install -r requirements.txt.

## Gợi ý mở rộng state-of-the-art

- Adaptive learning path: đề xuất lộ trình học theo điểm yếu theo tuần
- Spaced repetition: tạo lịch ôn tập theo thuật toán quên (SM-2)
- Semantic memory: vector store cho ngữ cảnh học dài hạn
- Multi-agent workflow: agent tóm tắt + phản biện + tạo quiz
- Progress dashboard web: weekly/monthly analytics, heatmap, goals
- Smart nudges: nhắc học cá nhân hoá theo giờ hoạt động thực tế
