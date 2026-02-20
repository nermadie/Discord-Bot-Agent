# Discord Agent Bot

Bot Discord phục vụ học tập và quản lý công việc với trọng tâm slash commands.

## Tính năng chính

- Calendar + Events (Google Calendar)
- Tasks (Google Tasks)
- Study Summary (AI) + câu hỏi ôn tập + chấm điểm + streak
- Chat / Reasoning + xử lý ảnh
- Weather hiện tại + forecast theo ngày/giờ
- Slogan động lực tự động khi idle + lệnh thủ công

## Cấu trúc dự án (đã tách module)

- discord_bot.py: entrypoint và orchestration chính
- bot_modules/constants.py: hằng số dùng chung, giảm magic numbers
- bot_modules/embed_builders.py: UI embed cho calendar/events/tasks
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
```

1. Chạy bot

   - python discord_bot.py

## Slash commands (chuẩn hoá)

- /help
- /calendar, /events, /add_event, /del_event, /move_event
- /tasks, /overdue, /add_task, /done, /del_task
- /summary, /continue_summary, /answer, /study_stats
- /chat, /reason
- /weather
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

## Lưu ý quan trọng

- Dữ liệu summary trong ngày không bị xoá khi chạy /summary; chỉ reset sang ngày mới.
- Nếu muốn slash command cập nhật nhanh trong server test, set APP_GUILD_ID.
- Nếu IDE báo thiếu import, kiểm tra đúng .venv và pip install -r requirements.txt.

## Gợi ý mở rộng state-of-the-art

- Adaptive learning path: đề xuất lộ trình học theo điểm yếu theo tuần
- Spaced repetition: tạo lịch ôn tập theo thuật toán quên (SM-2)
- Semantic memory: vector store cho ngữ cảnh học dài hạn
- Multi-agent workflow: agent tóm tắt + phản biện + tạo quiz
- Progress dashboard web: weekly/monthly analytics, heatmap, goals
- Smart nudges: nhắc học cá nhân hoá theo giờ hoạt động thực tế
