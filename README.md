# 🤖 Discord Agent Bot

> Trợ lý Discord đa năng cho học tập & quản lý công việc: **Calendar + Tasks + Weather + Summary + Chat + Countdown**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Discord.py](https://img.shields.io/badge/discord.py-2.x-5865F2)
![Google APIs](https://img.shields.io/badge/Google-Calendar%20%26%20Tasks-4285F4)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## ✨ Tính năng chính

### 📅 Calendar & Event
- Xem lịch theo ngày: hôm nay, ngày mai, thứ trong tuần, hoặc ngày cụ thể.
- Thêm / xóa / dời giờ event nhanh ngay trong Discord.
- Nhận nhắc nhở event trong ngày **trước 30 phút**.

### 📋 Tasks (Google Tasks)
- Xem tasks theo ngày hoặc toàn bộ tasks chưa hoàn thành.
- Tạo task có hạn ngày giờ.
- Đánh dấu hoàn thành, xóa task.
- Nhận nhắc nhở task có giờ trong ngày **trước 30 phút**.

### 🌤️ Weather
- Lấy thời tiết hiện tại tại Đà Nẵng (WeatherAPI).
- Dùng trong bản tin chào sáng tự động.

### 📚 Study Summary (AI)
- Theo dõi tin nhắn tại các channel học tập bạn chọn.
- Nếu tin nhắn có ảnh/file đính kèm, bot sẽ kèm URL/tên file vào ngữ cảnh gửi model.
- Tự động tổng hợp nội dung và tạo câu hỏi ôn tập.
- Trả lời từng câu hỏi bằng `!answer` và nhận đánh giá/nhận xét tự động.
- Hỗ trợ chia lô (batch) khi lượng tin nhắn lớn.

### 💬 Chatbot trực tiếp
- Dùng `!chat <nội dung>` để hỏi đáp trực tiếp với AI (có thể đính kèm ảnh).
- Dùng `!reason <nội dung>` cho reasoning mode (lọc phần `<think>` trước khi hiển thị).
- Dùng `!reason --show-thinking <nội dung>` để xổ phần `<think>` (owner + bật env).
- Hiển thị phản hồi bằng embed đẹp + thông tin model đang dùng.
- Tự động fallback model nếu model chính bị giới hạn lượt gọi.

### ⏰ Countdown thông minh
- Tạo countdown sự kiện bất kỳ.
- Mốc nhắc theo cấu trúc: **5' → 4' → 3' → 2' → 60s ... 0s**.
- Có tin nhắn kết thúc và có thể mention người dùng cấu hình.
- Hỗ trợ shortcut countdown cho `!newyear` và `!tet`.

### 🤖 Tự động hóa theo lịch
- Chào sáng + thời tiết + lịch + tasks.
- Nhắc lịch / task sắp đến.
- Review cuối ngày.
- Tổng hợp học tập buổi tối.

---

## 🧱 Cấu trúc dự án

```text
discord-agent-bot/
├─ discord_bot.py          # File chính chạy bot
├─ setup_calendar.py       # Hỗ trợ setup OAuth Google
├─ requirements.txt        # Danh sách thư viện
├─ .env                    # Biến môi trường (local, không commit)
├─ credentials.json        # OAuth client của Google (local, không commit)
├─ token.json              # Token người dùng Google (local, không commit)
└─ README.md
```

---

## ⚙️ Yêu cầu hệ thống

- Python 3.10 trở lên
- Discord Bot Token
- GitHub Models Token (cho tóm tắt AI)
- WeatherAPI key
- Google Calendar API + Google Tasks API đã bật

---

## 🚀 Hướng dẫn cài đặt nhanh

## 1) Clone & tạo môi trường ảo

### Windows (PowerShell)
```powershell
git clone <repo-url>
cd discord-agent-bot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux
```bash
git clone <repo-url>
cd discord-agent-bot
python -m venv .venv
source .venv/bin/activate
```

## 2) Cài dependencies
```bash
pip install -r requirements.txt
```

## 3) Tạo file `.env`
Tạo file `.env` ở thư mục gốc với nội dung mẫu:

```env
DISCORD_TOKEN=your_discord_bot_token
GITHUB_TOKEN=your_github_models_token

# Chat trực tiếp (!chat)
CHAT_MODEL_PRIMARY=openai/gpt-5
CHAT_MODEL_FALLBACKS=openai/gpt-5-mini,openai/gpt-5-nano,openai/gpt-4.1,openai/gpt-4o

# Vision/OCR cho ảnh trong chat (!chat, /chat) - xử lý từng ảnh một
VISION_MODEL_PRIMARY=meta/Llama-4-Maverick-17B-128E-Instruct-FP8
VISION_MODEL_FALLBACKS=openai/gpt-4.1-nano,openai/gpt-4o-mini

# Tổng hợp học tập (!summary)
SUMMARY_MODEL_PRIMARY=openai/gpt-5-chat
SUMMARY_MODEL_FALLBACKS=openai/gpt-5-mini,openai/gpt-5-nano,openai/gpt-4.1,openai/gpt-4o

# Chấm câu trả lời học tập (!answer)
ANSWER_MODEL_PRIMARY=openai/gpt-5-chat
ANSWER_MODEL_FALLBACKS=openai/gpt-5-mini,openai/gpt-5-nano,openai/gpt-4.1,openai/gpt-4o

# Reasoning mode (!reason, /reason)
REASONING_MODEL_PRIMARY=deepseek/DeepSeek-R1-0528
REASONING_MODEL_FALLBACKS=microsoft/Phi-4-reasoning,microsoft/Phi-4-mini-reasoning
ALLOW_SHOW_REASONING_THINKING=false

# Legacy optional
GITHUB_MODEL=gpt-4o-mini

WEATHER_API_KEY=your_weatherapi_key
WEATHER_PROVIDER=weatherapi
YOUR_USER_ID=123456789012345678
CHANNEL_MAIN=123456789012345678
CHANNELS_TO_MONITOR=111111111111111111,222222222222222222

# Optional: sync slash command ngay cho server test
APP_GUILD_ID=123456789012345678
```

> Bạn cũng có thể copy nhanh từ file `.env.example`.

> Gợi ý:
> - `YOUR_USER_ID`: ID Discord của bạn (để giới hạn lệnh nhạy cảm và mention đúng người).
> - `CHANNEL_MAIN`: channel bot gửi thông báo tự động.
> - `CHANNELS_TO_MONITOR`: các channel bot thu thập tin nhắn để summary.

## 4) Setup Google Calendar/Tasks OAuth
1. Truy cập Google Cloud Console.
2. Tạo Project mới.
3. Enable:
   - Google Calendar API
   - Google Tasks API
4. Tạo OAuth Client ID (Desktop App).
5. Tải file JSON và đổi tên thành `credentials.json` đặt ở thư mục gốc.
6. Chạy setup:
   ```bash
   python setup_calendar.py
   ```
7. Sau khi auth thành công, file `token.json` sẽ được tạo.

## 5) Chạy bot
```bash
python discord_bot.py
```

---

## 🔐 Bảo mật

Các file sau **không được commit**:
- `.env`
- `credentials.json`
- `token.json`

Dự án đã có `.gitignore` để tự động chặn các file này.

---

## 🕹️ Nhóm lệnh chính

## `!help`
- `!help`: danh sách tổng quan
- `!help calendar|tasks|countdown|weather|study|chatbot|automation`

## Slash Commands (`/`)
- Bot hỗ trợ slash commands để Discord tự hiện **Command Matching** như giao diện bạn mong muốn.
- Lệnh chính: `/calendar`, `/events`, `/tasks`, `/countdown`, `/chat`, `/reason`, `/summary`, `/answer`, `/weather`, `/ping`, `/help`.
- Để slash command hiện ngay (không phải chờ global cache), set `APP_GUILD_ID` trong `.env` rồi restart bot.

## Calendar
- `!calendar [date]`
- `!events [date]`
- `!add_event <title> | <date time-end> | <desc>`
- `!del_event <index>`
- `!move_event <index> | <date time>`

## Tasks
- `!tasks [date]`
- `!overdue`
- `!add_task <title> | <date time> | <notes>`
- `!done <index>`
- `!del_task <index>`

## Study
- `!summary`
- `!continue`
- `!answer <số câu> | <câu trả lời>`
- `!stats`

## Chatbot
- `!chat <câu hỏi của bạn>`
- `!chat <câu hỏi> + đính kèm 1-n ảnh` (bot tự trích xuất nội dung từng ảnh rồi trả lời)
- `!reason <câu hỏi/bài toán cần reasoning>`

## Slash Chatbot
- `/chat` với các trường ảnh `image_1`, `image_2`, `image_3`, `image_4`
- `/reason` (có tuỳ chọn `show_thinking`)

## Weather
- `!weather`

## Countdown
- `!countdown`
- `!add_countdown <name> | <date time> | <emoji>`
- `!del_countdown <name>`
- `!newyear [year month day hour minute]`
- `!tet`

## Utility
- `!ping`

---

## ⏱️ Lịch tự động hiện có

- **06:30**: chào sáng + thời tiết + lịch + task
- **Mỗi 1 phút**: kiểm tra nhắc event/task trong vòng 30 phút tới
- **20:00**: review task cuối ngày
- **21:00**: tổng hợp học tập
- **Mỗi giây**: kiểm tra countdown

---

## 🧠 Ghi chú vận hành

- Bot dùng timezone `Asia/Ho_Chi_Minh` cho xử lý lịch.
- Nếu thấy import lỗi trong editor, kiểm tra bạn đã activate đúng `.venv` và cài đủ package chưa.
- Nếu bot không gửi nhắc tự động:
  - Kiểm tra `CHANNEL_MAIN`
  - Kiểm tra bot có quyền gửi tin nhắn tại channel
  - Kiểm tra token/API key hợp lệ

---

## 🧪 Checklist test nhanh

1. `!ping` để kiểm tra bot online.
2. `!weather` để kiểm tra Weather API.
3. `!events` và `!tasks` để kiểm tra kết nối Google.
4. Tạo event/task sắp tới trong vòng 30 phút để kiểm tra reminder.
5. `!add_countdown Test | today 23:59 | 🎯` để kiểm tra countdown.
6. `!chat Giải thích OAuth2 là gì` để kiểm tra chat model + fallback.
7. `!summary` rồi `!answer 1 | ...` để kiểm tra cơ chế hỏi đáp ôn tập.

---

## 📌 Roadmap gợi ý

- Thêm logging chuẩn file + mức log.
- Thêm Dockerfile để deploy dễ hơn.
- Tách module lớn trong `discord_bot.py` để dễ bảo trì.

---

## 👤 Đóng góp

- Tạo branch mới từ `main`
- Commit nhỏ, rõ mục tiêu
- Mở Pull Request kèm mô tả test đã chạy

---

Nếu bạn muốn, mình có thể tách riêng module chatbot/summary thành file độc lập để dễ maintain hơn.