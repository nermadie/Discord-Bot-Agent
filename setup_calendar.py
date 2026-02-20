from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import os

# QUAN TRỌNG: Cần cả Calendar và Tasks scopes
SCOPES = [
    "https://www.googleapis.com/auth/calendar",  # Read/Write calendar
    "https://www.googleapis.com/auth/tasks",  # Read/Write tasks
    "https://www.googleapis.com/auth/gmail.readonly",  # Read unread/sent mails
    "https://www.googleapis.com/auth/gmail.modify",  # Mark read/checkpoint actions
]


def main():
    creds = None

    # Xóa token cũ nếu có (vì thêm scope mới)
    if os.path.exists("token.json"):
        print("⚠️ Phát hiện token cũ, đang xóa để tạo mới với scopes đầy đủ...")
        os.remove("token.json")

    # Tạo credentials mới
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists("credentials.json"):
                print("❌ Không tìm thấy credentials.json")
                print("Tải về từ: https://console.cloud.google.com/apis/credentials")
                return

            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=8080)

        # Lưu token
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    print("✅ Xác thực thành công!")
    print("✅ File token.json đã được tạo với quyền:")
    print("   - Google Calendar (đọc/ghi)")
    print("   - Google Tasks (đọc/ghi)")
    print("   - Gmail (chỉ đọc)")
    print("   - Gmail (đánh dấu đã đọc)")
    print("\n🎉 Bây giờ bạn có thể chạy bot!")


if __name__ == "__main__":
    main()
