import os
from datetime import timedelta, timezone

from dotenv import load_dotenv


load_dotenv()


def _parse_model_fallbacks(env_key, default_csv):
    raw = os.getenv(env_key, default_csv)
    return [item.strip() for item in str(raw).split(",") if item.strip()]


DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_MODEL = os.getenv("GITHUB_MODEL", "openai/gpt-4o-mini")
GOOGLE_AI_STUDIO_API_KEY = os.getenv("GOOGLE_AI_STUDIO_API_KEY", "")
THECATAPI_KEY = os.getenv("THECATAPI_KEY", "")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
WEATHER_PROVIDER = os.getenv("WEATHER_PROVIDER", "weatherapi")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
YOUR_USER_ID = int(os.getenv("YOUR_USER_ID", "0"))
MAIN_CHANNEL_ID = int(os.getenv("CHANNEL_MAIN", "0"))
APP_GUILD_ID = int(os.getenv("APP_GUILD_ID", "0"))

CHAT_MODEL_PRIMARY = os.getenv("CHAT_MODEL_PRIMARY", "openai/gpt-5")
CHAT_MODEL_FALLBACKS = _parse_model_fallbacks(
    "CHAT_MODEL_FALLBACKS", "openai/gpt-5-mini,openai/gpt-5-nano,openai/gpt-4o"
)

VISION_MODEL_PRIMARY = os.getenv(
    "VISION_MODEL_PRIMARY", "meta/Llama-4-Maverick-17B-128E-Instruct-FP8"
)
VISION_MODEL_FALLBACKS = _parse_model_fallbacks(
    "VISION_MODEL_FALLBACKS", "openai/gpt-4.1-nano,openai/gpt-4o-mini"
)

SUMMARY_MODEL_PRIMARY = os.getenv("SUMMARY_MODEL_PRIMARY", "openai/gpt-5-chat")
SUMMARY_MODEL_FALLBACKS = _parse_model_fallbacks(
    "SUMMARY_MODEL_FALLBACKS", "openai/gpt-5-mini,openai/gpt-5-nano,openai/gpt-4o"
)

ANSWER_MODEL_PRIMARY = os.getenv("ANSWER_MODEL_PRIMARY", "openai/gpt-5-chat")
ANSWER_MODEL_FALLBACKS = _parse_model_fallbacks(
    "ANSWER_MODEL_FALLBACKS", "openai/gpt-5-mini,openai/gpt-5-nano,openai/gpt-4o"
)

REASONING_MODEL_PRIMARY = os.getenv(
    "REASONING_MODEL_PRIMARY", "deepseek/DeepSeek-R1-0528"
)
REASONING_MODEL_FALLBACKS = _parse_model_fallbacks(
    "REASONING_MODEL_FALLBACKS",
    "microsoft/Phi-4-reasoning,microsoft/Phi-4-mini-reasoning",
)

GMAIL_MODEL_PRIMARY = os.getenv("GMAIL_MODEL_PRIMARY", "xai/grok-3")
GMAIL_MODEL_FALLBACKS = _parse_model_fallbacks(
    "GMAIL_MODEL_FALLBACKS", "openai/gpt-5-mini,openai/gpt-5-nano,openai/gpt-4o"
)
MAIL_LLM_MODELS = _parse_model_fallbacks(
    "MAIL_LLM_MODELS", "gemini-3.0-flash,gemini-2.5-pro"
)

MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "0"))
SUMMARY_MAX_OUTPUT_TOKENS = int(os.getenv("SUMMARY_MAX_OUTPUT_TOKENS", "16000"))
AI_REQUEST_TIMEOUT_SECONDS = int(os.getenv("AI_REQUEST_TIMEOUT_SECONDS", "45"))
REASONING_REQUEST_TIMEOUT_SECONDS = int(
    os.getenv("REASONING_REQUEST_TIMEOUT_SECONDS", "90")
)
STUDY_POINTS_PASS = int(os.getenv("STUDY_POINTS_PASS", "10"))
STUDY_POINTS_MISS = int(os.getenv("STUDY_POINTS_MISS", "3"))
STUDY_PASS_THRESHOLD = float(os.getenv("STUDY_PASS_THRESHOLD", "7"))
STUDY_METRICS_DIR = os.getenv("STUDY_METRICS_DIR", "study_metrics")
SLOGAN_IDLE_MINUTES = int(os.getenv("SLOGAN_IDLE_MINUTES", "180"))
SLOGAN_CHECK_INTERVAL_MINUTES = int(os.getenv("SLOGAN_CHECK_INTERVAL_MINUTES", "30"))
GMAIL_UNREAD_LIMIT = int(os.getenv("GMAIL_UNREAD_LIMIT", "10"))
GMAIL_READ_LIMIT = int(os.getenv("GMAIL_READ_LIMIT", "20"))
GMAIL_LOOKBACK_DAYS = int(os.getenv("GMAIL_LOOKBACK_DAYS", "30"))
GMAIL_SENT_TODAY_LIMIT = int(os.getenv("GMAIL_SENT_TODAY_LIMIT", "200"))
WEATHER_ALERT_INTERVAL_HOURS = int(os.getenv("WEATHER_ALERT_INTERVAL_HOURS", "3"))
WEATHER_ALERT_LOOKAHEAD_HOURS = int(os.getenv("WEATHER_ALERT_LOOKAHEAD_HOURS", "3"))
WEATHER_ALERT_RAIN_CHANCE_THRESHOLD = int(
    os.getenv("WEATHER_ALERT_RAIN_CHANCE_THRESHOLD", "60")
)
WEATHER_ALERT_WIND_KPH_THRESHOLD = float(
    os.getenv("WEATHER_ALERT_WIND_KPH_THRESHOLD", "25")
)

CHANNELS_TO_MONITOR_STR = os.getenv("CHANNELS_TO_MONITOR", "")
CHANNELS_TO_MONITOR = [
    int(ch.strip()) for ch in CHANNELS_TO_MONITOR_STR.split(",") if ch.strip()
]

VIETNAM_TZ = timezone(timedelta(hours=7))
