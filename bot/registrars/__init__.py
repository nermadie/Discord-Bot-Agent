from .events_tasks import register_events_and_tasks
from .prefix_commands import register_prefix_commands
from .slash_commands import register_slash_commands

__all__ = [
	"register_events_and_tasks",
	"register_prefix_commands",
	"register_slash_commands",
]
