import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gazeshop.toolkit.config import Config
from gazeshop.toolkit.event_bus import EventBus
from gazeshop.toolkit.fusion_engine import FusionEngine
from gazeshop.toolkit.dialogue_manager import DialogueManager
from tests.test_fusion_engine import gaze_lock, speech_intent, speech_confirm

bus = EventBus()
bus.enable_logging()

config = Config(
    FUSION_TIME_WINDOW_S=2.0,
    GAZE_DWELL_TO_LOCK_S=1.0,
    CONFIDENCE_THRESHOLD=0.60,
    MAX_REPAIR_ATTEMPTS=2,
    LOCK_TTL_S=4.0
)
engine = FusionEngine(event_bus=bus, config=config)
manager = DialogueManager(event_bus=bus, config=config)

commands = []
bus.subscribe("MultimodalCommandEvent", lambda e: commands.append(e))
prompts = []
bus.subscribe("PromptEvent", lambda e: prompts.append(e))
bus.subscribe("DisambiguationPromptEvent", lambda e: prompts.append(e))
bus.subscribe("ConfirmationPromptEvent", lambda e: prompts.append(e))
bus.subscribe("ActionCancelledEvent", lambda e: prompts.append(e))

print("\n--- Running test_confirmation_flow debug ---")
gaze_lock(bus, "test_item")
print("gaze_state:", engine._gaze_state, engine._locked_target)

speech_intent(bus, "BUY", requires_confirmation=True)
print("prompts len:", len(prompts))
print("manager state:", manager.state)

speech_confirm(bus)
print("commands len:", len(commands))
