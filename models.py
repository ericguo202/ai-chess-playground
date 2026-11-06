import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import (
    OpenAIChatCompletionsModel,
)

# Load environment variables
load_dotenv(override=True)

# Set up our models
# GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
# GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"
# GROK_BASE_URL = "https://api.x.ai/v1"

# Use OpenRouter for convenience
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# deepseek_client = AsyncOpenAI(base_url=DEEPSEEK_BASE_URL, api_key=os.getenv("DEEPSEEK_API_KEY"))
# gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=os.getenv("GOOGLE_API_KEY"))
# anthropic_client = AsyncOpenAI(base_url=ANTHROPIC_BASE_URL, api_key=os.getenv("ANTHROPIC_API_KEY"))
# grok_client = AsyncOpenAI(base_url=GROK_BASE_URL, api_key=os.getenv("GROK_API_KEY"))
openrouter_client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.getenv("OPENROUTER_API_KEY"))

deepseek_model = OpenAIChatCompletionsModel(model="deepseek-chat", openai_client=openrouter_client)
gemini_model = OpenAIChatCompletionsModel(model="gemini-2.5-pro", openai_client=openrouter_client)
grok_model = OpenAIChatCompletionsModel(model="grok-4-1-fast-non-reasoning", openai_client=openrouter_client)
claude_model = OpenAIChatCompletionsModel(model="claude-haiku-4-5-20251001", openai_client=openrouter_client)
oss_model = OpenAIChatCompletionsModel(model="openai/gpt-oss-120b", openai_client=openrouter_client)
qwen_model = OpenAIChatCompletionsModel(model="qwen/qwen3-235b-a22b-2507", openai_client=openrouter_client)
xiaomi_model = OpenAIChatCompletionsModel(model="xiaomi/mimo-v2-flash", openai_client=openrouter_client)
nano_model = OpenAIChatCompletionsModel(model="openai/gpt-5-nano", openai_client=openrouter_client)
haiku35_model = OpenAIChatCompletionsModel(model="anthropic/claude-3.5-haiku", openai_client=openrouter_client)
haiku45_model = OpenAIChatCompletionsModel(model="anthropic/claude-haiku-4.5", openai_client=openrouter_client)

MODEL_MAP = {
    "GPT-4.1-mini": "gpt-4.1-mini",
    "GPT-5-nano": "gpt-5-nano",
    "Gemini 2.5 Pro": gemini_model,
    "DeepSeek Chat": deepseek_model,
    "Grok 4.1": grok_model,
    "Claude Haiku 3.5": haiku35_model,
    "Claude Haiku 4.5": haiku45_model,
    "GPT-OSS-120b": oss_model,
    "Qwen3": qwen_model,
    "Xiaomi MiMo-V2-Flash": xiaomi_model,
}