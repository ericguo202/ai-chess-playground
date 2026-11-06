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

deepseek_model = OpenAIChatCompletionsModel(model="deepseek/deepseek-v4.1-flash", openai_client=openrouter_client)
gemini_model = OpenAIChatCompletionsModel(model="google/gemini-3.6-flash", openai_client=openrouter_client)
grok_model = OpenAIChatCompletionsModel(model="x-ai/grok-4.6", openai_client=openrouter_client)
claude_model = OpenAIChatCompletionsModel(model="claude-haiku-4-5-20251001", openai_client=openrouter_client)
oss_model = OpenAIChatCompletionsModel(model="openai/gpt-oss-120b", openai_client=openrouter_client)
qwen_model = OpenAIChatCompletionsModel(model="qwen/qwen3.8-27b", openai_client=openrouter_client)
nano_model = OpenAIChatCompletionsModel(model="openai/gpt-5-nano", openai_client=openrouter_client)
haiku_model = OpenAIChatCompletionsModel(model="anthropic/claude-haiku-5.5", openai_client=openrouter_client)
luna_model = OpenAIChatCompletionsModel(model="openai/gpt-5.6-luna", openai_client=openrouter_client)
gpt_model = OpenAIChatCompletionsModel(model="openai/gpt-4.1-mini", openai_client=openrouter_client)

MODEL_MAP = {
    "GPT-4.1-mini": gpt_model,
    "GPT-5.6-Luna": luna_model,
    "GPT-5-nano": nano_model,
    "Gemini 3.6 Flash": gemini_model,
    "DeepSeek v4.1 flash": deepseek_model,
    "Grok 4.6": grok_model,
    "Claude Haiku 5.5": haiku_model,
    "GPT-OSS-120b": oss_model,
    "Qwen 3.8": qwen_model,
}