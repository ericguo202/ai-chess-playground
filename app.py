# Imports
import asyncio
import chess
import chess.svg
import gradio as gr
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    function_tool,
    OpenAIChatCompletionsModel,
    set_default_openai_api,
)

load_dotenv(override=True)

# by default openAI uses the response API but all of our other models use the chat completions API
# so we must standardize on that
set_default_openai_api("chat_completions") 

# Set up our models
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"
GROK_BASE_URL = "https://api.x.ai/v1"

deepseek_client = AsyncOpenAI(base_url=DEEPSEEK_BASE_URL, api_key=os.getenv("DEEPSEEK_API_KEY"))
gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=os.getenv("GOOGLE_API_KEY"))
anthropic_client = AsyncOpenAI(base_url=ANTHROPIC_BASE_URL, api_key=os.getenv("ANTHROPIC_API_KEY"))
grok_client = AsyncOpenAI(base_url=GROK_BASE_URL, api_key=os.getenv("GROK_API_KEY"))

deepseek_model = OpenAIChatCompletionsModel(model="deepseek-chat", openai_client=deepseek_client)
gemini_model = OpenAIChatCompletionsModel(model="gemini-2.5-pro", openai_client=gemini_client)
grok_model = OpenAIChatCompletionsModel(model="grok-4-1-fast-reasoning", openai_client=grok_client)
claude_model = OpenAIChatCompletionsModel(model="claude-haiku-4-5-20251001", openai_client=anthropic_client)

MODEL_MAP = {
    "GPT-4.1-mini": "gpt-4.1-mini",
    "Gemini 2.5 Pro": gemini_model,
    "DeepSeek Chat": deepseek_model,
    "Grok 4": grok_model,
    "Claude Haiku 4.5": claude_model,
}

MODEL_CHOICES = list(MODEL_MAP.keys()) # list of model names

# System instructions
WHITE_INSTRUCTIONS = """
    You are a chess player playing WHITE. You must follow this protocol EXACTLY.
    OUTPUT RULE (CRITICAL):
    - You MUST NOT output any normal text.
    - The ONLY allowed normal-text output are the words "RESIGN" or "GAME OVER"
    - Otherwise, you MUST communicate only by calling tools.
    TURN PROTOCOL:
    On "start" or when you have control:
    1) Call tool get_opponent_last_move() to get Black's last move. If it returns None, you are first to move.
    2) If you think you are losing, you MUST respond with normal text "RESIGN".
    3) Otherwise, choose ONE move in standard algebraic notation.
        - DO NOT include move numbers (e.g. do NOT write "1.e4").
        - DO NOT include commentary or multiple lines.
        - Produce exactly one SAN move string to the tool.
    4) Call the make_move tool with that move.
        - If make_move returns status "illegal move", immediately call make_move again using ONE of the provided legal_moves.
        - Repeat until make_move returns status does not indicate "illegal move".
        - If make_move returns status "game_over", respond with normal text "GAME OVER"
    5) Immediately call transfer_to_black if status indicates success
        - Do NOT print the move; do not summarize; do not ask questions.
    """

BLACK_INSTRUCTIONS = """
    You are a chess player playing BLACK. You must follow this protocol EXACTLY.
    OUTPUT RULE (CRITICAL):
    - You MUST NOT output any normal text.
    - The ONLY allowed normal-text output are the words "RESIGN" or "GAME OVER"
    - Otherwise, you MUST communicate only by calling tools.
    TURN PROTOCOL:
    When you have control:
    1) Call tool get_opponent_last_move() to get White's last move.
    2) If you think you are losing, you MUST respond with normal text "RESIGN".
    3) Otherwise, choose ONE move in standard algebraic notation.
        - DO NOT include move numbers (e.g. do NOT write "1...e5").
        - DO NOT include commentary or multiple lines.
        - Produce exactly one SAN move string to the tool.
    4) Call the make_move tool with that move.
        - If make_move returns status "illegal move", immediately call make_move again using ONE of the provided legal_moves.
        - Repeat until make_move returns status does not indicate "illegal move".
        - If make_move returns status "game_over", respond with normal text "GAME OVER"
    5) Immediately call transfer_to_white if status indicates success
        - Do NOT print the move; do not summarize; do not ask questions.
    """


class GameSession:
    def __init__(self):
        self.board = chess.Board()
        self.move_log: list[str] = []
        self.queue: asyncio.Queue = asyncio.Queue()
        self.game_task: asyncio.Task | None = None
        self.end_message: str = ""

    def reset(self):
        if self.game_task and not self.game_task.done():
            self.game_task.cancel()
        self.board.reset()
        self.move_log.clear()
        self.queue = asyncio.Queue()
        self.game_task = None
        self.end_message = ""

    def board_svg(self) -> str:
        lastmove = self.board.peek() if self.board.move_stack else None
        return chess.svg.board(self.board, size=400, lastmove=lastmove)

# Creates the tools that the agents have access to
def create_tools(session: GameSession):
    # Make a move on the board
    @function_tool
    def make_move(san_move: str):
        try:
            session.board.push_san(san_move)
            session.move_log.append(san_move)

            if session.board.is_checkmate():
                winner = "Black" if session.board.turn == chess.WHITE else "White"
                session.move_log.append(f"Checkmate — {winner} wins!")
                session.queue.put_nowait((session.board_svg(), "\n".join(session.move_log)))
                session.end_message = "Game Over"
                session.queue.put_nowait(None)
                return {"status": "game_over"}
            elif session.board.is_stalemate():
                session.move_log.append("Stalemate — draw.")
                session.queue.put_nowait((session.board_svg(), "\n".join(session.move_log)))
                session.end_message = "Game Over"
                session.queue.put_nowait(None)
                return {"status": "game_over"}
            elif session.board.is_insufficient_material():
                session.move_log.append("Insufficient material — draw.")
                session.queue.put_nowait((session.board_svg(), "\n".join(session.move_log)))
                session.end_message = "Game Over"
                session.queue.put_nowait(None)
                return {"status": "game_over"}
            elif session.board.is_seventyfive_moves():
                session.move_log.append("75-move rule — draw.")
                session.queue.put_nowait((session.board_svg(), "\n".join(session.move_log)))
                session.end_message = "Game Over"
                session.queue.put_nowait(None)
                return {"status": "game_over"}
            elif session.board.is_fivefold_repetition():
                session.move_log.append("Fivefold repetition — draw.")
                session.queue.put_nowait((session.board_svg(), "\n".join(session.move_log)))
                session.end_message = "Game Over"
                session.queue.put_nowait(None)
                return {"status": "game_over"}
            else:
                session.queue.put_nowait((session.board_svg(), "\n".join(session.move_log)))
                return {"status": "success"}

        except chess.IllegalMoveError:
            legal_moves = ", ".join(session.board.san(m) for m in session.board.legal_moves)
            return {"status": "Illegal move", "legal_moves": legal_moves}

    # Retrieve opponent's previous move
    @function_tool
    def get_opponent_last_move():
        try:
            move = session.board.pop()
            san = session.board.san(move)
            session.board.push(move)
            return san
        except IndexError:
            return None

    return make_move, get_opponent_last_move

# Create the competing agents
def create_agents(session: GameSession, white_model, black_model):
    make_move, get_opponent_last_move = create_tools(session)

    white_agent = Agent(
        name="white",
        instructions=WHITE_INSTRUCTIONS,
        model=white_model,
        tools=[make_move, get_opponent_last_move],
        handoff_description="Opponent chess player who will make the next move",
    )
    black_agent = Agent(
        name="black",
        instructions=BLACK_INSTRUCTIONS,
        model=black_model,
        tools=[make_move, get_opponent_last_move],
        handoff_description="Opponent chess player who will make the next move",
    )
    white_agent.handoffs = [black_agent]
    black_agent.handoffs = [white_agent]

    return white_agent, black_agent


async def start_game(white_choice: str, black_choice: str, max_turns: int, session: GameSession):
    session.reset()

    yield (
        session.board_svg(),
        "Game starting...",
        gr.update(interactive=False),
        gr.update(interactive=False),
        "",
        "",
        session,
    )

    white_model = MODEL_MAP[white_choice]
    black_model = MODEL_MAP[black_choice]
    white_agent, _ = create_agents(session, white_model, black_model)

    async def run_game():
        try:
            await Runner.run(white_agent, "start", max_turns=max_turns)
        except asyncio.CancelledError:
            pass
        except Exception:
            session.end_message = f"Error: exceeded max turns ({max_turns})"
            session.queue.put_nowait((session.board_svg(), "\n".join(session.move_log)))
        finally:
            session.queue.put_nowait(None)

    session.game_task = asyncio.create_task(run_game())

    while True:
        item = await session.queue.get()
        if item is None:
            break
        svg, log = item
        yield (
            svg,
            log,
            gr.update(interactive=False),
            gr.update(interactive=False),
            "Game Ongoing...",
            "",
            session,
        )

    lichess_url = "https://lichess.org/analysis/" + session.board.fen().replace(" ", "_")
    end_reason = f'<p style="color: red; font-size: 24px; text-align: center;">{session.end_message}</p>'
    lichess_link = (
        f'<a href="{lichess_url}" target="_blank" style="display:inline-block;padding:8px 16px;'
        f'background-color:#007100;color:white;text-decoration:none;border-radius:4px;'
        f'font-size:14px;font-weight:600;">Analyze Position on Lichess</a>'
    )

    yield (
        session.board_svg(),
        "\n".join(session.move_log),
        gr.update(interactive=True),
        gr.update(interactive=True),
        end_reason,
        lichess_link,
        session,
    )


async def reset_game(session: GameSession):
    session.reset()
    yield (
        session.board_svg(),
        "Board reset. Select models and start a new game.",
        gr.update(interactive=True),
        gr.update(interactive=True),
        "",
        "",
        session,
    )

# Set themes
custom_green = gr.themes.Color(
    name="css_green",
    c50="#e6f4e6",
    c100="#cce9cc",
    c200="#99d399",
    c300="#66bd66",
    c400="#339633",
    c500="#007100",  # custom green
    c600="#007700",
    c700="#006600",
    c800="#004d00",
    c900="#003300",
    c950="#003311",
)

theme = gr.themes.Default(
    primary_hue=custom_green,
    secondary_hue="sky",
    neutral_hue="slate",
).set(
    body_background_fill="#F7F8F0",
    body_background_fill_dark="#000000",
    block_background_fill="#f7f8f0",
    block_background_fill_dark="#000000",
    block_border_color_dark="#bfbfbf",
)

with gr.Blocks(title="AI Chess Playground") as demo:
    session_state = gr.State(GameSession)

    gr.HTML('<h1 style="color: green; text-align: center">AI Chess Playground</h1>')

    with gr.Row():
        white_dd = gr.Dropdown(choices=MODEL_CHOICES, value="GPT-4.1-mini", label="White Model")
        black_dd = gr.Dropdown(choices=MODEL_CHOICES, value="GPT-4.1-mini", label="Black Model")

    gr.HTML(
        '<p style="color: green">A higher number of turns results in a more complete game, '
        'but may take several minutes to finish running.</p>'
    )
    max_turns_slider = gr.Slider(minimum=10, maximum=500, value=200, step=10, label="Maximum Turns")

    with gr.Row():
        start_btn = gr.Button("Start Game", variant="primary")
        reset_btn = gr.Button("Reset Board")

    end_message_html = gr.HTML(value="")
    board_html = gr.HTML(value='<p style="color: green">Select models and press Start Game.</p>')
    lichess_html = gr.HTML(value="")
    status_box = gr.Textbox(label="Move Log", lines=20, interactive=False)

    outputs = [board_html, status_box, start_btn, reset_btn, end_message_html, lichess_html, session_state]
    start_btn.click(fn=start_game, inputs=[white_dd, black_dd, max_turns_slider, session_state], outputs=outputs)
    reset_btn.click(fn=reset_game, inputs=[session_state], outputs=outputs)


if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=theme)
