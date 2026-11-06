# Imports
import asyncio
import chess
import chess.svg
import gradio as gr
from dotenv import load_dotenv
from agents import (
    Agent,
    Runner,
    function_tool,
    set_default_openai_api,
)

# System prompts
from instructions import WHITE_INSTRUCTIONS, BLACK_INSTRUCTIONS

# Dictionary of models
# Environment variables are loaded through models, including OPENAI_API_KEY which is needed in app.py
from models import MODEL_MAP

# Gradio theme
from themes import theme

load_dotenv(override=True)

# by default openAI uses the response API but all of our other models use the chat completions API
# so we must standardize on that
set_default_openai_api("chat_completions") 

MODEL_CHOICES = list(MODEL_MAP.keys()) # list of model names

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

            # Check for ending positions
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
            # Draw by insufficient material to checkmate
            elif session.board.is_insufficient_material():
                session.move_log.append("Insufficient material — draw.")
                session.queue.put_nowait((session.board_svg(), "\n".join(session.move_log)))
                session.end_message = "Game Over"
                session.queue.put_nowait(None)
                return {"status": "game_over"}
            # Draw by having 75 moves without pawn movements or captures
            elif session.board.is_seventyfive_moves():
                session.move_log.append("75-move rule — draw.")
                session.queue.put_nowait((session.board_svg(), "\n".join(session.move_log)))
                session.end_message = "Game Over"
                session.queue.put_nowait(None)
                return {"status": "game_over"}
            # Same position is repeated five times in the game
            elif session.board.is_fivefold_repetition():
                session.move_log.append("Fivefold repetition — draw.")
                session.queue.put_nowait((session.board_svg(), "\n".join(session.move_log)))
                session.end_message = "Game Over"
                session.queue.put_nowait(None)
                return {"status": "game_over"}
            # No stopping end position detected
            else:
                session.queue.put_nowait((session.board_svg(), "\n".join(session.move_log)))
                return {"status": "success"}

        # LLMs can commonly make illegal moves - provide hints for legal moves
        except chess.IllegalMoveError:
            legal_moves = ", ".join(session.board.san(m) for m in session.board.legal_moves)
            return {"status": "Illegal move", "legal_moves": legal_moves}

    # Retrieve opponent's previous move
    @function_tool
    def get_opponent_last_move():
        # Get the last move by temporarily popping from the move stack to retrieve the move
        # And then pushing the popped move onto the stack again
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


# Build the frontend UI using Gradio
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
