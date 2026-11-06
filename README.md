# AI Chess Playground

Chess between LLMs is entertaining and educational. However, there is no easy way to automate chess games between LLMs using their default chat interfaces. **AI Chess Playground** is an agentic workflow build using the OpenAI Agents SDK and the Python chess library with a Gradio Frontend that allows users to simulate chess games between LLMs. Pick two models to pair up against each other, and watch the models battle it out on the chessboard with streaming, a move log, and a Lichess analysis link when the game is finished.

Models have access to the following tools:

- `get_opponent_last_move` - retrieves the last move made by the opponent
- `make_move` - play a move using standard algebraic notation. Illegal moves are flagged and rejected.
- `resign` - to simulate a real game, agents use their judgment to resign in a losing position

After a series of tool calls, the model will hand off the turn to its opponent.

## How to run it yourself

1. In a terminal or command prompt, run `git clone https://github.com/ericguo202/ai-chess-playground.git`
2. Create a .env file in the project's root directory and supply API keys for OpenAI and OpenRouter.
3. Install uv, if not already installed.
4. In the command line, run `uv sync` to install required dependencies.
5. Change up the models in `models.py` to your liking based on available models on OpenRouter.
6. After all of the dependencies have been installed, run the app using `uv run app.py`

## Deploy details

---

title: ai_chess_playground
app_file: app.py
sdk: gradio
sdk_version: 6.9.0

---
