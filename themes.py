import gradio as gr

# Set Gradio themes
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