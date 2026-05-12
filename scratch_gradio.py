import gradio as gr

def test_fn(x):
    return x, "Done"

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Input")
            btn = gr.Button("Submit")
        with gr.Column():
            out_img = gr.Image(label="Output Image")
            out_txt = gr.Markdown("Status")
    
    btn.click(test_fn, inputs=inp, outputs=[out_img, out_txt])
    
    gr.Examples(
        examples=[["Hello"]],
        inputs=inp,
        outputs=[out_img, out_txt],
        fn=test_fn,
        cache_examples=False
    )

if __name__ == "__main__":
    demo.launch(server_port=7861)
