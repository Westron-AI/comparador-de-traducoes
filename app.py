import gradio as gr
from translator import traduzir_comparar

demo = gr.Interface(
    fn=traduzir_comparar,
    inputs=gr.Textbox(label="Texto em Inglês", placeholder="Digite a frase em inglês aqui..."),
    outputs=gr.Textbox(label="Tradução"),
    title="Comparação de Traduções",
    description="""Insira uma frase em inglês para visualizar a tradução gerada pelo modelo original e pelo modelo após passar por fine-tuning com textos de documentações técnicas:\n
                  Modelo original: Helsinki-NLP/opus-mt-tc-big-en-pt\n
                  Modelo fine-tuned: westronai/translation-en-pt""",
    theme=gr.themes.Monochrome()  
)

if __name__ == "__main__":
    demo.launch()
