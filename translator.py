from model import carrega_modelo

tokenizador_original, modelo_original = carrega_modelo('Helsinki-NLP/opus-mt-tc-big-en-pt')
tokenizador_finetuned, modelo_finetuned = carrega_modelo('westronai/translation-en-pt')

def traduz_sentenca(model, tokenizer, texto_ingles: str):
    """
    Traduz uma sentença do inglês para o português usando um modelo de tradução.

    Args:
        model: O modelo de tradução pré-treinado.
        tokenizer: O tokenizer responsável por preparar o texto.
        texto_ingles (str): A sentença em inglês que será traduzida.

    Returns:
        dict: Dicionário com a sentença original e sua tradução.
    """
    tokens = tokenizer(texto_ingles, return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    texto_portugues = tokenizer.decode(translated[0], skip_special_tokens=True)
    return {
        'texto_ingles': texto_ingles,
        'texto_portugues': texto_portugues
    }

def traduzir_comparar(texto_ingles):
    """
    Compara as traduções geradas pelo modelo original e pelo modelo fine-tuned.

    Args:
        texto_ingles (str): O texto em inglês a ser traduzido.

    Returns:
        str: String formatada contendo as traduções de ambos os modelos.
    """
    traducao_original = traduz_sentenca(modelo_original, tokenizador_original, texto_ingles)
    traducao_finetuned = traduz_sentenca(modelo_finetuned, tokenizador_finetuned, texto_ingles)
    return f"Modelo Original: {traducao_original['texto_portugues']}\n\nModelo Fine-Tuned: {traducao_finetuned['texto_portugues']}"
