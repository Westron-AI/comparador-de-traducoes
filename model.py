from transformers import MarianMTModel, MarianTokenizer

def carrega_modelo(nome_modelo: str) -> tuple:
    """
    Carrega um modelo de tradução pré-treinado e seu tokenizer.

    Args:
        nome_modelo (str): Nome do modelo pré-treinado a ser carregado.

    Returns:
        tuple: Uma tupla contendo o tokenizer e o modelo de tradução.
    """
    tokenizer = MarianTokenizer.from_pretrained(nome_modelo)
    model = MarianMTModel.from_pretrained(nome_modelo)
    return (tokenizer, model)
