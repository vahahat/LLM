from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Загрузка предварительно обученной модели и токенизатора
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def generate_text(prompt, length=50):
    """
    Функция для генерации текста с помощью GPT.
    :param prompt: текст-подсказка для модели
    :param length: желаемая длина вывода
    """
    # Кодирование подсказки (input text) в векторные представления
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Генерация текста на основе подсказки
    outputs = model.generate(inputs, max_length=length, num_return_sequences=1, no_repeat_ngram_size=2)

    # Декодирование и возврат сгенерированного текста
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Применение функции для генерации текста
prompt_text = "Чем знаменита Италия? "
generated_text = generate_text(prompt_text, length=100)
print(generated_text)
