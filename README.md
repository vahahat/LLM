# LLM
LLM training model 



В скрипте llm_learn мы:
1. Импортируем необходимые компоненты из библиотеки `transformers`.
2. Определяем функцию `train_gpt`, которая принимает путь к текстовому файлу, директорию для сохранения модели и количество эпох.
3. Загружаем предобученную модель GPT2 и соответствующий ей токенизатор.
4. Готовим набор данных для обучения с использованием класса `TextDataset`.
5. Создаём объект `Trainer` с параметрами обучения и запускаем процесс обучения.

Далее в файле agent используется `GPT2LMHeadModel` и `GPT2Tokenizer` из библиотеки `transformers`. 
Сначала загружаем модель и токенизатор с их предустановленными конфигурациями.
Затем в функции `generate_text` кодируем вводный текст с помощью токенизатора, генерируем ответ с помощью модели и декодируем его обратно в читаемый текст. 
Результатом работы будет текст, сгенерированный моделью, начиная с заданной подсказки.
