from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

def train_gpt(text_file, model_output_dir="gpt_finetuned", epochs=1):
    """
    Функция для обучения модели GPT.
    
    :param text_file: Путь к текстовому файлу для обучения
    :param model_output_dir: Директория, куда сохранится обученная модель
    :param epochs: Количество эпох
    """
    # Загрузка предобученной модели и токенизатора
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Подготовка данных
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=text_file,
        block_size=128)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False)
    
    # Настройка параметров обучения
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        save_steps=10000,
        save_total_limit=2,
    )
    
    # Создание объекта Trainer и запуск обучения
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    
    trainer.train()

# Вызов функции train_gpt с путём к вашему текстовому файлу
text_file_path = "path_to_your_text_file.txt"
train_gpt(text_file_path)

