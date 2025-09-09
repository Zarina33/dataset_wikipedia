#!/usr/bin/env python3
"""
Тестовый скрипт для проверки генерации Q&A на одной статье
"""

from wikipedia_qa_generator import WikipediaQAGenerator
import pandas as pd
import json

def test_single_article():
    """Тестирование на одной статье"""
    generator = WikipediaQAGenerator()
    
    # Проверяем подключение
    if not generator.test_ollama_connection():
        print("❌ Ошибка подключения к Ollama")
        return
    
    # Загружаем первую статью из CSV
    df = pd.read_csv("kyrgyz_wikipedia_data.csv")
    first_article = df.iloc[1]['Text']  # Берем вторую статью (первая может быть заголовком)
    
    print("Текст статьи:")
    print("-" * 50)
    print(first_article[:500] + "..." if len(first_article) > 500 else first_article)
    print("-" * 50)
    
    print("\nГенерация вопроса и ответа...")
    qa_pair = generator.generate_qa_from_text(first_article)
    
    if qa_pair:
        print("\n✅ Успешно сгенерировано:")
        print(f"Вопрос: {qa_pair['question']}")
        print(f"Ответ: {qa_pair['answer']}")
        
        # Сохраняем результат
        with open("test_result.json", "w", encoding="utf-8") as f:
            json.dump([qa_pair], f, ensure_ascii=False, indent=2)
        print("\nРезультат сохранен в test_result.json")
    else:
        print("❌ Не удалось сгенерировать Q&A пару")

if __name__ == "__main__":
    test_single_article()

