#!/usr/bin/env python3
"""
Демонстрация системы возобновления работы
"""

from wikipedia_qa_generator import WikipediaQAGenerator
import json
import os

def demo_resume():
    """Демонстрация возобновления работы"""
    generator = WikipediaQAGenerator()
    
    csv_file = "kyrgyz_wikipedia_data.csv"
    output_file = "demo_qa_dataset.json"
    
    print("=== ДЕМОНСТРАЦИЯ СИСТЕМЫ ВОЗОБНОВЛЕНИЯ ===\n")
    
    # Проверяем существующий файл
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            processed_indices = []
            for item in existing_data:
                if 'source_index' in item:
                    processed_indices.append(item['source_index'])
            
            if processed_indices:
                print(f"📁 Найден существующий файл с {len(existing_data)} Q&A парами")
                print(f"🔢 Обработанные статьи: {sorted(processed_indices)}")
                print(f"▶️  Работа будет продолжена с статьи {max(processed_indices) + 1}")
            else:
                print("📁 Найден пустой файл, начинаем с начала")
        except:
            print("📁 Файл поврежден, начинаем с начала")
    else:
        print("📁 Файл не найден, начинаем с начала")
    
    print(f"\n🚀 Запуск обработки...")
    
    # Запускаем обработку 3 статей
    try:
        generator.process_csv_file(
            csv_file_path=csv_file,
            output_file_path=output_file,
            start_from=0,  # Автоматически определится
            max_articles=3
        )
    except KeyboardInterrupt:
        print("\n⏸️  Процесс прерван пользователем")
        print("💡 При следующем запуске работа продолжится автоматически!")

if __name__ == "__main__":
    demo_resume()





