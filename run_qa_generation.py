#!/usr/bin/env python3
"""
Запуск генерации Q&A датасета с настройками по умолчанию
"""

from wikipedia_qa_generator import WikipediaQAGenerator

def main():
    """Запуск с рекомендуемыми настройками"""
    print("🚀 Запуск генерации Q&A датасета из Wikipedia статей")
    print("=" * 60)
    
    generator = WikipediaQAGenerator()
    
    # Настройки
    csv_file = "kyrgyz_wikipedia_data.csv"
    output_file = "kyrgyz_wikipedia_qa_dataset_old.json"
    
    # Рекомендуемые параметры для начала
    start_from = 0
    max_articles = 50  # Начинаем с небольшого количества
    
    print(f"📂 Входной файл: {csv_file}")
    print(f"💾 Выходной файл: {output_file}")
    print(f"🔢 Начать с статьи: {start_from}")
    print(f"📊 Максимум статей: {max_articles}")
    print(f"🤖 Модель: {generator.model}")
    print("=" * 60)
    
    # Подтверждение
    response = input("\nПродолжить? (y/n): ").lower().strip()
    if response != 'y':
        print("Отменено пользователем.")
        return
    
    try:
        generator.process_csv_file(
            csv_file_path=csv_file,
            output_file_path=output_file,
            start_from=start_from,
            max_articles=max_articles
        )
    except KeyboardInterrupt:
        print("\n⏹️  Обработка прервана пользователем")
        print("Промежуточные результаты сохранены в файле.")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        print("Проверьте лог файл qa_generation.log для подробностей.")

if __name__ == "__main__":
    main()
