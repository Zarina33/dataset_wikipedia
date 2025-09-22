import json
import os


def detect_file_format(file_path):
    """
    Определяет формат файла: JSON массив или JSONL
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        if first_char == '[':
            return 'json_array'
        else:
            return 'jsonl'


def read_data_file(file_path):
    """
    Читает данные из файла, автоматически определяя формат
    """
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден")
        return None

    file_format = detect_file_format(file_path)
    print(f"Определен формат файла {file_path}: {file_format}")

    try:
        if file_format == 'json_array':
            # Читаем как обычный JSON файл
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"Загружено {len(data) if isinstance(data, list) else 1} записей")
                return data
        else:
            # Читаем как JSONL
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Ошибка в строке {line_num}: {e}")
                            return None
            print(f"Загружено {len(data)} записей из JSONL")
            return data
    except Exception as e:
        print(f"Ошибка при чтении файла {file_path}: {e}")
        return None


def merge_json_files(file1_path, file2_path, output_path, merge_mode='structured'):
    """
    Объединяет два файла с данными JSON

    Args:
        file1_path: путь к первому файлу
        file2_path: путь к второму файлу
        output_path: путь для сохранения результата
        merge_mode: 'structured' или 'array'
    """
    print("=" * 60)
    print("ОБЪЕДИНЕНИЕ ФАЙЛОВ")
    print("=" * 60)

    # Читаем первый файл
    data1 = read_data_file(file1_path)
    if data1 is None:
        return

    # Читаем второй файл
    data2 = read_data_file(file2_path)
    if data2 is None:
        return

    # Объединяем данные
    if merge_mode == 'structured':
        # Структурированное объединение
        merged_data = {
            "file1_data": data1,
            "file2_data": data2,
            "metadata": {
                "file1_records": len(data1) if isinstance(data1, list) else 1,
                "file2_records": len(data2) if isinstance(data2, list) else 1,
                "total_records": (len(data1) if isinstance(data1, list) else 1) +
                                 (len(data2) if isinstance(data2, list) else 1),
                "source_files": {
                    "file1": file1_path,
                    "file2": file2_path
                }
            }
        }
    else:
        # Объединение в один массив
        merged_data = []

        # Добавляем данные из первого файла
        if isinstance(data1, list):
            merged_data.extend(data1)
        else:
            merged_data.append(data1)

        # Добавляем данные из второго файла
        if isinstance(data2, list):
            merged_data.extend(data2)
        else:
            merged_data.append(data2)

    # Сохраняем результат
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)

        total_records = len(merged_data) if merge_mode == 'array' else merged_data['metadata']['total_records']
        print(f"\n✓ Успешно объединено {total_records} записей")
        print(f"✓ Результат сохранен в: {output_path}")
        print(f"✓ Размер выходного файла: {os.path.getsize(output_path)} байт")

    except Exception as e:
        print(f"Ошибка при сохранении: {e}")


def show_sample_data(file_path, num_samples=3):
    """
    Показывает несколько примеров данных из файла
    """
    data = read_data_file(file_path)
    if data and isinstance(data, list) and len(data) > 0:
        print(f"\nПримеры данных из {file_path}:")
        print("-" * 40)
        for i, item in enumerate(data[:num_samples]):
            print(f"Запись {i + 1}:")
            if isinstance(item, dict):
                for key, value in item.items():
                    value_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                    print(f"  {key}: {value_str}")
            else:
                print(f"  {item}")
            print()


if __name__ == "__main__":
    # Ваши файлы
    jsonl_file = "/home/zarina/Рабочий стол/data_gemma/kyrgyz_wikipedia_qa_datasetGEMINI.jsonl"  # На самом деле JSON массив
    json_file = "/home/zarina/Рабочий стол/data_gemma/kyrgyz_wikipedia_qa_dataset.json"  # Обычный JSON файл

    print("АНАЛИЗ ФАЙЛОВ:")
    print("=" * 60)

    # Показываем примеры данных
    if os.path.exists(jsonl_file):
        show_sample_data(jsonl_file, 2)

    if os.path.exists(json_file):
        show_sample_data(json_file, 2)

    # Объединяем файлы
    print("\nВЫБЕРИТЕ РЕЖИМ ОБЪЕДИНЕНИЯ:")
    print("1. Структурированное объединение (разные секции)")
    print("2. Объединение в один массив")

    try:
        choice = input("Введите номер (1 или 2): ").strip()
        mode = 'structured' if choice == '1' else 'array'
        output_file = f"merged_{mode}.json"

        merge_json_files(jsonl_file, json_file, output_file, mode)

    except KeyboardInterrupt:
        print("\nОтменено пользователем")
    except Exception as e:
        print(f"Ошибка: {e}")
        # Используем режим по умолчанию
        print("Использую режим по умолчанию: объединение в массив")
        merge_json_files(jsonl_file, json_file, "merged_array.json", 'array')