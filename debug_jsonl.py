import json
import os


def debug_jsonl_file(file_path="/home/zarina/Рабочий стол/data_gemma/kyrgyz_wikipedia_qa_datasetGEMINI.jsonl"):
    print(f"Отладка файла: {file_path}")
    print("=" * 60)

    # Проверяем существование файла
    if not os.path.exists(file_path):
        print(f"ОШИБКА: Файл {file_path} не существует!")
        return

    # Размер файла
    file_size = os.path.getsize(file_path)
    print(f"Размер файла: {file_size} байт")

    if file_size == 0:
        print("ОШИБКА: Файл пустой!")
        return

    # Читаем первые 200 байтов в бинарном режиме
    with open(file_path, 'rb') as f:
        raw_bytes = f.read(200)
        print(f"\nПервые {len(raw_bytes)} байт (hex):")
        print(' '.join(f'{b:02x}' for b in raw_bytes))

        print(f"\nПервые байты как текст:")
        try:
            print(repr(raw_bytes.decode('utf-8')))
        except:
            print("Не удается декодировать как UTF-8")

    # Пробуем разные кодировки
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ КОДИРОВОК:")

    for encoding in ['utf-8', 'utf-8-sig', 'cp1251', 'cp1252', 'latin1']:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                first_line = f.readline()
                print(f"{encoding:12}: '{first_line.strip()[:80]}'")
                if first_line.strip():
                    # Пробуем парсить первую строку как JSON
                    try:
                        json.loads(first_line.strip())
                        print(f"             ✓ Первая строка - валидный JSON!")
                    except json.JSONDecodeError as e:
                        print(f"             ✗ JSON ошибка: {e}")
        except Exception as e:
            print(f"{encoding:12}: ОШИБКА - {e}")

    # Попробуем прочитать несколько строк с UTF-8
    print("\n" + "=" * 60)
    print("СОДЕРЖИМОЕ ФАЙЛА (первые 5 строк с UTF-8):")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                print(f"Строка {i + 1}: {repr(line)}")

                # Проверяем каждую строку на валидность JSON
                line_clean = line.strip()
                if line_clean:
                    try:
                        parsed = json.loads(line_clean)
                        print(f"         ✓ Валидный JSON, тип: {type(parsed).__name__}")
                    except json.JSONDecodeError as e:
                        print(f"         ✗ Невалидный JSON: {e}")
                        # Показываем проблемное место
                        error_pos = getattr(e, 'pos', 0)
                        if error_pos < len(line_clean):
                            problem_char = line_clean[error_pos] if error_pos < len(line_clean) else 'EOF'
                            print(
                                f"         Проблемный символ на позиции {error_pos}: '{problem_char}' (ord: {ord(problem_char) if problem_char != 'EOF' else 'EOF'})")
    except Exception as e:
        print(f"Ошибка при чтении: {e}")


if __name__ == "__main__":
    debug_jsonl_file()