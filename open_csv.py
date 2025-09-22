import pandas as pd

# Настройки pandas для полного отображения
pd.set_option('display.max_columns', None)  # Показать все столбцы
pd.set_option('display.max_rows', None)  # Показать все строки
pd.set_option('display.width', None)  # Убрать ограничение по ширине
pd.set_option('display.max_colwidth', None)  # Показать полный текст в ячейках


def read_csv_full_display(filename):
    try:
        # Читаем CSV файл
        df = pd.read_csv(filename)

        # Выводим первые 3 записи с полным текстом
        print("Первые 3 записи (полный текст):")
        print(df.head(3))

        # Дополнительно можно вывести информацию о структуре данных
        print(f"\nОбщая информация:")
        print(f"Количество строк: {len(df)}")
        print(f"Количество столбцов: {len(df.columns)}")
        print(f"Названия столбцов: {list(df.columns)}")

    except FileNotFoundError:
        print(f"Файл {filename} не найден")
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")


# Альтернативный способ - выводить каждую запись отдельно
def read_csv_detailed(filename):
    try:
        df = pd.read_csv(filename)

        print("Первые 3 записи (детальный вывод):")
        for i in range(min(3, len(df))):
            print(f"\n--- Запись {i + 1} ---")
            for column in df.columns:
                print(f"{column}: {df.iloc[i][column]}")

    except FileNotFoundError:
        print(f"Файл {filename} не найден")
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")


# Использование
if __name__ == "__main__":
    filename = "kyrgyz_wikipedia_data.csv"  # Замените на имя вашего файла

    # Способ 1: Полное отображение в табличном виде
    read_csv_full_display(filename)

    print("\n" + "=" * 50 + "\n")

    # Способ 2: Детальный вывод каждой записи
    read_csv_detailed(filename)