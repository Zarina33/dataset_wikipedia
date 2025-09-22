#!/usr/bin/env python3
"""
Скрипт для генерации вопросов-ответов из статей Wikipedia с помощью Gemini API
"""

import pandas as pd
import json
import time
import logging
from typing import Dict, List, Optional
import os
from tqdm import tqdm
import copy
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qa_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Глобальные переменные для управления API ключами и моделями
API_KEYS = []
MODEL_LIST = ["gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash", "gemini-1.5-pro"]
current_api_key_index = 0
current_model_index = 0
model = None

# Константы
MAX_REQUESTS_PER_RUN = 5000
REQUEST_DELAY_SECONDS = 2  # Задержка между запросами


class InvalidApiResponseError(Exception):
    """Исключение для случаев, когда API возвращает невалидный JSON."""

    def __init__(self, message, response_text):
        super().__init__(message)
        self.response_text = response_text


class WikipediaQAGenerator:
    def __init__(self):
        self.requests_count = 0

    def load_and_configure_keys(self) -> bool:
        """Загружает ключи из .env и инициализирует первую модель из списка."""
        global API_KEYS, current_api_key_index, current_model_index, model

        load_dotenv()
        api_key_names = [key for key in os.environ if key.startswith("GOOGLE_API_KEY_")]
        api_key_names.sort()
        API_KEYS = [os.getenv(key) for key in api_key_names]

        if not API_KEYS:
            logger.error("Не найдены переменные окружения формата GOOGLE_API_KEY_1, GOOGLE_API_KEY_2 и т.д.")
            return False

        logger.info(f"🔑 Найдено {len(API_KEYS)} API ключей.")
        logger.info(f"🧠 Список моделей для использования: {MODEL_LIST}")

        current_api_key_index = 0
        current_model_index = 0

        try:
            genai.configure(api_key=API_KEYS[current_api_key_index])
            model_name = MODEL_LIST[current_model_index]
            model = genai.GenerativeModel(model_name=model_name)
            logger.info(f"✅ Успешная инициализация с ключом #1 и моделью '{model_name}'")
            return True
        except Exception as e:
            logger.error(f"Ошибка при инициализации с первым ключом и моделью: {e}")
            return self.switch_model_or_key()

    def switch_model_or_key(self) -> bool:
        """
        Переключается на следующую модель в списке. Если все модели для текущего ключа
        испробованы, переключается на следующий API ключ и начинает с первой модели.
        """
        global current_api_key_index, current_model_index, model

        # Пытаемся переключиться на следующую модель с текущим ключом
        current_model_index += 1
        if current_model_index < len(MODEL_LIST):
            model_name = MODEL_LIST[current_model_index]
            key_index_for_log = current_api_key_index + 1
            logger.info(f"🔄 Переключение на модель '{model_name}' с ключом #{key_index_for_log}...")
            try:
                model = genai.GenerativeModel(model_name=model_name)
                logger.info(f"✅ Успешная инициализация с моделью '{model_name}'")
                return True
            except Exception as e:
                logger.error(f"Не удалось инициализировать модель '{model_name}' с ключом #{key_index_for_log}: {e}")
                return self.switch_model_or_key()

        # Если все модели для текущего ключа исчерпаны, переключаемся на следующий ключ
        current_api_key_index += 1
        if current_api_key_index < len(API_KEYS):
            key_index_for_log = current_api_key_index + 1
            logger.info(f"🔄 Все модели исчерпаны для текущего ключа. Переключение на API ключ #{key_index_for_log}...")

            # Сбрасываем индекс моделей
            current_model_index = 0
            model_name = MODEL_LIST[current_model_index]

            try:
                genai.configure(api_key=API_KEYS[current_api_key_index])
                model = genai.GenerativeModel(model_name=model_name)
                logger.info(f"✅ Успешная инициализация с ключом #{key_index_for_log} и моделью '{model_name}'")
                return True
            except Exception as e:
                logger.error(f"Не удалось инициализировать ключ #{key_index_for_log} с моделью '{model_name}': {e}")
                return self.switch_model_or_key()
        else:
            logger.critical("🚫 Все доступные модели и API ключи исчерпали свои лимиты.")
            return False

    def generate_qa_from_text(self, text: str) -> Optional[Dict[str, str]]:
        """Генерация вопроса и ответа из текста статьи"""
        global model

        if self.requests_count >= MAX_REQUESTS_PER_RUN:
            logger.warning(f"Достигнут лимит запросов ({MAX_REQUESTS_PER_RUN})")
            return None

        # Очистка текста от лишних символов
        cleaned_text = text.strip().replace('"', '').replace('\n', ' ')

        # Ограничиваем длину текста для обработки
        if len(cleaned_text) > 5000:
            cleaned_text = cleaned_text[:5000] + "..."

        prompt = f"""Generate a natural question-answer pair in Kyrgyz language for training data collection.

        Text: {cleaned_text}

        Instructions:
        1. Analyze the content and create a broad, contextual question about the main topic:
           - Focus on the SUBJECT/THEME rather than specific objects ("this book", "this document")
           - Ask about general concepts, processes, or categories mentioned
           - Avoid questions that assume specific context unknown to the reader

        2. Question generation logic:
           - If content lists items → "What are the main [category] in [field/area]?"
           - If content explains process → "How does [process] work?"
           - If content describes concept → "What is [concept]?"
           - If content gives overview → "What can you tell about [topic area]?"

        3. Answer requirements:
           - Transform the original text into a natural response
           - Remove any references to specific documents/sources
           - Make minimal grammatical corrections only
           - Present information as general knowledge, not as "this text says"

        4. Language requirements:
           - Use only Kyrgyz language
           - Make both question and answer sound completely natural
           - Never reference source ("бул китепте", "документте", "тексте")
           - Ensure proper Kyrgyz grammar and sentence flow

        5. Example transformations:
           - Bad: "Бул китепте кандай ырчылар бар?" 
           - Good: "Кыргызстандын белгилүү композиторлору жана ырчылары кимдер?"

        Return response in strict JSON format:
        {{
            "question": "your natural question here",
            "answer": "your comprehensive answer here"
        }}"""

        try:
            response = model.generate_content(prompt)
            self.requests_count += 1

            if response.text:
                cleaned_response = response.text.strip()
                qa_pair = self.parse_qa_response(cleaned_response)
                return qa_pair
            else:
                logger.error("API вернул пустой ответ")
                return None

        except (google_exceptions.ResourceExhausted, google_exceptions.PermissionDenied,
                google_exceptions.InvalidArgument) as e:
            logger.warning(f"Проблема с API: {e}. Попытка переключения...")
            if self.switch_model_or_key():
                # Повторяем запрос с новым ключом/моделью
                return self.generate_qa_from_text(text)
            else:
                return None
        except Exception as e:
            logger.error(f"Ошибка при обращении к Gemini API: {e}")
            return None

    def parse_qa_response(self, response_text: str) -> Optional[Dict[str, str]]:
        """Парсинг JSON ответа модели для извлечения вопроса и ответа"""
        try:
            # Очистка ответа от возможных markdown блоков
            cleaned_response = response_text.strip().lstrip("```json").rstrip("```").strip()

            # Попытка найти JSON в ответе
            json_start = cleaned_response.find('{')
            json_end = cleaned_response.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = cleaned_response[json_start:json_end]
                try:
                    qa_data = json.loads(json_str)
                    if 'question' in qa_data and 'answer' in qa_data:
                        return {
                            "question": qa_data['question'].strip(),
                            "answer": qa_data['answer'].strip()
                        }
                except json.JSONDecodeError:
                    pass

            # Fallback: попытка парсинга старого формата
            lines = response_text.split('\n')
            question = None
            answer = None

            for line in lines:
                line = line.strip()
                if line.startswith('Суроо:'):
                    question = line.replace('Суроо:', '').strip()
                elif line.startswith('Жооп:'):
                    answer = line.replace('Жооп:', '').strip()
                    # Собираем многострочный ответ
                    idx = lines.index(line)
                    if idx < len(lines) - 1:
                        remaining_lines = lines[idx + 1:]
                        for remaining_line in remaining_lines:
                            remaining_line = remaining_line.strip()
                            if remaining_line and not remaining_line.startswith('Суроо:'):
                                answer += ' ' + remaining_line
                            else:
                                break

            if question and answer:
                return {
                    "question": question,
                    "answer": answer
                }
            else:
                logger.warning(f"Не удалось распарсить ответ: {response_text}")
                raise InvalidApiResponseError(
                    "Failed to parse QA response",
                    response_text=response_text
                )

        except Exception as e:
            logger.error(f"Ошибка парсинга ответа: {e}")
            return None

    def process_csv_file(self, csv_file_path: str, output_file_path: str,
                         start_from: int = 0, max_articles: Optional[int] = None):
        """Обработка CSV файла и генерация датасета Q&A"""

        if not os.path.exists(csv_file_path):
            logger.error(f"CSV файл не найден: {csv_file_path}")
            return

        # Инициализация API ключей
        if not self.load_and_configure_keys():
            logger.error("Не удалось инициализировать Gemini API")
            return

        logger.info(f"Загрузка CSV файла: {csv_file_path}")

        try:
            # Загрузка CSV файла
            df = pd.read_csv(csv_file_path)
            logger.info(f"Загружено {len(df)} статей")

            # Фильтрация данных
            df = df.dropna(subset=['Text'])
            df = df[df['Text'].str.len() > 50]

            # Загрузка существующих результатов
            qa_dataset = []
            processed_indices = set()

            if os.path.exists(output_file_path):
                try:
                    with open(output_file_path, 'r', encoding='utf-8') as f:
                        qa_dataset = json.load(f)

                    for item in qa_dataset:
                        if 'source_index' in item:
                            processed_indices.add(item['source_index'])

                    logger.info(f"Загружено {len(qa_dataset)} существующих Q&A пар")

                    # Автоматически определяем, с какой статьи продолжить
                    if processed_indices:
                        last_processed = max(processed_indices)
                        if start_from <= last_processed:
                            start_from = last_processed + 1
                            logger.info(f"🔄 Автоматическое возобновление с статьи {start_from}")

                except Exception as e:
                    logger.warning(f"Не удалось загрузить существующий файл: {e}")

            # Определяем диапазон обработки
            if max_articles:
                end_idx = min(start_from + max_articles, len(df))
            else:
                end_idx = len(df)

            articles_to_process = df.iloc[start_from:end_idx]
            logger.info(f"Будет обработано статей: {len(articles_to_process)} (с {start_from} по {end_idx - 1})")

            # Обработка статей
            successful_generations = 0
            failed_generations = 0

            with tqdm(total=len(articles_to_process), desc="Генерация Q&A") as pbar:
                for idx, row in articles_to_process.iterrows():
                    if self.requests_count >= MAX_REQUESTS_PER_RUN:
                        logger.warning(f"Достигнут лимит запросов ({MAX_REQUESTS_PER_RUN})")
                        break

                    # Пропускаем уже обработанные статьи
                    if idx in processed_indices:
                        pbar.set_description(f"Пропуск статьи {idx} (уже обработана)")
                        pbar.update(1)
                        continue

                    text = row['Text']
                    pbar.set_description(f"Обработка статьи {idx}")

                    # Генерация Q&A пары
                    qa_pair = self.generate_qa_from_text(text)

                    if qa_pair:
                        # Добавляем метаинформацию
                        qa_pair['source_index'] = int(idx)
                        qa_pair['source_text_length'] = len(text)
                        qa_pair['generated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')

                        qa_dataset.append(qa_pair)
                        successful_generations += 1

                        # Сохраняем промежуточные результаты
                        if successful_generations % 5 == 0:
                            self.save_dataset(qa_dataset, output_file_path)
                            logger.info(f"Промежуточное сохранение: {successful_generations} Q&A пар")
                    else:
                        failed_generations += 1
                        logger.warning(f"Не удалось сгенерировать Q&A для статьи {idx}")

                    pbar.update(1)

                    # Задержка между запросами
                    if self.requests_count < MAX_REQUESTS_PER_RUN:
                        time.sleep(REQUEST_DELAY_SECONDS)

            # Финальное сохранение
            self.save_dataset(qa_dataset, output_file_path)

            logger.info(f"✅ Обработка завершена!")
            logger.info(f"Успешно сгенерировано: {successful_generations} Q&A пар")
            logger.info(f"Ошибок: {failed_generations}")
            logger.info(f"Всего запросов к API: {self.requests_count}")
            logger.info(f"Всего в датасете: {len(qa_dataset)} Q&A пар")
            logger.info(f"Результаты сохранены в: {output_file_path}")

        except Exception as e:
            logger.error(f"Критическая ошибка при обработке файла: {e}")
            raise

    def save_dataset(self, dataset: List[Dict], output_file_path: str):
        """Сохранение датасета в JSON файл"""
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения файла: {e}")
            raise


def main():
    """Основная функция"""
    generator = WikipediaQAGenerator()

    # Пути к файлам
    csv_file = "kyrgyz_wikipedia_data.csv"
    output_file = "kyrgyz_wikipedia_qa_datasetGEMINI.jsonl"

    # Параметры обработки
    start_from = 7545  # Начать с какой статьи (для возобновления)
    max_articles = None  # Максимальное количество статей для обработки (None = ВСЕ статьи!)

    try:
        generator.process_csv_file(
            csv_file_path=csv_file,
            output_file_path=output_file,
            start_from=start_from,
            max_articles=max_articles
        )
    except KeyboardInterrupt:
        logger.info("Обработка прервана пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")


if __name__ == "__main__":
    main()
