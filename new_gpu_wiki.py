#!/usr/bin/env python3
"""
Скрипт для генерации вопросов-ответов из статей Wikipedia с помощью Gemma3:27 через Ollama
"""

import pandas as pd
import json
import requests
import time
import logging
from typing import Dict, List, Optional
import os
from tqdm import tqdm

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


class WikipediaQAGenerator:
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "gemma3:27b"):
        self.ollama_url = ollama_url
        self.model = model
        self.session = requests.Session()

    def test_ollama_connection(self) -> bool:
        """Проверка соединения с Ollama"""
        try:
            response = self.session.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                if self.model in model_names:
                    logger.info(f"✅ Подключение к Ollama успешно. Модель {self.model} доступна.")
                    return True
                else:
                    logger.error(f"❌ Модель {self.model} не найдена. Доступные модели: {model_names}")
                    return False
            else:
                logger.error(f"❌ Ошибка подключения к Ollama: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Не удается подключиться к Ollama: {e}")
            return False

    def generate_qa_from_text(self, text: str) -> Optional[Dict[str, str]]:
        """Генерация вопроса и ответа из текста статьи"""
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
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 8000
                }
            }

            response = self.session.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=300
            )

            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()

                # Парсинг ответа
                qa_pair = self.parse_qa_response(generated_text)
                return qa_pair
            else:
                logger.error(f"Ошибка API Ollama: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.Timeout:
            logger.error("Тайм-аут при запросе к Ollama")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка запроса к Ollama: {e}")
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}")
            return None

    def parse_qa_response(self, response_text: str) -> Optional[Dict[str, str]]:
        """Парсинг JSON ответа модели для извлечения вопроса и ответа"""
        try:
            # Очистка ответа от возможных лишних символов
            cleaned_response = response_text.strip()

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

            # Fallback: попытка парсинга старого формата (на случай если модель не следует новому формату)
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
                return None

        except Exception as e:
            logger.error(f"Ошибка парсинга ответа: {e}")
            return None

    def process_csv_file(self, csv_file_path: str, output_file_path: str,
                         start_from: int = 0, max_articles: Optional[int] = None):
        """Обработка CSV файла и генерация датасета Q&A"""

        if not os.path.exists(csv_file_path):
            logger.error(f"CSV файл не найден: {csv_file_path}")
            return

        # Проверка подключения к Ollama
        if not self.test_ollama_connection():
            return

        logger.info(f"Загрузка CSV файла: {csv_file_path}")

        try:
            # Загрузка CSV файла
            df = pd.read_csv(csv_file_path)
            logger.info(f"Загружено {len(df)} статей")

            # Фильтрация данных
            df = df.dropna(subset=['Text'])  # Удаляем пустые статьи
            df = df[df['Text'].str.len() > 50]  # Оставляем только статьи длиннее 50 символов

            # Загрузка существующих результатов если файл существует
            qa_dataset = []
            processed_indices = set()

            if os.path.exists(output_file_path):
                try:
                    with open(output_file_path, 'r', encoding='utf-8') as f:
                        qa_dataset = json.load(f)

                    # Собираем индексы уже обработанных статей
                    for item in qa_dataset:
                        if 'source_index' in item:
                            processed_indices.add(item['source_index'])

                    logger.info(f"Загружено {len(qa_dataset)} существующих Q&A пар")
                    logger.info(f"Уже обработано статей с индексами: {sorted(list(processed_indices))}")

                    # Автоматически определяем, с какой статьи продолжить
                    if processed_indices:
                        last_processed = max(processed_indices)
                        if start_from <= last_processed:
                            start_from = last_processed + 1
                            logger.info(f"🔄 Автоматическое возобновление с статьи {start_from}")

                except Exception as e:
                    logger.warning(f"Не удалось загрузить существующий файл: {e}")

            # Определяем диапазон обработки после загрузки существующих результатов
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

                        # Сохраняем промежуточные результаты каждую статью (для надежности)
                        if successful_generations % 1 == 0:
                            self.save_dataset(qa_dataset, output_file_path)
                            logger.info(f"Промежуточное сохранение: {successful_generations} Q&A пар")
                    else:
                        failed_generations += 1
                        logger.warning(f"Не удалось сгенерировать Q&A для статьи {idx}")

                    pbar.update(1)

                    # Небольшая задержка между запросами
                    time.sleep(0.5)

            # Финальное сохранение
            self.save_dataset(qa_dataset, output_file_path)

            logger.info(f"✅ Обработка завершена!")
            logger.info(f"Успешно сгенерировано: {successful_generations} Q&A пар")
            logger.info(f"Ошибок: {failed_generations}")
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
    output_file = "kyrgyz_wikipedia_qa_datasetGPU.json"

    # Параметры обработки
    start_from = 80000  # Начать с 70,000-й статьи
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