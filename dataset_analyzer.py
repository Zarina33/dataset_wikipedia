import json
import re
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import statistics
import matplotlib.pyplot as plt
import seaborn as sns


class KyrgyzQAAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = []
        self.analysis_results = {}

    def load_data(self):
        """Загружает данные из JSON файла"""
        print("📥 Загрузка данных...")

        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                # Определяем формат файла
                first_char = f.read(1)
                f.seek(0)

                if first_char == '[':
                    # JSON массив
                    self.data = json.load(f)
                else:
                    # JSONL формат
                    for line in f:
                        line = line.strip()
                        if line:
                            self.data.append(json.loads(line))

            print(f"✅ Загружено {len(self.data):,} Q&A пар")
            return True

        except Exception as e:
            print(f"❌ Ошибка при загрузке: {e}")
            return False

    def basic_statistics(self):
        """Базовая статистика о Q&A датасете"""
        print("\n📊 БАЗОВАЯ СТАТИСТИКА")
        print("=" * 60)

        total_pairs = len(self.data)
        print(f"Общее количество Q&A пар: {total_pairs:,}")

        # Проверка уникальности source_index
        source_indices = [item.get('source_index') for item in self.data if 'source_index' in item]
        if source_indices:
            unique_indices = len(set(source_indices))
            duplicates = len(source_indices) - unique_indices
            print(f"Уникальных source_index: {unique_indices:,}")
            if duplicates > 0:
                print(f"⚠️  Дубликаты source_index: {duplicates:,}")

        # Анализ длины исходного текста
        text_lengths = [item.get('source_text_length', 0) for item in self.data if 'source_text_length' in item]
        if text_lengths:
            print(f"Средняя длина исходного текста: {statistics.mean(text_lengths):.0f} символов")
            print(f"Медианная длина исходного текста: {statistics.median(text_lengths):.0f} символов")
            print(f"Минимальная длина: {min(text_lengths)} символов")
            print(f"Максимальная длина: {max(text_lengths):,} символов")

        self.analysis_results['basic'] = {
            'total_pairs': total_pairs,
            'unique_source_indices': len(set(source_indices)) if source_indices else None,
            'duplicate_indices': len(source_indices) - len(set(source_indices)) if source_indices else None,
            'avg_source_length': statistics.mean(text_lengths) if text_lengths else None,
            'median_source_length': statistics.median(text_lengths) if text_lengths else None,
            'min_source_length': min(text_lengths) if text_lengths else None,
            'max_source_length': max(text_lengths) if text_lengths else None
        }

    def analyze_questions(self):
        """Анализ вопросов"""
        print("\n❓ АНАЛИЗ ВОПРОСОВ")
        print("=" * 60)

        questions = [item.get('question', '') for item in self.data if 'question' in item]

        if not questions:
            print("Вопросы не найдены")
            return

        # Длина вопросов
        question_lengths = [len(q) for q in questions]
        word_counts = [len(q.split()) for q in questions]

        print(f"Средняя длина вопроса: {statistics.mean(question_lengths):.0f} символов")
        print(f"Медианная длина вопроса: {statistics.median(question_lengths):.0f} символов")
        print(f"Самый короткий вопрос: {min(question_lengths)} символов")
        print(f"Самый длинный вопрос: {max(question_lengths)} символов")

        print(f"\nСреднее количество слов в вопросе: {statistics.mean(word_counts):.1f}")
        print(f"Медианное количество слов: {statistics.median(word_counts):.1f}")

        # Анализ вопросительных слов
        question_words = []
        question_patterns = {
            'эмне': 'что',
            'кайда': 'где',
            'качан': 'когда',
            'кандай': 'какой',
            'канча': 'сколько',
            'ким': 'кто',
            'эмнеге': 'почему',
            'кантип': 'как',
            'жөнүндө': 'о чем',
            'деген': 'что такое'
        }

        pattern_counts = Counter()
        for question in questions:
            question_lower = question.lower()
            for pattern, meaning in question_patterns.items():
                if pattern in question_lower:
                    pattern_counts[f"{pattern} ({meaning})"] += 1

        print(f"\nТоп-10 вопросительных конструкций:")
        for pattern, count in pattern_counts.most_common(10):
            percentage = (count / len(questions)) * 100
            print(f"  {pattern}: {count:,} ({percentage:.1f}%)")

        # Примеры коротких и длинных вопросов
        questions_with_length = [(q, len(q)) for q in questions]
        questions_with_length.sort(key=lambda x: x[1])

        print(f"\n📝 Примеры самых коротких вопросов:")
        for q, length in questions_with_length[:3]:
            print(f"  ({length} симв.): {q}")

        print(f"\n📝 Примеры самых длинных вопросов:")
        for q, length in questions_with_length[-3:]:
            print(f"  ({length} симв.): {q[:100]}{'...' if len(q) > 100 else ''}")

        self.analysis_results['questions'] = {
            'avg_length': statistics.mean(question_lengths),
            'median_length': statistics.median(question_lengths),
            'min_length': min(question_lengths),
            'max_length': max(question_lengths),
            'avg_words': statistics.mean(word_counts),
            'median_words': statistics.median(word_counts),
            'question_patterns': dict(pattern_counts)
        }

    def analyze_answers(self):
        """Анализ ответов"""
        print("\n💬 АНАЛИЗ ОТВЕТОВ")
        print("=" * 60)

        answers = [item.get('answer', '') for item in self.data if 'answer' in item]

        if not answers:
            print("Ответы не найдены")
            return

        # Длина ответов
        answer_lengths = [len(a) for a in answers]
        word_counts = [len(a.split()) for a in answers]

        print(f"Средняя длина ответа: {statistics.mean(answer_lengths):.0f} символов")
        print(f"Медианная длина ответа: {statistics.median(answer_lengths):.0f} символов")
        print(f"Самый короткий ответ: {min(answer_lengths)} символов")
        print(f"Самый длинный ответ: {max(answer_lengths)} символов")

        print(f"\nСреднее количество слов в ответе: {statistics.mean(word_counts):.1f}")
        print(f"Медианное количество слов: {statistics.median(word_counts):.1f}")

        # Анализ содержания ответов
        sentences_per_answer = []
        for answer in answers:
            # Подсчет предложений (по точкам)
            sentences = len([s for s in answer.split('.') if s.strip()])
            sentences_per_answer.append(sentences)

        print(f"\nСреднее количество предложений в ответе: {statistics.mean(sentences_per_answer):.1f}")

        # Распределение по длине ответов
        length_ranges = {
            'Очень короткие (≤50 симв.)': sum(1 for l in answer_lengths if l <= 50),
            'Короткие (51-150 симв.)': sum(1 for l in answer_lengths if 50 < l <= 150),
            'Средние (151-300 симв.)': sum(1 for l in answer_lengths if 150 < l <= 300),
            'Длинные (301-500 симв.)': sum(1 for l in answer_lengths if 300 < l <= 500),
            'Очень длинные (>500 симв.)': sum(1 for l in answer_lengths if l > 500)
        }

        print(f"\nРаспределение ответов по длине:")
        for range_name, count in length_ranges.items():
            percentage = (count / len(answers)) * 100
            print(f"  {range_name}: {count:,} ({percentage:.1f}%)")

        # Примеры ответов
        answers_with_length = [(a, len(a)) for a in answers]
        answers_with_length.sort(key=lambda x: x[1])

        print(f"\n📝 Примеры коротких ответов:")
        for a, length in answers_with_length[:3]:
            print(f"  ({length} симв.): {a}")

        print(f"\n📝 Примеры длинных ответов:")
        for a, length in answers_with_length[-3:]:
            print(f"  ({length} симв.): {a[:150]}{'...' if len(a) > 150 else ''}")

        self.analysis_results['answers'] = {
            'avg_length': statistics.mean(answer_lengths),
            'median_length': statistics.median(answer_lengths),
            'min_length': min(answer_lengths),
            'max_length': max(answer_lengths),
            'avg_words': statistics.mean(word_counts),
            'median_words': statistics.median(word_counts),
            'avg_sentences': statistics.mean(sentences_per_answer),
            'length_distribution': length_ranges
        }

    def analyze_temporal_patterns(self):
        """Анализ временных паттернов генерации"""
        print("\n🕒 АНАЛИЗ ВРЕМЕННЫХ ПАТТЕРНОВ")
        print("=" * 60)

        timestamps = []
        for item in self.data:
            if 'generated_at' in item:
                try:
                    timestamp = datetime.strptime(item['generated_at'], '%Y-%m-%d %H:%M:%S')
                    timestamps.append(timestamp)
                except:
                    pass

        if not timestamps:
            print("Временные метки не найдены")
            return

        timestamps.sort()

        print(f"Период генерации: {timestamps[0]} - {timestamps[-1]}")
        print(f"Общая продолжительность: {timestamps[-1] - timestamps[0]}")

        # Анализ по дням
        days = defaultdict(int)
        hours = defaultdict(int)

        for ts in timestamps:
            days[ts.date()] += 1
            hours[ts.hour] += 1

        print(f"\nДней генерации: {len(days)}")
        print(f"Среднее количество Q&A в день: {len(timestamps) / len(days):.1f}")

        # Топ дни по активности
        top_days = sorted(days.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nТоп-5 дней по количеству генераций:")
        for date, count in top_days:
            print(f"  {date}: {count:,} Q&A пар")

        # Распределение по часам
        print(f"\nРаспределение по часам дня:")
        for hour in sorted(hours.keys()):
            count = hours[hour]
            percentage = (count / len(timestamps)) * 100
            print(f"  {hour:02d}:00: {count:,} ({percentage:.1f}%)")

        self.analysis_results['temporal'] = {
            'start_date': timestamps[0].isoformat(),
            'end_date': timestamps[-1].isoformat(),
            'total_days': len(days),
            'avg_per_day': len(timestamps) / len(days),
            'top_days': [(str(date), count) for date, count in top_days],
            'hourly_distribution': dict(hours)
        }

    def analyze_topics_and_entities(self):
        """Анализ тем и сущностей"""
        print("\n🏷️ АНАЛИЗ ТЕМ И СУЩНОСТЕЙ")
        print("=" * 60)

        all_text = ""
        for item in self.data:
            question = item.get('question', '')
            answer = item.get('answer', '')
            all_text += f"{question} {answer} "

        # Поиск географических названий (заканчиваются на характерные суффиксы)
        geo_patterns = [
            r'\b\w+стан\b',  # страны на -стан
            r'\b\w+ия\b',  # страны на -ия
            r'\b\w+ль\b',  # города типа "шаар"
            r'\b\w+орд\b',  # исторические места
        ]

        geographic_entities = set()
        for pattern in geo_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            geographic_entities.update(matches)

        # Поиск чисел и дат
        numbers = re.findall(r'\b\d+\b', all_text)
        years = re.findall(r'\b(19|20)\d{2}\b', all_text)

        # Наиболее частые слова
        words = re.findall(r'\b\w{3,}\b', all_text.lower())
        word_counts = Counter(words)

        # Исключаем служебные слова
        stop_words = {'жана', 'үчүн', 'менен', 'болгон', 'эмне', 'болуп', 'кылып',
                      'деп', 'башка', 'ошол', 'анын', 'бирок', 'дагы'}

        content_words = [(word, count) for word, count in word_counts.items()
                         if word not in stop_words and len(word) > 3]
        content_words.sort(key=lambda x: x[1], reverse=True)

        print(f"Найдено географических названий: {len(geographic_entities)}")
        print(f"Найдено чисел: {len(set(numbers))}")
        print(f"Найдено годов: {len(set(years))}")

        print(f"\nТоп-20 наиболее частых содержательных слов:")
        for word, count in content_words[:20]:
            print(f"  {word}: {count:,}")

        print(f"\nПримеры географических названий:")
        for entity in sorted(geographic_entities)[:10]:
            print(f"  {entity}")

        self.analysis_results['topics'] = {
            'geographic_entities': len(geographic_entities),
            'unique_numbers': len(set(numbers)),
            'unique_years': len(set(years)),
            'top_words': content_words[:50],
            'sample_geo_entities': list(sorted(geographic_entities)[:20])
        }

    def analyze_qa_relationship(self):
        """Анализ соотношения длины вопросов и ответов"""
        print("\n🔗 АНАЛИЗ СООТНОШЕНИЯ ВОПРОС-ОТВЕТ")
        print("=" * 60)

        qa_pairs = []
        for item in self.data:
            if 'question' in item and 'answer' in item:
                q_len = len(item['question'])
                a_len = len(item['answer'])
                qa_pairs.append((q_len, a_len))

        if not qa_pairs:
            print("Q&A пары не найдены")
            return

        q_lengths = [pair[0] for pair in qa_pairs]
        a_lengths = [pair[1] for pair in qa_pairs]

        # Корреляция длин
        if len(qa_pairs) > 1:
            correlation = statistics.correlation(q_lengths, a_lengths)
            print(f"Корреляция длины вопроса и ответа: {correlation:.3f}")

        # Соотношение длин
        ratios = [a_len / q_len if q_len > 0 else 0 for q_len, a_len in qa_pairs]
        avg_ratio = statistics.mean(ratios)

        print(f"Среднее соотношение длины ответа к вопросу: {avg_ratio:.2f}")
        print(f"Медианное соотношение: {statistics.median(ratios):.2f}")

        # Категории по соотношению
        ratio_categories = {
            'Ответ короче вопроса (<1)': sum(1 for r in ratios if r < 1),
            'Ответ примерно равен (1-2)': sum(1 for r in ratios if 1 <= r < 2),
            'Ответ в 2-3 раза длиннее': sum(1 for r in ratios if 2 <= r < 3),
            'Ответ в 3-5 раз длиннее': sum(1 for r in ratios if 3 <= r < 5),
            'Ответ в >5 раз длиннее': sum(1 for r in ratios if r >= 5)
        }

        print(f"\nРаспределение по соотношению длин:")
        for category, count in ratio_categories.items():
            percentage = (count / len(ratios)) * 100
            print(f"  {category}: {count:,} ({percentage:.1f}%)")

        self.analysis_results['qa_relationship'] = {
            'correlation': correlation if len(qa_pairs) > 1 else None,
            'avg_ratio': avg_ratio,
            'median_ratio': statistics.median(ratios),
            'ratio_distribution': ratio_categories
        }

    def estimate_tokens(self):
        """Оценка количества токенов"""
        print("\n🔢 ОЦЕНКА ТОКЕНОВ")
        print("=" * 60)

        total_chars = 0
        total_words = 0

        for item in self.data:
            question = item.get('question', '')
            answer = item.get('answer', '')

            total_chars += len(question) + len(answer)
            total_words += len(question.split()) + len(answer.split())

        # Разные методы оценки токенов для кыргызского языка
        tokens_by_chars = total_chars // 3  # ~3 символа = 1 токен для кириллицы
        tokens_by_words = int(total_words * 1.3)  # ~1.3 токена на слово для агглютинативных языков

        print(f"Общее количество символов: {total_chars:,}")
        print(f"Общее количество слов: {total_words:,}")
        print(f"Оценка токенов (по символам): {tokens_by_chars:,}")
        print(f"Оценка токенов (по словам): {tokens_by_words:,}")
        print(f"Средняя оценка токенов: {(tokens_by_chars + tokens_by_words) // 2:,}")

        # Токены на одну Q&A пару
        avg_tokens_per_pair = (tokens_by_chars + tokens_by_words) // 2 // len(self.data)
        print(f"Среднее количество токенов на Q&A пару: {avg_tokens_per_pair}")

        self.analysis_results['tokens'] = {
            'total_chars': total_chars,
            'total_words': total_words,
            'estimated_tokens_chars': tokens_by_chars,
            'estimated_tokens_words': tokens_by_words,
            'avg_estimated_tokens': (tokens_by_chars + tokens_by_words) // 2,
            'avg_tokens_per_pair': avg_tokens_per_pair
        }

    def generate_summary_report(self):
        """Генерирует итоговый отчет"""
        print("\n" + "=" * 80)
        print("📋 ИТОГОВЫЙ ОТЧЕТ ПО КЫРГЫЗСКОМУ Q&A ДАТАСЕТУ")
        print("=" * 80)

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Дата анализа: {current_time}")
        print(f"Файл данных: {os.path.basename(self.file_path)}")
        print(f"Размер файла: {os.path.getsize(self.file_path):,} байт")

        # Сохраняем детальный отчет в JSON
        report_file = f"kyrgyz_qa_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.analysis_results['metadata'] = {
            'analysis_date': current_time,
            'source_file': self.file_path,
            'file_size_bytes': os.path.getsize(self.file_path),
            'analyzer_version': '1.0'
        }

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, ensure_ascii=False, indent=2)

        print(f"\n📄 Подробная статистика сохранена в: {report_file}")

        # Краткая сводка
        basic = self.analysis_results.get('basic', {})
        questions = self.analysis_results.get('questions', {})
        answers = self.analysis_results.get('answers', {})
        tokens = self.analysis_results.get('tokens', {})
        temporal = self.analysis_results.get('temporal', {})

        print(f"\n📊 КРАТКАЯ СВОДКА:")
        print(f"  🔢 Всего Q&A пар: {basic.get('total_pairs', 0):,}")
        print(f"  ❓ Средняя длина вопроса: {questions.get('avg_length', 0):.0f} символов")
        print(f"  💬 Средняя длина ответа: {answers.get('avg_length', 0):.0f} символов")
        print(f"  🔢 Примерное количество токенов: {tokens.get('avg_estimated_tokens', 0):,}")

        if temporal:
            print(f"  📅 Период генерации: {temporal.get('total_days', 0)} дней")
            print(f"  ⚡ Среднее Q&A в день: {temporal.get('avg_per_day', 0):.1f}")

        print(f"\n🎯 РЕКОМЕНДАЦИИ ДЛЯ ИСПОЛЬЗОВАНИЯ:")
        avg_length = basic.get('total_pairs', 0)
        if avg_length > 50000:
            print("  • Датасет достаточно большой для обучения языковых моделей")
        if questions.get('avg_length', 0) > 50:
            print("  • Вопросы достаточно детализированы")
        if answers.get('avg_length', 0) > 100:
            print("  • Ответы содержат развернутую информацию")

        print("\n✅ Анализ завершен!")

    def run_full_analysis(self):
        """Запускает полный анализ Q&A датасета"""
        if not self.load_data():
            return

        self.basic_statistics()
        self.analyze_questions()
        self.analyze_answers()
        self.analyze_temporal_patterns()
        self.analyze_topics_and_entities()
        self.analyze_qa_relationship()
        self.estimate_tokens()
        self.generate_summary_report()


# Пример использования
if __name__ == "__main__":
    # Укажите путь к вашему JSON файлу
    file_path = "/home/zarina/Рабочий стол/data_gemma/merged_array.json"

    if not file_path:
        # Ищем файлы автоматически
        json_files = [f for f in os.listdir('.') if f.endswith(('.json', '.jsonl')) and 'kyrgyz' in f.lower()]
        if json_files:
            file_path = json_files[0]
            print(f"Найден файл: {file_path}")
        else:
            print("Доступные JSON файлы:")
            for f in os.listdir('.'):
                if f.endswith(('.json', '.jsonl')):
                    print(f"  - {f}")
            exit()

    if not os.path.exists(file_path):
        print(f"❌ Файл {file_path} не найден!")
    else:
        analyzer = KyrgyzQAAnalyzer(file_path)
        analyzer.run_full_analysis()