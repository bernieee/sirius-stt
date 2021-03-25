# DeepSpeech2

В данном репозитории содержится реализация модели [DeepSpeech2](#1) с использованием фреймворка pytorch.

## Описание структуры репозитория 

Репозиторий организован следующим образом:
* ``src`` -- модули с реализацией компонент модели
    * ``src/datasets`` -- реализация датасета для аудиофайлов
    * ``src/audio_utils`` -- аугментации аудио и спектрограмм
    * ``src/decoding`` -- beam search и greedy декодер
    * ``src/deepspeech`` -- реализация модели DeepSpeech2
    * ``src/optimization`` -- метрики и training loop
    * ``src/logging_my`` -- логгер процесса обучения
    * ``src/inference`` -- api для инференса модели
* ``tests`` -- тесты основных для компонент модели   
* ``examples`` -- примеры запуска обучения и инференса
    * ``examples/scripts`` -- примеры скриптов обучения
    * ``examples/notebooks`` -- эксперименты с моделью и обучение из jupyter notebook
* ``presentation`` -- презентация с защиты проекта

## Использование модели

Для использования модели реализован удобный интерфейс:
```python
from src.inference import InferenceModel

model = InferenceModel(checkpoint_path='/path/to/checkpoint')
results = model.run(audio_path='/path/to/you/audio/file')
print(results)
# >> "Ваш отлично распознанный голос"
```

## Установка зависимостей

Для запуска проекта необходимо установить зависимости:
```shell
sh setup.sh
pip install -r requirements.txt
```

## Ссылки

На основе данного репозитория реализован [telegram бот](#2).

<a id="1">[1]</a> Deep Speech 2: End-to-End Speech Recognition in English and Mandarin. https://arxiv.org/pdf/1512.02595.pdf

<a id="1">[2]</a> https://github.com/Polly42Rose/sirius-stt-bot