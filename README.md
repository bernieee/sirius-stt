# DeepSpeech2

В данном репозитории содержится реализация модели DeepSpeech2 с использованием фреймворка pytorch.

Репозиторий организован следующим образом:
* ``src`` -- модули с реализацией компонент модели
    * ``src/datasets`` -- реализация датасета для аудиофайлов
    * ``src/audio_utils`` -- аугментации аудио и спектрограмм
    * ``src/decoding`` -- beam search и greedy декодер
    * ``src/deepspeech`` -- реализация модели DeepSpeech2
    * ``src/optimization`` -- метрики и training loop
    * ``src/logging_my`` -- логгер процесса обучения
    * ``src/inference`` -- api для инференса модели
    
* ``examples`` -- примеры запуска обучения и инференса
    * ``examples/scripts`` -- примеры скриптов обучения
    * ``examples/notebooks`` -- эксперименты с моделью и обучение из jupyter notebook

* ``tests`` -- тесты основных для компонент модели


Для использования модели реализован удобный интерфейс:
```python
from src.inference import InferenceModel

model = InferenceModel(checkpoint_path='/path/to/checkpoint')
results = model.run(audio_path='/path/to/you/audio/file')
# print(results)
# >> "Ваш отлично распознанный голос"
```
