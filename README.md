# panoramic

Опис работи скрипта.

---

## 🛠️ Можливості

- Створення чанків з фотографій
- Обчислення траєкорії руху склеювання фотографій/чанків
- Додавання синтетичної координати в кожен знімок

## ▶️ Використання
- Вказати path до фотографій в параметрі -img
- Вказати процент фотографій для аналізу -p

Команда для запуска створення чанків:
python main_v7.py

Команда для запуска обчислення ходу:
python main_v9.py

Файл config.py маэ бути сконфігурованим в залежності від даних для обробки. 

Це дефолтний сет параметрів:
descriptor:
  algo: "sift"
  max_num_keypoints: 256
  nfeatures: 2000
  device: "cpu"
  dist_value: 0.6
  trees: 5
  k: 2
  checks: 50
  good_matches_quantity: 30
  min_match_count: 5
  iter_value: 7.0


image:
  size_edje: 30000
  resize_coef: 0.7

mode:
  ortho_type: "way"
  type: "fast_mapping"
  step: 30
  chunk_size: 3
  overlap_size: 3
  chunk_ortho_side_size: 30000
  start: 0
  last: 750
  part_or_full: True
  reverse: False

