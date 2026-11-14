# DriveSmartAI ML preparation

Цей модуль готує GTSRB для майбутнього навчання власної CNN і MobileNetV3.
Навчання моделей, ONNX та числові результати свідомо не входять до цього етапу.
Датасет і згенеровані звіти зберігаються локально та ігноруються Git.

## Windows PowerShell

Команди виконуються з `C:\Projects\DriveSmartAI\ml-training`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Завантаження офіційних train і test наборів GTSRB виконується окремо:

```powershell
python -m drivesmart_ml.cli download --root ..\datasets\gtsrb
```

Аудит файлів, форматів, розмірів, пошкоджених зображень і розподілу класів:

```powershell
python -m drivesmart_ml.cli audit --root ..\datasets\gtsrb --output outputs\audit
```

Перевірка DataLoader на першому train batch (на Windows за замовчуванням `num_workers=0`):

```powershell
python -m drivesmart_ml.cli smoke --root ..\datasets\gtsrb --batch-size 8 --num-workers 0
```

Створення group-aware train/validation manifest із training набору:

```powershell
python -m drivesmart_ml.cli split --root ..\datasets\gtsrb --output outputs\split.csv --seed 42
```

Запуск синтетичних unit-тестів без завантаження GTSRB:

```powershell
python -m pytest
```

## Контракти та структура

- `src/drivesmart_ml/config.py` — типізована конфігурація та перевірки.
- `src/drivesmart_ml/data.py` — офіційний `torchvision.datasets.GTSRB` і train/validation/test DataLoader.
- `src/drivesmart_ml/audit.py` — аудит зображень і пошкоджених файлів.
- `src/drivesmart_ml/split.py` — відтворюваний split за групами track, окремо для кожного класу.
- `src/drivesmart_ml/preprocessing.py` — спільний RGB/resize/normalization tensor contract.
- `src/drivesmart_ml/contracts/` — JSON для майбутнього Java/ONNX споживача.
- `tests/` — перевірки mapping, preprocessing, конфігурації та відсутності leakage.

Офіційний test набір не використовується для train/validation split і має бути
зарезервований для фінального оцінювання після реалізації моделей.
