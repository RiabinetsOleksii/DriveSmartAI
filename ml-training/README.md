# DriveSmartAI ML preparation

Цей модуль готує GTSRB і навчає першу власну CNN; офіційний test набір
залишається зарезервованим для фінального оцінювання.
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

Навчання власної CNN (за замовчуванням 15 епох, CUDA з CPU fallback,
batch size 16, mixed precision на CUDA):

```powershell
python -m drivesmart_ml.cli train --root ..\datasets\gtsrb --split-manifest outputs\split.csv --output outputs\custom-cnn --epochs 15 --batch-size 16 --seed 42
```

Для пробного запуску використовуйте `--epochs 1`. У вихідній папці створюються
`best_cnn.pth`, `history.csv`, `training_curves.png` і `config.json`.

Навчання MobileNetV3-Small із ImageNet transfer learning виконується окремо:

```powershell
python -m drivesmart_ml.cli train-mobilenet --root ..\datasets\gtsrb --split-manifest outputs\split.csv --output outputs\mobilenetv3 --image-size 64 --epochs 15 --classifier-epochs 2 --fine-tune-blocks 3 --batch-size 16 --seed 42
```

Перші `classifier-epochs` епох навчають лише classifier, після чого
розморожуються останні `fine-tune-blocks` блоки backbone. Результати MobileNet
зберігаються окремо: `best_mobilenetv3.pth`, `history.csv`,
`training_curves.png` та `config.json`. Для проби використовуйте
`--epochs 1 --classifier-epochs 0`.

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

Фінальне незалежне оцінювання двох checkpoint-ів на офіційному GTSRB test:

```powershell
python -m drivesmart_ml.cli evaluate-models --root ..\datasets\gtsrb --custom-checkpoint outputs\custom-cnn\best_cnn.pth --mobilenet-checkpoint outputs\mobilenetv3-128-full\best_mobilenetv3.pth --output outputs\evaluation --batch-size 8
```

Команда створює aggregate/per-class metrics, CSV/JSON comparison,
confusion matrix CSV/PNG і вимірювання GPU inference latency.

## Експорт Custom CNN у ONNX

Після встановлення залежностей експорт і перевірка виконуються з
`C:\Projects\DriveSmartAI\ml-training`:

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 -m pip install -e .
py -3.11 -m drivesmart_ml.cli export-cnn-onnx --checkpoint outputs\custom-cnn\best_cnn.pth --output outputs\custom-cnn\best_cnn.onnx --parity-tensors 4 --seed 42
```

Команда перевіряє checksum/checkpoint architecture, експортує динамічну batch
вісь, перевіряє ONNX model/shape та порівнює PyTorch і ONNX Runtime CPU inference
на детермінованих тестових тензорах.

Створюються:

- `outputs/custom-cnn/best_cnn.onnx` — модель із входом `[batch, 3, 64, 64]`
  і виходом `[batch, 43]`;
- `outputs/custom-cnn/best_cnn_metadata.json` — preprocessing, class mapping,
  імена tensor-ів, opset, checksum-и та parity;
- `outputs/custom-cnn/best_cnn_parity.json` — числове порівняння PyTorch/ONNX.

Автоматична перевірка ONNX включена в unit-тести:

```powershell
py -3.11 -m pytest
```
