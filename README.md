# DriveSmartAI

DriveSmartAI — кросплатформений PWA-застосунок для класифікації дорожніх знаків на фотографіях. Проєкт розробляється в межах курсової роботи «Розробка моделей машинного навчання для задач розпізнавання образів».

## Мета

Розробити та порівняти дві моделі класифікації на датасеті GTSRB із 43 класами:

- власну CNN;
- донавчену MobileNetV3.

Підготовлені моделі експортуються в ONNX та використовуються backend-ом через ONNX Runtime Java.

## Планований стек

| Частина | Технології |
|---|---|
| Backend | Java 21 LTS, Spring Boot, Maven, ONNX Runtime Java |
| Frontend | React, TypeScript, Vite, Material UI, PWA |
| ML | Python 3.11, PyTorch |
| Дані | GTSRB |
| Зберігання | MySQL |
| Авторизація | Google OAuth 2.0 |

## Функції

- класифікація одного знака із завантаженого фото або камери;
- назва, впевненість, top-3 прогнози та пояснення;
- каталог 43 дорожніх знаків;
- навчальні тести;
- Google-авторизація;
- історія розпізнавань і прогрес;
- порівняння ML-моделей за фактичними експериментами;
- адаптивний PWA-інтерфейс.

YOLO, відео та детекція кількох знаків не входять до початкового обсягу.

## Структура репозиторію

```text
.
├── backend/       # Spring Boot backend (GET /api/health)
├── datasets/      # Локальні дані; не комітяться
├── docs/          # Деталі експериментів та технічні матеріали
├── frontend/      # React + TypeScript frontend skeleton
├── ml-training/   # PyTorch data preparation and future training scripts
├── AGENTS.md
├── ARCHITECTURE.md
├── PROJECT_SPEC.md
├── TASKS.md
├── README.md
└── .gitignore
```

## Поточний стан

Реалізовано мінімальний Spring Boot backend із `GET /api/health`, React + TypeScript frontend skeleton із health API integration та ML-модуль підготовки і навчання GTSRB.

Для фінального checkpoint Custom CNN додано ONNX-експорт, ONNX Runtime parity-перевірку
CPU та JSON metadata для майбутнього Java inference. Backend і Frontend у цей етап не змінювалися.

## Документація

- [Технічне завдання](./PROJECT_SPEC.md)
- [Архітектура та API](./ARCHITECTURE.md)
- [План робіт](./TASKS.md)
- [Правила для AI-агента та розробників](./AGENTS.md)
- [Інструкції ML-модуля](./ml-training/README.md)

## Експорт фінальної Custom CNN

Команди виконуються в PowerShell із кореня репозиторію:

```powershell
py -3.11 -m pip install -r ml-training\requirements.txt
py -3.11 -m pip install -e .\ml-training
Set-Location .\ml-training
py -3.11 -m pytest
py -3.11 -m drivesmart_ml.cli export-cnn-onnx --checkpoint outputs\custom-cnn\best_cnn.pth --output outputs\custom-cnn\best_cnn.onnx --parity-tensors 4 --seed 42
```

## Запуск Backend із ONNX inference

Команди виконуються з `C:\Projects\DriveSmartAI\backend`:

```powershell
.\mvnw.cmd test
.\mvnw.cmd spring-boot:run
Invoke-RestMethod -Uri "http://localhost:8080/api/health" -Method Get
```

Застосунок завантажує ONNX-модель один раз при старті. За замовчуванням
використовується локальний шлях
`..\ml-training\outputs\custom-cnn\best_cnn.onnx`.

Для іншого розташування можна задати environment variable:

```powershell
$env:DRIVESMART_ONNX_MODEL_PATH = "C:\models\drivesmart\best_cnn.onnx"
$env:DRIVESMART_ONNX_MODEL_SHA256 = "<SHA-256 ONNX-моделі>"
.\mvnw.cmd spring-boot:run
```

Перевірка розпізнавання локального GTSRB-зображення:

```powershell
curl.exe -s -F "file=@src\test\resources\gtsrb-00000.png;type=image/png" http://localhost:8080/api/predict
```

## Принципи

- ML-результати публікуються лише після реального відтворюваного експерименту.
- Дані, моделі та секрети не зберігаються в Git.
- MVP обмежений класифікацією одного знака.
- Якість, безпека, тестованість і відтворюваність важливіші за додавання зайвих технологій.
