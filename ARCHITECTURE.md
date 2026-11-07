# DriveSmartAI — архітектура

## 1. Архітектурний стиль

Проєкт поділяється на три логічні частини в одному репозиторії:

```text
frontend (PWA) -> backend (REST API, auth, persistence, inference) -> MySQL
                                      |
                                      +-> ONNX Runtime Java -> ONNX model artifacts
ml-training (experiments, training, evaluation) ----------------^
```

ML-навчання відбувається окремо від runtime backend. Backend не навчає моделі: він завантажує затверджений ONNX-артефакт і виконує інференс.

## 2. Компоненти

### `frontend/`

React + TypeScript + Vite + Material UI + PWA tooling.

- **App shell:** маршрутизація, layout, тема, адаптивна навігація.
- **Recognition UI:** вибір файлу, камера, preview, валідація, стани запиту.
- **Results:** основний прогноз, top-3, confidence, пояснення.
- **Catalog:** довідник 43 класів.
- **Tests:** навчальні питання, результат спроби.
- **History/Progress:** дані авторизованого користувача.
- **Model comparison:** таблиці та графіки фактичних метрик.
- **API client:** типізовані HTTP-запити й обробка помилок.

### `backend/`

Java 25 LTS + Spring Boot + Maven.

- **Web/API layer:** REST controllers, DTO, validation, error responses.
- **Application layer:** use cases розпізнавання, каталогу, тестів, історії та порівняння.
- **Inference adapter:** preprocessing, ONNX Runtime Java session, output mapping, top-k.
- **Model registry:** конфігурація активних моделей, версій і preprocessing contract.
- **Security:** Google OAuth 2.0, session/token policy, authorization rules.
- **Persistence:** JPA/repositories для користувачів, результатів, тестів і прогресу.
- **Infrastructure:** MySQL, migrations, logging, configuration profiles.

Рекомендована внутрішня структура: `controller -> service/use case -> repository/adapter`; controller не містить ML або SQL-логіки.

### `ml-training/`

Python 3.11 + PyTorch.

- підготовка та аудит GTSRB;
- спільний preprocessing і class mapping;
- реалізація власної CNN;
- transfer learning/fine-tuning MobileNetV3;
- train/evaluate scripts;
- збереження checkpoints;
- експорт у ONNX;
- звіти експериментів і перевірка parity PyTorch/ONNX.

### `datasets/`

Локальне або зовнішнє розташування GTSRB та похідних індексів. Дані не комітяться; у Git дозволяються лише README/маніфести метаданих за потреби.

### `docs/`

Деталі експериментів, preprocessing contract, API приклади, рішення та інструкції відтворення. Фактичні ML-результати додаються тільки після запуску експериментів.

## 3. Потік розпізнавання

1. Користувач завантажує файл або робить фото у PWA.
2. Frontend перевіряє розширення/MIME/розмір і надсилає `multipart/form-data`.
3. Backend перевіряє файл повторно та нормалізує зображення відповідно до contract моделі.
4. Inference adapter виконує ONNX Runtime inference.
5. Output mapper перетворює logits/probabilities на class IDs, назви та top-3.
6. Backend повертає типізований JSON і, за наявності користувача, зберігає metadata результату.
7. Frontend показує результат або структуровану помилку.

## 4. Плановані API

Базовий префікс: `/api/v1`.

| Метод | Endpoint | Призначення |
|---|---|---|
| `POST` | `/recognitions` | Класифікувати одне зображення |
| `GET` | `/recognitions/{id}` | Отримати збережений результат |
| `GET` | `/recognitions/history` | Історія поточного користувача |
| `GET` | `/signs` | Каталог із фільтрами/пагінацією |
| `GET` | `/signs/{classId}` | Деталі класу знака |
| `GET` | `/tests` | Отримати доступні тести |
| `POST` | `/tests/{testId}/attempts` | Завершити спробу тесту |
| `GET` | `/progress` | Прогрес поточного користувача |
| `GET` | `/models/comparison` | Фактичні метрики моделей |
| `GET` | `/auth/me` | Поточний користувач |
| `POST` | `/auth/logout` | Завершити сесію |

### Контракт `POST /recognitions`

Request: `multipart/form-data`, поле `image`, опційне поле `modelVersion`.

Response shape:

```json
{
  "recognitionId": "uuid",
  "model": {"name": "custom-cnn", "version": "version-id"},
  "prediction": {
    "classId": 14,
    "name": "Class name",
    "confidence": 0.0
  },
  "top3": [
    {"classId": 14, "name": "Class name", "confidence": 0.0}
  ],
  "explanation": {
    "summary": "Explanation based on the returned classification.",
    "limitations": ["Single-sign classification only."]
  },
  "processingTimeMs": 0
}
```

Числові значення у прикладі є заповнювачами контракту, а не результатами експерименту.

## 5. Дані та зберігання

Основні сутності MySQL:

- `users` — ідентичність Google та локальні налаштування;
- `recognitions` — metadata запиту, модель, час і головний прогноз;
- `recognition_predictions` — top-k передбачення;
- `signs` — 43 класи каталогу;
- `tests`, `test_questions`, `test_attempts` — навчальні тести;
- `model_versions`, `model_metrics` — registry і виміряні метрики.

Оригінальне зображення за замовчуванням не зберігається. Якщо політика зміниться, потрібні окремі правила retention, доступу та видалення.

## 6. Безпека та межі відповідальності

- Google OAuth credentials надходять тільки через environment/secret manager.
- Backend не довіряє MIME, назві файлу або class ID із клієнта.
- Користувач має доступ лише до власної історії та прогресу.
- Логи не містять OAuth токенів, cookies, персональних даних і binary image content.
- ONNX-файли проходять перевірку checksum/version перед активацією.
- CORS, CSRF/session policy, rate limiting і file limits визначаються до production.

## 7. Версіонування ML-артефактів

Кожна модель має ідентифікатор, commit/config reference, список класів, preprocessing contract, ONNX opset, метрики та checksum. Backend активує тільки повний і перевірений запис registry.
