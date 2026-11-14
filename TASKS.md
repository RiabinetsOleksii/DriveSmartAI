# DriveSmartAI — план робіт

Позначення: `[ ]` не виконано, `[-]` у роботі, `[x]` виконано.

## Етап 0. Планування та базові правила

- [x] Зафіксувати межі MVP: один знак, 43 класи, без detection/video.
- [x] Описати вимоги, архітектуру та правила агента.
- [ ] Узгодити формат версій моделей і preprocessing contract.
- [ ] Узгодити політику зберігання зображень і retention.

## Етап 1. Підготовка даних

- [x] Додати reproducible спосіб отримання GTSRB без коміту датасету.
- [-] Перевірити структуру, пошкоджені файли та розподіл 43 класів.
- [-] Зафіксувати train/validation/test split.
- [x] Реалізувати спільний preprocessing і class mapping.
- [x] Додати тести відсутності data leakage.

## Етап 2. Власна CNN

- [ ] Реалізувати конфігуровану архітектуру власної CNN.
- [ ] Додати train/validation loop, checkpoints і seed.
- [ ] Додати логування loss та метрик.
- [ ] Провести baseline-експеримент.
- [ ] Оцінити accuracy, precision, recall, macro-F1 і confusion matrix.
- [ ] Зберегти конфігурацію та фактичні результати експерименту.

## Етап 3. MobileNetV3

- [ ] Підготувати MobileNetV3 для 43 класів.
- [ ] Реалізувати transfer learning і fine-tuning stages.
- [ ] Провести експерименти з документованими параметрами.
- [ ] Порівняти модель з CNN на однаковому test split.
- [ ] Зафіксувати фактичні метрики, latency та розмір артефакту.

## Етап 4. ONNX та inference contract

- [ ] Експортувати обидві затверджені моделі в ONNX.
- [ ] Додати parity-перевірку PyTorch проти ONNX Runtime.
- [ ] Описати input shape, normalization, labels і output semantics.
- [ ] Перевірити checksum та version metadata.

## Етап 5. Backend

- [x] Створити Spring Boot/Maven skeleton на Java 21 LTS.
- [ ] Додати конфігурацію профілів і секретів без hardcode.
- [ ] Реалізувати ONNX Runtime adapter і model registry.
- [ ] Реалізувати `POST /api/v1/recognitions` з валідацією.
- [ ] Реалізувати каталог, history, tests, progress і comparison API.
- [ ] Додати Google OAuth 2.0 та authorization rules.
- [ ] Додати MySQL schema/migrations та repository tests.
- [ ] Додати unit, integration і API contract tests.

## Етап 6. Frontend/PWA

- [x] Створити React + TypeScript + Vite skeleton.
- [x] Налаштувати Material UI theme та responsive layout.
- [x] Реалізувати upload flow і client-side validation.
- [ ] Реалізувати result, catalog, tests, history, progress і comparison pages.
- [ ] Додати auth flow і стани loading/error/empty.
- [ ] Додати manifest, service worker та installability checks.
- [ ] Додати component та integration tests.

## Етап 7. Інтеграція та якість

- [ ] Перевірити повний сценарій: photo → API → ONNX → result.
- [ ] Перевірити помилки великих/невалідних/пошкоджених файлів.
- [ ] Перевірити ізоляцію історії та прогресу різних користувачів.
- [ ] Перевірити доступність клавіатурою та mobile layout.
- [ ] Додати performance smoke test для inference.
- [ ] Оновити документацію фактичними результатами без вигадування метрик.

## Етап 8. Підготовка курсової

- [ ] Описати методику, архітектуру та preprocessing.
- [ ] Описати експерименти й обмеження.
- [ ] Додати порівняльні таблиці лише з відтворюваними даними.
- [ ] Підготувати демонстраційний сценарій.
- [ ] Перевірити ліцензії датасету, бібліотек і ресурсів каталогу.
