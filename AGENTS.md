 # Правила розробки DriveSmartAI

## Контекст проєкту

DriveSmartAI — PWA для класифікації одного дорожнього знака на зображенні. Основний дослідницький результат — власна CNN, донавчена MobileNetV3 і чесне порівняння їхніх фактичних результатів на GTSRB.

Детальні межі та критерії приймання: [PROJECT_SPEC.md](./PROJECT_SPEC.md). Компоненти й API: [ARCHITECTURE.md](./ARCHITECTURE.md). План робіт: [TASKS.md](./TASKS.md).

## Загальні правила

- Не додавати YOLO, object detection, video або multi-sign classification без окремої зміни вимог.
- Не вигадувати accuracy, latency, confidence чи інші ML-результати.
- Не створювати backend/frontend/ML-код у межах документаційного етапу.
- Не зберігати secrets, токени, OAuth credentials, датасети або великі model artifacts у Git.
- Перед зміною прочитати відповідну специфікацію та перевірити поточний стан репозиторію.
- Зміни мають бути вузькими, типобезпечними та сумісними з існуючими контрактами.

## Якість коду

- Backend: Java 25 LTS, Spring Boot, Maven; розділяти controller, use case/service, repository та infrastructure.
- Frontend: TypeScript без `any`, React-компоненти з явними станами loading/error/empty, UI через Material UI.
- ML: Python 3.11, PyTorch; фіксувати seed, конфігурацію, split, preprocessing і class mapping.
- Повторно використовувати спільні типи/контракти, не дублювати preprocessing і mapping.
- Не приховувати помилки широкими `catch`, мовчазними fallback або success-shaped responses.
- Коментарі додавати лише для нетривіальної логіки; очевидний код не коментувати.

## Тестування та перевірка

- Перед завершенням запускати найменший релевантний build/test/lint для зміненого шару.
- Для ML перевіряти data leakage, форму tensor, label mapping, reproducibility та parity ONNX.
- Для backend тестувати валідацію файлів, авторизацію, ізоляцію даних і помилки inference.
- Для frontend тестувати upload/camera, top-3 rendering, error states, responsive-critical flows і PWA build.
- Не оновлювати документацію числовими результатами, якщо тест/експеримент їх не створив.
- Якщо перевірку неможливо запустити, явно описати причину та неперевірений ризик.

## Безпека

- Секрети передавати через environment або secret manager; не додавати їх у `.env`, код чи логи.
- Повторно валідовувати на backend MIME, розмір і вміст завантаженого файлу.
- Авторизація перевіряється на кожному endpoint із приватними даними.
- Не логувати токени, cookies, персональні дані або вміст зображень.
- Перевіряти checksum і metadata моделей перед активацією.
- Для production використовувати HTTPS, обмеження запитів і контроль CORS/CSRF.

## Git і документація

- Не виконувати `git commit` або `git push` без прямого запиту користувача.
- Не видаляти чужі незв’язані зміни та не робити destructive reset/checkout.
- Оновлювати пов’язані документи разом зі зміною вимог, API або структури.
- Усі експериментальні висновки мають посилатися на конкретну конфігурацію та артефакти.
