# DriveSmartAI Frontend

React + TypeScript + Vite інтерфейс для реального розпізнавання дорожніх знаків
через Spring Boot API.

## Запуск

```powershell
Set-Location C:\Projects\DriveSmartAI\frontend
npm install
npm run dev
```

Сайт доступний на `http://localhost:5173`. Для повної перевірки також запустіть
Spring Boot backend на `http://localhost:8080`.

## Перевірка

```powershell
npm run build
npm test
npm run lint
```

Vite проксує `/api` до backend-у, тому запит `POST /api/predict` надсилається
через `multipart/form-data` із полем `file` без ручного встановлення `Content-Type`.

Сторінка `/recognition` підтримує перетягування, вибір фото і popередній перегляд.
Допустимі формати: JPEG і PNG; максимальний розмір: 5 МБ.

# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some Oxlint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the Oxlint configuration

If you are developing a production application, we recommend enabling type-aware lint rules by installing `oxlint-tsgolint` and editing `.oxlintrc.json`:

```json
{
  "$schema": "./node_modules/oxlint/configuration_schema.json",
  "plugins": ["react", "typescript", "oxc"],
  "options": {
    "typeAware": true
  },
  "rules": {
    "react/rules-of-hooks": "error",
    "react/only-export-components": ["warn", { "allowConstantExport": true }]
  }
}
```

See the [Oxlint rules documentation](https://oxc.rs/docs/guide/usage/linter/rules) for the full list of rules and categories.
