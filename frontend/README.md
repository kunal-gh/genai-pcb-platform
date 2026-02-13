# GenAI PCB Design Platform - Frontend

React-based web application for the GenAI PCB Design Platform.

## Features

- Natural language prompt input with validation
- Real-time processing status with progress indicators
- Design file preview and download
- Design history management
- Responsive Material-UI interface

## Setup

1. Install dependencies:
```bash
npm install
```

2. Create `.env` file:
```bash
cp .env.example .env
```

3. Update API URL in `.env` if needed

4. Start development server:
```bash
npm start
```

The application will open at http://localhost:3000

## Build

Create production build:
```bash
npm run build
```

## Testing

Run tests:
```bash
npm test
```

## Project Structure

```
src/
├── components/       # Reusable UI components
│   ├── PromptInput.tsx
│   ├── ProcessingStatus.tsx
│   └── DesignPreview.tsx
├── pages/           # Page components
│   ├── DesignPage.tsx
│   └── HistoryPage.tsx
├── services/        # API integration
│   └── api.ts
├── App.tsx          # Main application component
└── index.tsx        # Application entry point
```
