# SavVio Frontend: The Financial Advocate Interface

This directory contains the production-ready React frontend for **SavVio**, an AI-driven financial fiduciary application. The interface is designed to provide users with responsible, non-biased purchase recommendations by integrating their real-time financial health with product utility data.

---

## Tech Stack

*   **Framework**: [React 18](https://react.dev/) with [Vite](https://vitejs.dev/) for high-performance development.
*   **Styling**: [Tailwind CSS](https://tailwindcss.com/) for a sleek, utility-first UI.
*   **Components**: [shadcn/ui](https://ui.shadcn.com/) (Radix UI) for accessible, premium-quality interactions.
*   **State & Query**: [TanStack Query (v5)](https://tanstack.com/query/latest) for robust server-state management and caching.
*   **Navigation**: [React Router](https://reactrouter.com/) for SPA-style transitions.
*   **Visualizations**: [Recharts](https://recharts.org/) for financial health dashboards and "What-If" scenario modeling.
*   **Animations**: [Framer Motion](https://www.framer.com/motion/) for smooth micro-interactions.

---

## Architecture & Integration

The frontend acts as a thin, highly-visual wrapper around the **SavVio Inference Pipeline**. It communicates with the FastAPI backend via a centralized API service layer.

### 1. Backend Integration (FastAPI)
The frontend connects to the backend running at `localhost:3500` (proxied via `/api` in development):
*   **`POST /predict`**: Powers the conversational "Advocate" hub. It sends user queries about products and receives a structured inference response (Decision + Reasoning + Suggestions).
*   **`GET /user/{user_id}/profile`**: Retrieves the user's financial profile from PostgreSQL (income, expenses, savings) to populate the dashboard.
*   **`GET /health`**: Used by the frontend to verify that the ML models and DB connections are active before allowing evaluations.

### 2. LLM & Inference Flow
When a user asks, *"Should I buy this $1,500 laptop?"*:
1.  The frontend sends the query to `/predict`.
2.  The backend's LLM parses the intent and resolves the specific product.
3.  The **authoritative deterministic engine** computes the financial impact (`Residual Utility Score`).
4.  The LLM generates a conversational "advocate" response based on the math.
5.  The frontend renders this response with clear **Traffic Light** (Green/Yellow/Red) visual cues.

---

## Core Features

### The AI Advocate Hub
A conversational interface where users interact with the fiduciary AI. The UI dynamically adjusts its color scheme (Emerald Green, Soft Amber, Alert Red) based on the calculated safety of the purchase.

### Financial Health Dashboard
A data-rich view showing:
*   **Income vs. Essential Obligations**: Visual breakdown of the budget.
*   **"What-If" Analysis**: Shows how a proposed purchase would impact the user's emergency fund over time.
*   **Friction Success Rate**: Tracks how many impulse purchases SavVio successfully prevented.

---

## Getting Started

### Prerequisites
*   Node.js (v18+)
*   The SavVio Backend running at `localhost:3500`

### Installation
```bash
cd deployment_pipeline/frontend
npm install
```

### Local Development
```bash
npm run dev
```
The app will be available at [http://localhost:3000](http://localhost:3000). API requests to `/api/*` are automatically proxied to the backend at `:3500`.

### Testing
*   **Unit/Integration**: `npm run test` (Vitest + React Testing Library)
*   **End-to-End**: `npx playwright test` (E2E flows through `e2e/`)

---

## Production Deployment
The application is containerized using the [Docker/Cloud Run] setup in the `deployment_pipeline/docker` directory.
```bash
npm run build
```
The build artifacts in `dist/` are served by a lightweight Nginx container in production.
