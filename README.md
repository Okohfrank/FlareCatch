# FlareCatch — AI-Optimized Gas-to-Methanol Reactor Control

[![React](https://img.shields.io/badge/React-19-blue?logo=react)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-8-646CFF?logo=vite)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-v4-38B2AC?logo=tailwindcss)](https://tailwindcss.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**FlareCatch** is an AI-driven reactor control and live decision-support web interface designed for micro-scale Gas-to-Methanol (GTM) chemical plants. The system dynamically predicts and regulates optimal reactor operating conditions—reformer temperature and steam-to-carbon (S/C) ratio—in response to real-time fluctuations in flare gas flow rate and methane composition.

---

## 🌟 Key Features

* **🤖 Trained Machine Learning Model:** Driven by a Random Forest Regressor trained on 18,432 process chemistry simulation runs ($R^2 = 0.896$, $\text{RMSE} = 0.0817\text{ kg/h}$).
* **⚡ 2D Bilinear Grid Interpolation:** Instantaneous client-side predictions ($<1\text{ ms}$) via a 50×50 lookup grid (`flarecatch_model.json`).
* **🔄 Autonomous Real-Time Fluctuation:** Multi-frequency sine-wave engine simulating natural flare-gas composition and flow-rate drift.
* **📊 Live Impact Analytics:**
  * **Methanol Production Rate** ($\text{kg/day}$)
  * **Revenue Generation** ($\text{₦/day}$ based on ₦1,825/kg domestic retail benchmark)
  * **$\text{CO}_2$ Emissions Avoided** ($\text{kg/day}$ based on $0.62\text{ kg CO}_2/\text{kg}$)
* **📈 Rolling Area Chart:** High-contrast 24-hour visual comparison between **AI-Optimized** and **Fixed-Setting** operational baselines.
* **💾 Local Storage Persistence:** Automatically preserves simulation history, chart data, and settings across browser sessions.

---

## ⚙️ System Architecture

1. **Data & Chemistry Engine:** Simulates Steam Methane Reforming (SMR) reaction kinetics ($\text{CH}_4 + \text{H}_2\text{O} \rightarrow \text{CO} + 3\text{H}_2$) and catalytic methanol synthesis ($\text{CO} + 2\text{H}_2 \rightarrow \text{CH}_3\text{OH}$).
2. **Model Layer:** Scikit-Learn Random Forest model (`100 trees`, `max_depth=10`) trained in Google Colab and exported to JSON.
3. **Web Dashboard Layer:** Modern Navy Blue & White responsive SPA built with React 19, Vite, and Tailwind CSS.

---

## 🚀 Getting Started

### Prerequisites
* [Node.js](https://nodejs.org/) (v18 or higher)
* `npm` or `pnpm`

### Installation & Local Run

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ChemXAI.git

# Navigate into project directory
cd ChemXAI

# Install dependencies
npm install

# Start local development server
npm run dev
```

Open [http://localhost:8443](http://localhost:8443) in your browser.

---

## 🛠️ Built With

* **Frontend:** React 19, TypeScript, Recharts
* **Styling:** Tailwind CSS v4, Glassmorphism UI
* **Build Tool:** Vite 8
* **AI/ML:** Python, Scikit-Learn, Pandas, NumPy (trained on Google Colab)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
