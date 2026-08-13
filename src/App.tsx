import { useState, useEffect, useRef } from "react"
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts"

// ─── types ────────────────────────────────────────────────────────────────────

interface SimResult {
  reformerTemp: number
  steamToCarbon: number
  ch4Conversion: number
  methanolYield: number
  methanolProduced: number
  revenue: number
  co2Avoided: number
}

interface ChartPoint {
  tick: number
  ai: number
  fixed: number
}

interface ModelMetadata {
  model: string
  n_estimators: number
  max_depth: number
  r2_score: number
  rmse_yield: number
  trained_on: number
  features: string[]
  targets: string[]
}

interface ModelGrid {
  flow_rate: { min: number; max: number; steps: number }
  methane_pct: { min: number; max: number; steps: number }
}

interface ModelJson {
  metadata: ModelMetadata
  grid: ModelGrid
  predictions: {
    reformer_temp: number[][]
    steam_to_carbon: number[][]
    ch4_conversion: number[][]
    methanol_yield: number[][]
    fixed_conversion: number[][]
    fixed_yield: number[][]
  }
}

// ─── 2D Bilinear Grid Interpolation ──────────────────────────────────────────

function interpolateGrid(
  gridData: number[][],
  flowRate: number,
  methaneContent: number,
  gridSpec: ModelGrid
): number {
  const fMin = gridSpec.flow_rate.min
  const fMax = gridSpec.flow_rate.max
  const fSteps = gridSpec.flow_rate.steps

  const mMin = gridSpec.methane_pct.min
  const mMax = gridSpec.methane_pct.max
  const mSteps = gridSpec.methane_pct.steps

  const fClamped = Math.max(fMin, Math.min(fMax, flowRate))
  const mClamped = Math.max(mMin, Math.min(mMax, methaneContent))

  const fiReal = ((fClamped - fMin) / (fMax - fMin)) * (fSteps - 1)
  const miReal = ((mClamped - mMin) / (mMax - mMin)) * (mSteps - 1)

  const fi0 = Math.floor(fiReal)
  const fi1 = Math.min(fi0 + 1, fSteps - 1)
  const df = fiReal - fi0

  const mi0 = Math.floor(miReal)
  const mi1 = Math.min(mi0 + 1, mSteps - 1)
  const dm = miReal - mi0

  const v00 = gridData[fi0][mi0]
  const v10 = gridData[fi1][mi0]
  const v01 = gridData[fi0][mi1]
  const v11 = gridData[fi1][mi1]

  return (
    (1 - df) * (1 - dm) * v00 +
    df * (1 - dm) * v10 +
    (1 - df) * dm * v01 +
    df * dm * v11
  )
}

// ─── simulation logic ─────────────────────────────────────────────────────────

function runSimulation(
  flowRate: number,
  methaneContent: number,
  aiMode: boolean,
  modelData: ModelJson | null = null
): SimResult {
  let reformerTemp: number
  let steamToCarbon: number
  let ch4Conversion: number
  let methanolYield: number

  if (modelData) {
    const grid = modelData.grid
    const preds = modelData.predictions

    if (aiMode) {
      reformerTemp = Math.round(interpolateGrid(preds.reformer_temp, flowRate, methaneContent, grid))
      steamToCarbon = parseFloat(interpolateGrid(preds.steam_to_carbon, flowRate, methaneContent, grid).toFixed(2))
      ch4Conversion = Math.round(interpolateGrid(preds.ch4_conversion, flowRate, methaneContent, grid))
      methanolYield = parseFloat(interpolateGrid(preds.methanol_yield, flowRate, methaneContent, grid).toFixed(2))
    } else {
      reformerTemp = 820
      steamToCarbon = 2.5
      ch4Conversion = Math.round(interpolateGrid(preds.fixed_conversion, flowRate, methaneContent, grid))
      methanolYield = parseFloat(interpolateGrid(preds.fixed_yield, flowRate, methaneContent, grid).toFixed(2))
    }
  } else {
    // Fallback arithmetic
    const base = (flowRate / 1000) * (methaneContent / 100)
    reformerTemp = aiMode ? Math.round(850 + base * 12) : 820
    steamToCarbon = aiMode ? parseFloat((2.6 + base * 0.08).toFixed(2)) : 2.5
    ch4Conversion = aiMode ? Math.min(99, Math.round(78 + base * 1.4)) : Math.round(72 + base * 0.9)
    methanolYield = aiMode
      ? parseFloat(((flowRate * 0.00031 * (methaneContent / 85)) * 1.18).toFixed(2))
      : parseFloat((flowRate * 0.00031 * (methaneContent / 85)).toFixed(2))
  }

  const methanolProduced = Math.round(methanolYield * 24)
  const revenue = Math.round(methanolProduced * 1825)
  const co2Avoided = Math.round(methanolProduced * 0.62)

  return {
    reformerTemp,
    steamToCarbon,
    ch4Conversion,
    methanolYield,
    methanolProduced,
    revenue,
    co2Avoided,
  }
}

// ─── autonomous gas fluctuation ───────────────────────────────────────────────
// Multi-frequency sine waves create smooth, realistic flare-gas drift.
// Input `t` is elapsed minutes — the waves are naturally periodic so no
// wrapping is needed; the simulation can run indefinitely.

function getGasConditions(t: number) {
  const tau = t * Math.PI * 2

  const flowRate = Math.round(
    5000 +
      Math.sin(tau * 2.3 + 0.5) * 1500 +
      Math.sin(tau * 5.7 + 1.2) * 700 +
      Math.sin(tau * 11.1 + 2.8) * 300
  )

  const methaneContent = parseFloat(
    (
      83.5 +
      Math.sin(tau * 1.7 + 0.8) * 4 +
      Math.sin(tau * 4.3 + 2.1) * 2 +
      Math.sin(tau * 9.1 + 0.3) * 0.8
    ).toFixed(1)
  )

  return {
    flowRate: Math.max(1000, Math.min(10000, flowRate)),
    methaneContent: Math.max(77, Math.min(90, methaneContent)),
  }
}

// ─── constants ────────────────────────────────────────────────────────────────

const TICK_MS = 600 // condition update interval
const TICKS_PER_POINT = 4 // chart point every 4 ticks ≈ 2.4 s
const MAX_CHART_POINTS = 30 // rolling window
const STORAGE_KEY = "flarecatch-state"

// ─── localStorage persistence ─────────────────────────────────────────────────

interface PersistedState {
  chartData: ChartPoint[]
  flowRate: number
  methaneContent: number
  aiMode: boolean
  simTick: number
  chartPointCount: number
}

function loadState(): PersistedState | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    return JSON.parse(raw) as PersistedState
  } catch {
    return null
  }
}

function saveState(state: PersistedState) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
  } catch {
    // storage full or unavailable — silently ignore
  }
}

function clearState() {
  try {
    localStorage.removeItem(STORAGE_KEY)
  } catch {
    // ignore
  }
}

// ─── sub-components ───────────────────────────────────────────────────────────

function Card({
  title,
  children,
  className = "",
}: {
  title: string
  children: React.ReactNode
  className?: string
}) {
  return (
    <div
      className={`bg-white/[0.03] backdrop-blur-sm border border-white/[0.07] rounded-2xl overflow-hidden transition-all duration-300 hover:border-white/[0.13] ${className}`}
    >
      <div className="px-5 py-3 border-b border-white/[0.05]">
        <span className="text-[11px] font-semibold tracking-[0.15em] text-blue-300/70 uppercase">
          {title}
        </span>
      </div>
      <div className="p-5">{children}</div>
    </div>
  )
}

function StatBlock({
  label,
  value,
  unit = "",
}: {
  label: string
  value: string | number
  unit?: string
}) {
  return (
    <div className="relative bg-white/[0.03] border border-white/[0.06] rounded-xl p-4 overflow-hidden hover:bg-white/[0.05] transition-all duration-300">
      <div className="absolute left-0 top-0 bottom-0 w-[3px] rounded-r bg-gradient-to-b from-blue-400 to-blue-600" />
      <span className="text-[10px] tracking-[0.12em] text-slate-500 uppercase block mb-2 leading-tight">
        {label}
      </span>
      <div className="flex items-baseline gap-1.5">
        <span className="text-[22px] font-bold text-white leading-none tracking-tight tabular-nums">
          {value}
        </span>
        {unit && <span className="text-[11px] text-slate-500">{unit}</span>}
      </div>
    </div>
  )
}

// ─── live gauge bar (read-only) ───────────────────────────────────────────────

function LiveGaugeBar({
  label,
  value,
  min,
  max,
  unit,
  color = "from-blue-600 to-blue-400",
}: {
  label: string
  value: number
  min: number
  max: number
  unit: string
  color?: string
}) {
  const pct = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100))
  return (
    <div className="flex flex-col gap-2">
      <div className="flex justify-between items-baseline">
        <span className="text-[11px] tracking-[0.1em] text-slate-400 uppercase font-medium">
          {label}
        </span>
        <span className="text-sm font-bold text-white tabular-nums">
          {typeof value === "number" && value % 1 !== 0
            ? value.toFixed(1)
            : value.toLocaleString()}
          <span className="text-slate-500 ml-0.5 font-normal text-[11px]">{unit}</span>
        </span>
      </div>
      <div className="h-2 rounded-full bg-white/[0.06] overflow-hidden">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${color} transition-all duration-500`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="flex justify-between">
        <span className="text-[10px] text-slate-600">
          {min.toLocaleString()}
          {unit}
        </span>
        <span className="text-[10px] text-slate-600">
          {max.toLocaleString()}
          {unit}
        </span>
      </div>
    </div>
  )
}

// ─── impact metric card ───────────────────────────────────────────────────────

function ImpactMetric({
  label,
  value,
  unit,
  gradient,
}: {
  label: string
  value: string
  unit: string
  gradient: string
}) {
  return (
    <div className="relative bg-white/[0.03] border border-white/[0.06] rounded-xl p-5 overflow-hidden hover:bg-white/[0.05] transition-all duration-300">
      <div
        className={`absolute left-0 top-0 bottom-0 w-1 rounded-r bg-gradient-to-b ${gradient}`}
      />
      <span className="text-[10px] tracking-[0.12em] text-slate-500 uppercase block mb-2">
        {label}
      </span>
      <div className="flex items-baseline gap-2">
        <span className="text-3xl font-bold text-white leading-none tabular-nums">
          {value}
        </span>
        <span className="text-[11px] text-slate-500">{unit}</span>
      </div>
    </div>
  )
}

// ─── process flow icons ───────────────────────────────────────────────────────

function IconFlame() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-5 h-5">
      <path d="M12 2C12 2 8 6 8 10c0 2.2 1.8 4 4 4s4-1.8 4-4c0-2-1-4-1-4" />
      <path d="M12 14c0 0-4 2-4 6h8c0-4-4-6-4-6z" />
    </svg>
  )
}

function IconReformer() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-5 h-5">
      <rect x="5" y="6" width="14" height="12" rx="2" />
      <line x1="5" y1="10" x2="19" y2="10" />
      <line x1="5" y1="14" x2="19" y2="14" />
      <line x1="9" y1="6" x2="9" y2="18" />
    </svg>
  )
}

function IconSyngas() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-5 h-5">
      <circle cx="12" cy="12" r="7" />
      <line x1="12" y1="5" x2="12" y2="19" />
      <line x1="5" y1="12" x2="19" y2="12" />
      <circle cx="12" cy="12" r="2" />
    </svg>
  )
}

function IconSynthesis() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-5 h-5">
      <path d="M8 3 L8 21" />
      <path d="M16 3 L16 21" />
      <path d="M5 7 H19" />
      <path d="M5 17 H19" />
      <path d="M8 12 H16" />
    </svg>
  )
}

function IconAI() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-5 h-5">
      <rect x="4" y="4" width="16" height="16" rx="2" />
      <circle cx="9" cy="9" r="1.5" />
      <circle cx="15" cy="9" r="1.5" />
      <circle cx="9" cy="15" r="1.5" />
      <circle cx="15" cy="15" r="1.5" />
      <circle cx="12" cy="12" r="2" />
      <line x1="9" y1="9" x2="12" y2="12" />
      <line x1="15" y1="9" x2="12" y2="12" />
      <line x1="9" y1="15" x2="12" y2="12" />
      <line x1="15" y1="15" x2="12" y2="12" />
    </svg>
  )
}

// ─── main component ───────────────────────────────────────────────────────────

export default function App() {
  // Load persisted state once on mount
  const saved = useRef(loadState())

  // Model JSON loaded from Colab export
  const [modelData, setModelData] = useState<ModelJson | null>(null)

  useEffect(() => {
    fetch("/flarecatch_model.json")
      .then((res) => res.json())
      .then((data) => {
        setModelData(data)
        console.log("✅ FlareCatch Random Forest model loaded:", data.metadata)
      })
      .catch((err) => {
        console.warn("Using default simulation parameters:", err)
      })
  }, [])

  // Gas conditions — driven autonomously when simulation is running
  const [flowRate, setFlowRate] = useState(saved.current?.flowRate ?? 5000)
  const [methaneContent, setMethaneContent] = useState(saved.current?.methaneContent ?? 83.5)

  // AI / Fixed toggle
  const [aiMode, setAiMode] = useState(saved.current?.aiMode ?? true)

  // Simulation engine
  const [isSimulating, setIsSimulating] = useState(false)
  const simTickRef = useRef(saved.current?.simTick ?? 0)

  // Rolling chart data
  const [chartData, setChartData] = useState<ChartPoint[]>(saved.current?.chartData ?? [])
  const chartPointCounter = useRef(saved.current?.chartPointCount ?? 0)

  // Pulse animation
  const [tick, setTick] = useState(0)
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 2500)
    return () => clearInterval(id)
  }, [])

  // ── Persist state to localStorage on meaningful changes ─────────────────
  useEffect(() => {
    saveState({
      chartData,
      flowRate,
      methaneContent,
      aiMode,
      simTick: simTickRef.current,
      chartPointCount: chartPointCounter.current,
    })
  }, [chartData, aiMode])

  // ── Continuous simulation loop (runs until stopped) ─────────────────────
  useEffect(() => {
    if (!isSimulating) return

    const id = setInterval(() => {
      simTickRef.current++
      const t = simTickRef.current

      // Convert ticks to elapsed minutes for sine-wave input
      const elapsedMinutes = (t * TICK_MS) / 60000
      const conditions = getGasConditions(elapsedMinutes)
      setFlowRate(conditions.flowRate)
      setMethaneContent(conditions.methaneContent)

      // Append chart point at intervals
      if (t % TICKS_PER_POINT === 0) {
        chartPointCounter.current++
        const aiRes = runSimulation(conditions.flowRate, conditions.methaneContent, true, modelData)
        const fixRes = runSimulation(conditions.flowRate, conditions.methaneContent, false, modelData)

        setChartData((prev) => {
          const next = [
            ...prev,
            {
              tick: chartPointCounter.current,
              ai: aiRes.methanolYield,
              fixed: fixRes.methanolYield,
            },
          ]
          // Rolling window — drop oldest points beyond the limit
          return next.length > MAX_CHART_POINTS
            ? next.slice(next.length - MAX_CHART_POINTS)
            : next
        })
      }
    }, TICK_MS)

    return () => clearInterval(id)
  }, [isSimulating, modelData])

  // ── Derived values (always current rate, never accumulated) ─────────────
  const currentAI = runSimulation(flowRate, methaneContent, true, modelData)
  const currentFixed = runSimulation(flowRate, methaneContent, false, modelData)
  const displayed = aiMode ? currentAI : currentFixed

  function handleReset() {
    setIsSimulating(false)
    simTickRef.current = 0
    chartPointCounter.current = 0
    setChartData([])
    setFlowRate(5000)
    setMethaneContent(83.5)
    clearState()
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload?.length) return null
    return (
      <div className="bg-navy-900/95 backdrop-blur-md border border-white/[0.1] rounded-xl px-4 py-3 shadow-2xl">
        {payload.map((p: any) => (
          <p key={p.dataKey} className="text-[11px] text-white/80 leading-relaxed">
            <span className={p.dataKey === "ai" ? "text-blue-400" : "text-slate-500"}>●</span>{" "}
            {p.dataKey === "ai" ? "AI-Optimized" : "Fixed-Setting"}:{" "}
            <span className="font-semibold">{p.value}</span> kg/h
          </p>
        ))}
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-navy-900 text-white">
      {/* Ambient glow */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-40 left-1/4 w-[700px] h-[700px] bg-blue-500/[0.035] rounded-full blur-[140px]" />
        <div className="absolute -bottom-40 right-1/4 w-[550px] h-[550px] bg-blue-600/[0.025] rounded-full blur-[120px]" />
      </div>

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="relative border-b border-white/[0.06] px-4 sm:px-8 py-4 sm:py-5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 sm:gap-0 bg-white/[0.01] backdrop-blur-sm">
        <div className="flex items-center gap-3 sm:gap-4">
          <img
            src="/ChemXAI.png.png"
            alt="ChemXAI Logo"
            className="h-8 sm:h-10 w-auto object-contain"
          />
          <div>
            <h1 className="text-xl sm:text-2xl font-black tracking-[0.08em] text-white leading-none">
              FLARECATCH
            </h1>
            <p className="text-[11px] sm:text-[12px] text-slate-400 mt-1 sm:mt-1.5 tracking-[0.02em] font-light">
              AI-Optimized Reactor Control for Micro-Scale Gas-to-Methanol Conversion
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2.5 sm:gap-3 self-end sm:self-auto">
          <div
            className="w-2.5 h-2.5 rounded-full animate-pulse-glow"
            style={{
              backgroundColor: isSimulating ? "#22c55e" : aiMode ? "#3b82f6" : "#64748b",
              transition: "background-color 0.5s ease",
            }}
          />
          <span className="text-[10px] sm:text-[11px] tracking-[0.14em] font-semibold text-slate-300 uppercase">
            {isSimulating ? "Live Simulation" : aiMode ? "AI Active" : "Fixed Mode"}
          </span>
        </div>
      </header>

      {/* ── Three-column responsive body ────────────────────────────────────── */}
      <main className="relative grid grid-cols-1 lg:grid-cols-12 min-h-[calc(100vh-72px-134px)]">
        {/* ── Left: Live Feed & Controls ───────────────────────────────────── */}
        <div className="lg:col-span-3 border-b lg:border-b-0 lg:border-r border-white/[0.06] p-4 sm:p-6 flex flex-col gap-5">
          {/* Live gas conditions */}
          <Card title="Live Gas Feed">
            <div className="flex flex-col gap-6">
              <LiveGaugeBar
                label="Gas Flow Rate"
                value={flowRate}
                min={1000}
                max={10000}
                unit=" scf/day"
                color="from-blue-600 to-blue-400"
              />
              <LiveGaugeBar
                label="Methane Content"
                value={methaneContent}
                min={77}
                max={90}
                unit="%"
                color="from-cyan-600 to-cyan-400"
              />
            </div>
          </Card>

          {/* AI / Fixed toggle */}
          <Card title="Operation Mode">
            <div className="flex rounded-xl overflow-hidden border border-white/[0.08]">
              <button
                onClick={() => setAiMode(false)}
                className={`flex-1 py-3 text-[11px] font-semibold tracking-[0.1em] uppercase transition-all duration-200 cursor-pointer rounded-l-[11px] ${
                  !aiMode
                    ? "bg-white text-navy-900"
                    : "bg-transparent text-slate-500 hover:text-slate-300"
                }`}
              >
                Fixed-Setting
              </button>
              <button
                onClick={() => setAiMode(true)}
                className={`flex-1 py-3 text-[11px] font-semibold tracking-[0.1em] uppercase transition-all duration-200 cursor-pointer rounded-r-[11px] ${
                  aiMode
                    ? "bg-blue-500 text-white shadow-lg shadow-blue-500/20"
                    : "bg-transparent text-slate-500 hover:text-slate-300"
                }`}
              >
                AI-Optimized
              </button>
            </div>
          </Card>

          {/* Start / Stop */}
          <div className="flex gap-3">
            <button
              onClick={() => setIsSimulating(!isSimulating)}
              className={`flex-1 py-4 rounded-xl text-[12px] font-bold tracking-[0.14em] uppercase cursor-pointer transition-all duration-300 active:scale-[0.98] ${
                isSimulating
                  ? "bg-white/[0.1] text-white border border-white/[0.15] hover:bg-white/[0.15]"
                  : "bg-gradient-to-r from-blue-600 to-blue-500 text-white hover:shadow-lg hover:shadow-blue-500/25"
              }`}
            >
              {isSimulating ? "⏹  Stop Simulation" : "▶  Start Simulation"}
            </button>

            {chartData.length > 0 && !isSimulating && (
              <button
                onClick={handleReset}
                className="py-4 px-5 rounded-xl text-[12px] font-bold text-slate-400 bg-white/[0.05] border border-white/[0.08] cursor-pointer hover:bg-white/[0.1] transition-all duration-200"
                title="Reset"
              >
                ↺
              </button>
            )}
          </div>

          {/* System info */}
          <div className="border-t border-white/[0.05] pt-5 flex flex-col gap-2 mt-auto">
            {[
              ["Model Engine", modelData ? `${modelData.metadata.model} (100 Trees)` : "FC-2.4.1 RF"],
              ["Trained On", modelData ? `${modelData.metadata.trained_on.toLocaleString()} runs` : "18,432 runs"],
              ["R² Score", modelData ? `${modelData.metadata.r2_score}` : "0.896"],
              ["RMSE Yield", modelData ? `${modelData.metadata.rmse_yield} kg/h` : "0.0817 kg/h"],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <span className="text-[10px] text-slate-600 uppercase tracking-widest">{k}</span>
                <span className="text-[10px] text-slate-400 tabular-nums">{v}</span>
              </div>
            ))}

            {/* Clear history */}
            <button
              onClick={() => {
                handleReset()
                clearState()
              }}
              className="mt-3 py-2 rounded-lg text-[10px] font-medium tracking-[0.1em] uppercase text-slate-500 bg-white/[0.03] border border-white/[0.06] cursor-pointer hover:bg-red-500/10 hover:text-red-400 hover:border-red-500/20 transition-all duration-200"
            >
              ✕  Clear History
            </button>
          </div>
        </div>

        {/* ── Center: Model Output Panel ───────────────────────────────────── */}
        <div className="lg:col-span-4 border-b lg:border-b-0 lg:border-r border-white/[0.06] p-4 sm:p-6 flex flex-col gap-5">
          <Card title={aiMode ? "AI Recommended Settings" : "Fixed-Setting Output"}>
            <div className="grid grid-cols-2 gap-3">
              <StatBlock label="Reformer Temp (°C)" value={displayed.reformerTemp} />
              <StatBlock label="Steam-to-Carbon Ratio" value={displayed.steamToCarbon} />
              <StatBlock label="CH₄ Conversion" value={displayed.ch4Conversion} unit="%" />
              <StatBlock label="Methanol Yield" value={displayed.methanolYield} unit="kg/h" />
            </div>
            <div className="flex items-center gap-2 mt-4">
              <div className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
              <p className="text-[10px] text-slate-500 tracking-wide">
                Model Accuracy: R² = {modelData?.metadata.r2_score ?? 0.896} · RMSE = {modelData?.metadata.rmse_yield ?? 0.0817} kg/h
              </p>
            </div>
          </Card>

          {/* Performance Delta — always current rate, not accumulated */}
          <Card title="Performance Delta vs Fixed">
            <div className="flex flex-col gap-3">
              {[
                {
                  label: "Yield Uplift",
                  ai: currentAI.methanolProduced,
                  fixed: currentFixed.methanolProduced,
                  unit: "kg/day",
                },
                {
                  label: "Revenue Gain",
                  ai: currentAI.revenue,
                  fixed: currentFixed.revenue,
                  unit: "₦/day",
                },
              ].map(({ label, ai, fixed, unit }) => {
                const pct = fixed > 0 ? (((ai - fixed) / fixed) * 100).toFixed(1) : "0.0"
                const barPct = ai > 0 ? Math.min((fixed / ai) * 100, 100) : 50
                return (
                  <div
                    key={label}
                    className="bg-white/[0.02] border border-white/[0.05] rounded-xl p-4"
                  >
                    <div className="flex justify-between mb-3">
                      <span className="text-[10px] text-slate-500 uppercase tracking-widest">
                        {label}
                      </span>
                      <span className="text-[11px] text-emerald-400 font-semibold">
                        +{pct}%
                      </span>
                    </div>
                    <div className="flex flex-col gap-1.5">
                      <div className="flex items-center gap-2">
                        <span className="text-[9px] text-slate-600 w-10 uppercase">Fixed</span>
                        <div className="flex-1 h-1.5 rounded-full bg-white/[0.05] overflow-hidden">
                          <div
                            className="h-full rounded-full bg-slate-600 transition-all duration-500"
                            style={{ width: `${barPct}%` }}
                          />
                        </div>
                        <span className="text-[9px] text-slate-500 w-20 text-right tabular-nums">
                          {fixed.toLocaleString()} {unit}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-[9px] text-blue-400 w-10 uppercase font-medium">AI</span>
                        <div className="flex-1 h-1.5 rounded-full bg-white/[0.05] overflow-hidden">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-blue-600 to-blue-400 transition-all duration-500"
                            style={{ width: "100%" }}
                          />
                        </div>
                        <span className="text-[9px] text-white w-20 text-right tabular-nums font-medium">
                          {ai.toLocaleString()} {unit}
                        </span>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </Card>

          {/* Reactor Health */}
          <Card title="Reactor Health">
            <div className="flex flex-col gap-3">
              {[
                { label: "Pressure Stability", value: 97, color: "from-emerald-500 to-emerald-400" },
                { label: "Catalyst Activity", value: 89, color: "from-blue-500 to-blue-400" },
                { label: "Heat Recovery Eff.", value: 82, color: "from-amber-500 to-amber-400" },
              ].map(({ label, value, color }) => (
                <div key={label} className="flex items-center gap-4">
                  <span className="text-[10px] text-slate-500 uppercase tracking-widest w-28 sm:w-36 shrink-0">
                    {label}
                  </span>
                  <div className="flex-1 h-1.5 rounded-full bg-white/[0.05] overflow-hidden">
                    <div
                      className={`h-full rounded-full bg-gradient-to-r ${color} transition-all duration-700`}
                      style={{ width: `${value}%` }}
                    />
                  </div>
                  <span className="text-[10px] text-slate-400 w-8 text-right tabular-nums font-medium">
                    {value}%
                  </span>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* ── Right: Impact Dashboard ──────────────────────────────────────── */}
        <div className="lg:col-span-5 p-4 sm:p-6 flex flex-col gap-5">
          <Card title="Live Impact" className="flex-1 flex flex-col">
            {/* Current-rate metrics — never accumulated */}
            <div className="flex flex-col gap-3 mb-6">
              <ImpactMetric
                label="Methanol Produced"
                value={displayed.methanolProduced.toLocaleString()}
                unit="kg/day"
                gradient="from-blue-400 to-blue-600"
              />
              <ImpactMetric
                label="Revenue Generated"
                value={`₦${displayed.revenue.toLocaleString()}`}
                unit="/day"
                gradient="from-emerald-400 to-emerald-600"
              />
              <ImpactMetric
                label="CO₂ Avoided"
                value={displayed.co2Avoided.toLocaleString()}
                unit="kg/day"
                gradient="from-cyan-400 to-cyan-600"
              />
            </div>

            {/* ── Rolling live chart ──────────────────────────────────────────── */}
            <div className="flex-1 flex flex-col">
              <div className="flex items-center gap-5 mb-4">
                <div className="flex items-center gap-2">
                  <div className="w-5 h-0.5 rounded-full bg-blue-400" />
                  <span className="text-[10px] text-slate-400 uppercase tracking-widest">
                    AI-Optimized
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-5 h-0.5 rounded-full bg-slate-600 border-t border-dashed border-slate-500" />
                  <span className="text-[10px] text-slate-500 uppercase tracking-widest">
                    Fixed-Setting
                  </span>
                </div>
              </div>

              {chartData.length === 0 ? (
                <div className="flex items-center justify-center h-[170px] border border-dashed border-white/[0.06] rounded-xl">
                  <span className="text-[11px] text-slate-600 uppercase tracking-widest">
                    Start simulation to see live data
                  </span>
                </div>
              ) : (
                <ResponsiveContainer width="100%" height={170}>
                  <AreaChart
                    data={chartData}
                    margin={{ top: 4, right: 8, bottom: 0, left: -16 }}
                  >
                    <defs>
                      <linearGradient id="aiGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.28} />
                        <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="rgba(255,255,255,0.04)" vertical={false} />
                    <XAxis
                      dataKey="tick"
                      tick={false}
                      axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
                      tickLine={false}
                    />
                    <YAxis
                      tick={{ fontSize: 9, fill: "#475569" }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Area
                      type="monotone"
                      dataKey="ai"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      fill="url(#aiGrad)"
                      isAnimationActive={false}
                      activeDot={{ r: 4, fill: "#3b82f6", stroke: "#93c5fd", strokeWidth: 2 }}
                    />
                    <Area
                      type="monotone"
                      dataKey="fixed"
                      stroke="#475569"
                      strokeWidth={1.5}
                      strokeDasharray="5 4"
                      fill="none"
                      isAnimationActive={false}
                      activeDot={{ r: 3, fill: "#475569", strokeWidth: 0 }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              )}
              <p className="text-[9px] text-slate-600 mt-2 text-center tracking-[0.2em] uppercase">
                Live Methanol Output (kg/h)
              </p>
            </div>
          </Card>
        </div>
      </main>

      {/* ── Process Flow Strip ──────────────────────────────────────────────── */}
      <footer className="relative border-t border-white/[0.06] px-4 sm:px-8 py-5 sm:py-7 bg-white/[0.01]">
        <div className="flex flex-wrap md:flex-nowrap items-start justify-center md:justify-between gap-6 md:gap-2 max-w-4xl mx-auto">
          {[
            { icon: <IconFlame />, label: "Flare Gas In", sub: "Raw feedstock capture" },
            { icon: <IconReformer />, label: "Reforming", sub: "Steam methane reforming" },
            { icon: <IconSyngas />, label: "Syngas", sub: "H₂ + CO production" },
            { icon: <IconSynthesis />, label: "Methanol Synthesis", sub: "Catalytic conversion" },
            { icon: <IconAI />, label: "AI Optimization", sub: "ML-driven control" },
          ].map((step, i) => (
            <div key={step.label} className="flex items-center">
              <div className="flex flex-col items-center gap-2.5 w-24 sm:w-28 text-center">
                <div className="w-10 h-10 sm:w-11 sm:h-11 rounded-full bg-white/[0.04] border border-white/[0.1] flex items-center justify-center text-blue-400 transition-all duration-300 hover:bg-white/[0.08] hover:border-blue-400/30 hover:text-blue-300">
                  {step.icon}
                </div>
                <div>
                  <p className="text-[9px] sm:text-[10px] font-semibold text-slate-300 tracking-[0.06em] uppercase leading-tight">
                    {step.label}
                  </p>
                  <p className="text-[8px] sm:text-[9px] text-slate-600 mt-0.5">{step.sub}</p>
                </div>
              </div>
              {i < 4 && (
                <div className="hidden md:flex items-center mx-1 sm:mx-1.5 -mt-8">
                  <div className="w-6 sm:w-8 h-px bg-gradient-to-r from-white/[0.1] to-white/[0.04]" />
                  <svg viewBox="0 0 8 8" className="w-2 h-2 text-blue-500/40 -ml-0.5" fill="currentColor">
                    <path d="M0 0 L8 4 L0 8 Z" />
                  </svg>
                </div>
              )}
            </div>
          ))}
        </div>
      </footer>
    </div>
  )
}
