
import { useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  CheckCircle2,
  Cpu,
  Gauge,
  RotateCcw,
  ShieldCheck,
  Thermometer,
  Zap,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
} from "recharts";
import "./App.css";

const initialSensors = {
  airTemperature: "300",
  processTemperature: "310",
  rotationalSpeed: "1500",
  torque: "40",
  toolWear: "100",
};

const chartData = [
  { time: "08:00", value: 24 },
  { time: "09:00", value: 31 },
  { time: "10:00", value: 28 },
  { time: "11:00", value: 42 },
  { time: "12:00", value: 36 },
  { time: "13:00", value: 48 },
  { time: "14:00", value: 41 },
  { time: "15:00", value: 35 },
  { time: "16:00", value: 44 },
];

const sensorFields = [
  {
    key: "airTemperature",
    label: "Air temperature",
    unit: "K",
    icon: Thermometer,
  },
  {
    key: "processTemperature",
    label: "Process temperature",
    unit: "K",
    icon: Thermometer,
  },
  {
    key: "rotationalSpeed",
    label: "Rotational speed",
    unit: "RPM",
    icon: RotateCcw,
  },
  { key: "torque", label: "Torque", unit: "Nm", icon: Gauge },
  { key: "toolWear", label: "Tool wear", unit: "min", icon: Activity },
];

function Dashboard() {
  const [sensors, setSensors] = useState(initialSensors);
  const [machineType, setMachineType] = useState("M");
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function updateSensor(key, value) {
    setSensors((previous) => ({ ...previous, [key]: value }));
  }

  async function handlePrediction(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setPrediction(null);

    const payload = {
      machine_type: machineType,
      air_temperature: Number(sensors.airTemperature),
      process_temperature: Number(sensors.processTemperature),
      rotational_speed: Number(sensors.rotationalSpeed),
      torque: Number(sensors.torque),
      tool_wear: Number(sensors.toolWear),
    };

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const result = await response.json();

      if (!response.ok) {
        const detail = Array.isArray(result.detail)
          ? result.detail.map((item) => item.msg).join(", ")
          : result.detail;

        throw new Error(detail || "Prediction request failed.");
      }

      setPrediction(result);
    } catch (err) {
      setError(
        err.message ||
          "Could not connect to the prediction API. Make sure the FastAPI backend is running."
      );
    } finally {
      setLoading(false);
    }
  }

  const failureDetected = prediction?.prediction === "MACHINE FAILURE";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="/">
          <span className="brand-mark">
            <Cpu size={22} />
          </span>
          <span>
            <strong>NEXFORGE</strong>
            <small>INDUSTRIAL INTELLIGENCE</small>
          </span>
        </a>

        <div className="sidebar-label">WORKSPACE</div>
        <nav className="sidebar-nav">
          <a className="nav-item active" href="#overview">
            <Gauge size={18} /> Overview
          </a>
          <a className="nav-item" href="#prediction">
            <Activity size={18} /> AI Prediction
          </a>
          <a className="nav-item" href="#sensors">
            <Cpu size={18} /> Machine Sensors
          </a>
        </nav>

        <div className="sidebar-bottom">
          <span className="status-dot" />
          <span>Predictive engine</span>
          <span className="online-label">READY</span>
        </div>
      </aside>

      <main className="main-content" id="overview">
        <header className="topbar">
          <div>
            <div className="eyebrow">INTELLIGENCE CENTER / OVERVIEW</div>
            <h1>Machine Intelligence</h1>
          </div>
          <div className="topbar-status">
            <span className="status-dot" /> AI workspace
          </div>
        </header>

        <section className="hero-section">
          <div className="hero-copy">
            <span className="section-kicker">
              <Zap size={14} /> PREDICTIVE MAINTENANCE
            </span>
            <h2>
              Detect failure
              <br />
              <span>before it happens.</span>
            </h2>
            <p>
              Analyze machine operating conditions with the optimized XGBoost
              model to identify potential equipment failure risks.
            </p>
            <a className="hero-button" href="#prediction">
              Run an analysis <ArrowUpRight size={17} />
            </a>
          </div>

          <div className="hero-visual" aria-hidden="true">
            <div className="orbit orbit-one" />
            <div className="orbit orbit-two" />
            <div className="orbit orbit-three" />
            <div className="core-glow">
              <Cpu size={64} strokeWidth={1.2} />
            </div>
            <span className="orbit-node node-one" />
            <span className="orbit-node node-two" />
            <span className="orbit-node node-three" />
          </div>

          <div className="hero-index">SYSTEM 01 / XGBOOST</div>
        </section>

        <section className="metrics-grid">
          <article className="metric-card">
            <div className="metric-top">
              <span>MONITORING ENGINE</span>
              <Cpu size={18} />
            </div>
            <h3>XGBoost</h3>
            <p>Optimized classification model</p>
          </article>

          <article className="metric-card">
            <div className="metric-top">
              <span>INPUT PARAMETERS</span>
              <Activity size={18} />
            </div>
            <h3>05 + TYPE</h3>
            <p>Operating condition features</p>
          </article>

          <article className="metric-card">
            <div className="metric-top">
              <span>OUTPUT</span>
              <ShieldCheck size={18} />
            </div>
            <h3>Binary</h3>
            <p>Failure / no failure</p>
          </article>
        </section>

        <section className="analysis-grid">
          <article className="panel chart-panel">
            <div className="panel-heading">
              <div>
                <div className="eyebrow">MACHINE TELEMETRY</div>
                <h3>Operating signal</h3>
              </div>
              <span className="panel-tag">ILLUSTRATIVE DATA</span>
            </div>

            <div className="chart-legend">
              <span className="legend-dot" /> Relative signal index
            </div>

            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient
                      id="signalGradient"
                      x1="0"
                      y1="0"
                      x2="0"
                      y2="1"
                    >
                      <stop
                        offset="0%"
                        stopColor="#D3F477"
                        stopOpacity={0.3}
                      />
                      <stop
                        offset="95%"
                        stopColor="#D3F477"
                        stopOpacity={0}
                      />
                    </linearGradient>
                  </defs>
                  <CartesianGrid
                    stroke="#25313c"
                    strokeDasharray="4 5"
                    vertical={false}
                  />
                  <XAxis
                    dataKey="time"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: "#81909e", fontSize: 11 }}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "#101820",
                      border: "1px solid #25313c",
                      borderRadius: 10,
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="value"
                    stroke="#D3F477"
                    strokeWidth={2}
                    fill="url(#signalGradient)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <p className="chart-note">
              Sample visualization only. Live telemetry is not connected.
            </p>
          </article>

          <article className="panel engine-panel">
            <div className="panel-heading">
              <div>
                <div className="eyebrow">MODEL STATUS</div>
                <h3>XGBoost engine</h3>
              </div>
              <span className="engine-icon">
                <Cpu size={20} />
              </span>
            </div>

            <div className="engine-status">
              <span className="status-dot" />
              {loading ? "Processing input" : "Ready for assessment"}
            </div>

            <div className="engine-graphic">
              <div className="engine-ring">
                <Activity size={42} />
              </div>
            </div>

            <div className="engine-detail">
              <span>Inference mode</span>
              <strong>Binary classification</strong>
            </div>

            <div className="engine-detail">
              <span>Dataset</span>
              <strong>AI4I 2020</strong>
            </div>
          </article>
        </section>

        <section className="panel prediction-panel" id="prediction">
          <div className="panel-heading">
            <div>
              <div className="eyebrow">XGBOOST INFERENCE</div>
              <h3>Run machine assessment</h3>
              <p className="panel-description">
                Enter the five operating features used by the trained model.
              </p>
            </div>
            <span className="panel-tag">LIVE API REQUEST</span>
          </div>

          <form onSubmit={handlePrediction}>
            <div className="machine-type-select">
              <span className="field-label">MACHINE TYPE</span>
              <div className="type-options">
                {["L", "M", "H"].map((type) => (
                  <button
                    type="button"
                    key={type}
                    className={
                      machineType === type
                        ? "type-option selected"
                        : "type-option"
                    }
                    onClick={() => setMachineType(type)}
                  >
                    Type {type}
                  </button>
                ))}
              </div>
            </div>

            <div className="sensor-grid" id="sensors">
              {sensorFields.map(({ key, label, unit, icon: Icon }) => (
                <label className="sensor-field" key={key}>
                  <span className="field-label">
                    <Icon size={15} /> {label}
                  </span>
                  <span className="sensor-input-wrap">
                    <input
                      type="number"
                      step="any"
                      required
                      value={sensors[key]}
                      onChange={(event) =>
                        updateSensor(key, event.target.value)
                      }
                    />
                    <span className="sensor-unit">{unit}</span>
                  </span>
                </label>
              ))}
            </div>

            {error && <p className="prediction-error">{error}</p>}

            <button
              className="predict-button"
              type="submit"
              disabled={loading}
            >
              {loading ? "Analyzing machine..." : "Run XGBoost prediction"}
              {!loading && <ArrowUpRight size={18} />}
            </button>
          </form>

          {prediction && (
            <div className="prediction-result">
              <div className="result-icon">
                {failureDetected ? (
                  <AlertTriangle size={25} />
                ) : (
                  <CheckCircle2 size={25} />
                )}
              </div>

              <div className="result-copy">
                <div className="eyebrow">PREDICTION RESULT</div>
                <h4>
                  {failureDetected
                    ? "Potential machine failure detected"
                    : "No machine failure predicted"}
                </h4>
                <p>
                  Model output: {prediction.prediction}
                  {prediction.failure_probability_percent != null
                    ? ` · Failure probability: ${Number(
                        prediction.failure_probability_percent
                      ).toFixed(2)}%`
                    : prediction.failure_probability != null
                      ? ` · Failure probability: ${(
                          Number(prediction.failure_probability) * 100
                        ).toFixed(2)}%`
                      : ""}
                </p>
                {prediction.model && (
                  <p>Model: {prediction.model}</p>
                )}
              </div>
            </div>
          )}
        </section>

        <footer className="dashboard-footer">
          <span>NEXFORGE / INDUSTRIAL AI</span>
          <span>AI4I 2020 · XGBOOST-POWERED ANALYSIS</span>
        </footer>
      </main>
    </div>
  );
}

export default Dashboard;
