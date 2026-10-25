
import {
  Activity,
  ArrowDown,
  ArrowRight,
  ArrowUpRight,
  Cpu,
  Gauge,
  ShieldCheck,
  Sparkles,
  Zap,
} from "lucide-react";
import "./LandingPage.css";

function LandingPage({ onLogin, onGetStarted }) {
  return (
    <div className="landing-page">
      <nav className="landing-nav">
        <a className="landing-brand" href="#home">
          <span className="landing-brand-mark">
            <Activity size={22} strokeWidth={2.5} />
          </span>
          <span className="landing-brand-name">
            nex<span>forge</span>
            <small>INDUSTRIAL INTELLIGENCE</small>
          </span>
        </a>

        <div className="landing-nav-links">
          <a href="#platform">Platform</a>
          <a href="#technology">Technology</a>
          <a href="#about">About</a>
        </div>

        <div className="landing-nav-actions">
          <button className="landing-login" onClick={onLogin}>
            Log in
          </button>
          <button className="landing-get-started" onClick={onGetStarted}>
            Get started <ArrowUpRight size={16} />
          </button>
        </div>
      </nav>

      <main id="home">
        <section className="landing-hero">
          <div className="landing-hero-copy">
            <div className="landing-announcement">
              <span>
                <Sparkles size={13} />
              </span>
              INTELLIGENCE FOR INDUSTRY 4.0
              <ArrowRight size={13} />
            </div>

            <h1>
              The future of
              <br />
              maintenance is
              <br />
              <span>predictive.</span>
            </h1>

            <p className="landing-description">
              Turn machine signals into meaningful insights. Detect failure
              risks earlier, make smarter decisions, and keep your operations
              moving forward.
            </p>

            <div className="landing-hero-actions">
              <button
                className="landing-primary-cta"
                onClick={onGetStarted}
              >
                Explore the platform <ArrowRight size={17} />
              </button>
              <a className="landing-secondary-cta" href="#platform">
                Discover how it works <ArrowDown size={16} />
              </a>
            </div>

            <div className="landing-trust-row">
              <span>
                <ShieldCheck size={16} /> AI-driven insights
              </span>
              <span>
                <Zap size={16} /> Smarter maintenance
              </span>
            </div>
          </div>

          <div className="landing-visual">
            <div className="landing-visual-grid" />
            <div className="landing-visual-glow" />

            <div className="landing-visual-label visual-label-top">
              <span className="landing-pulse-dot" />
              NEURAL SYSTEM
              <strong>READY</strong>
            </div>

            <div className="landing-orbit landing-orbit-outer" />
            <div className="landing-orbit landing-orbit-middle" />
            <div className="landing-orbit landing-orbit-inner" />

            <div className="landing-ai-core">
              <div className="landing-core-icon">
                <Cpu size={39} strokeWidth={1.3} />
              </div>
              <span>NEURAL ENGINE</span>
              <strong>AI / 01</strong>
            </div>

            <div className="landing-orbit-node node-one" />
            <div className="landing-orbit-node node-two" />
            <div className="landing-orbit-node node-three" />

            <div className="landing-floating-card floating-card-left">
              <span className="floating-card-icon">
                <Gauge size={17} />
              </span>
              <div>
                <small>Machine intelligence</small>
                <strong>Predictive insights</strong>
              </div>
            </div>

            <div className="landing-floating-card floating-card-right">
              <span className="floating-card-icon">
                <Activity size={17} />
              </span>
              <div>
                <small>Neural network</small>
                <strong>Inference engine</strong>
              </div>
              <span className="landing-pulse-dot" />
            </div>

            <div className="landing-visual-caption">
              <span>01 — INTELLIGENT SYSTEMS</span>
              <span>PRECISION / PREDICTION</span>
            </div>
          </div>

          <div className="landing-hero-bottom">
            <span>ENGINEERED FOR THE NEXT INDUSTRIAL ERA</span>
            <span className="landing-scroll-indicator">
              <span /> SCROLL TO EXPLORE
            </span>
          </div>
        </section>

        <section className="landing-platform" id="platform">
          <div className="landing-section-heading">
            <span className="landing-section-kicker">01 / THE PLATFORM</span>
            <h2>
              Machines don't have to
              <br />
              fail <span>without warning.</span>
            </h2>
            <p>
              Predictive intelligence helps transform operating parameters
              into actionable signals — before problems turn into downtime.
            </p>
          </div>

          <div className="landing-feature-grid">
            <article className="landing-feature-card">
              <span className="landing-feature-number">01</span>
              <div className="landing-feature-icon">
                <Activity size={22} />
              </div>
              <h3>Understand machine signals</h3>
              <p>
                Explore sensor parameters such as temperature, speed, torque,
                and tool wear.
              </p>
              <span className="landing-feature-link">
                SENSOR INTELLIGENCE <ArrowUpRight size={14} />
              </span>
            </article>

            <article className="landing-feature-card">
              <span className="landing-feature-number">02</span>
              <div className="landing-feature-icon">
                <Cpu size={22} />
              </div>
              <h3>Predict with AI</h3>
              <p>
                Use an artificial neural network to estimate machine failure
                probability from input readings.
              </p>
              <span className="landing-feature-link">
                NEURAL INFERENCE <ArrowUpRight size={14} />
              </span>
            </article>

            <article className="landing-feature-card">
              <span className="landing-feature-number">03</span>
              <div className="landing-feature-icon">
                <ShieldCheck size={22} />
              </div>
              <h3>Make informed decisions</h3>
              <p>
                Review prediction results and build a clearer picture of
                potential machine risks.
              </p>
              <span className="landing-feature-link">
                DECISION SUPPORT <ArrowUpRight size={14} />
              </span>
            </article>
          </div>
        </section>

        <section className="landing-technology" id="technology">
          <div className="landing-tech-visual">
            <div className="landing-tech-grid" />
            <div className="landing-tech-core">
              <Cpu size={32} />
              <span>ANN</span>
            </div>
            <div className="landing-tech-orbit tech-orbit-one" />
            <div className="landing-tech-orbit tech-orbit-two" />
            <div className="landing-tech-tag">DENSE NEURAL NETWORK</div>
          </div>

          <div className="landing-tech-copy">
            <span className="landing-section-kicker">
              02 / THE TECHNOLOGY
            </span>
            <h2>
              Built on data.
              <br />
              <span>Powered by intelligence.</span>
            </h2>
            <p>
              A trained artificial neural network analyses machine parameters
              and produces a failure probability to support predictive
              maintenance workflows.
            </p>

            <div className="landing-tech-points">
              <span>
                <ShieldCheck size={17} /> Structured sensor inputs
              </span>
              <span>
                <Zap size={17} /> Neural network inference
              </span>
              <span>
                <Gauge size={17} /> Probability-based assessment
              </span>
            </div>

            <a className="landing-tech-link" href="#about">
              Explore the approach <ArrowRight size={16} />
            </a>
          </div>
        </section>

        <section className="landing-cta" id="about">
          <div className="landing-cta-glow" />
          <span className="landing-section-kicker">
            YOUR NEXT STEP STARTS HERE
          </span>
          <h2>
            Make every machine
            <br />
            <span>more predictable.</span>
          </h2>
          <p>Discover a smarter way to approach industrial maintenance.</p>
          <button
            className="landing-primary-cta"
            onClick={onGetStarted}
          >
            Get started <ArrowRight size={17} />
          </button>
        </section>
      </main>

      <footer className="landing-footer">
        <a className="landing-brand" href="#home">
          <span className="landing-brand-mark">
            <Activity size={19} />
          </span>
          <span className="landing-brand-name">
            nex<span>forge</span>
            <small>INDUSTRIAL INTELLIGENCE</small>
          </span>
        </a>
        <p>AI-POWERED PREDICTIVE MAINTENANCE</p>
        <span>
          BUILT FOR INDUSTRY 4.0{" "}
          <span className="landing-footer-dot">●</span>
        </span>
      </footer>
    </div>
  );
}

export default LandingPage;
