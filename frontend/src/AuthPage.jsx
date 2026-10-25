
import { useState } from "react";
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  Eye,
  EyeOff,
  LockKeyhole,
  Mail,
  ShieldCheck,
  UserRound,
} from "lucide-react";
import {
  createUserWithEmailAndPassword,
  GoogleAuthProvider,
  signInWithEmailAndPassword,
  signInWithPopup,
  updateProfile,
} from "firebase/auth";
import { auth } from "./firebase";
import "./AuthPage.css";

const googleProvider = new GoogleAuthProvider();

function AuthPage({ onBack, onSuccess }) {
  const [mode, setMode] = useState("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setLoading(true);

    try {
      let result;

      if (mode === "signup") {
        result = await createUserWithEmailAndPassword(
          auth,
          email.trim(),
          password
        );

        if (name.trim()) {
          await updateProfile(result.user, {
            displayName: name.trim(),
          });
        }
      } else {
        result = await signInWithEmailAndPassword(
          auth,
          email.trim(),
          password
        );
      }

      onSuccess?.(result.user);
    } catch (err) {
      const messages = {
        "auth/email-already-in-use": "This email is already registered. Try logging in.",
        "auth/invalid-email": "Please enter a valid email address.",
        "auth/weak-password": "Use a stronger password with at least 6 characters.",
        "auth/invalid-credential": "Email or password is incorrect.",
        "auth/user-not-found": "No account found. Please sign up first.",
        "auth/wrong-password": "Email or password is incorrect.",
        "auth/popup-closed-by-user": "Google sign-in was cancelled.",
        "auth/popup-blocked": "Your browser blocked the sign-in popup. Allow popups and try again.",
        "auth/unauthorized-domain": "This domain is not authorized in Firebase Authentication settings.",
        "auth/operation-not-allowed": "This sign-in method is not enabled in Firebase Console.",
        "auth/network-request-failed": "Network error. Check your internet connection.",
      };

      setError(
        messages[err.code] ||
          err.message ||
          "Authentication failed. Please try again."
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleGoogleSignIn() {
    setError("");
    setLoading(true);

    try {
      const result = await signInWithPopup(auth, googleProvider);
      onSuccess?.(result.user);
    } catch (err) {
      const messages = {
        "auth/popup-closed-by-user": "Google sign-in was cancelled.",
        "auth/popup-blocked": "Allow popups in your browser and try again.",
        "auth/unauthorized-domain": "This domain is not authorized in Firebase settings.",
        "auth/operation-not-allowed": "Enable Google sign-in in Firebase Console.",
      };

      setError(
        messages[err.code] ||
          err.message ||
          "Google sign-in failed. Please try again."
      );
    } finally {
      setLoading(false);
    }
  }

  function switchMode(nextMode) {
    setMode(nextMode);
    setError("");
    setPassword("");
  }

  return (
    <div className="auth-page">
      <header className="auth-nav">
        <button className="auth-back" onClick={onBack} type="button">
          <ArrowLeft size={17} /> Back to home
        </button>

        <a className="auth-brand" href="#auth">
          <span className="auth-brand-icon">
            <Activity size={21} />
          </span>
          <span>
            nex<span>forge</span>
            <small>INDUSTRIAL INTELLIGENCE</small>
          </span>
        </a>

        <span className="auth-secure-label">
          <ShieldCheck size={16} /> SECURE ACCESS
        </span>
      </header>

      <main className="auth-main" id="auth">
        <section className="auth-intro">
          <div className="auth-kicker">
            <span /> INDUSTRIAL INTELLIGENCE PLATFORM
          </div>

          <h1>
            Smarter machines.
            <br />
            <span>Stronger decisions.</span>
          </h1>

          <p>
            Access your predictive maintenance workspace and turn machine
            operating data into actionable insights.
          </p>

          <div className="auth-benefits">
            <div>
              <span className="auth-benefit-icon">
                <Activity size={19} />
              </span>
              <span>
                <strong>Predictive insights</strong>
                <small>Analyze machine failure risk</small>
              </span>
            </div>

            <div>
              <span className="auth-benefit-icon">
                <LockKeyhole size={19} />
              </span>
              <span>
                <strong>Protected access</strong>
                <small>Your account is managed by Firebase</small>
              </span>
            </div>
          </div>
        </section>

        <section className="auth-card">
          <div className="auth-card-heading">
            <span className="auth-card-kicker">
              {mode === "login" ? "WELCOME BACK" : "CREATE YOUR ACCOUNT"}
            </span>

            <h2>
              {mode === "login" ? "Sign in to NexForge" : "Join NexForge"}
            </h2>

            <p>
              {mode === "login"
                ? "Enter your credentials to continue."
                : "Set up your account to get started."}
            </p>
          </div>

          <div className="auth-mode-switch">
            <button
              type="button"
              className={mode === "login" ? "active" : ""}
              onClick={() => switchMode("login")}
            >
              Log in
            </button>

            <button
              type="button"
              className={mode === "signup" ? "active" : ""}
              onClick={() => switchMode("signup")}
            >
              Sign up
            </button>
          </div>

          <button
            type="button"
            className="auth-google-button"
            onClick={handleGoogleSignIn}
            disabled={loading}
          >
            <svg viewBox="0 0 48 48" aria-hidden="true">
              <path
                fill="#EA4335"
                d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"
                transform="translate(4 4) scale(.83)"
              />
              <path
                fill="#4285F4"
                d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.72 7.18l7.62 5.91c4.45-4.11 7.14-10.16 7.14-17.56z"
              />
              <path
                fill="#FBBC05"
                d="M10.53 28.59A14.4 14.4 0 0 1 9.75 24c0-1.59.27-3.13.76-4.59l-7.98-6.2A23.9 23.9 0 0 0 0 24c0 3.87.93 7.52 2.56 10.78l7.97-6.19z"
                transform="translate(4 4) scale(.83)"
              />
              <path
                fill="#34A853"
                d="M24 48c6.48 0 11.93-2.14 15.91-5.81l-7.62-5.91c-2.11 1.41-4.8 2.25-8.29 2.25-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.2C6.51 42.62 14.62 48 24 48z"
                transform="translate(4 4) scale(.83)"
              />
            </svg>
            Continue with Google
          </button>

          <div className="auth-divider">
            <span /> <small>OR CONTINUE WITH EMAIL</small> <span />
          </div>

          <form className="auth-form" onSubmit={handleSubmit}>
            {mode === "signup" && (
              <label className="auth-field">
                <span>Full name</span>
                <div className="auth-input-wrap">
                  <UserRound size={17} />
                  <input
                    type="text"
                    placeholder="Your name"
                    autoComplete="name"
                    value={name}
                    onChange={(event) => setName(event.target.value)}
                    required
                  />
                </div>
              </label>
            )}

            <label className="auth-field">
              <span>Email address</span>
              <div className="auth-input-wrap">
                <Mail size={17} />
                <input
                  type="email"
                  placeholder="you@example.com"
                  autoComplete="email"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  required
                />
              </div>
            </label>

            <label className="auth-field">
              <span>Password</span>
              <div className="auth-input-wrap">
                <LockKeyhole size={17} />
                <input
                  type={showPassword ? "text" : "password"}
                  placeholder="At least 6 characters"
                  autoComplete={
                    mode === "login" ? "current-password" : "new-password"
                  }
                  minLength={6}
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  required
                />
                <button
                  type="button"
                  className="auth-password-toggle"
                  aria-label={showPassword ? "Hide password" : "Show password"}
                  onClick={() => setShowPassword((previous) => !previous)}
                >
                  {showPassword ? <EyeOff size={17} /> : <Eye size={17} />}
                </button>
              </div>
            </label>

            {error && (
              <div className="auth-error" role="alert">
                {error}
              </div>
            )}

            <button
              className="auth-submit"
              type="submit"
              disabled={loading}
            >
              {loading
                ? "Please wait..."
                : mode === "login"
                  ? "Sign in"
                  : "Create account"}
              {!loading && <ArrowRight size={17} />}
            </button>
          </form>

          <p className="auth-switch-copy">
            {mode === "login"
              ? "New to NexForge?"
              : "Already have an account?"}{" "}
            <button
              type="button"
              onClick={() =>
                switchMode(mode === "login" ? "signup" : "login")
              }
            >
              {mode === "login" ? "Create an account" : "Log in"}
            </button>
          </p>

          <div className="auth-security-note">
            <ShieldCheck size={15} />
            Authentication secured by Firebase
          </div>
        </section>
      </main>

      <footer className="auth-footer">
        <span>NEXFORGE / INDUSTRIAL AI</span>
        <span>BUILT FOR INDUSTRY 4.0</span>
      </footer>
    </div>
  );
}

export default AuthPage;
