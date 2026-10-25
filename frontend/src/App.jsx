
import { useEffect, useState } from "react";
import { onAuthStateChanged, signOut } from "firebase/auth";
import LandingPage from "./LandingPage";
import AuthPage from "./AuthPage";
import Dashboard from "./Dashboard";
import { auth } from "./firebase";
import "./App.css";

function App() {
  const [page, setPage] = useState("landing");
  const [user, setUser] = useState(null);
  const [authLoading, setAuthLoading] = useState(true);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      setAuthLoading(false);

      if (currentUser) {
        setPage((currentPage) =>
          currentPage === "landing" ? "dashboard" : currentPage
        );
      } else {
        setPage((currentPage) =>
          currentPage === "dashboard" ? "landing" : currentPage
        );
      }
    });

    return unsubscribe;
  }, []);

  function handleAuthSuccess() {
    setPage("dashboard");
  }

  async function handleLogout() {
    try {
      await signOut(auth);
      setPage("landing");
    } catch (error) {
      console.error("Logout failed:", error);
    }
  }

  if (authLoading) {
    return (
      <div
        style={{
          minHeight: "100vh",
          display: "grid",
          placeItems: "center",
          background: "#090e13",
          color: "#5ce5d0",
          fontFamily: "Inter, sans-serif",
        }}
      >
        Loading NexForge...
      </div>
    );
  }

  if (page === "auth" && !user) {
    return (
      <AuthPage
        onBack={() => setPage("landing")}
        onSuccess={handleAuthSuccess}
      />
    );
  }

  if (user && page === "dashboard") {
    return (
      <div>
        <div
          style={{
            position: "fixed",
            top: 16,
            right: 20,
            zIndex: 1000,
            display: "flex",
            alignItems: "center",
            gap: 12,
            padding: "8px 12px",
            border: "1px solid #293329",
            borderRadius: 10,
            background: "#101512",
            color: "#edf0e6",
            fontSize: 12,
          }}
        >
          <span>{user.displayName || user.email}</span>
          <button
            onClick={handleLogout}
            style={{
              background: "#D3F477",
              color: "#10150B",
              border: "1px solid #D3F477",
              borderRadius: 8,
              padding: "9px 12px",
              fontWeight: 800,
              cursor: "pointer",
              transition: "all 0.2s ease",
            }}
          >
            Log out
          </button>
        </div>
        <Dashboard />
      </div>
    );
  }

  return (
    <LandingPage
      onLogin={() => setPage("auth")}
      onGetStarted={() => setPage("auth")}
    />
  );
}

export default App;
