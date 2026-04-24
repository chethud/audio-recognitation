import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";
import * as api from "../api/api.js";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [ready, setReady] = useState(false);

  const applySession = useCallback((token, u) => {
    api.setAuthToken(token);
    setUser(u);
  }, []);

  const login = useCallback(
    async (email, password) => {
      const data = await api.loginRequest(email, password);
      applySession(data.access_token, data.user);
      return data;
    },
    [applySession]
  );

  const signup = useCallback(
    async (email, password) => {
      const data = await api.signupRequest(email, password);
      applySession(data.access_token, data.user);
      return data;
    },
    [applySession]
  );

  const logout = useCallback(async () => {
    try {
      await api.logoutRequest();
    } catch {
      api.clearAuthToken();
    }
    setUser(null);
  }, []);

  useEffect(() => {
    const t = localStorage.getItem(api.AUTH_STORAGE_KEY);
    if (!t) {
      setReady(true);
      return;
    }
    api.setAuthToken(t);
    api
      .fetchMe()
      .then(setUser)
      .catch(() => {
        api.clearAuthToken();
        setUser(null);
      })
      .finally(() => setReady(true));
  }, []);

  const value = { user, ready, login, signup, logout };
  return (
    <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return ctx;
}
