import axios from "axios";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

const getPersistedToken = () => {
  if (typeof window === "undefined") return null;
  const raw = localStorage.getItem("auth-storage");
  if (!raw) return null;

  try {
    const parsed = JSON.parse(raw);
    return parsed?.state?.token ?? null;
  } catch {
    return null;
  }
};

api.interceptors.request.use((config) => {
  const token = getPersistedToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Auth API
export const authApi = {
  login: async (email: string, password: string) => {
    const formData = new URLSearchParams();
    formData.append("username", email);
    formData.append("password", password);

    const response = await api.post("/auth/login", formData, {
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
    });
    return response.data;
  },

  register: async (email: string, password: string) => {
    const response = await api.post("/auth/register", {
      email,
      password,
    });
    return response.data;
  },

  getMe: async () => {
    const response = await api.get("/auth/me");
    return response.data;
  },
};

// Response interceptor for auth failures
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      if (typeof window !== "undefined") {
        localStorage.removeItem("auth-storage");
        document.cookie = "auth_token=; Path=/; Max-Age=0; SameSite=Lax";
        window.location.href = "/auth/login";
      }
    }
    return Promise.reject(error);
  }
);

// Trading API
export const tradingApi = {
  getSignal: async (
    symbol: string,
    useSentiment: boolean = false,
    params?: Record<string, string | number | boolean | undefined>,
  ) => {
    const response = await api.get(`/trading/signals/${symbol}`, {
      params: { use_sentiment: useSentiment, ...(params || {}) },
    });
    return response.data;
  },

  getMarketData: async (symbol: string, period: string = "1mo") => {
    const response = await api.get(`/trading/market/${symbol}`, {
      params: { period },
    });
    return response.data;
  },

  getWatchlist: async (params?: Record<string, string | number | boolean | undefined>) => {
    const response = await api.get("/trading/watchlist", { params: params || {} });
    return response.data;
  },
};

// Backtest API
export const backtestApi = {
  runBacktest: async (config: {
    symbol: string;
    start_date: string;
    end_date: string;
    initial_capital: number;
    risk_tolerance: number;
  }) => {
    const response = await api.post("/backtest/run", config);
    return response.data;
  },
};

// Profile API
export const profileApi = {
  getProfile: async () => {
    const response = await api.get("/profile");
    return response.data;
  },

  submitRiskAssessment: async (answers: number[]) => {
    const response = await api.post("/profile/risk-assessment", { answers });
    return response.data;
  },

  submitBehaviorAssessment: async (answers: Record<string, unknown>) => {
    const response = await api.post("/profile/behavior-assessment", { answers });
    return response.data;
  },

  getModelTrainingStatus: async () => {
    const response = await api.get("/profile/model-training-status");
    return response.data;
  },

  updatePreferences: async (preferences: {
    use_sentiment: boolean;
    preferred_timeframe: string;
    symbols: string[];
  }) => {
    const response = await api.put("/profile/preferences", preferences);
    return response.data;
  },
};

// Trades API
export const tradesApi = {
  getTradeHistory: async () => {
    const response = await api.get("/profile/trades");
    return response.data;
  },
};

export default api;
