import axios from "axios";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// Trading API
export const tradingApi = {
  getSignal: async (symbol: string, useSentiment: boolean = false) => {
    const response = await api.get(`/trading/signals/${symbol}`, {
      params: { use_sentiment: useSentiment },
    });
    return response.data;
  },

  getMarketData: async (symbol: string, period: string = "1mo") => {
    const response = await api.get(`/trading/market/${symbol}`, {
      params: { period },
    });
    return response.data;
  },

  getWatchlist: async () => {
    const response = await api.get("/trading/watchlist");
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

  updatePreferences: async (preferences: {
    use_sentiment: boolean;
    preferred_timeframe: string;
    symbols: string[];
  }) => {
    const response = await api.put("/profile/preferences", preferences);
    return response.data;
  },
};

export default api;
