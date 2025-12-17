import { create } from "zustand";

interface Signal {
    symbol: string;
    action: string;
    confidence: number;
    price: number;
    change_pct: number;
}

interface RiskProfile {
    tolerance: number;
    category: string;
}

interface AppState {
    // Watchlist
    signals: Signal[];
    isLoadingSignals: boolean;
    setSignals: (signals: Signal[]) => void;
    setLoadingSignals: (loading: boolean) => void;

    // Risk Profile
    riskProfile: RiskProfile | null;
    setRiskProfile: (profile: RiskProfile) => void;

    // Preferences
    useSentiment: boolean;
    setUseSentiment: (use: boolean) => void;

    // Selected Symbol
    selectedSymbol: string;
    setSelectedSymbol: (symbol: string) => void;
}

export const useAppStore = create<AppState>((set) => ({
    // Watchlist
    signals: [],
    isLoadingSignals: false,
    setSignals: (signals) => set({ signals }),
    setLoadingSignals: (loading) => set({ isLoadingSignals: loading }),

    // Risk Profile
    riskProfile: null,
    setRiskProfile: (profile) => set({ riskProfile: profile }),

    // Preferences
    useSentiment: false,
    setUseSentiment: (use) => set({ useSentiment: use }),

    // Selected Symbol
    selectedSymbol: "RELIANCE.NS",
    setSelectedSymbol: (symbol) => set({ selectedSymbol: symbol }),
}));
