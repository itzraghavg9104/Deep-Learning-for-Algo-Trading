"use client";

import { useEffect, useState } from "react";
import { SignalCard, StatsCard } from "@/components/dashboard";
import { tradingApi } from "@/lib/api";
import { RefreshCw } from "lucide-react";

interface Signal {
    symbol: string;
    price: number;
    change_pct: number;
    action: string;
    confidence: number;
}

export default function DashboardPage() {
    const [signals, setSignals] = useState<Signal[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchSignals = async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await tradingApi.getWatchlist();
            setSignals(data.signals || []);
        } catch (err) {
            setError("Failed to fetch signals. Make sure the backend is running.");
            // Demo data for UI preview
            setSignals([
                { symbol: "RELIANCE.NS", price: 2485.50, change_pct: 1.25, action: "BUY", confidence: 0.78 },
                { symbol: "TCS.NS", price: 3890.25, change_pct: -0.45, action: "HOLD", confidence: 0.65 },
                { symbol: "INFY.NS", price: 1520.80, change_pct: 2.10, action: "BUY", confidence: 0.82 },
                { symbol: "HDFCBANK.NS", price: 1680.15, change_pct: -1.20, action: "SELL", confidence: 0.71 },
                { symbol: "ICICIBANK.NS", price: 1095.60, change_pct: 0.85, action: "HOLD", confidence: 0.58 },
            ]);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchSignals();
    }, []);

    return (
        <div className="max-w-7xl mx-auto">
            {/* Header */}
            <div className="flex justify-between items-center mb-8">
                <div>
                    <h1 className="text-3xl font-bold text-white">Dashboard</h1>
                    <p className="text-gray-400 mt-1">AI-Powered Trading Signals for NSE/BSE</p>
                </div>
                <button
                    onClick={fetchSignals}
                    disabled={loading}
                    className="flex items-center gap-2 px-4 py-2 bg-blue-500/20 text-blue-400 rounded-lg border border-blue-500/30 hover:bg-blue-500/30 transition-all disabled:opacity-50"
                >
                    <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
                    Refresh
                </button>
            </div>

            {/* Error Banner */}
            {error && (
                <div className="mb-6 p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg text-yellow-400 text-sm">
                    ⚠️ {error}
                </div>
            )}

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                <StatsCard
                    title="Portfolio Value"
                    value="₹10,00,000"
                    change={5.2}
                    icon="up"
                    color="green"
                />
                <StatsCard
                    title="Today's P&L"
                    value="₹12,450"
                    change={1.25}
                    icon="activity"
                    color="blue"
                />
                <StatsCard
                    title="Sharpe Ratio"
                    value="1.45"
                    icon="target"
                    color="purple"
                />
                <StatsCard
                    title="Win Rate"
                    value="62%"
                    icon="up"
                    color="green"
                />
            </div>

            {/* Signals Section */}
            <div className="mb-6">
                <h2 className="text-xl font-semibold text-white mb-4">Trading Signals</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {signals.map((signal) => (
                        <SignalCard
                            key={signal.symbol}
                            symbol={signal.symbol}
                            price={signal.price}
                            change_pct={signal.change_pct}
                            action={signal.action}
                            confidence={signal.confidence}
                        />
                    ))}
                </div>
            </div>

            {/* Model Info */}
            <div className="mt-8 p-6 bg-gray-800/30 border border-gray-700 rounded-xl">
                <h3 className="text-lg font-semibold text-white mb-2">About the Model</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                    <div>
                        <p className="text-gray-400">Layer 1: Prediction</p>
                        <p className="text-white">DeepAR-Attention + 30+ Indicators</p>
                    </div>
                    <div>
                        <p className="text-gray-400">Layer 2: Decision</p>
                        <p className="text-white">PPO Agent (Sharpe Optimized)</p>
                    </div>
                    <div>
                        <p className="text-gray-400">Behavior Module</p>
                        <p className="text-white">Risk-Adjusted Position Sizing</p>
                    </div>
                </div>
            </div>
        </div>
    );
}
