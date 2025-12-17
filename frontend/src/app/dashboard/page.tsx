"use client";

import { useEffect, useState, useCallback } from "react";
import { SignalCard, StatsCard } from "@/components/dashboard";
import { tradingApi } from "@/lib/api";
import { RefreshCw, Clock, Wifi, WifiOff } from "lucide-react";

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
    const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
    const [isConnected, setIsConnected] = useState(true);
    const [autoRefresh, setAutoRefresh] = useState(true);

    const fetchSignals = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await tradingApi.getWatchlist();
            setSignals(data.signals || []);
            setLastUpdated(new Date());
            setIsConnected(true);
        } catch (err) {
            setError("Backend offline - showing cached data");
            setIsConnected(false);
            // Demo data when backend is down
            setSignals([
                { symbol: "RELIANCE.NS", price: 1544.40, change_pct: 0.14, action: "HOLD", confidence: 0.50 },
                { symbol: "TCS.NS", price: 3217.80, change_pct: 0.40, action: "HOLD", confidence: 0.50 },
                { symbol: "INFY.NS", price: 1602.00, change_pct: 0.57, action: "HOLD", confidence: 0.50 },
                { symbol: "HDFCBANK.NS", price: 984.00, change_pct: -1.04, action: "HOLD", confidence: 0.50 },
                { symbol: "ICICIBANK.NS", price: 1352.40, change_pct: -1.00, action: "HOLD", confidence: 0.50 },
                { symbol: "SBIN.NS", price: 838.50, change_pct: 0.32, action: "BUY", confidence: 0.65 },
            ]);
        } finally {
            setLoading(false);
        }
    }, []);

    // Initial fetch
    useEffect(() => {
        fetchSignals();
    }, [fetchSignals]);

    // Auto-refresh every 30 seconds
    useEffect(() => {
        if (!autoRefresh) return;

        const interval = setInterval(() => {
            fetchSignals();
        }, 30000);

        return () => clearInterval(interval);
    }, [autoRefresh, fetchSignals]);

    // Calculate portfolio metrics from signals
    const totalChange = signals.reduce((acc, s) => acc + s.change_pct, 0) / (signals.length || 1);
    const buySignals = signals.filter(s => s.action === "BUY").length;
    const sellSignals = signals.filter(s => s.action === "SELL").length;

    return (
        <div className="max-w-7xl mx-auto">
            {/* Header */}
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-8">
                <div>
                    <h1 className="text-3xl font-bold text-white">Dashboard</h1>
                    <p className="text-gray-400 mt-1">Real-Time Trading Signals for NSE/BSE</p>
                </div>
                <div className="flex items-center gap-3">
                    {/* Connection Status */}
                    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${isConnected
                            ? "bg-green-500/10 text-green-400 border border-green-500/30"
                            : "bg-red-500/10 text-red-400 border border-red-500/30"
                        }`}>
                        {isConnected ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
                        {isConnected ? "Live" : "Offline"}
                    </div>

                    {/* Auto Refresh Toggle */}
                    <button
                        onClick={() => setAutoRefresh(!autoRefresh)}
                        className={`px-3 py-1.5 rounded-full text-sm border transition-all ${autoRefresh
                                ? "bg-blue-500/20 text-blue-400 border-blue-500/30"
                                : "bg-gray-800 text-gray-400 border-gray-700"
                            }`}
                    >
                        Auto: {autoRefresh ? "ON" : "OFF"}
                    </button>

                    {/* Manual Refresh */}
                    <button
                        onClick={fetchSignals}
                        disabled={loading}
                        className="flex items-center gap-2 px-4 py-2 bg-blue-500/20 text-blue-400 rounded-lg border border-blue-500/30 hover:bg-blue-500/30 transition-all disabled:opacity-50"
                    >
                        <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
                        Refresh
                    </button>
                </div>
            </div>

            {/* Last Updated */}
            {lastUpdated && (
                <div className="flex items-center gap-2 text-sm text-gray-500 mb-6">
                    <Clock className="w-4 h-4" />
                    Last updated: {lastUpdated.toLocaleTimeString("en-IN", {
                        hour: "2-digit",
                        minute: "2-digit",
                        second: "2-digit"
                    })}
                    {autoRefresh && <span className="text-gray-600">• Auto-refreshing every 30s</span>}
                </div>
            )}

            {/* Error Banner */}
            {error && (
                <div className="mb-6 p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg text-yellow-400 text-sm">
                    ⚠️ {error}
                </div>
            )}

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                <StatsCard
                    title="Stocks Tracked"
                    value={signals.length.toString()}
                    icon="activity"
                    color="blue"
                />
                <StatsCard
                    title="Avg. Change"
                    value={`${totalChange >= 0 ? "+" : ""}${totalChange.toFixed(2)}%`}
                    change={totalChange}
                    icon={totalChange >= 0 ? "up" : "down"}
                    color={totalChange >= 0 ? "green" : "red"}
                />
                <StatsCard
                    title="Buy Signals"
                    value={buySignals.toString()}
                    icon="up"
                    color="green"
                />
                <StatsCard
                    title="Sell Signals"
                    value={sellSignals.toString()}
                    icon="down"
                    color="red"
                />
            </div>

            {/* Signals Section */}
            <div className="mb-6">
                <div className="flex justify-between items-center mb-4">
                    <h2 className="text-xl font-semibold text-white">Trading Signals</h2>
                    <span className="text-sm text-gray-500">{signals.length} stocks</span>
                </div>

                {loading && signals.length === 0 ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {[1, 2, 3, 4, 5, 6].map((i) => (
                            <div key={i} className="bg-gray-800/50 rounded-xl p-4 animate-pulse">
                                <div className="h-6 bg-gray-700 rounded w-2/3 mb-3"></div>
                                <div className="h-8 bg-gray-700 rounded w-1/2 mb-2"></div>
                                <div className="h-4 bg-gray-700 rounded w-1/3"></div>
                            </div>
                        ))}
                    </div>
                ) : (
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
                )}
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

            {/* Market Hours Notice */}
            <div className="mt-4 text-center text-sm text-gray-500">
                NSE Trading Hours: 9:15 AM - 3:30 PM IST (Mon-Fri)
            </div>
        </div>
    );
}
