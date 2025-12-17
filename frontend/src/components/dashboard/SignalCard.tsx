"use client";

import { TrendingUp, TrendingDown, Minus } from "lucide-react";

interface SignalCardProps {
    symbol: string;
    price: number;
    change_pct: number;
    action: string;
    confidence: number;
    onClick?: () => void;
}

export function SignalCard({
    symbol,
    price,
    change_pct,
    action,
    confidence,
    onClick,
}: SignalCardProps) {
    const getActionColor = () => {
        switch (action) {
            case "BUY":
                return "bg-green-500/20 text-green-400 border-green-500/30";
            case "SELL":
                return "bg-red-500/20 text-red-400 border-red-500/30";
            default:
                return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
        }
    };

    const getActionIcon = () => {
        switch (action) {
            case "BUY":
                return <TrendingUp className="w-5 h-5" />;
            case "SELL":
                return <TrendingDown className="w-5 h-5" />;
            default:
                return <Minus className="w-5 h-5" />;
        }
    };

    const isPositive = change_pct >= 0;

    return (
        <div
            onClick={onClick}
            className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-4 hover:border-gray-600 transition-all cursor-pointer group"
        >
            <div className="flex justify-between items-start mb-3">
                <div>
                    <h3 className="text-lg font-semibold text-white group-hover:text-blue-400 transition-colors">
                        {symbol.replace(".NS", "")}
                    </h3>
                    <span className="text-xs text-gray-500">NSE</span>
                </div>
                <div
                    className={`px-3 py-1 rounded-full text-sm font-medium border ${getActionColor()} flex items-center gap-1`}
                >
                    {getActionIcon()}
                    {action}
                </div>
            </div>

            <div className="flex justify-between items-end">
                <div>
                    <p className="text-2xl font-bold text-white">₹{price.toFixed(2)}</p>
                    <p
                        className={`text-sm ${isPositive ? "text-green-400" : "text-red-400"}`}
                    >
                        {isPositive ? "+" : ""}
                        {change_pct.toFixed(2)}%
                    </p>
                </div>
                <div className="text-right">
                    <p className="text-xs text-gray-500">Confidence</p>
                    <p className="text-lg font-semibold text-blue-400">
                        {(confidence * 100).toFixed(0)}%
                    </p>
                </div>
            </div>

            {/* Confidence bar */}
            <div className="mt-3 h-1 bg-gray-700 rounded-full overflow-hidden">
                <div
                    className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full transition-all"
                    style={{ width: `${confidence * 100}%` }}
                />
            </div>
        </div>
    );
}
