"use client";

import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { Sparkline } from "@/components/charts/Sparkline";
import { ACTIONS, formatChangePct, normalizeAction } from "@/lib/trading-format";

interface SignalCardProps {
    symbol: string;
    price: number;
    change_pct: number;
    action: string;
    confidence: number;
    target_price?: number | null;
    trade_plan?: {
        capital_per_trade_pct?: number;
        tp_sl_ratio_target?: number;
        capital_amount_inr?: number;
        profit_target_exit_price?: number;
    } | null;
    onClick?: () => void;
    sparkline?: number[];
    flash?: "up" | "down";
}

export function SignalCard({
    symbol,
    price,
    change_pct,
    action,
    confidence,
    target_price,
    trade_plan,
    onClick,
    sparkline,
    flash,
}: SignalCardProps) {
    const normalizedAction = normalizeAction(action);

    const getActionColor = () => {
        switch (normalizedAction) {
            case ACTIONS.BUY:
            case ACTIONS.HOLD_BUY:
                return "bg-green-500/20 text-green-400 border-green-500/30";
            case ACTIONS.SELL:
            case ACTIONS.HOLD_SELL:
                return "bg-red-500/20 text-red-400 border-red-500/30";
            default:
                return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
        }
    };

    const getActionIcon = () => {
        switch (normalizedAction) {
            case ACTIONS.BUY:
            case ACTIONS.HOLD_BUY:
                return <TrendingUp className="w-5 h-5" />;
            case ACTIONS.SELL:
            case ACTIONS.HOLD_SELL:
                return <TrendingDown className="w-5 h-5" />;
            default:
                return <Minus className="w-5 h-5" />;
        }
    };

    const isPositive = change_pct >= 0;

    return (
        <div
            onClick={onClick}
            className={`bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-4 hover:border-gray-600 transition-all cursor-pointer group ${
                flash === "up"
                    ? "ring-2 ring-green-400/60 shadow-[0_0_12px_rgba(34,197,94,0.35)]"
                    : flash === "down"
                      ? "ring-2 ring-red-400/60 shadow-[0_0_12px_rgba(248,113,113,0.35)]"
                      : ""
            }`}
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
                    {normalizedAction}
                </div>
            </div>

            <div className="flex justify-between items-end">
                <div>
                    <p className="text-2xl font-bold text-white">₹{price.toFixed(2)}</p>
                    <p
                        className={`text-sm ${isPositive ? "text-green-400" : "text-red-400"}`}
                    >
                        {formatChangePct(change_pct)}
                    </p>
                </div>
                <div className="text-right">
                    <p className="text-xs text-gray-500">Confidence</p>
                    <p className="text-lg font-semibold text-blue-400">
                        {(confidence * 100).toFixed(0)}%
                    </p>
                    {(normalizedAction === ACTIONS.BUY || normalizedAction === ACTIONS.SELL || normalizedAction === ACTIONS.HOLD_BUY || normalizedAction === ACTIONS.HOLD_SELL) && target_price ? (
                        <p className="text-xs text-gray-400 mt-1">
                            Target: ₹{target_price.toFixed(2)}
                        </p>
                    ) : null}
                </div>
            </div>

            {sparkline && sparkline.length > 0 && (
                <div className="mt-3">
                    <Sparkline values={sparkline} positive={isPositive} />
                </div>
            )}

            {trade_plan && (
                <div className="mt-3 grid grid-cols-3 gap-2">
                    <div className="rounded-md border border-gray-700 bg-gray-900/60 p-2">
                        <p className="text-[10px] uppercase text-gray-500">Capital</p>
                        <p className="text-xs font-semibold text-cyan-300">
                            ₹{(trade_plan.capital_amount_inr ?? 0).toFixed(0)}
                        </p>
                    </div>
                    <div className="rounded-md border border-gray-700 bg-gray-900/60 p-2">
                        <p className="text-[10px] uppercase text-gray-500">TP/SL</p>
                        <p className="text-xs font-semibold text-cyan-300">
                            {(trade_plan.tp_sl_ratio_target ?? 0).toFixed(2)}
                        </p>
                    </div>
                    <div className="rounded-md border border-gray-700 bg-gray-900/60 p-2">
                        <p className="text-[10px] uppercase text-gray-500">Exit</p>
                        <p className="text-xs font-semibold text-cyan-300">
                            ₹{(trade_plan.profit_target_exit_price ?? 0).toFixed(2)}
                        </p>
                    </div>
                </div>
            )}

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
