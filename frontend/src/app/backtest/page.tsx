"use client";

import { useState } from "react";
import axios from "axios";
import { backtestApi } from "@/lib/api";
import { BacktestConfig } from "@/components/forms/BacktestConfig";
import { EquityCurve } from "@/components/charts/EquityCurve";

type BacktestResult = {
  backtest_id?: string;
  symbol: string;
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  profit_factor: number;
  total_trades: number;
  final_value: number;
  trades: Record<string, unknown>[];
  equity_curve: number[];
};

const formatPct = (value: number) => `${(value * 100).toFixed(2)}%`;

export default function BacktestPage() {
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runBacktest = async (payload: {
    symbol: string;
    start_date: string;
    end_date: string;
    initial_capital: number;
    risk_tolerance: number;
  }) => {
    setLoading(true);
    setError(null);

    try {
      const data = await backtestApi.runBacktest(payload);
      setResult(data);
    } catch (err: unknown) {
      if (axios.isAxiosError(err) && typeof err.response?.data?.detail === "string") {
        setError(err.response.data.detail);
      } else {
        setError("Backtest request failed.");
      }
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-950 p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold text-white mb-2">Backtest</h1>
        <p className="text-gray-400 mb-6">Run historical strategy simulation with configurable risk settings.</p>

        <BacktestConfig loading={loading} onSubmit={runBacktest} />

        {error && (
          <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
            {error}
          </div>
        )}

        {result && (
          <div className="mt-6 space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
              <MetricCard title="Total Return" value={formatPct(result.total_return)} />
              <MetricCard title="Sharpe Ratio" value={result.sharpe_ratio.toFixed(2)} />
              <MetricCard title="Max Drawdown" value={formatPct(result.max_drawdown)} />
              <MetricCard title="Win Rate" value={formatPct(result.win_rate)} />
              <MetricCard title="Profit Factor" value={result.profit_factor.toFixed(2)} />
              <MetricCard title="Trades" value={result.total_trades.toString()} />
              <MetricCard title="Final Value" value={`INR ${result.final_value.toFixed(2)}`} />
              <MetricCard title="Symbol" value={result.symbol} />
            </div>

            <EquityCurve values={result.equity_curve} />

            <div className="p-4 border border-gray-800 rounded-xl bg-gray-900/40">
              <p className="text-sm text-gray-300 mb-3">Recent Trades</p>
              {result.trades.length === 0 ? (
                <p className="text-gray-400 text-sm">No trades generated for this backtest window.</p>
              ) : (
                <div className="overflow-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left text-gray-400 border-b border-gray-800">
                        <th className="py-2 pr-3">#</th>
                        <th className="py-2 pr-3">Action</th>
                        <th className="py-2 pr-3">Price</th>
                        <th className="py-2 pr-3">Qty</th>
                        <th className="py-2 pr-3">P&L</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.trades.slice(0, 20).map((trade, index) => (
                        <tr key={index} className="border-b border-gray-900 text-gray-200">
                          <td className="py-2 pr-3">{index + 1}</td>
                          <td className="py-2 pr-3">{String(trade.action ?? "-")}</td>
                          <td className="py-2 pr-3">{String(trade.price ?? "-")}</td>
                          <td className="py-2 pr-3">{String(trade.quantity ?? "-")}</td>
                          <td className="py-2 pr-3">{String(trade.pnl ?? "-")}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </main>
  );
}

function MetricCard({ title, value }: { title: string; value: string }) {
  return (
    <div className="p-3 border border-gray-800 rounded-lg bg-gray-900/40">
      <p className="text-xs text-gray-400">{title}</p>
      <p className="text-lg font-semibold text-white">{value}</p>
    </div>
  );
}
