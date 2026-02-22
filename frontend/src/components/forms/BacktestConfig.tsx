"use client";

import { useState, type FormEvent } from "react";

type BacktestInput = {
  symbol: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  risk_tolerance: number;
};

type BacktestConfigProps = {
  loading: boolean;
  onSubmit: (input: BacktestInput) => Promise<void>;
};

export function BacktestConfig({ loading, onSubmit }: BacktestConfigProps) {
  const [form, setForm] = useState<BacktestInput>({
    symbol: "RELIANCE.NS",
    start_date: "2024-01-01",
    end_date: "2024-12-31",
    initial_capital: 100000,
    risk_tolerance: 0.5,
  });

  const handleChange = (key: keyof BacktestInput, value: string | number) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    await onSubmit(form);
  };

  return (
    <form onSubmit={handleSubmit} className="p-5 bg-gray-900/50 border border-gray-800 rounded-xl space-y-4">
      <h2 className="text-lg font-semibold text-white">Backtest Configuration</h2>

      <div>
        <label className="block text-sm text-gray-300 mb-1">Symbol</label>
        <input
          value={form.symbol}
          onChange={(event) => handleChange("symbol", event.target.value)}
          className="w-full px-3 py-2 rounded-lg bg-gray-800 border border-gray-700 text-white"
          placeholder="RELIANCE.NS"
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div>
          <label className="block text-sm text-gray-300 mb-1">Start Date</label>
          <input
            type="date"
            value={form.start_date}
            onChange={(event) => handleChange("start_date", event.target.value)}
            className="w-full px-3 py-2 rounded-lg bg-gray-800 border border-gray-700 text-white"
          />
        </div>
        <div>
          <label className="block text-sm text-gray-300 mb-1">End Date</label>
          <input
            type="date"
            value={form.end_date}
            onChange={(event) => handleChange("end_date", event.target.value)}
            className="w-full px-3 py-2 rounded-lg bg-gray-800 border border-gray-700 text-white"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div>
          <label className="block text-sm text-gray-300 mb-1">Initial Capital (INR)</label>
          <input
            type="number"
            min={1000}
            step={1000}
            value={form.initial_capital}
            onChange={(event) => handleChange("initial_capital", Number(event.target.value))}
            className="w-full px-3 py-2 rounded-lg bg-gray-800 border border-gray-700 text-white"
          />
        </div>
        <div>
          <label className="block text-sm text-gray-300 mb-1">
            Risk Tolerance ({Math.round(form.risk_tolerance * 100)}%)
          </label>
          <input
            type="range"
            min={0.1}
            max={1}
            step={0.1}
            value={form.risk_tolerance}
            onChange={(event) => handleChange("risk_tolerance", Number(event.target.value))}
            className="w-full"
          />
        </div>
      </div>

      <button
        type="submit"
        disabled={loading}
        className="w-full py-2.5 rounded-lg bg-gradient-to-r from-blue-500 to-indigo-500 text-white font-semibold disabled:opacity-50"
      >
        {loading ? "Running Backtest..." : "Run Backtest"}
      </button>
    </form>
  );
}
