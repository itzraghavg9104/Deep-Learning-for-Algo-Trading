"use client";

import {
  LineChart,
  Line,
  ResponsiveContainer,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";

type EquityCurveProps = {
  values: number[];
};

export function EquityCurve({ values }: EquityCurveProps) {
  const data = values.map((value, index) => ({
    step: index + 1,
    equity: Number(value.toFixed(2)),
  }));

  if (data.length === 0) {
    return (
      <div className="h-72 flex items-center justify-center text-gray-400 border border-gray-800 rounded-xl bg-gray-900/40">
        No equity curve data available.
      </div>
    );
  }

  return (
    <div className="h-72 p-4 border border-gray-800 rounded-xl bg-gray-900/40">
      <p className="text-sm text-gray-300 mb-3">Equity Curve</p>
      <ResponsiveContainer width="100%" height="90%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="step" stroke="#9CA3AF" tick={{ fontSize: 12 }} />
          <YAxis stroke="#9CA3AF" tick={{ fontSize: 12 }} />
          <Tooltip
            contentStyle={{
              background: "#111827",
              border: "1px solid #374151",
              borderRadius: 8,
            }}
          />
          <Line type="monotone" dataKey="equity" stroke="#60A5FA" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
