"use client";

import {
    ResponsiveContainer,
    ComposedChart,
    Line,
    Bar,
    XAxis,
    YAxis,
    Tooltip,
    CartesianGrid,
} from "recharts";

type PricePoint = {
    timestamp: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
};

type PriceChartProps = {
    data: PricePoint[];
};

const formatDate = (value: string) => {
    const date = new Date(value);
    return date.toLocaleDateString("en-IN", { month: "short", day: "numeric" });
};

export function PriceChart({ data }: PriceChartProps) {
    if (!data.length) {
        return (
            <div className="h-72 flex items-center justify-center text-gray-400 border border-gray-800 rounded-xl bg-gray-900/40">
                No price history available.
            </div>
        );
    }

    const chartData = data.map((point) => ({
        ...point,
        close: Number(point.close.toFixed(2)),
    }));

    return (
        <div className="h-80 p-4 border border-gray-800 rounded-xl bg-gray-900/40">
            <p className="text-sm text-gray-300 mb-3">Price & Volume</p>
            <ResponsiveContainer width="100%" height="90%">
                <ComposedChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="timestamp" tickFormatter={formatDate} stroke="#9CA3AF" tick={{ fontSize: 12 }} />
                    <YAxis yAxisId="price" stroke="#9CA3AF" tick={{ fontSize: 12 }} width={48} />
                    <YAxis yAxisId="volume" orientation="right" stroke="#6B7280" tick={{ fontSize: 10 }} width={40} />
                    <Tooltip
                        contentStyle={{
                            background: "#111827",
                            border: "1px solid #374151",
                            borderRadius: 8,
                        }}
                        labelFormatter={formatDate}
                    />
                    <Bar yAxisId="volume" dataKey="volume" fill="#1F2937" opacity={0.8} />
                    <Line yAxisId="price" type="monotone" dataKey="close" stroke="#60A5FA" strokeWidth={2} dot={false} />
                </ComposedChart>
            </ResponsiveContainer>
        </div>
    );
}
