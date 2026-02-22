"use client";

import { LineChart, Line, ResponsiveContainer } from "recharts";

type SparklineProps = {
    values: number[];
    positive?: boolean;
};

export function Sparkline({ values, positive }: SparklineProps) {
    if (!values.length) {
        return <div className="h-10" />;
    }

    const data = values.map((value, index) => ({ index, value }));
    const stroke = positive === undefined ? "#60A5FA" : positive ? "#34D399" : "#F87171";

    return (
        <div className="h-10 w-full">
            <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data}>
                    <Line type="monotone" dataKey="value" stroke={stroke} strokeWidth={2} dot={false} />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}
