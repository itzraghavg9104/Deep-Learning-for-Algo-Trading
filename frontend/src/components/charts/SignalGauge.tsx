"use client";

type SignalGaugeProps = {
    confidence: number;
    action: string;
};

const getColor = (action: string) => {
    if (action === "BUY") return "from-green-400 to-emerald-500";
    if (action === "SELL") return "from-red-400 to-rose-500";
    return "from-yellow-400 to-orange-400";
};

export function SignalGauge({ confidence, action }: SignalGaugeProps) {
    const pct = Math.max(0, Math.min(1, confidence)) * 100;

    return (
        <div className="p-4 border border-gray-800 rounded-xl bg-gray-900/40">
            <div className="flex items-center justify-between mb-2">
                <p className="text-sm text-gray-300">Signal Confidence</p>
                <span className="text-sm text-gray-400">{action}</span>
            </div>
            <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                <div
                    className={`h-full bg-gradient-to-r ${getColor(action)}`}
                    style={{ width: `${pct}%` }}
                />
            </div>
            <p className="text-xs text-gray-500 mt-2">{pct.toFixed(0)}% confidence</p>
        </div>
    );
}
