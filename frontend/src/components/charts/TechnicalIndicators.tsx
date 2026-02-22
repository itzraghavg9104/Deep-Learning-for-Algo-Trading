"use client";

type IndicatorValue = string | number | boolean | null | undefined;

type TechnicalIndicatorsProps = {
    indicators: Record<string, IndicatorValue>;
};

const INDICATOR_LABELS: { key: string; label: string }[] = [
    { key: "trend", label: "Trend" },
    { key: "rsi_14", label: "RSI (14)" },
    { key: "macd_line", label: "MACD" },
    { key: "macd_signal", label: "MACD Signal" },
    { key: "sma_20", label: "SMA 20" },
    { key: "sma_50", label: "SMA 50" },
    { key: "ema_12", label: "EMA 12" },
    { key: "ema_26", label: "EMA 26" },
    { key: "bb_upper", label: "BB Upper" },
    { key: "bb_middle", label: "BB Mid" },
    { key: "bb_lower", label: "BB Lower" },
    { key: "adx", label: "ADX" },
    { key: "atr_14", label: "ATR" },
    { key: "volume_ratio", label: "Volume Ratio" },
];

const formatValue = (value: IndicatorValue) => {
    if (value === null || value === undefined) return "—";
    if (typeof value === "boolean") return value ? "Yes" : "No";
    if (typeof value === "number") return Number.isFinite(value) ? value.toFixed(2) : "—";
    return value.toString();
};

export function TechnicalIndicators({ indicators }: TechnicalIndicatorsProps) {
    const items = INDICATOR_LABELS.filter(({ key }) => key in indicators);

    if (!items.length) {
        return (
            <div className="p-4 border border-gray-800 rounded-xl bg-gray-900/40 text-sm text-gray-400">
                Indicators not available.
            </div>
        );
    }

    return (
        <div className="p-4 border border-gray-800 rounded-xl bg-gray-900/40">
            <p className="text-sm text-gray-300 mb-3">Technical Indicators</p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                {items.map(({ key, label }) => (
                    <div key={key} className="rounded-lg border border-gray-800 bg-gray-950/40 p-3">
                        <p className="text-xs text-gray-500">{label}</p>
                        <p className="text-base text-white font-semibold">{formatValue(indicators[key])}</p>
                    </div>
                ))}
            </div>
        </div>
    );
}
