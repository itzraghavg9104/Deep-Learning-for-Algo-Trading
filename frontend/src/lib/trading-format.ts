export const ACTIONS = {
  HOLD_BUY: "HOLD BUY",
  HOLD_SELL: "HOLD SELL",
  BUY: "BUY",
  SELL: "SELL",
  IDLE: "IDLE",
} as const;

export function normalizeAction(action: string): string {
  const trimmed = (action || "").trim().toUpperCase();
  if (trimmed === "HOLD") return ACTIONS.IDLE;
  return trimmed;
}

export function isBuyishAction(action: string): boolean {
  const a = normalizeAction(action);
  return a === ACTIONS.BUY || a === ACTIONS.HOLD_BUY;
}

export function isSellishAction(action: string): boolean {
  const a = normalizeAction(action);
  return a === ACTIONS.SELL || a === ACTIONS.HOLD_SELL;
}

export function formatChangePct(changePct: number): string {
  const sign = changePct >= 0 ? "+" : "";
  return `${sign}${changePct.toFixed(2)}%`;
}
