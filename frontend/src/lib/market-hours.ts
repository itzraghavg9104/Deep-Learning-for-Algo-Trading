const IST_TZ = "Asia/Kolkata";

const getISTParts = (date: Date) => {
    const formatter = new Intl.DateTimeFormat("en-US", {
        timeZone: IST_TZ,
        weekday: "short",
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
    });
    const parts = formatter.formatToParts(date);
    const part = (type: string) => parts.find((p) => p.type === type)?.value ?? "";
    return {
        weekday: part("weekday"),
        hour: Number(part("hour")),
        minute: Number(part("minute")),
    };
};

const weekdayIndex = (weekday: string) => {
    const map: Record<string, number> = {
        Sun: 0,
        Mon: 1,
        Tue: 2,
        Wed: 3,
        Thu: 4,
        Fri: 5,
        Sat: 6,
    };
    return map[weekday] ?? 0;
};

export const isMarketOpen = (date: Date = new Date()) => {
    const { weekday, hour, minute } = getISTParts(date);
    const day = weekdayIndex(weekday);

    const isWeekday = day >= 1 && day <= 5;
    const currentMinutes = hour * 60 + minute;
    const openMinutes = 9 * 60 + 15;
    const closeMinutes = 15 * 60 + 30;

    return isWeekday && currentMinutes >= openMinutes && currentMinutes <= closeMinutes;
};

export const getMarketStatusLabel = (date: Date = new Date()) =>
    isMarketOpen(date) ? "Market Open" : "Market Closed";
