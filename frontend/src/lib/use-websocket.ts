"use client";

import { useEffect, useRef, useState, useCallback } from "react";

type WsStatus = "connecting" | "open" | "closed";

type UseWebsocketOptions = {
    onMessage?: (data: unknown) => void;
    shouldConnect?: boolean;
};

const getWsUrl = () => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";
    const base = apiUrl.replace(/\/api\/v1$/, "");
    const wsBase = base.replace(/^http/, "ws");
    return `${wsBase}/api/v1/ws/prices`;
};

export const useWebsocket = ({ onMessage, shouldConnect = true }: UseWebsocketOptions) => {
    const socketRef = useRef<WebSocket | null>(null);
    const retryRef = useRef(0);
    const [status, setStatus] = useState<WsStatus>("closed");

    const sendJson = useCallback((payload: unknown) => {
        if (socketRef.current?.readyState === WebSocket.OPEN) {
            socketRef.current.send(JSON.stringify(payload));
        }
    }, []);

    const subscribe = useCallback(
        (symbols: string[]) => sendJson({ action: "subscribe", symbols }),
        [sendJson],
    );

    const unsubscribe = useCallback(
        (symbols: string[]) => sendJson({ action: "unsubscribe", symbols }),
        [sendJson],
    );

    useEffect(() => {
        if (!shouldConnect) return;

        let closedByUser = false;

        const connect = () => {
            setStatus("connecting");
            const socket = new WebSocket(getWsUrl());
            socketRef.current = socket;

            socket.onopen = () => {
                retryRef.current = 0;
                setStatus("open");
            };

            socket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    onMessage?.(data);
                } catch {
                    // Ignore malformed messages
                }
            };

            socket.onclose = () => {
                socketRef.current = null;
                if (closedByUser) {
                    setStatus("closed");
                    return;
                }
                setStatus("closed");
                const retryDelay = Math.min(10000, 1000 * Math.pow(2, retryRef.current));
                retryRef.current += 1;
                setTimeout(connect, retryDelay);
            };
        };

        connect();

        return () => {
            closedByUser = true;
            socketRef.current?.close();
        };
    }, [onMessage, shouldConnect]);

    return {
        status,
        subscribe,
        unsubscribe,
        sendJson,
    };
};
