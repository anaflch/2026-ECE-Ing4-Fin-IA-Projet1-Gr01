"use client";

import { useState, useEffect, useRef } from "react";
import { ChevronDown, Search } from "lucide-react"; // Assuming lucide-react is available, or use SVG

interface TickerInputProps {
    onSearch: (ticker: string) => void;
    isLoading: boolean;
}

interface SearchResult {
    symbol: string;
    name: string;
}

export default function TickerInput({ onSearch, isLoading }: TickerInputProps) {
    const [ticker, setTicker] = useState("");
    const [query, setQuery] = useState("");
    const [results, setResults] = useState<SearchResult[]>([]);
    const [availableTickers, setAvailableTickers] = useState<string[]>([]);
    const [showResults, setShowResults] = useState(false);
    const searchRef = useRef<HTMLDivElement>(null);

    // 1. Fetch all available tickers on mount
    useEffect(() => {
        const fetchTickers = async () => {
            try {
                const res = await fetch("http://localhost:8000/tickers");
                if (res.ok) {
                    const data = await res.json();
                    setAvailableTickers(data);
                }
            } catch (e) {
                console.error("Error fetching tickers:", e);
            }
        };
        fetchTickers();
    }, []);

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
                setShowResults(false);
            }
        };
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, []);

    const handleSearch = async (val: string) => {
        // If empty, show full list
        if (!val) {
            setResults(availableTickers.map(t => ({ symbol: t, name: "Available Asset" })));
            return;
        }

        // If < 2 chars, clear
        if (val.length < 2) {
            setResults([]);
            return;
        }

        try {
            const res = await fetch(`http://localhost:8000/search?q=${val}`);
            if (res.ok) {
                const data = await res.json();
                setResults(data);
                setShowResults(true);
            }
        } catch (e) {
            console.error(e);
        }
    };

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const val = e.target.value;
        setTicker(val.toUpperCase());
        handleSearch(val);
    };

    const handleFocus = () => {
        // If empty, show available tickers immediately
        if (!ticker) {
            setResults(availableTickers.map(t => ({ symbol: t, name: "Available Asset" })));
            setShowResults(true);
        } else {
            // If has text, trigger search or show existing results
            if (results.length > 0) setShowResults(true);
            else handleSearch(ticker);
        }
    };

    const toggleDropdown = () => {
        if (showResults) {
            setShowResults(false);
        } else {
            // Force show all
            setResults(availableTickers.map(t => ({ symbol: t, name: "Available Asset" })));
            setShowResults(true);
        }
    };

    const handleSelect = (symbol: string) => {
        setTicker(symbol);
        setShowResults(false);
        onSearch(symbol);
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (ticker.trim()) {
            onSearch(ticker.trim());
            setShowResults(false);
        }
    };

    return (
        <div className="relative w-full max-w-md" ref={searchRef}>
            <form onSubmit={handleSubmit} className="flex gap-2">
                <div className="relative flex-grow">
                    <div className="relative">
                        <input
                            type="text"
                            value={ticker}
                            onChange={handleChange}
                            onFocus={handleFocus}
                            placeholder="Select or Search Asset..."
                            className="w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 uppercase font-mono text-black font-semibold pr-10" // added PR for icon
                            disabled={isLoading}
                        />

                        {/* Dropdown Chevron Icon */}
                        <button
                            type="button"
                            onClick={toggleDropdown}
                            className="absolute right-2 top-1/2 transform -translate-y-1/2 p-1 text-gray-400 hover:text-gray-600 focus:outline-none"
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m6 9 6 6 6-6" /></svg>
                        </button>
                    </div>

                    {/* Dropdown */}
                    {showResults && (
                        <div className="absolute z-50 w-full bg-white mt-1 border border-gray-200 rounded-lg shadow-xl max-h-80 overflow-y-auto">
                            {/* Header if showing full list */}
                            {results.length > 0 && ticker === "" && (
                                <div className="px-4 py-2 bg-gray-50 text-xs font-semibold text-gray-500 uppercase tracking-wider border-b">
                                    All Available Assets ({availableTickers.length})
                                </div>
                            )}

                            {results.map((r, idx) => (
                                <div
                                    key={idx}
                                    className="px-4 py-3 hover:bg-blue-50 cursor-pointer border-b border-gray-100 last:border-0 flex justify-between items-center group"
                                    onClick={() => handleSelect(r.symbol)}
                                >
                                    <div>
                                        <div className="font-bold text-gray-900 group-hover:text-blue-700">{r.symbol}</div>
                                        <div className="text-xs text-gray-500">{r.name}</div>
                                    </div>
                                    {/* Visual cue for selection */}
                                    <div className="opacity-0 group-hover:opacity-100 text-blue-500">
                                        →
                                    </div>
                                </div>
                            ))}

                            {results.length === 0 && ticker.length > 0 && (
                                <div className="p-4 text-center text-gray-400 text-sm">No results</div>
                            )}
                        </div>
                    )}
                </div>

                <button
                    type="submit"
                    disabled={isLoading || !ticker}
                    className="px-6 py-2 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors whitespace-nowrap shadow-sm"
                >
                    {isLoading ? "Running..." : "Analyze"}
                </button>
            </form>
        </div>
    );
}
