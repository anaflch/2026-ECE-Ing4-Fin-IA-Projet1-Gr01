"use client";

interface AICommentaryProps {
    text: string;
}

export default function AICommentary({ text }: AICommentaryProps) {
    return (
        <div className="bg-blue-50 border border-blue-100 rounded-xl p-6">
            <h3 className="flex items-center gap-2 text-sm font-bold text-blue-800 uppercase tracking-wide mb-3">
                🤖 AI Analyst Verdict
            </h3>
            <p className="text-gray-800 leading-relaxed text-sm md:text-base">
                {text}
            </p>
        </div>
    );
}
