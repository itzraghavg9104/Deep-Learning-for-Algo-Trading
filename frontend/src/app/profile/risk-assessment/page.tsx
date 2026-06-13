import { RiskQuestionnaire } from "@/components/forms/RiskQuestionnaire";
import Link from "next/link";

export default function RiskAssessmentPage() {
  return (
    <main className="min-h-screen bg-gray-950 p-6">
      <div className="max-w-5xl mx-auto mb-4">
        <Link
          href="/profile"
          className="inline-flex items-center px-3 py-1.5 rounded-lg bg-gray-800 border border-gray-700 text-gray-200 hover:bg-gray-700"
        >
          Back to Profile
        </Link>
      </div>
      <RiskQuestionnaire />
    </main>
  );
}
