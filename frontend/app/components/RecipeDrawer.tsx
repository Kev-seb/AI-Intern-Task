"use client";

import { useEffect, useState } from "react";
import { api, type Recipe } from "../lib/api";

const SOURCES: Record<string, string> = {
  ar: "AllRecipes",
  epi: "Epicurious",
  fn: "Food Network",
};

export default function RecipeDrawer({ id, onClose }: { id: string; onClose: () => void }) {
  const [recipe, setRecipe] = useState<Recipe | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .recipe(id)
      .then((r) => !cancelled && setRecipe(r))
      .catch((e: Error) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [id]);

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/40" onClick={onClose}>
      <aside
        className="h-full w-full max-w-xl overflow-y-auto bg-white p-6 shadow-xl dark:bg-stone-900"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-start justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-wide text-stone-500">Original dataset record</p>
            <h2 className="text-xl font-semibold">{recipe?.title ?? "Loading…"}</h2>
          </div>
          <button
            onClick={onClose}
            className="rounded-md border border-stone-300 px-2 py-1 text-sm hover:bg-stone-100 dark:border-stone-700 dark:hover:bg-stone-800"
          >
            Close
          </button>
        </div>
        <p className="mb-4 break-all font-mono text-xs text-stone-500">{id}</p>
        {error && <p className="text-red-600">{error}</p>}
        {recipe && (
          <>
            <p className="mb-4 text-sm text-stone-600 dark:text-stone-400">
              {SOURCES[recipe.source] ?? recipe.source} · {recipe.n_ingredients} ingredients · {recipe.n_steps} steps
              (counted by the sentence rule)
            </p>
            <h3 className="mb-2 font-semibold">Ingredients</h3>
            <ul className="mb-5 list-disc space-y-1 pl-5 text-sm">
              {recipe.ingredients
                .map((i) => i.replace(/ADVERTISEMENT/g, "").trim())
                .filter(Boolean)
                .map((i, k) => (
                  <li key={k}>{i}</li>
                ))}
            </ul>
            <h3 className="mb-2 font-semibold">Instructions</h3>
            <p className="whitespace-pre-line text-sm leading-6">{recipe.instructions}</p>
          </>
        )}
      </aside>
    </div>
  );
}
