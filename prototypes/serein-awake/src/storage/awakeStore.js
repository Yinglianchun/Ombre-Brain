import { compositionPresets, defaultCompositionItems } from "../data/awake.js";

export function readLocalPreference(key, fallback) {
  try {
    const saved = window.localStorage.getItem(key);
    return saved === null ? fallback : saved;
  } catch {
    return fallback;
  }
}

export function storeLocalPreference(key, value) {
  try {
    const serializedValue = String(value);
    window.localStorage.setItem(key, serializedValue);
    window.dispatchEvent(new CustomEvent("serein:preference-change", {
      detail: { key, value: serializedValue },
    }));
  } catch {
    // The prototype remains usable when browser storage is unavailable.
  }
}

export function readCompositionItems() {
  try {
    const saved = window.localStorage.getItem("serein.awake.composition.items");
    if (!saved) return defaultCompositionItems;
    const parsed = JSON.parse(saved);
    return Array.isArray(parsed) && parsed.every((item) => (
      typeof item.id === "string"
      && compositionPresets[item.kind]
      && Number.isFinite(item.x)
      && Number.isFinite(item.y)
      && Number.isFinite(item.width)
      && Number.isFinite(item.height)
    )) ? parsed.map((item) => ({ ...item, layer: item.layer === "behind" ? "behind" : "front" })) : defaultCompositionItems;
  } catch {
    return defaultCompositionItems;
  }
}

export function storeCompositionItems(items) {
  try {
    window.localStorage.setItem("serein.awake.composition.items", JSON.stringify(items));
  } catch {
    // A future Settings store can own this key when browser storage is unavailable.
  }
}
