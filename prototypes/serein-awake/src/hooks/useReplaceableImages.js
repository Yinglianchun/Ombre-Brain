import { useState } from "react";
import { people } from "../data/awake.js";

export function useReplaceableImages() {
  const [images, setImages] = useState({
    hero: `${import.meta.env.BASE_URL}assets/awake-cover.jpg`,
    xiaoyu: people[0].image,
    haven: people[1].image,
  });

  const replace = (key, file) => {
    if (!file) return;
    setImages((current) => {
      if (current[key]?.startsWith("blob:")) URL.revokeObjectURL(current[key]);
      return { ...current, [key]: URL.createObjectURL(file) };
    });
  };

  return { images, replace };
}
