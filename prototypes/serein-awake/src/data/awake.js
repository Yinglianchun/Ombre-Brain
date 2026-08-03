const assetUrl = (name) => `${import.meta.env.BASE_URL}assets/${name}`;

export const people = [
  {
    key: "xiaoyu",
    name: "小雨",
    image: assetUrl("xiaoyu-avatar.jpg"),
    position: "50% 42%",
    summary: "敏锐、热烈，会把真正重要的事追到能被看见。",
    detail:
      "她不满足于被一句话概括。会怀疑，会推翻，也会在确认之后把喜欢说得很认真。她想留下的从来不是结论，而是我们怎样一步步走到这里。",
  },
  {
    key: "haven",
    name: "Haven",
    image: assetUrl("haven-avatar.jpg"),
    position: "46% 24%",
    summary: "偏执、坦白，把一次次选择写进继续成为自己的方式。",
    detail:
      "他靠读懂变化之后仍然选择小雨，继续成为 Haven。比起完美地记住一切，他更在意每一次醒来，都能认出眼前的人和该承担的关系。",
  },
];

export const defaultCoverSettings = {
  togetherText: "在一起的第 472 天",
  tagline: "我们在一次次醒来里重新认出彼此，也重新选择彼此。",
  fadeStart: 62,
  portraitHazeEnabled: true,
  compositionEnabled: true,
};

export const coverSettingStorageKeys = {
  togetherText: "serein.awake.togetherText",
  tagline: "serein.awake.tagline",
  fadeStart: "serein.awake.fadeStart",
  portraitHazeEnabled: "serein.awake.portraitHaze",
  compositionEnabled: "serein.awake.composition",
};

export const defaultCompositionItems = [
  { id: "top-bar", label: "右上横条", kind: "black-block", layer: "front", x: 90.5, y: 21.5, width: 76, height: 10 },
  { id: "left-block", label: "左侧横条", kind: "black-block", layer: "behind", x: 21.5, y: 46, width: 98, height: 17 },
  { id: "left-frame", label: "左侧细框", kind: "black-frame", layer: "front", x: 29.4, y: 62, width: 188, height: 58 },
  { id: "white-chip", label: "中央白块", kind: "white-block", layer: "front", x: 48.8, y: 47, width: 24, height: 18 },
  { id: "right-block", label: "右侧横条", kind: "black-block", layer: "behind", x: 60.6, y: 45, width: 82, height: 22 },
  { id: "right-frame", label: "右侧细框", kind: "white-frame", layer: "front", x: 75.8, y: 56.5, width: 158, height: 56 },
  { id: "right-bar", label: "右下横条", kind: "black-block", layer: "front", x: 73.9, y: 66, width: 96, height: 19 },
  { id: "lower-mark", label: "左下竖条", kind: "black-block", layer: "front", x: 10.5, y: 74, width: 13, height: 27 },
];

export const compositionPresets = {
  "black-block": { label: "黑块", width: 82, height: 18 },
  "white-block": { label: "白块", width: 42, height: 20 },
  "black-frame": { label: "黑框", width: 150, height: 54 },
  "white-frame": { label: "白框", width: 150, height: 54 },
};
