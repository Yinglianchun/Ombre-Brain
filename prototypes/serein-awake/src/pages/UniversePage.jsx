import { useEffect, useMemo, useRef, useState } from "react";
import { CaretDown, Heart, Sparkle, X } from "@phosphor-icons/react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { loadMemorySnapshot, readMemoryScenes } from "../storage/memoryStore.js";

const galaxyRadius = 5;
const branchCount = 2;
const spin = 1.15;

const palette = {
  relationship: "#ff84c8",
  love: "#ff84c8",
  family: "#f4c982",
  continuity: "#c08cf0",
  identity: "#c08cf0",
  engineering: "#8fa8ff",
  work: "#8fa8ff",
  daily: "#82d6df",
  diary: "#82d6df",
  general: "#d7a0e8",
};

function hashString(value) {
  let hash = 2166136261;
  for (const character of String(value || "")) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}
function seededRandom(seed) {
  let state = seed || 1;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function colorForScene(scene) {
  const key = String(scene.bucketDomain || "general").toLowerCase();
  if (palette[key]) return palette[key];
  if (/love|relationship|romance|我们|恋爱|关系/u.test(key)) return palette.relationship;
  if (/family|home|家庭|家/u.test(key)) return palette.family;
  if (/engineer|code|work|编程|工程/u.test(key)) return palette.engineering;
  if (/daily|diary|日常|日记/u.test(key)) return palette.daily;
  return palette.general;
}

function pointMaterial(sizeScale = 1, depthInfluence = 0.45) {
  return new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uPixelRatio: { value: Math.min(window.devicePixelRatio, 2) * sizeScale },
      uDepthInfluence: { value: depthInfluence },
    },
    vertexShader: `
      attribute vec3 aColor;
      attribute float aSize;
      attribute float aPhase;
      varying vec3 vColor;
      varying float vAlpha;
      uniform float uTime;
      uniform float uPixelRatio;
      uniform float uDepthInfluence;
      void main() {
        vColor = aColor;
        float pulse = 0.92 + 0.08 * sin(uTime * 0.72 + aPhase);
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        float depthSize = clamp(4.8 / max(0.8, -mv.z), 0.78, 1.08);
        gl_PointSize = aSize * pulse * uPixelRatio * mix(1.0, depthSize, uDepthInfluence);
        gl_Position = projectionMatrix * mv;
        vAlpha = pulse;
      }
    `,
    fragmentShader: `
      varying vec3 vColor;
      varying float vAlpha;
      void main() {
        float d = length(gl_PointCoord - vec2(0.5));
        float alpha = pow(smoothstep(0.5, 0.0, d), 1.6) * vAlpha;
        if (alpha < 0.01) discard;
        gl_FragColor = vec4(vColor, alpha);
      }
    `,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
}

function buildGalaxyDisk(count, radius, seed) {
  const random = seededRandom(seed);
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const sizes = new Float32Array(count);
  const phases = new Float32Array(count);
  const inner = new THREE.Color("#ffdcf2");
  const middle = new THREE.Color("#bd91f4");
  const outer = new THREE.Color("#86a8ff");

  for (let index = 0; index < count; index += 1) {
    const radial = Math.pow(random(), 1.7) * radius;
    const branch = (index % branchCount) / branchCount * Math.PI * 2;
    const scatter = 0.55;
    const angle = branch + radial * spin;
    const signedScatter = (power, amount) => (
      Math.pow(random(), power) * (random() < 0.5 ? -1 : 1) * amount
    );
    const progress = Math.min(1, radial / radius);
    positions[index * 3] = Math.cos(angle) * radial
      + signedScatter(2.6, scatter * (radial + 0.4));
    positions[index * 3 + 1] = signedScatter(3, scatter * 0.34 * (radial + 0.3));
    positions[index * 3 + 2] = Math.sin(angle) * radial
      + signedScatter(2.6, scatter * (radial + 0.4));

    const color = progress < 0.5
      ? inner.clone().lerp(middle, progress * 2)
      : middle.clone().lerp(outer, (progress - 0.5) * 2);
    const edgeQuieting = 1 - progress * 0.32;
    const softness = (0.72 + random() * 0.48) * edgeQuieting;
    colors[index * 3] = color.r * softness;
    colors[index * 3 + 1] = color.g * softness;
    colors[index * 3 + 2] = color.b * softness;
    const radialScale = 1.16 - progress * 0.42;
    sizes[index] = (0.55 + random() * 1.75) * radialScale;
    phases[index] = random() * Math.PI * 2;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("aColor", new THREE.BufferAttribute(colors, 3));
  geometry.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
  geometry.setAttribute("aPhase", new THREE.BufferAttribute(phases, 1));
  return new THREE.Points(geometry, pointMaterial(1, 0));
}

function buildDustCloud(count, radius, seed) {
  const random = seededRandom(seed);
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const sizes = new Float32Array(count);
  const phases = new Float32Array(count);
  const palette = ["#e8b8ff", "#9fb8ff", "#ffd0ec", "#c8a0f0", "#ffe0b0"]
    .map((color) => new THREE.Color(color));

  for (let index = 0; index < count; index += 1) {
    const radial = Math.pow(random(), 1.1) * radius * 1.7 + 0.5;
    const angle = random() * Math.PI * 2;
    positions[index * 3] = Math.cos(angle) * radial;
    positions[index * 3 + 1] = (random() - 0.5) * radius * 1.1;
    positions[index * 3 + 2] = Math.sin(angle) * radial;
    const color = palette[Math.floor(random() * palette.length)];
    const brightness = 0.38 + random() * 0.54;
    colors.set([color.r * brightness, color.g * brightness, color.b * brightness], index * 3);
    sizes[index] = 0.42 + random() * 1.18;
    phases[index] = random() * Math.PI * 2;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("aColor", new THREE.BufferAttribute(colors, 3));
  geometry.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
  geometry.setAttribute("aPhase", new THREE.BufferAttribute(phases, 1));
  return new THREE.Points(geometry, pointMaterial(0.9, 0.45));
}

function buildHaloCloud(count, radius, seed) {
  const random = seededRandom(seed);
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const sizes = new Float32Array(count);
  const phases = new Float32Array(count);
  const violet = new THREE.Color("#c59cf2");
  const rose = new THREE.Color("#f0a6d6");
  const warm = new THREE.Color("#f3d39a");

  for (let index = 0; index < count; index += 1) {
    const distance = radius * (0.72 + Math.pow(random(), 0.62) * 1.42);
    const yDirection = random() * 2 - 1;
    const horizontal = Math.sqrt(1 - yDirection * yDirection);
    const angle = random() * Math.PI * 2;
    positions[index * 3] = Math.cos(angle) * horizontal * distance;
    positions[index * 3 + 1] = yDirection * distance * 0.72;
    positions[index * 3 + 2] = Math.sin(angle) * horizontal * distance;

    const tintRoll = random();
    const color = tintRoll > 0.965 ? warm : violet.clone().lerp(rose, random() * 0.42);
    const brightness = 0.38 + random() * 0.72;
    colors.set([color.r * brightness, color.g * brightness, color.b * brightness], index * 3);
    const flare = random();
    sizes[index] = flare > 0.986 ? 2.4 + random() * 2.5 : 0.42 + random() * 1.05;
    phases[index] = random() * Math.PI * 2;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("aColor", new THREE.BufferAttribute(colors, 3));
  geometry.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
  geometry.setAttribute("aPhase", new THREE.BufferAttribute(phases, 1));
  return new THREE.Points(geometry, pointMaterial(1.08, 0.62));
}

function buildMemoryPoints(scenes) {
  const positions = new Float32Array(scenes.length * 3);
  const colors = new Float32Array(scenes.length * 3);
  const sizes = new Float32Array(scenes.length);
  const phases = new Float32Array(scenes.length);
  const points = [];

  scenes.forEach((scene, index) => {
    const random = seededRandom(hashString(scene.id));
    const radius = 0.46 + Math.pow(random(), 1.35) * galaxyRadius * 0.88;
    const branch = hashString(scene.id) % branchCount;
    const angle = branch / branchCount * Math.PI * 2
      + radius * spin
      + (random() - 0.5) * 0.72 * (radius + 0.4);
    const position = new THREE.Vector3(
      Math.cos(angle) * radius + (random() - 0.5) * 0.42,
      (random() + random() - 1) * (0.035 + radius * 0.012),
      Math.sin(angle) * radius + (random() - 0.5) * 0.42,
    );
    const color = new THREE.Color(colorForScene(scene));
    const brightness = (0.88 + random() * 0.16) * (scene.selfAnchor || scene.favorite ? 1.12 : 1);

    positions.set([position.x, position.y, position.z], index * 3);
    colors.set([color.r * brightness, color.g * brightness, color.b * brightness], index * 3);
    sizes[index] = 1.55 + random() * 0.45;
    phases[index] = random() * Math.PI * 2;
    points.push({ scene, position, color, index });
  });

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("aColor", new THREE.BufferAttribute(colors, 3));
  geometry.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
  geometry.setAttribute("aPhase", new THREE.BufferAttribute(phases, 1));
  return {
    object: new THREE.Points(geometry, pointMaterial(1.18, 0)),
    points,
    originalColors: Float32Array.from(colors),
    originalSizes: Float32Array.from(sizes),
  };
}

function makeCoreGlow() {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 128;
  const context = canvas.getContext("2d");
  const glow = context.createRadialGradient(64, 64, 0, 64, 64, 64);
  glow.addColorStop(0, "rgba(255,226,246,.32)");
  glow.addColorStop(0.28, "rgba(234,178,242,.14)");
  glow.addColorStop(0.62, "rgba(186,142,236,.045)");
  glow.addColorStop(1, "rgba(162,126,226,0)");
  context.fillStyle = glow;
  context.fillRect(0, 0, 128, 128);
  const material = new THREE.SpriteMaterial({
    map: new THREE.CanvasTexture(canvas),
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    transparent: true,
    opacity: 0.44,
  });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(1.8, 1.8, 1);
  return sprite;
}

function relatedIndexes(selectedPoint, points) {
  const relatedIds = new Set([
    ...(selectedPoint.scene.relatedSceneIds || []),
    ...(selectedPoint.scene.relatedScenes || []).map((item) => item.id),
  ]);
  const reviewed = points.filter((point) => point.index !== selectedPoint.index && relatedIds.has(point.scene.id));
  if (reviewed.length) return reviewed.slice(0, 10);
  const domain = selectedPoint.scene.bucketDomain;
  if (!domain) return [];
  return points
    .filter((point) => point.index !== selectedPoint.index && point.scene.bucketDomain === domain)
    .sort((left, right) => hashString(left.scene.id) - hashString(right.scene.id))
    .slice(0, 6);
}

export function UniversePage() {
  const stageRef = useRef(null);
  const canvasRef = useRef(null);
  const selectPointRef = useRef(null);
  const clearPointRef = useRef(null);
  const [scenes, setScenes] = useState(readMemoryScenes);
  const [selectedSceneId, setSelectedSceneId] = useState("");
  const [detailOpen, setDetailOpen] = useState(false);
  const [likedSceneIds, setLikedSceneIds] = useState(() => new Set());
  const [ready, setReady] = useState(false);

  const selectedScene = useMemo(
    () => scenes.find((scene) => scene.id === selectedSceneId) || null,
    [scenes, selectedSceneId],
  );

  useEffect(() => {
    let cancelled = false;
    loadMemorySnapshot().then((snapshot) => {
      if (!cancelled && snapshot?.length) setScenes(snapshot);
    });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    const stage = stageRef.current;
    const canvas = canvasRef.current;
    if (!stage || !canvas || !scenes.length) return undefined;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#050209");
    scene.fog = new THREE.FogExp2("#050209", 0.022);
    const camera = new THREE.PerspectiveCamera(55, stage.clientWidth / stage.clientHeight, 0.1, 100);
    camera.position.set(0, 4.65, 8.3);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(stage.clientWidth, stage.clientHeight, false);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;

    const controls = new OrbitControls(camera, canvas);
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    controls.enableDamping = true;
    controls.dampingFactor = 0.055;
    controls.enablePan = false;
    controls.minDistance = 2.3;
    controls.maxDistance = 13;
    controls.minPolarAngle = 0.08;
    controls.maxPolarAngle = Math.PI - 0.08;
    controls.autoRotate = !reduceMotion;
    controls.autoRotateSpeed = 0.28;
    controls.target.set(0, 0, 0);

    const galaxy = buildGalaxyDisk(14200, galaxyRadius, 8226);
    const dust = buildDustCloud(6400, galaxyRadius, 1024);
    const halo = buildHaloCloud(2800, galaxyRadius, 7319);
    const memory = buildMemoryPoints(scenes);
    const coreGlow = makeCoreGlow();
    scene.add(halo, galaxy, dust, memory.object, coreGlow);

    const ringMaterials = [];
    [1.3, 2.4, 3.6, 4.7].forEach((radius) => {
      const geometry = new THREE.RingGeometry(radius - 0.007, radius + 0.007, 160);
      const material = new THREE.MeshBasicMaterial({
        color: "#ffd9a0",
        transparent: true,
        opacity: 0.055,
        side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const ring = new THREE.Mesh(geometry, material);
      ring.rotation.x = -Math.PI / 2;
      ringMaterials.push(material);
      scene.add(ring);
    });

    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    composer.addPass(new UnrealBloomPass(
      new THREE.Vector2(stage.clientWidth, stage.clientHeight),
      0.6,
      0.62,
      0.12,
    ));
    composer.addPass(new OutputPass());

    const raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.16;
    const pointer = new THREE.Vector2();
    let pointerDown = null;
    let highlightLines = null;
    let activeIndexes = [];
    const targetColors = Float32Array.from(memory.originalColors);
    const targetSizes = Float32Array.from(memory.originalSizes);

    const clearHighlight = () => {
      if (highlightLines) {
        scene.remove(highlightLines);
        highlightLines.geometry.dispose();
        highlightLines.material.dispose();
        highlightLines = null;
      }
      activeIndexes.forEach((index) => {
        targetColors.set(memory.originalColors.slice(index * 3, index * 3 + 3), index * 3);
        targetSizes[index] = memory.originalSizes[index];
      });
      activeIndexes = [];
    };
    clearPointRef.current = clearHighlight;

    const selectPoint = (point) => {
      clearHighlight();
      const related = relatedIndexes(point, memory.points);
      activeIndexes = [point.index, ...related.map((item) => item.index)];
      activeIndexes.forEach((index, relatedIndex) => {
        for (let channel = 0; channel < 3; channel += 1) {
          targetColors[index * 3 + channel] = Math.min(
            2.2,
            memory.originalColors[index * 3 + channel] * 2.5,
          );
        }
        targetSizes[index] = relatedIndex === 0
          ? memory.originalSizes[index] * 3.2 + 2.4
          : memory.originalSizes[index] * 1.35 + 0.4;
      });

      if (related.length) {
        const linePositions = [];
        related.forEach((item) => linePositions.push(
          point.position.x, point.position.y, point.position.z,
          item.position.x, item.position.y, item.position.z,
        ));
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.Float32BufferAttribute(linePositions, 3));
        highlightLines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({
          color: point.color,
          transparent: true,
          opacity: 0.44,
          blending: THREE.AdditiveBlending,
          depthWrite: false,
        }));
        scene.add(highlightLines);
      }
      setSelectedSceneId(point.scene.id);
      setDetailOpen(true);
    };
    selectPointRef.current = selectPoint;

    const onPointerDown = (event) => { pointerDown = [event.clientX, event.clientY]; };
    const onPointerUp = (event) => {
      if (!pointerDown) return;
      const moved = Math.abs(event.clientX - pointerDown[0]) + Math.abs(event.clientY - pointerDown[1]);
      pointerDown = null;
      if (moved > 9) return;
      const bounds = canvas.getBoundingClientRect();
      pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
      pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObject(memory.object)[0];
      if (hit) selectPoint(memory.points[hit.index]);
    };
    const onKeyDown = (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      selectPoint(memory.points.at(-1));
    };
    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("keydown", onKeyDown);

    const resize = () => {
      const width = stage.clientWidth;
      const height = stage.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
      composer.setSize(width, height);
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(stage);

    const clock = new THREE.Clock();
    let frame = 0;
    let elapsed = 0;
    const animate = () => {
      frame = window.requestAnimationFrame(animate);
      const delta = Math.min(clock.getDelta(), 0.05);
      elapsed += delta;
      galaxy.material.uniforms.uTime.value = elapsed;
      dust.material.uniforms.uTime.value = elapsed;
      halo.material.uniforms.uTime.value = elapsed;
      memory.object.material.uniforms.uTime.value = elapsed;
      const colorAttribute = memory.object.geometry.attributes.aColor;
      const sizeAttribute = memory.object.geometry.attributes.aSize;
      const response = reduceMotion ? 1 : 1 - Math.exp(-delta * 7.5);
      let attributesChanged = false;
      for (let index = 0; index < memory.points.length; index += 1) {
        const nextSize = THREE.MathUtils.lerp(sizeAttribute.array[index], targetSizes[index], response);
        if (Math.abs(nextSize - sizeAttribute.array[index]) > 0.0005) {
          sizeAttribute.array[index] = nextSize;
          attributesChanged = true;
        }
        for (let channel = 0; channel < 3; channel += 1) {
          const attributeIndex = index * 3 + channel;
          const nextColor = THREE.MathUtils.lerp(
            colorAttribute.array[attributeIndex],
            targetColors[attributeIndex],
            response,
          );
          if (Math.abs(nextColor - colorAttribute.array[attributeIndex]) > 0.0005) {
            colorAttribute.array[attributeIndex] = nextColor;
            attributesChanged = true;
          }
        }
      }
      if (attributesChanged) {
        colorAttribute.needsUpdate = true;
        sizeAttribute.needsUpdate = true;
      }
      controls.update();
      const elevation = Math.abs(camera.position.y - controls.target.y)
        / Math.max(0.001, camera.position.distanceTo(controls.target));
      const pulse = 1 + Math.sin(elapsed * 0.82) * 0.07;
      coreGlow.scale.setScalar(2.15 * pulse);
      coreGlow.material.opacity = 0.3 + Math.sin(elapsed * 0.82) * 0.045;
      const ringOpacity = 0.004 + Math.pow(elevation, 0.82) * 0.072;
      ringMaterials.forEach((material) => { material.opacity = ringOpacity; });
      composer.render();
    };
    animate();
    window.setTimeout(() => setReady(true), 180);

    return () => {
      window.cancelAnimationFrame(frame);
      resizeObserver.disconnect();
      canvas.removeEventListener("pointerdown", onPointerDown);
      canvas.removeEventListener("pointerup", onPointerUp);
      canvas.removeEventListener("keydown", onKeyDown);
      controls.dispose();
      composer.dispose();
      renderer.dispose();
      scene.traverse((object) => {
        object.geometry?.dispose?.();
        if (Array.isArray(object.material)) object.material.forEach((material) => material.dispose());
        else object.material?.dispose?.();
      });
      selectPointRef.current = null;
      clearPointRef.current = null;
    };
  }, [scenes]);

  const closeDetail = () => setDetailOpen(false);
  const clearSelection = () => {
    clearPointRef.current?.();
    setSelectedSceneId("");
    setDetailOpen(false);
  };
  const toggleLike = () => {
    if (!selectedScene) return;
    setLikedSceneIds((current) => {
      const next = new Set(current);
      if (next.has(selectedScene.id)) next.delete(selectedScene.id);
      else next.add(selectedScene.id);
      return next;
    });
  };

  return (
    <div className={`universe-experience${ready ? " is-ready" : ""}`} ref={stageRef}>
      <canvas
        className="universe-canvas"
        ref={canvasRef}
        tabIndex={0}
        aria-label="记忆银河。拖动旋转，滚动缩放，点击星星查看 Scene；按回车聚焦最新一颗星。"
      />

      <header className="universe-heading">
        <h1>Memory</h1>
        <p>A constellation of us</p>
        <span>我们留下的每一段，都在这里继续发光。</span>
      </header>

      <button
        className={`universe-deselect${selectedScene ? " is-visible" : ""}`}
        type="button"
        onClick={clearSelection}
        aria-label="取消选中"
        aria-hidden={!selectedScene}
        tabIndex={selectedScene ? 0 : -1}
      >
        <X size={18} weight="light" aria-hidden="true" />
      </button>

      <div className="universe-hint" aria-hidden={selectedScene ? "true" : undefined}>
        drag to orbit · scroll to fly · tap a star
      </div>

      <button
        className={`universe-scrim${detailOpen && selectedScene ? " is-visible" : ""}`}
        type="button"
        tabIndex={detailOpen && selectedScene ? 0 : -1}
        aria-label="收起记忆"
        onClick={closeDetail}
      />

      <article
        className={`universe-memory${detailOpen && selectedScene ? " is-visible" : ""}`}
        aria-hidden={!detailOpen || !selectedScene}
      >
        {selectedScene ? (
          <>
            <header className="universe-memory__head">
              <Sparkle size={30} weight="fill" color={colorForScene(selectedScene)} aria-hidden="true" />
              <div>
                <h2>{selectedScene.title}</h2>
                <p>{selectedScene.id === scenes.at(-1)?.id ? "The newest light" : "A memory in orbit"}</p>
              </div>
              <button type="button" onClick={closeDetail} aria-label="收起记忆">×</button>
            </header>
            <div className="universe-memory__meta">
              <time dateTime={selectedScene.date}>{selectedScene.date.replaceAll("-", ".")}</time>
              <span>{selectedScene.selfAnchor ? "自我锚点" : selectedScene.bucketDomain || selectedScene.status || "Scene"}</span>
              {selectedScene.relationCount ? <small>{selectedScene.relationCount} 条真实关系边</small> : null}
            </div>
            <div className="universe-memory__body">
              {(selectedScene.body?.length ? selectedScene.body : [selectedScene.excerpt]).map((paragraph, index) => (
                <p key={`${selectedScene.id}-paragraph-${index}`}>{paragraph}</p>
              ))}
            </div>
            <footer className="universe-memory__actions">
              <button
                className={likedSceneIds.has(selectedScene.id) || selectedScene.favorite ? "is-liked" : ""}
                type="button"
                onClick={toggleLike}
                aria-pressed={likedSceneIds.has(selectedScene.id) || selectedScene.favorite}
                aria-label="心动"
              >
                <Heart size={21} weight={likedSceneIds.has(selectedScene.id) || selectedScene.favorite ? "fill" : "light"} aria-hidden="true" />
              </button>
              <button type="button" onClick={closeDetail} aria-label="收起">
                <CaretDown size={21} weight="light" aria-hidden="true" />
              </button>
            </footer>
          </>
        ) : null}
      </article>

      <div className="universe-loading" aria-hidden={ready}>entering our galaxy…</div>
    </div>
  );
}
