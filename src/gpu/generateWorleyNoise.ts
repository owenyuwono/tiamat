export function generateWorleyNoise(size = 256, cellCount = 12): ImageData {
  const points: [number, number][] = [];
  for (let i = 0; i < cellCount * cellCount; i++) {
    points.push([Math.random(), Math.random()]);
  }

  const data = new Uint8ClampedArray(size * size * 4);
  const invSize = 1 / size;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const px = (x + 0.5) * invSize;
      const py = (y + 0.5) * invSize;

      let minDist = 1.0;
      for (const [cx, cy] of points) {
        for (let ox = -1; ox <= 1; ox++) {
          for (let oy = -1; oy <= 1; oy++) {
            const dx = px - (cx + ox);
            const dy = py - (cy + oy);
            const d = Math.sqrt(dx * dx + dy * dy);
            if (d < minDist) minDist = d;
          }
        }
      }

      const cellSize = 1 / cellCount;
      const normalized = Math.min(minDist / (cellSize * 0.7), 1.0);
      const value = 1.0 - normalized * normalized;
      const byte = Math.floor(value * 255);
      const idx = (y * size + x) * 4;
      data[idx] = byte;
      data[idx + 1] = byte;
      data[idx + 2] = byte;
      data[idx + 3] = 255;
    }
  }

  return new ImageData(data, size, size);
}
