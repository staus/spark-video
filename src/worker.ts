import init_wasm, { sort_splats, sort32_splats } from "spark-internal-rs";
import type {
  SOGVideoMetadata,
  SOGVideoTiles,
  SplatEncoding,
} from "./PackedSplats";
import type { PcSogsJson, TranscodeSpzInput } from "./SplatLoader";
import { unpackAntiSplat } from "./antisplat";
import { LN_SCALE_MAX, LN_SCALE_MIN, WASM_SPLAT_SORT } from "./defines";
import { unpackKsplat } from "./ksplat";
import { unpackPcSogs, unpackPcSogsZip } from "./pcsogs";
import { PlyReader } from "./ply";
import { SpzReader, transcodeSpz } from "./spz";
import {
  computeMaxSplats,
  encodeSh1Rgb,
  encodeSh2Rgb,
  encodeSh3Rgb,
  getArrayBuffers,
  setPackedSplat,
  setPackedSplatCenter,
  setPackedSplatOpacity,
  setPackedSplatQuat,
  setPackedSplatRgb,
  setPackedSplatScales,
  toHalf,
} from "./utils";

// WebWorker for Spark's background CPU tasks, such as Gsplat file decoding
// and sorting.

async function onMessage(event: MessageEvent) {
  // Unpack RPC function name, arguments, and ID from the main thread.
  const { name, args, id }: { name: string; args: unknown; id: number } =
    event.data;
  // console.log(`worker.onMessage(${id}, ${name}):`, args);

  // Initialize return result/error, to be filled out below.
  let result = undefined;
  let error = undefined;

  try {
    switch (name) {
      case "unpackPly": {
        const { packedArray, fileBytes, splatEncoding } = args as {
          packedArray: Uint32Array;
          fileBytes: Uint8Array;
          splatEncoding: SplatEncoding;
        };
        const decoded = await unpackPly({
          packedArray,
          fileBytes,
          splatEncoding,
        });
        result = {
          id,
          numSplats: decoded.numSplats,
          packedArray: decoded.packedArray,
          extra: decoded.extra,
        };
        break;
      }
      case "decodeSpz": {
        const { fileBytes, splatEncoding } = args as {
          fileBytes: Uint8Array;
          splatEncoding: SplatEncoding;
        };
        const decoded = await unpackSpz(fileBytes, splatEncoding);
        result = {
          id,
          numSplats: decoded.numSplats,
          packedArray: decoded.packedArray,
          extra: decoded.extra,
        };
        break;
      }
      case "decodeAntiSplat": {
        const { fileBytes, splatEncoding } = args as {
          fileBytes: Uint8Array;
          splatEncoding: SplatEncoding;
        };
        const decoded = unpackAntiSplat(fileBytes, splatEncoding);
        result = {
          id,
          numSplats: decoded.numSplats,
          packedArray: decoded.packedArray,
        };
        break;
      }
      case "decodeKsplat": {
        const { fileBytes, splatEncoding } = args as {
          fileBytes: Uint8Array;
          splatEncoding: SplatEncoding;
        };
        const decoded = unpackKsplat(fileBytes, splatEncoding);
        result = {
          id,
          numSplats: decoded.numSplats,
          packedArray: decoded.packedArray,
          extra: decoded.extra,
        };
        break;
      }
      case "decodePcSogs": {
        const { fileBytes, extraFiles, splatEncoding } = args as {
          fileBytes: Uint8Array;
          extraFiles: Record<string, ArrayBuffer>;
          splatEncoding: SplatEncoding;
        };
        const json = JSON.parse(
          new TextDecoder().decode(fileBytes),
        ) as PcSogsJson;
        const decoded = await unpackPcSogs(json, extraFiles, splatEncoding);
        result = {
          id,
          numSplats: decoded.numSplats,
          packedArray: decoded.packedArray,
          extra: decoded.extra,
        };
        break;
      }
      case "decodePcSogsZip": {
        const { fileBytes, splatEncoding } = args as {
          fileBytes: Uint8Array;
          splatEncoding: SplatEncoding;
        };
        const decoded = await unpackPcSogsZip(fileBytes, splatEncoding);
        result = {
          id,
          numSplats: decoded.numSplats,
          packedArray: decoded.packedArray,
          extra: decoded.extra,
        };
        break;
      }
      case "sortSplats": {
        // Sort maxSplats splats using readback data, which encodes one uint32 per
        // Gsplats, with the low bytes encoding a float16 distance sort metric.
        const { maxSplats, totalSplats, readback, ordering } = args as {
          maxSplats: number;
          totalSplats: number;
          readback: Uint8Array[];
          ordering: Uint32Array;
        };
        // Sort totalSplats splats each with 4 bytes of readback, and outputs Uint32Array ordering of splat indices
        result = {
          id,
          readback,
          ...sortSplats({ totalSplats, readback, ordering }),
        };
        break;
      }
      case "sortDoubleSplats": {
        // Sort numSplats splats using the readback distance metric, which encodes
        // one float16 per splat (no unused high bytes like for sortSplats).
        const { numSplats, readback, ordering } = args as {
          numSplats: number;
          readback: Uint16Array;
          ordering: Uint32Array;
        };
        if (WASM_SPLAT_SORT) {
          result = {
            id,
            readback,
            ordering,
            activeSplats: sort_splats(numSplats, readback, ordering),
          };
        } else {
          result = {
            id,
            readback,
            ...sortDoubleSplats({ numSplats, readback, ordering }),
          };
        }
        break;
      }
      case "sort32Splats": {
        const { maxSplats, numSplats, readback, ordering } = args as {
          maxSplats: number;
          numSplats: number;
          readback: Uint32Array;
          ordering: Uint32Array;
        };
        // Benchmark sort
        // benchmarkSort(numSplats, readback, ordering);
        if (WASM_SPLAT_SORT) {
          result = {
            id,
            readback,
            ordering,
            activeSplats: sort32_splats(numSplats, readback, ordering),
          };
        } else {
          result = {
            id,
            readback,
            ...sort32Splats({ maxSplats, numSplats, readback, ordering }),
          };
        }
        break;
      }
      case "transcodeSpz": {
        const input = args as TranscodeSpzInput;
        const spzBytes = await transcodeSpz(input);
        result = {
          id,
          fileBytes: spzBytes,
          input,
        };
        break;
      }
      case "initVideoMode": {
        const { metadata } = args as { metadata: SOGVideoMetadata };
        initVideoModeWorker(metadata);
        result = { id, success: true };
        break;
      }
      case "decodeVideoFrame": {
        const { tiles, packedArray } = args as {
          tiles: SOGVideoTiles;
          packedArray: Uint32Array;
        };
        decodeVideoFrameWorker(tiles, packedArray);
        result = { id, packedArray };
        break;
      }
      default: {
        throw new Error(`Unknown name: ${name}`);
      }
    }
  } catch (e) {
    error = e;
    console.error(error);
  }

  // Send the result or error back to the main thread, making sure to transfer any ArrayBuffers
  self.postMessage(
    { id, result, error },
    { transfer: getArrayBuffers(result) },
  );
}

function benchmarkSort(
  numSplats: number,
  readback32: Uint32Array,
  ordering: Uint32Array,
) {
  if (numSplats > 0) {
    console.log("Running sort benchmark");
    const readbackF32 = new Float32Array(readback32.buffer);
    const readback16 = new Uint16Array(readback32.length);
    for (let i = 0; i < numSplats; ++i) {
      readback16[i] = toHalf(readbackF32[i]);
    }

    const WARMUP = 10;
    for (let i = 0; i < WARMUP; ++i) {
      const activeSplats = sort_splats(numSplats, readback16, ordering);
      const activeSplats32 = sort32_splats(numSplats, readback32, ordering);
      const results = sortDoubleSplats({
        numSplats,
        readback: readback16,
        ordering,
      });
      const results32 = sort32Splats({
        maxSplats: numSplats,
        numSplats,
        readback: readback32,
        ordering,
      });
    }

    const TIMING_SAMPLES = 1000;
    let start: number;

    start = performance.now();
    for (let i = 0; i < TIMING_SAMPLES; ++i) {
      const activeSplats = sort_splats(numSplats, readback16, ordering);
    }
    const wasmTime = (performance.now() - start) / TIMING_SAMPLES;

    start = performance.now();
    for (let i = 0; i < TIMING_SAMPLES; ++i) {
      const results = sortDoubleSplats({
        numSplats,
        readback: readback16,
        ordering,
      });
    }
    const jsTime = (performance.now() - start) / TIMING_SAMPLES;

    console.log(
      `JS: ${jsTime} ms, WASM: ${wasmTime} ms, numSplats: ${numSplats}`,
    );

    start = performance.now();
    for (let i = 0; i < TIMING_SAMPLES; ++i) {
      const activeSplats32 = sort32_splats(numSplats, readback32, ordering);
    }
    const wasm32Time = (performance.now() - start) / TIMING_SAMPLES;

    start = performance.now();
    for (let i = 0; i < TIMING_SAMPLES; ++i) {
      const results = sort32Splats({
        maxSplats: numSplats,
        numSplats,
        readback: readback32,
        ordering,
      });
    }
    const js32Time = (performance.now() - start) / TIMING_SAMPLES;

    console.log(
      `JS32: ${js32Time} ms, WASM32: ${wasm32Time} ms, numSplats: ${numSplats}`,
    );
  }
}

async function unpackPly({
  packedArray,
  fileBytes,
  splatEncoding,
}: {
  packedArray: Uint32Array;
  fileBytes: Uint8Array;
  splatEncoding: SplatEncoding;
}): Promise<{
  packedArray: Uint32Array;
  numSplats: number;
  extra: Record<string, unknown>;
}> {
  const ply = new PlyReader({ fileBytes });
  await ply.parseHeader();
  const numSplats = ply.numSplats;

  const extra: Record<string, unknown> = {};

  ply.parseSplats(
    (
      index,
      x,
      y,
      z,
      scaleX,
      scaleY,
      scaleZ,
      quatX,
      quatY,
      quatZ,
      quatW,
      opacity,
      r,
      g,
      b,
    ) => {
      setPackedSplat(
        packedArray,
        index,
        x,
        y,
        z,
        scaleX,
        scaleY,
        scaleZ,
        quatX,
        quatY,
        quatZ,
        quatW,
        opacity,
        r,
        g,
        b,
        splatEncoding,
      );
    },
    (index, sh1, sh2, sh3) => {
      if (sh1) {
        if (!extra.sh1) {
          extra.sh1 = new Uint32Array(numSplats * 2);
        }
        encodeSh1Rgb(extra.sh1 as Uint32Array, index, sh1, splatEncoding);
      }
      if (sh2) {
        if (!extra.sh2) {
          extra.sh2 = new Uint32Array(numSplats * 4);
        }
        encodeSh2Rgb(extra.sh2 as Uint32Array, index, sh2, splatEncoding);
      }
      if (sh3) {
        if (!extra.sh3) {
          extra.sh3 = new Uint32Array(numSplats * 4);
        }
        encodeSh3Rgb(extra.sh3 as Uint32Array, index, sh3, splatEncoding);
      }
    },
  );

  return { packedArray, numSplats, extra };
}

async function unpackSpz(
  fileBytes: Uint8Array,
  splatEncoding: SplatEncoding,
): Promise<{
  packedArray: Uint32Array;
  numSplats: number;
  extra: Record<string, unknown>;
}> {
  const spz = new SpzReader({ fileBytes });
  await spz.parseHeader();
  const numSplats = spz.numSplats;
  const maxSplats = computeMaxSplats(numSplats);
  const packedArray = new Uint32Array(maxSplats * 4);
  const extra: Record<string, unknown> = {};

  await spz.parseSplats(
    (index, x, y, z) => {
      setPackedSplatCenter(packedArray, index, x, y, z);
    },
    (index, alpha) => {
      setPackedSplatOpacity(packedArray, index, alpha);
    },
    (index, r, g, b) => {
      setPackedSplatRgb(packedArray, index, r, g, b, splatEncoding);
    },
    (index, scaleX, scaleY, scaleZ) => {
      setPackedSplatScales(
        packedArray,
        index,
        scaleX,
        scaleY,
        scaleZ,
        splatEncoding,
      );
    },
    (index, quatX, quatY, quatZ, quatW) => {
      setPackedSplatQuat(packedArray, index, quatX, quatY, quatZ, quatW);
    },
    (index, sh1, sh2, sh3) => {
      if (sh1) {
        if (!extra.sh1) {
          extra.sh1 = new Uint32Array(numSplats * 2);
        }
        encodeSh1Rgb(extra.sh1 as Uint32Array, index, sh1, splatEncoding);
      }
      if (sh2) {
        if (!extra.sh2) {
          extra.sh2 = new Uint32Array(numSplats * 4);
        }
        encodeSh2Rgb(extra.sh2 as Uint32Array, index, sh2, splatEncoding);
      }
      if (sh3) {
        if (!extra.sh3) {
          extra.sh3 = new Uint32Array(numSplats * 4);
        }
        encodeSh3Rgb(extra.sh3 as Uint32Array, index, sh3, splatEncoding);
      }
    },
  );
  return { packedArray, numSplats, extra };
}

// Array of buckets for sorting float16 distances with range [0, DEPTH_INFINITY].
const DEPTH_INFINITY_F16 = 0x7c00;
const DEPTH_SIZE_16 = DEPTH_INFINITY_F16 + 1;
let depthArray16: Uint32Array | null = null;

function sortSplats({
  totalSplats,
  readback,
  ordering,
}: { totalSplats: number; readback: Uint8Array[]; ordering: Uint32Array }): {
  activeSplats: number;
  ordering: Uint32Array;
} {
  // Sort totalSplats Gsplats, each with 4 bytes of readback, and outputs Uint32Array
  // of indices from most distant to nearest. Each 4 bytes encode a float16 distance
  // and unused high bytes.
  if (!depthArray16) {
    depthArray16 = new Uint32Array(DEPTH_SIZE_16);
  }
  depthArray16.fill(0);

  const readbackUint32 = readback.map((layer) => new Uint32Array(layer.buffer));
  const layerSize = readbackUint32[0].length;
  const numLayers = Math.ceil(totalSplats / layerSize);

  let layerBase = 0;
  for (let layer = 0; layer < numLayers; ++layer) {
    const readbackLayer = readbackUint32[layer];
    const layerSplats = Math.min(readbackLayer.length, totalSplats - layerBase);
    for (let i = 0; i < layerSplats; ++i) {
      const pri = readbackLayer[i] & 0x7fff;
      if (pri < DEPTH_INFINITY_F16) {
        depthArray16[pri] += 1;
      }
    }
    layerBase += layerSplats;
  }

  let activeSplats = 0;
  for (let j = 0; j < DEPTH_SIZE_16; ++j) {
    const nextIndex = activeSplats + depthArray16[j];
    depthArray16[j] = activeSplats;
    activeSplats = nextIndex;
  }

  layerBase = 0;
  for (let layer = 0; layer < numLayers; ++layer) {
    const readbackLayer = readbackUint32[layer];
    const layerSplats = Math.min(readbackLayer.length, totalSplats - layerBase);
    for (let i = 0; i < layerSplats; ++i) {
      const pri = readbackLayer[i] & 0x7fff;
      if (pri < DEPTH_INFINITY_F16) {
        ordering[depthArray16[pri]] = layerBase + i;
        depthArray16[pri] += 1;
      }
    }
    layerBase += layerSplats;
  }
  if (depthArray16[DEPTH_SIZE_16 - 1] !== activeSplats) {
    throw new Error(
      `Expected ${activeSplats} active splats but got ${depthArray16[DEPTH_SIZE_16 - 1]}`,
    );
  }

  return { activeSplats, ordering };
}

// Sort numSplats splats, each with 2 bytes of float16 readback for distance metric,
// using one bucket sort pass, outputting Uint32Array of indices.
function sortDoubleSplats({
  numSplats,
  readback,
  ordering,
}: { numSplats: number; readback: Uint16Array; ordering: Uint32Array }): {
  activeSplats: number;
  ordering: Uint32Array;
} {
  // Ensure depthArray is allocated and zeroed out for our buckets.
  if (!depthArray16) {
    depthArray16 = new Uint32Array(DEPTH_SIZE_16);
  }
  depthArray16.fill(0);

  // Count the number of splats in each bucket (cull Gsplats at infinity).
  for (let i = 0; i < numSplats; ++i) {
    const pri = readback[i];
    if (pri < DEPTH_INFINITY_F16) {
      depthArray16[pri] += 1;
    }
  }

  // Compute the beginning index of each bucket in the output array and the
  // total number of active (non-infinity) splats, going in reverse order
  // because we want most distant Gsplats to be first in the output array.
  let activeSplats = 0;
  for (let j = DEPTH_INFINITY_F16 - 1; j >= 0; --j) {
    const nextIndex = activeSplats + depthArray16[j];
    depthArray16[j] = activeSplats;
    activeSplats = nextIndex;
  }

  // Write out the sorted indices into the output array according
  // bucket order.
  for (let i = 0; i < numSplats; ++i) {
    const pri = readback[i];
    if (pri < DEPTH_INFINITY_F16) {
      ordering[depthArray16[pri]] = i;
      depthArray16[pri] += 1;
    }
  }
  // Sanity check that the end of the closest bucket is the same as
  // our total count of active splats (not at infinity).
  if (depthArray16[0] !== activeSplats) {
    throw new Error(
      `Expected ${activeSplats} active splats but got ${depthArray16[0]}`,
    );
  }

  return { activeSplats, ordering };
}

const DEPTH_INFINITY_F32 = 0x7f800000;
let bucket16lo: Uint32Array | null = null;
let bucket16hi: Uint32Array | null = null;
let scratchSplats: Uint32Array | null = null;

// two-pass radix sort (base 65536) of 32-bit keys in readback,
// but placing largest values first.
function sort32Splats({
  maxSplats,
  numSplats,
  readback, // Uint32Array of bit‑patterns
  ordering, // Uint32Array to fill with sorted indices
}: {
  maxSplats: number;
  numSplats: number;
  readback: Uint32Array;
  ordering: Uint32Array;
}): { activeSplats: number; ordering: Uint32Array } {
  const BASE = 1 << 16; // 65536

  // allocate once
  if (!bucket16lo) {
    bucket16lo = new Uint32Array(BASE);
  }
  if (!bucket16hi) {
    bucket16hi = new Uint32Array(BASE);
  }
  if (!scratchSplats || scratchSplats.length < maxSplats) {
    scratchSplats = new Uint32Array(maxSplats);
  }

  // tally low and high buckets
  bucket16lo.fill(0);
  bucket16hi.fill(0);
  for (let i = 0; i < numSplats; ++i) {
    const key = readback[i];
    if (key < DEPTH_INFINITY_F32) {
      const inv = ~key >>> 0;
      bucket16lo[inv & 0xffff] += 1;
      bucket16hi[inv >>> 16] += 1;
    }
  }

  //
  // ——— Pass #1: bucket by inv(lo 16 bits) ———
  //
  // exclusive prefix‑sum → starting offsets
  let total = 0;
  for (let b = 0; b < BASE; ++b) {
    const c = bucket16lo[b];
    bucket16lo[b] = total;
    total += c;
  }
  const activeSplats = total;

  // scatter into scratch by low bits of inv
  for (let i = 0; i < numSplats; ++i) {
    const key = readback[i];
    if (key < DEPTH_INFINITY_F32) {
      const inv = ~key >>> 0;
      scratchSplats[bucket16lo[inv & 0xffff]++] = i;
    }
  }

  //
  // ——— Pass #2: bucket by inv(hi 16 bits) ———
  //
  // exclusive prefix‑sum again
  let sum = 0;
  for (let b = 0; b < BASE; ++b) {
    const c = bucket16hi[b];
    bucket16hi[b] = sum;
    sum += c;
  }

  // scatter into final ordering by high bits of inv
  for (let k = 0; k < activeSplats; ++k) {
    const idx = scratchSplats[k];
    const inv = ~readback[idx] >>> 0;
    ordering[bucket16hi[inv >>> 16]++] = idx;
  }

  // sanity‑check: the last bucket should have eaten all entries
  if (bucket16hi[BASE - 1] !== activeSplats) {
    throw new Error(
      `Expected ${activeSplats} active splats but got ${bucket16hi[BASE - 1]}`,
    );
  }

  return { activeSplats, ordering };
}

// =============================================
// Video Frame Decoding (Worker-side)
// =============================================

// Pre-computed lookup tables for video mode
let videoModeData: {
  count: number;
  scaleLookup: Uint8Array;
  sh0Lookup: Uint8Array;
  quatLookup: Float32Array;
  posFloat16LookupX: Uint16Array;
  posFloat16LookupY: Uint16Array;
  posFloat16LookupZ: Uint16Array;
} | null = null;

// Shared typed arrays for float16 conversion
const f32View = new Float32Array(1);
const i32View = new Int32Array(f32View.buffer);

function toFloat16Worker(value: number): number {
  f32View[0] = value;
  const f = i32View[0];
  const sign = (f >> 16) & 0x8000;
  const exponent = ((f >> 23) & 0xff) - 127 + 15;
  const mantissa = f & 0x7fffff;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    const m = (mantissa | 0x800000) >> (1 - exponent);
    return sign | (m >> 13);
  }
  if (exponent >= 31) return sign | 0x7c00;
  return sign | (exponent << 10) | (mantissa >> 13);
}

function encodeQuatOctWorker(
  qx: number,
  qy: number,
  qz: number,
  qw: number,
): number {
  const len = Math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
  let nx = qx / len;
  let ny = qy / len;
  let nz = qz / len;
  let nw = qw / len;
  if (nw < 0) {
    nx = -nx;
    ny = -ny;
    nz = -nz;
    nw = -nw;
  }
  const theta = 2 * Math.acos(Math.min(1, nw));
  const xyzNorm = Math.sqrt(nx * nx + ny * ny + nz * nz);
  let axisX: number;
  let axisY: number;
  let axisZ: number;
  if (xyzNorm < 1e-6) {
    axisX = 1;
    axisY = 0;
    axisZ = 0;
  } else {
    const invNorm = 1 / xyzNorm;
    axisX = nx * invNorm;
    axisY = ny * invNorm;
    axisZ = nz * invNorm;
  }
  const sum = Math.abs(axisX) + Math.abs(axisY) + Math.abs(axisZ);
  let px = axisX / sum;
  let py = axisY / sum;
  if (axisZ < 0) {
    const tmp = px;
    px = (1 - Math.abs(py)) * (px >= 0 ? 1 : -1);
    py = (1 - Math.abs(tmp)) * (py >= 0 ? 1 : -1);
  }
  const quantU = Math.round((px * 0.5 + 0.5) * 255);
  const quantV = Math.round((py * 0.5 + 0.5) * 255);
  const angleInt = Math.round((theta / Math.PI) * 255);
  return quantU | (quantV << 8) | (angleInt << 16);
}

function initVideoModeWorker(metadata: SOGVideoMetadata) {
  const SH_C0 = 0.28209479177387814;
  const SQRT2 = Math.sqrt(2);

  const scaleLookup = new Uint8Array(256);
  const lnScaleScale = 254.0 / (LN_SCALE_MAX - LN_SCALE_MIN);
  for (let i = 0; i < 256; i++) {
    const logScale = metadata.scaleCodebook[i] ?? metadata.scaleCodebook[0];
    const sparkScale = Math.min(
      255,
      Math.max(1, Math.round((logScale - LN_SCALE_MIN) * lnScaleScale) + 1),
    );
    scaleLookup[i] = sparkScale;
  }

  const sh0Lookup = new Uint8Array(256);
  for (let i = 0; i < 256; i++) {
    const shVal = metadata.sh0Codebook[i] ?? metadata.sh0Codebook[0];
    const rgb = Math.min(
      255,
      Math.max(0, Math.round((SH_C0 * shVal + 0.5) * 255)),
    );
    sh0Lookup[i] = rgb;
  }

  const quatLookup = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    quatLookup[i] = (i / 255 - 0.5) * SQRT2;
  }

  const posFloat16LookupX = new Uint16Array(65536);
  const posFloat16LookupY = new Uint16Array(65536);
  const posFloat16LookupZ = new Uint16Array(65536);

  const rangeX = metadata.maxs[0] - metadata.mins[0];
  const rangeY = metadata.maxs[1] - metadata.mins[1];
  const rangeZ = metadata.maxs[2] - metadata.mins[2];

  for (let i = 0; i < 65536; i++) {
    const fx = i / 65535;
    let posX = metadata.mins[0] + rangeX * fx;
    let posY = metadata.mins[1] + rangeY * fx;
    let posZ = metadata.mins[2] + rangeZ * fx;
    posX = Math.sign(posX) * (Math.exp(Math.abs(posX)) - 1);
    posY = Math.sign(posY) * (Math.exp(Math.abs(posY)) - 1);
    posZ = Math.sign(posZ) * (Math.exp(Math.abs(posZ)) - 1);
    posFloat16LookupX[i] = toFloat16Worker(posX);
    posFloat16LookupY[i] = toFloat16Worker(posY);
    posFloat16LookupZ[i] = toFloat16Worker(posZ);
  }

  videoModeData = {
    count: metadata.count,
    scaleLookup,
    sh0Lookup,
    quatLookup,
    posFloat16LookupX,
    posFloat16LookupY,
    posFloat16LookupZ,
  };
}

function decodeVideoFrameWorker(tiles: SOGVideoTiles, packed: Uint32Array) {
  if (!videoModeData) {
    throw new Error("Call initVideoMode first");
  }

  const {
    count,
    scaleLookup,
    sh0Lookup,
    quatLookup,
    posFloat16LookupX,
    posFloat16LookupY,
    posFloat16LookupZ,
  } = videoModeData;

  const meansL = tiles.means_l;
  const meansU = tiles.means_u;
  const quats = tiles.quats;
  const scales = tiles.scales;
  const sh0 = tiles.sh0;

  for (let i = 0; i < count; i++) {
    const p = i << 2;

    const posXU16 = meansL[p] | (meansU[p] << 8);
    const posYU16 = meansL[p + 1] | (meansU[p + 1] << 8);
    const posZU16 = meansL[p + 2] | (meansU[p + 2] << 8);
    const posXF16 = posFloat16LookupX[posXU16];
    const posYF16 = posFloat16LookupY[posYU16];
    const posZF16 = posFloat16LookupZ[posZU16];

    const r0 = quatLookup[quats[p]];
    const r1 = quatLookup[quats[p + 1]];
    const r2 = quatLookup[quats[p + 2]];
    const rr = Math.sqrt(Math.max(0, 1.0 - r0 * r0 - r1 * r1 - r2 * r2));
    const rOrder = quats[p + 3] - 252;
    const qx = rOrder === 0 ? r0 : rOrder === 1 ? rr : r1;
    const qy = rOrder <= 1 ? r1 : rOrder === 2 ? rr : r2;
    const qz = rOrder <= 2 ? r2 : rr;
    const qw = rOrder === 0 ? rr : r0;
    const uQuat = encodeQuatOctWorker(qx, qy, qz, qw);

    const uScaleX = scaleLookup[scales[p]];
    const uScaleY = scaleLookup[scales[p + 1]];
    const uScaleZ = scaleLookup[scales[p + 2]];

    const colorR = sh0Lookup[sh0[p]];
    const colorG = sh0Lookup[sh0[p + 1]];
    const colorB = sh0Lookup[sh0[p + 2]];
    const opacity = sh0[p + 3];

    packed[p] = colorR | (colorG << 8) | (colorB << 16) | (opacity << 24);
    packed[p + 1] = posXF16 | (posYF16 << 16);
    packed[p + 2] =
      posZF16 | ((uQuat & 0xff) << 16) | (((uQuat >> 8) & 0xff) << 24);
    packed[p + 3] =
      uScaleX |
      (uScaleY << 8) |
      (uScaleZ << 16) |
      (((uQuat >> 16) & 0xff) << 24);
  }
}

// Buffer to queue any messages received while initializing, for example
// early messages to unpack a Gsplat file while still initializing the WASM code.
const messageBuffer: MessageEvent[] = [];

function bufferMessage(event: MessageEvent) {
  messageBuffer.push(event);
}

async function initialize() {
  // Hold any messages received while initializing
  self.addEventListener("message", bufferMessage);

  await init_wasm();

  self.removeEventListener("message", bufferMessage);
  self.addEventListener("message", onMessage);

  // Process any buffered messages
  for (const event of messageBuffer) {
    onMessage(event);
  }
  messageBuffer.length = 0;
}

initialize().catch(console.error);
