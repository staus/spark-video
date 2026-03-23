/**
 * Delta-encoded 4DGS video decoder.
 *
 * Each frame contains only newly-born gaussians with motion vectors and lifetimes.
 * The decoder accumulates births, interpolates positions, and reconstructs full SOG textures.
 *
 * Uses canvas 2D with careful color space handling to read raw pixel data.
 */

import { unzipSync } from "fflate";

// WebCodecs API types (not yet complete in lib.dom.d.ts)
interface ImageDecoderInit {
  data: ArrayBuffer | ArrayBufferView;
  type: string;
}

interface ImageDecodeResult {
  image: VideoFrame;
}

interface ImageTrack {
  frameCount: number;
}

interface ImageTrackList {
  ready: Promise<void>;
  selectedTrack: ImageTrack | null;
}

interface ImageDecoderInterface {
  tracks: ImageTrackList;
  decode(options: { frameIndex: number }): Promise<ImageDecodeResult>;
  close(): void;
}

declare const ImageDecoder: {
  new (init: ImageDecoderInit): ImageDecoderInterface;
};

/**
 * Keyframe metadata - frames extracted separately from the video stream
 */
export interface KeyframeInfo {
  frame_index: number;
  path: string;
  tile_size: number;
  birth_count: number;
}

/**
 * Metadata format for delta-encoded 4DGS video files (JSON sidecar)
 */
export interface Delta4DGSMetadata {
  tile_size: number;
  grid: [number, number]; // [cols, rows] for delta frame layout
  layout: Record<string, [number, number]>; // tile name -> [col, row]
  video: {
    frames: number;
    fps: number;
    start_frame?: number; // First frame index in video (after keyframes)
    frame_map?: number[]; // Maps video frame index to original frame index
  };
  sog: {
    means: {
      mins: [number, number, number];
      maxs: [number, number, number];
    };
    motion: {
      mins: [number, number, number];
      maxs: [number, number, number];
    };
    scales: {
      codebook: number[];
    };
    sh0: {
      codebook: number[];
    };
  };
  "4dgs": {
    max_active_gaussians: number;
    birth_counts: number[];
    total_frames?: number; // Total frames including keyframes
  };
  encoding: "delta";
  keyframes?: KeyframeInfo[]; // Separate keyframe images
}

export class DeltaSplatDecoder {
  private metadata: Delta4DGSMetadata;
  private tileSize: number; // Delta frame tile size (based on max_births)
  private sogTileSize: number; // SOG output tile size (based on max_active)

  private posMins: Float32Array;
  private posMaxs: Float32Array;
  private posRange: Float32Array;

  private motionMins: Float32Array;
  private motionMaxs: Float32Array;
  private motionRange: Float32Array;

  private maxActive: number;
  private activeIndices: number[]; // Compact list of occupied slot indices
  private freeSlots: number[];
  private activeCount: number;
  currentFrameIndex: number;

  // Struct-of-Arrays storage for active gaussians (zero per-frame allocations)
  private positions: Float32Array; // maxActive * 3
  private motions: Float32Array; // maxActive * 3
  private quatsEncoded: Uint8Array; // maxActive * 4
  private scalesEncoded: Uint8Array; // maxActive * 4
  private sh0Encoded: Uint8Array; // maxActive * 4
  private remainingFrames: Int16Array; // maxActive (-1 = inactive)
  private justBorn: Uint8Array; // maxActive (0 or 1)

  // Pre-allocated temp buffers for birth decoding (avoids per-frame allocation)
  private tempSlots: number[]; // Reused each frame
  private tempMotionMags: Float32Array; // max births per frame

  // Store raw pixel data for each video frame (pre-decoded at load time)
  private framePixelData: Uint8ClampedArray[];
  private frameWidth: number;
  private frameHeight: number;

  // Keyframe support: separate images with potentially different tile sizes
  private keyframeData: Map<number, Uint8ClampedArray>; // frameIndex -> pixel data
  private keyframeTileSizes: Map<number, number>; // frameIndex -> tile_size
  private keyframeIndices: Set<number>; // Set of frame indices that are keyframes
  private totalFrames: number; // Total frames including keyframes
  private frameIndexToVideoIndex: Map<number, number> | null = null; // Reverse lookup for O(1) frame mapping

  // SOG output: 3x2 tiles (standard SOG layout for GPU decode)
  private sogWidth: number;
  private sogHeight: number;
  private sogTileData: Uint8Array;

  // Pre-allocated position encoding buffer
  private posEncodeBuf: Uint8Array;

  // GPU mode output buffers (float positions, no CPU encoding)
  private gpuPositionBuffer: Float32Array;
  private gpuAttributeBuffer: Uint8Array; // 3 rows: quats, scales, sh0
  private gpuTextureSize: number; // Square texture side length

  // Static/dynamic separation for GPU upload optimization
  // Static gaussians: keyframe births with zero/minimal motion (uploaded once)
  // Dynamic gaussians: all others (uploaded each frame)
  private staticSlots: number[] = []; // Slot indices of static gaussians
  private staticCount = 0;
  private staticPositionBuffer: Float32Array | null = null;
  private staticAttributeBuffer: Uint8Array | null = null;
  private staticTextureSize = 0;
  private staticDataReady = false;

  // Dynamic gaussian tracking (excludes static keyframe gaussians)
  private dynamicTextureSize = 0;
  private dynamicPositionBuffer: Float32Array | null = null;
  private dynamicAttributeBuffer: Uint8Array | null = null;

  // Motion magnitude threshold for classifying keyframe births as static
  // Gaussians with motion below this are uploaded once and never updated
  // 0.001 captures ~86% of keyframe births (motion below this is imperceptible)
  private static readonly STATIC_MOTION_THRESHOLD = 0.001;

  // Cache for dynamic keyframe births (reused on subsequent loops to avoid re-decoding)
  // Stores template data that can be copied to new slots without re-decoding
  private dynamicKeyframeBirthCache: Map<
    number,
    {
      count: number;
      lifetimes: Int16Array;
      positions: Float32Array; // count * 3
      motions: Float32Array; // count * 3
      quats: Uint8Array; // count * 4
      scales: Uint8Array; // count * 4
      sh0: Uint8Array; // count * 4
    }
  > = new Map();

  constructor(metadata: Delta4DGSMetadata) {
    this.metadata = metadata;
    this.tileSize = metadata.tile_size;

    // Parse bounds
    this.posMins = new Float32Array(metadata.sog.means.mins);
    this.posMaxs = new Float32Array(metadata.sog.means.maxs);
    this.posRange = new Float32Array([
      this.posMaxs[0] - this.posMins[0],
      this.posMaxs[1] - this.posMins[1],
      this.posMaxs[2] - this.posMins[2],
    ]);

    this.motionMins = new Float32Array(metadata.sog.motion.mins);
    this.motionMaxs = new Float32Array(metadata.sog.motion.maxs);
    this.motionRange = new Float32Array([
      this.motionMaxs[0] - this.motionMins[0],
      this.motionMaxs[1] - this.motionMins[1],
      this.motionMaxs[2] - this.motionMins[2],
    ]);

    // Pre-allocate for max active gaussians
    const maxActive = metadata["4dgs"].max_active_gaussians;
    this.maxActive = maxActive;
    this.activeIndices = [];
    this.freeSlots = [];
    for (let i = maxActive - 1; i >= 0; i--) {
      this.freeSlots.push(i);
    }
    this.activeCount = 0;
    this.currentFrameIndex = 0;

    // Struct-of-Arrays buffers (zero per-frame allocations)
    this.positions = new Float32Array(maxActive * 3);
    this.motions = new Float32Array(maxActive * 3);
    this.quatsEncoded = new Uint8Array(maxActive * 4);
    this.scalesEncoded = new Uint8Array(maxActive * 4);
    this.sh0Encoded = new Uint8Array(maxActive * 4);
    this.remainingFrames = new Int16Array(maxActive);
    this.remainingFrames.fill(-1); // -1 = inactive
    this.justBorn = new Uint8Array(maxActive);

    // Pre-allocated temp buffers for birth decoding
    const maxBirths = Math.max(...metadata["4dgs"].birth_counts);
    this.tempSlots = new Array(maxBirths);
    this.tempMotionMags = new Float32Array(maxBirths);

    // Frame dimensions (for delta frames, based on max_births)
    this.frameWidth = this.tileSize * metadata.grid[0];
    this.frameHeight = this.tileSize * metadata.grid[1];

    // Pre-decoded frame pixel data (filled during loadDeltaFrames)
    this.framePixelData = [];

    // Initialize keyframe support
    this.keyframeData = new Map();
    this.keyframeTileSizes = new Map();
    this.keyframeIndices = new Set();
    this.totalFrames = metadata["4dgs"].total_frames ?? metadata.video.frames;

    // Register keyframes from metadata
    if (metadata.keyframes) {
      for (const kf of metadata.keyframes) {
        this.keyframeIndices.add(kf.frame_index);
        this.keyframeTileSizes.set(kf.frame_index, kf.tile_size);
      }
      console.log(
        `Keyframes registered: ${metadata.keyframes.length} (indices: ${Array.from(this.keyframeIndices).join(", ")})`,
      );
    }

    // Build reverse lookup map for O(1) frame index -> video index mapping
    if (metadata.video.frame_map) {
      this.frameIndexToVideoIndex = new Map();
      metadata.video.frame_map.forEach((frameIdx, videoIdx) => {
        this.frameIndexToVideoIndex?.set(frameIdx, videoIdx);
      });
    }

    // SOG output tile size: based on max_active, not delta frame tile_size
    const sogTileSide = Math.ceil(Math.sqrt(maxActive));
    const sogTileSizePower = Math.ceil(Math.log2(Math.max(sogTileSide, 1)));
    this.sogTileSize = 2 ** sogTileSizePower;

    // SOG output: 3x2 tiles
    const sogWidth = this.sogTileSize * 3;
    const sogHeight = this.sogTileSize * 2;
    this.sogWidth = sogWidth;
    this.sogHeight = sogHeight;
    this.sogTileData = new Uint8Array(sogWidth * sogHeight * 4);

    // Pre-allocated position encoding buffer
    this.posEncodeBuf = new Uint8Array(6);

    // GPU mode output buffers
    // Position texture: square, power-of-2, fits maxActive gaussians
    const gpuTexSide = Math.ceil(Math.sqrt(maxActive));
    const gpuTexSizePow = Math.ceil(Math.log2(Math.max(gpuTexSide, 1)));
    this.gpuTextureSize = 2 ** gpuTexSizePow;
    // 4 floats per gaussian (RGBA32F, alpha unused)
    this.gpuPositionBuffer = new Float32Array(
      this.gpuTextureSize * this.gpuTextureSize * 4,
    );
    // Attribute texture: width = gpuTextureSize, height = gpuTextureSize * 3
    // 3 "planes": quats, scales, sh0 - each plane is gpuTextureSize x gpuTextureSize
    // Total: gpuTextureSize^2 * 3 * 4 bytes
    this.gpuAttributeBuffer = new Uint8Array(
      this.gpuTextureSize * this.gpuTextureSize * 3 * 4,
    );

    console.log(
      `DeltaSplatDecoder initialized: maxActive=${maxActive}, deltaTile=${this.tileSize}, sogTile=${this.sogTileSize}, gpuTex=${this.gpuTextureSize}`,
    );
    console.log(
      `GPU buffers: position=${this.gpuPositionBuffer.length} floats (${this.gpuTextureSize}x${this.gpuTextureSize}), ` +
        `attributes=${this.gpuAttributeBuffer.length} bytes (${this.gpuTextureSize}x${this.gpuTextureSize * 3})`,
    );
    console.log(
      `Total frames: ${this.totalFrames}, Video frames: ${metadata.video.frames}, Keyframes: ${this.keyframeIndices.size}`,
    );
  }

  async loadDeltaFrames(webpBlob: Blob): Promise<void> {
    const arrayBuffer = await webpBlob.arrayBuffer();
    const decoder = new ImageDecoder({ data: arrayBuffer, type: "image/webp" });
    await decoder.tracks.ready;

    const frameCount =
      decoder.tracks.selectedTrack?.frameCount || this.metadata.video.frames;
    this.framePixelData = [];

    console.log(`Decoding ${frameCount} delta frames...`);
    console.log(`Frame dimensions: ${this.frameWidth}x${this.frameHeight}`);

    for (let i = 0; i < frameCount; i++) {
      const result = await decoder.decode({ frameIndex: i });
      const frame = result.image;

      // Log format on first frame (before closing)
      const format = frame.format ?? "unknown";
      if (i === 0) {
        console.log(`VideoFrame format: ${format}`);
      }

      // Use copyTo() to get raw bytes directly from VideoFrame
      // This bypasses ImageBitmap color space conversion
      const byteLength = this.frameWidth * this.frameHeight * 4;
      const buffer = new ArrayBuffer(byteLength);

      // copyTo gives us raw bytes in the frame's native format
      await frame.copyTo(buffer);

      frame.close();

      // Convert to Uint8ClampedArray and handle BGRA->RGBA if needed
      const pixels = new Uint8ClampedArray(buffer);

      // BGRA is common for browsers - swap R and B channels
      if (format === "BGRA" || format === "BGRX") {
        for (let j = 0; j < pixels.length; j += 4) {
          const b = pixels[j];
          pixels[j] = pixels[j + 2]; // R = B
          pixels[j + 2] = b; // B = R
        }
      }

      this.framePixelData.push(pixels);

      // Debug: log first frame's pixels from different tiles
      if (i === 0) {
        console.log(
          `Frame 0 first pixel (means_l): R=${pixels[0]} G=${pixels[1]} B=${pixels[2]} A=${pixels[3]}`,
        );
        // Pixel from tile (2,0) = quats tile
        const ts = this.tileSize;
        const quatTileX = 2 * ts;
        const quatIdx = quatTileX * 4;
        console.log(
          `Frame 0 quat tile first pixel: R=${pixels[quatIdx]} G=${pixels[quatIdx + 1]} B=${pixels[quatIdx + 2]} A=${pixels[quatIdx + 3]}`,
        );
        // Pixel from tile (0,1) = scales tile
        const scaleTileY = ts;
        const scaleIdx = scaleTileY * this.frameWidth * 4;
        console.log(
          `Frame 0 scale tile first pixel: R=${pixels[scaleIdx]} G=${pixels[scaleIdx + 1]} B=${pixels[scaleIdx + 2]} A=${pixels[scaleIdx + 3]}`,
        );
        // Pixel from tile (1,1) = sh0 tile
        const sh0Idx = (scaleTileY * this.frameWidth + ts) * 4;
        console.log(
          `Frame 0 sh0 tile first pixel: R=${pixels[sh0Idx]} G=${pixels[sh0Idx + 1]} B=${pixels[sh0Idx + 2]} A=${pixels[sh0Idx + 3]}`,
        );
        console.log(
          `Birth count for frame 0: ${this.metadata["4dgs"].birth_counts[0]}`,
        );
      }
    }

    decoder.close();
    console.log(`Decoded ${this.framePixelData.length} delta frames`);

    // Log total births
    const totalBirths = this.metadata["4dgs"].birth_counts.reduce(
      (a, b) => a + b,
      0,
    );
    console.log(`Total births across all frames: ${totalBirths}`);
  }

  /**
   * Load keyframe PNG images.
   * Call this after loadDeltaFrames() if metadata contains keyframes.
   *
   * @param baseUrl Base URL for fetching keyframe files (directory containing the JSON)
   */
  async loadKeyframes(baseUrl: string): Promise<void> {
    if (!this.metadata.keyframes || this.metadata.keyframes.length === 0) {
      console.log("No keyframes to load");
      return;
    }

    console.log(`Loading ${this.metadata.keyframes.length} keyframe(s)...`);

    for (const kf of this.metadata.keyframes) {
      const url = `${baseUrl}/${kf.path}`;
      console.log(`  Loading keyframe ${kf.frame_index}: ${url}`);

      try {
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        const blob = await response.blob();

        // Decode PNG to ImageBitmap with settings to preserve raw data
        const bitmap = await createImageBitmap(blob, {
          premultiplyAlpha: "none",
          colorSpaceConversion: "none",
        });

        // Draw to canvas to get raw pixel data
        const kfTileSize = kf.tile_size;
        const kfWidth = kfTileSize * this.metadata.grid[0];
        const kfHeight = kfTileSize * this.metadata.grid[1];

        const canvas = new OffscreenCanvas(kfWidth, kfHeight);
        const ctx = canvas.getContext("2d", {
          willReadFrequently: true,
        });
        if (!ctx) {
          throw new Error("Failed to get 2D context for keyframe canvas");
        }

        ctx.drawImage(bitmap, 0, 0);
        const imageData = ctx.getImageData(0, 0, kfWidth, kfHeight);

        this.keyframeData.set(kf.frame_index, imageData.data);

        console.log(
          `  Keyframe ${kf.frame_index}: ${kfWidth}x${kfHeight}, tile_size=${kfTileSize}, births=${kf.birth_count}`,
        );
      } catch (error) {
        console.error(`Failed to load keyframe ${kf.frame_index}:`, error);
        throw error;
      }
    }

    console.log(`Loaded ${this.keyframeData.size} keyframe(s)`);
  }

  /**
   * Get pixel value from pre-decoded frame data.
   * @param frameData Raw pixel data
   * @param x X coordinate
   * @param y Y coordinate
   * @param width Frame width (optional, defaults to this.frameWidth)
   */
  private _getPixel(
    frameData: Uint8ClampedArray,
    x: number,
    y: number,
    width?: number,
  ): [number, number, number, number] {
    const w = width ?? this.frameWidth;
    const idx = (y * w + x) * 4;
    return [
      frameData[idx],
      frameData[idx + 1],
      frameData[idx + 2],
      frameData[idx + 3],
    ];
  }

  /**
   * Decode births directly into SoA buffers at specified slots.
   * Zero per-frame allocations - all data written to pre-allocated arrays.
   * @returns Number of births decoded, motion magnitudes stored in motionMags parameter
   */
  private _decodeBirthsToSoA(
    frameIndex: number,
    slots: number[],
    motionMags: Float32Array,
  ): number {
    const birthCount = this.metadata["4dgs"].birth_counts[frameIndex];
    if (birthCount === 0) return 0;

    // Determine frame data source
    const isKeyframe = this.keyframeIndices.has(frameIndex);
    let frameData: Uint8ClampedArray;
    let ts: number;
    let frameWidth: number;

    if (isKeyframe) {
      const kfData = this.keyframeData.get(frameIndex);
      if (!kfData) {
        console.warn(`Keyframe ${frameIndex} not loaded`);
        return 0;
      }
      frameData = kfData;
      ts = this.keyframeTileSizes.get(frameIndex) ?? this.tileSize;
      frameWidth = ts * this.metadata.grid[0];
    } else {
      let videoFrameIndex: number;
      if (this.frameIndexToVideoIndex) {
        const mappedIndex = this.frameIndexToVideoIndex.get(frameIndex);
        if (mappedIndex !== undefined) {
          videoFrameIndex = mappedIndex;
        } else {
          console.warn(
            `Frame ${frameIndex} not found in frame_map, using direct index`,
          );
          videoFrameIndex = frameIndex;
        }
      } else {
        videoFrameIndex = frameIndex;
      }

      if (videoFrameIndex >= this.framePixelData.length) {
        console.warn(
          `Video frame index ${videoFrameIndex} out of bounds (${this.framePixelData.length} frames)`,
        );
        return 0;
      }

      frameData = this.framePixelData[videoFrameIndex];
      ts = this.tileSize;
      frameWidth = this.frameWidth;
    }

    const layout = this.metadata.layout;
    const maxLifetime = this.totalFrames - frameIndex;

    // Pre-compute tile offsets (no allocation - stack variables)
    const meansLOffX = layout.means_l[0] * ts;
    const meansLOffY = layout.means_l[1] * ts;
    const meansUOffX = layout.means_u[0] * ts;
    const meansUOffY = layout.means_u[1] * ts;
    const quatsOffX = layout.quats[0] * ts;
    const quatsOffY = layout.quats[1] * ts;
    const motionLOffX = layout.motion_l[0] * ts;
    const motionLOffY = layout.motion_l[1] * ts;
    const scalesOffX = layout.scales[0] * ts;
    const scalesOffY = layout.scales[1] * ts;
    const sh0OffX = layout.sh0[0] * ts;
    const sh0OffY = layout.sh0[1] * ts;
    const motionUOffX = layout.motion_u[0] * ts;
    const motionUOffY = layout.motion_u[1] * ts;
    const metaOffX = layout.meta[0] * ts;
    const metaOffY = layout.meta[1] * ts;

    const count = Math.min(birthCount, slots.length);

    for (let i = 0; i < count; i++) {
      const slot = slots[i];
      const col = i % ts;
      const row = Math.floor(i / ts);

      // Direct pixel access (no tuple allocation)
      // meansL
      let idx = ((meansLOffY + row) * frameWidth + meansLOffX + col) * 4;
      const meansL0 = frameData[idx];
      const meansL1 = frameData[idx + 1];
      const meansL2 = frameData[idx + 2];

      // meansU
      idx = ((meansUOffY + row) * frameWidth + meansUOffX + col) * 4;
      const meansU0 = frameData[idx];
      const meansU1 = frameData[idx + 1];
      const meansU2 = frameData[idx + 2];

      // quats
      idx = ((quatsOffY + row) * frameWidth + quatsOffX + col) * 4;
      const quats0 = frameData[idx];
      const quats1 = frameData[idx + 1];
      const quats2 = frameData[idx + 2];
      const quats3 = frameData[idx + 3];

      // motionL
      idx = ((motionLOffY + row) * frameWidth + motionLOffX + col) * 4;
      const motionL0 = frameData[idx];
      const motionL1 = frameData[idx + 1];
      const motionL2 = frameData[idx + 2];

      // scales
      idx = ((scalesOffY + row) * frameWidth + scalesOffX + col) * 4;
      const scales0 = frameData[idx];
      const scales1 = frameData[idx + 1];
      const scales2 = frameData[idx + 2];
      const scales3 = frameData[idx + 3];

      // sh0
      idx = ((sh0OffY + row) * frameWidth + sh0OffX + col) * 4;
      const sh00 = frameData[idx];
      const sh01 = frameData[idx + 1];
      const sh02 = frameData[idx + 2];
      const sh03 = frameData[idx + 3];

      // motionU
      idx = ((motionUOffY + row) * frameWidth + motionUOffX + col) * 4;
      const motionU0 = frameData[idx];
      const motionU1 = frameData[idx + 1];
      const motionU2 = frameData[idx + 2];

      // meta
      idx = ((metaOffY + row) * frameWidth + metaOffX + col) * 4;
      const meta0 = frameData[idx];
      const meta1 = frameData[idx + 1];

      // Decode position (16-bit signed-log)
      const posU16X = meansL0 + meansU0 * 256;
      const posU16Y = meansL1 + meansU1 * 256;
      const posU16Z = meansL2 + meansU2 * 256;
      const posLogX = this.posMins[0] + (posU16X / 65535) * this.posRange[0];
      const posLogY = this.posMins[1] + (posU16Y / 65535) * this.posRange[1];
      const posLogZ = this.posMins[2] + (posU16Z / 65535) * this.posRange[2];
      const posX = Math.sign(posLogX) * (Math.exp(Math.abs(posLogX)) - 1);
      const posY = Math.sign(posLogY) * (Math.exp(Math.abs(posLogY)) - 1);
      const posZ = Math.sign(posLogZ) * (Math.exp(Math.abs(posLogZ)) - 1);

      // Decode motion (16-bit signed-log)
      const motU16X = motionL0 + motionU0 * 256;
      const motU16Y = motionL1 + motionU1 * 256;
      const motU16Z = motionL2 + motionU2 * 256;
      const motLogX =
        this.motionMins[0] + (motU16X / 65535) * this.motionRange[0];
      const motLogY =
        this.motionMins[1] + (motU16Y / 65535) * this.motionRange[1];
      const motLogZ =
        this.motionMins[2] + (motU16Z / 65535) * this.motionRange[2];
      const motX = Math.sign(motLogX) * (Math.exp(Math.abs(motLogX)) - 1);
      const motY = Math.sign(motLogY) * (Math.exp(Math.abs(motLogY)) - 1);
      const motZ = Math.sign(motLogZ) * (Math.exp(Math.abs(motLogZ)) - 1);

      // Decode lifetime
      const rawLifetime = meta0 + meta1 * 256;
      const lifetime = Math.min(rawLifetime, maxLifetime);

      // Write to SoA buffers at slot index
      const posBase = slot * 3;
      this.positions[posBase] = posX;
      this.positions[posBase + 1] = posY;
      this.positions[posBase + 2] = posZ;

      this.motions[posBase] = motX;
      this.motions[posBase + 1] = motY;
      this.motions[posBase + 2] = motZ;

      const attrBase = slot * 4;
      this.quatsEncoded[attrBase] = quats0;
      this.quatsEncoded[attrBase + 1] = quats1;
      this.quatsEncoded[attrBase + 2] = quats2;
      this.quatsEncoded[attrBase + 3] = quats3;

      this.scalesEncoded[attrBase] = scales0;
      this.scalesEncoded[attrBase + 1] = scales1;
      this.scalesEncoded[attrBase + 2] = scales2;
      this.scalesEncoded[attrBase + 3] = scales3;

      this.sh0Encoded[attrBase] = sh00;
      this.sh0Encoded[attrBase + 1] = sh01;
      this.sh0Encoded[attrBase + 2] = sh02;
      this.sh0Encoded[attrBase + 3] = sh03;

      this.remainingFrames[slot] = lifetime;
      this.justBorn[slot] = 1;

      // Compute motion magnitude for static classification
      motionMags[i] = Math.sqrt(motX * motX + motY * motY + motZ * motZ);

      // Debug: log first birth of first frame
      if (frameIndex === 0 && i === 0) {
        console.log(
          `First birth: pos=(${posX.toFixed(3)}, ${posY.toFixed(3)}, ${posZ.toFixed(3)})`,
        );
        console.log(
          `  lifetime=${lifetime}, quats=[${quats0},${quats1},${quats2},${quats3}], scales=[${scales0},${scales1},${scales2},${scales3}]`,
        );
      }
    }

    return count;
  }

  /**
   * Process a single frame: decode births, update positions, assemble SOG texture.
   * Returns { data: Uint8Array, count: number }
   */
  processFrame(frameIndex: number): { data: Uint8Array; count: number } {
    // Reset on backwards seek (includes loop wrap: e.g., 253 -> 2)
    if (frameIndex < this.currentFrameIndex) {
      this.reset();
    }

    // Process all frames from current to target (handles sequential playback)
    while (this.currentFrameIndex <= frameIndex) {
      this._processOneFrame(this.currentFrameIndex);
      this.currentFrameIndex++;
    }

    // Assemble SOG texture
    this._assembleSogTexture();

    return { data: this.sogTileData, count: this.activeCount };
  }

  /**
   * GPU-optimized frame processing: returns float positions and uint8 attributes.
   * Skips CPU-side signed-log encoding - positions uploaded as floats directly.
   */
  processFrameGPU(frameIndex: number): {
    positions: Float32Array;
    attributes: Uint8Array;
    count: number;
    textureSize: number;
  } {
    // Reset on backwards seek (includes loop wrap: e.g., 253 -> 2)
    if (frameIndex < this.currentFrameIndex) {
      this.reset();
    }

    // Process all frames from current to target (handles sequential playback)
    while (this.currentFrameIndex <= frameIndex) {
      this._processOneFrame(this.currentFrameIndex);
      this.currentFrameIndex++;
    }

    // Fill GPU buffers (no encoding, just copy floats and bytes)
    this._fillGPUBuffers();

    return {
      positions: this.gpuPositionBuffer,
      attributes: this.gpuAttributeBuffer,
      count: this.activeCount,
      textureSize: this.gpuTextureSize,
    };
  }

  /**
   * Fill GPU output buffers with float positions and uint8 attributes.
   * Much faster than _assembleSogTexture() - no position encoding.
   */
  private _fillGPUBuffers(): void {
    const texSize = this.gpuTextureSize;
    const planeSize = texSize * texSize * 4; // bytes per attribute plane

    // No buffer clear needed - shader uses splatCount guard to ignore stale data

    // Iterate active gaussians using SoA
    for (let idx = 0; idx < this.activeIndices.length; idx++) {
      const slot = this.activeIndices[idx];
      const srcPosBase = slot * 3;
      const srcAttrBase = slot * 4;

      // Position buffer: RGBA32F, 2D grid layout (alpha = 1.0)
      const row = Math.floor(idx / texSize);
      const col = idx % texSize;
      const posBase = (row * texSize + col) * 4;
      this.gpuPositionBuffer[posBase] = this.positions[srcPosBase];
      this.gpuPositionBuffer[posBase + 1] = this.positions[srcPosBase + 1];
      this.gpuPositionBuffer[posBase + 2] = this.positions[srcPosBase + 2];
      this.gpuPositionBuffer[posBase + 3] = 1.0;

      // Attribute buffer: 3 planes (quats, scales, sh0), same 2D layout as position
      const pixelOffset = (row * texSize + col) * 4;

      // Plane 0: quats
      const quatsBase = pixelOffset;
      this.gpuAttributeBuffer[quatsBase] = this.quatsEncoded[srcAttrBase];
      this.gpuAttributeBuffer[quatsBase + 1] =
        this.quatsEncoded[srcAttrBase + 1];
      this.gpuAttributeBuffer[quatsBase + 2] =
        this.quatsEncoded[srcAttrBase + 2];
      this.gpuAttributeBuffer[quatsBase + 3] =
        this.quatsEncoded[srcAttrBase + 3];

      // Plane 1: scales
      const scalesBase = planeSize + pixelOffset;
      this.gpuAttributeBuffer[scalesBase] = this.scalesEncoded[srcAttrBase];
      this.gpuAttributeBuffer[scalesBase + 1] =
        this.scalesEncoded[srcAttrBase + 1];
      this.gpuAttributeBuffer[scalesBase + 2] =
        this.scalesEncoded[srcAttrBase + 2];
      this.gpuAttributeBuffer[scalesBase + 3] =
        this.scalesEncoded[srcAttrBase + 3];

      // Plane 2: sh0
      const sh0Base = planeSize * 2 + pixelOffset;
      this.gpuAttributeBuffer[sh0Base] = this.sh0Encoded[srcAttrBase];
      this.gpuAttributeBuffer[sh0Base + 1] = this.sh0Encoded[srcAttrBase + 1];
      this.gpuAttributeBuffer[sh0Base + 2] = this.sh0Encoded[srcAttrBase + 2];
      this.gpuAttributeBuffer[sh0Base + 3] = this.sh0Encoded[srcAttrBase + 3];
    }
  }

  /**
   * Get GPU texture dimensions for creating THREE.DataTexture
   */
  getGPUTextureDimensions(): {
    positionSize: number; // Square texture side (positionSize x positionSize)
    attributeWidth: number; // Attribute texture width
    attributeHeight: number; // Attribute texture height (gpuTextureSize * 3)
  } {
    return {
      positionSize: this.gpuTextureSize,
      attributeWidth: this.gpuTextureSize,
      attributeHeight: this.gpuTextureSize * 3, // 3 planes: quats, scales, sh0
    };
  }

  /**
   * Initialize and fill static buffers after keyframe processing.
   * Call this after processFrameGPU(0) to finalize static data.
   */
  initStaticBuffers(): void {
    if (this.staticDataReady) return;
    if (this.staticCount === 0) {
      console.log("No static gaussians to buffer");
      this.staticDataReady = true;
      return;
    }

    // Calculate static texture size (power of 2, square)
    const staticTexSide = Math.ceil(Math.sqrt(this.staticCount));
    const staticTexSizePow = Math.ceil(Math.log2(Math.max(staticTexSide, 1)));
    this.staticTextureSize = 2 ** staticTexSizePow;

    // Allocate static buffers
    const texSize = this.staticTextureSize;
    this.staticPositionBuffer = new Float32Array(texSize * texSize * 4);
    this.staticAttributeBuffer = new Uint8Array(texSize * texSize * 3 * 4);

    // Fill static buffers from SoA (positions never change)
    const planeSize = texSize * texSize * 4;

    for (let idx = 0; idx < this.staticSlots.length; idx++) {
      const slot = this.staticSlots[idx];
      const srcPosBase = slot * 3;
      const srcAttrBase = slot * 4;

      const row = Math.floor(idx / texSize);
      const col = idx % texSize;

      // Position buffer
      const posBase = (row * texSize + col) * 4;
      this.staticPositionBuffer[posBase] = this.positions[srcPosBase];
      this.staticPositionBuffer[posBase + 1] = this.positions[srcPosBase + 1];
      this.staticPositionBuffer[posBase + 2] = this.positions[srcPosBase + 2];
      this.staticPositionBuffer[posBase + 3] = 1.0;

      // Attribute buffer (3 planes)
      const pixelOffset = (row * texSize + col) * 4;

      // Plane 0: quats
      const quatsBase = pixelOffset;
      this.staticAttributeBuffer[quatsBase] = this.quatsEncoded[srcAttrBase];
      this.staticAttributeBuffer[quatsBase + 1] =
        this.quatsEncoded[srcAttrBase + 1];
      this.staticAttributeBuffer[quatsBase + 2] =
        this.quatsEncoded[srcAttrBase + 2];
      this.staticAttributeBuffer[quatsBase + 3] =
        this.quatsEncoded[srcAttrBase + 3];

      // Plane 1: scales
      const scalesBase = planeSize + pixelOffset;
      this.staticAttributeBuffer[scalesBase] = this.scalesEncoded[srcAttrBase];
      this.staticAttributeBuffer[scalesBase + 1] =
        this.scalesEncoded[srcAttrBase + 1];
      this.staticAttributeBuffer[scalesBase + 2] =
        this.scalesEncoded[srcAttrBase + 2];
      this.staticAttributeBuffer[scalesBase + 3] =
        this.scalesEncoded[srcAttrBase + 3];

      // Plane 2: sh0
      const sh0Base = planeSize * 2 + pixelOffset;
      this.staticAttributeBuffer[sh0Base] = this.sh0Encoded[srcAttrBase];
      this.staticAttributeBuffer[sh0Base + 1] =
        this.sh0Encoded[srcAttrBase + 1];
      this.staticAttributeBuffer[sh0Base + 2] =
        this.sh0Encoded[srcAttrBase + 2];
      this.staticAttributeBuffer[sh0Base + 3] =
        this.sh0Encoded[srcAttrBase + 3];

      // Return slot to free pool (static data is copied, slot no longer needed)
      this.freeSlots.push(slot);
      this.remainingFrames[slot] = -1;
    }

    // Clear staticSlots - data is in static buffers now
    this.staticSlots = [];

    this.staticDataReady = true;

    console.log(
      `Static buffers initialized: ${this.staticCount} gaussians, ` +
        `${this.staticTextureSize}x${this.staticTextureSize} texture, ` +
        `${((this.staticPositionBuffer.byteLength + this.staticAttributeBuffer.byteLength) / 1024).toFixed(1)} KB`,
    );
  }

  /**
   * Get static gaussian data for one-time GPU upload.
   * Returns null if no static gaussians or not yet initialized.
   */
  getStaticData(): {
    positions: Float32Array;
    attributes: Uint8Array;
    count: number;
    textureSize: number;
  } | null {
    if (
      !this.staticDataReady ||
      !this.staticPositionBuffer ||
      !this.staticAttributeBuffer
    ) {
      return null;
    }
    return {
      positions: this.staticPositionBuffer,
      attributes: this.staticAttributeBuffer,
      count: this.staticCount,
      textureSize: this.staticTextureSize,
    };
  }

  /**
   * Get static texture dimensions for creating THREE.DataTexture.
   * Returns null if no static gaussians.
   */
  getStaticTextureDimensions(): {
    positionSize: number;
    attributeWidth: number;
    attributeHeight: number;
  } | null {
    if (this.staticCount === 0) return null;

    // Ensure static texture size is calculated
    if (this.staticTextureSize === 0) {
      const staticTexSide = Math.ceil(Math.sqrt(this.staticCount));
      const staticTexSizePow = Math.ceil(Math.log2(Math.max(staticTexSide, 1)));
      this.staticTextureSize = 2 ** staticTexSizePow;
    }

    return {
      positionSize: this.staticTextureSize,
      attributeWidth: this.staticTextureSize,
      attributeHeight: this.staticTextureSize * 3,
    };
  }

  /**
   * Get counts for static vs dynamic gaussians.
   */
  getGaussianCounts(): {
    static: number;
    dynamic: number;
    total: number;
  } {
    return {
      static: this.staticCount,
      dynamic: this.activeCount,
      total: this.staticCount + this.activeCount,
    };
  }

  reset(): void {
    // Clear dynamic gaussians only - static gaussians persist across loops
    this.remainingFrames.fill(-1); // Mark all slots as inactive
    this.activeIndices = [];
    this.freeSlots = [];
    for (let i = this.maxActive - 1; i >= 0; i--) {
      this.freeSlots.push(i);
    }
    this.activeCount = 0;
    this.currentFrameIndex = 0;
    // Note: staticSlots, staticCount, and staticDataReady are NOT reset
    // Static data is uploaded once and reused across animation loops
  }

  private _processOneFrame(frameIndex: number): void {
    const isKeyframe = this.keyframeIndices.has(frameIndex);
    const birthCount = this.metadata["4dgs"].birth_counts[frameIndex];

    // For keyframes on subsequent loops, use cached dynamic births (avoids re-decoding 205k births)
    if (isKeyframe && this.staticDataReady) {
      const cached = this.dynamicKeyframeBirthCache.get(frameIndex);
      if (cached) {
        // Copy cached template data to new slots
        for (let i = 0; i < cached.count; i++) {
          if (this.freeSlots.length === 0) {
            console.warn(
              `No free slots for cached birth at frame ${frameIndex}`,
            );
            break;
          }
          const slot = this.freeSlots.pop();
          if (slot === undefined) break;
          const srcPosBase = i * 3;
          const dstPosBase = slot * 3;
          const srcAttrBase = i * 4;
          const dstAttrBase = slot * 4;

          // Copy position (need fresh copy since it gets modified)
          this.positions[dstPosBase] = cached.positions[srcPosBase];
          this.positions[dstPosBase + 1] = cached.positions[srcPosBase + 1];
          this.positions[dstPosBase + 2] = cached.positions[srcPosBase + 2];

          // Copy motion (shared reference is fine, motion doesn't change)
          this.motions[dstPosBase] = cached.motions[srcPosBase];
          this.motions[dstPosBase + 1] = cached.motions[srcPosBase + 1];
          this.motions[dstPosBase + 2] = cached.motions[srcPosBase + 2];

          // Copy attributes
          this.quatsEncoded[dstAttrBase] = cached.quats[srcAttrBase];
          this.quatsEncoded[dstAttrBase + 1] = cached.quats[srcAttrBase + 1];
          this.quatsEncoded[dstAttrBase + 2] = cached.quats[srcAttrBase + 2];
          this.quatsEncoded[dstAttrBase + 3] = cached.quats[srcAttrBase + 3];

          this.scalesEncoded[dstAttrBase] = cached.scales[srcAttrBase];
          this.scalesEncoded[dstAttrBase + 1] = cached.scales[srcAttrBase + 1];
          this.scalesEncoded[dstAttrBase + 2] = cached.scales[srcAttrBase + 2];
          this.scalesEncoded[dstAttrBase + 3] = cached.scales[srcAttrBase + 3];

          this.sh0Encoded[dstAttrBase] = cached.sh0[srcAttrBase];
          this.sh0Encoded[dstAttrBase + 1] = cached.sh0[srcAttrBase + 1];
          this.sh0Encoded[dstAttrBase + 2] = cached.sh0[srcAttrBase + 2];
          this.sh0Encoded[dstAttrBase + 3] = cached.sh0[srcAttrBase + 3];

          this.remainingFrames[slot] = cached.lifetimes[i];
          this.justBorn[slot] = 1;

          this.activeIndices.push(slot);
          this.activeCount++;
        }
        console.log(
          `Keyframe ${frameIndex}: ${cached.count} dynamic births (from cache)`,
        );
        this._updateActiveGaussians();
        return;
      }
    }

    // First loop or non-keyframe: decode births directly into SoA
    // Allocate slots for all births
    const slotsNeeded = Math.min(birthCount, this.freeSlots.length);
    for (let i = 0; i < slotsNeeded; i++) {
      const slot = this.freeSlots.pop();
      if (slot === undefined) break;
      this.tempSlots[i] = slot;
    }

    // Decode births directly into SoA at allocated slots
    const decodedCount = this._decodeBirthsToSoA(
      frameIndex,
      this.tempSlots.slice(0, slotsNeeded),
      this.tempMotionMags,
    );

    // Track dynamic births for keyframe cache (first pass only)
    let dynamicCount = 0;

    // Classify births as static or dynamic based on motion magnitude
    for (let i = 0; i < decodedCount; i++) {
      const slot = this.tempSlots[i];
      const motionMag = this.tempMotionMags[i];

      if (isKeyframe && motionMag < DeltaSplatDecoder.STATIC_MOTION_THRESHOLD) {
        if (!this.staticDataReady) {
          // First loop: add to static pool
          this.staticSlots.push(slot);
          this.staticCount++;
        } else {
          // Subsequent loops: return slot to free pool (shouldn't happen with cache)
          this.freeSlots.push(slot);
          this.remainingFrames[slot] = -1;
        }
      } else {
        // Dynamic gaussian: add to active set
        this.activeIndices.push(slot);
        this.activeCount++;
        dynamicCount++;
      }
    }

    // Cache dynamic keyframe births for subsequent loops (first pass only)
    if (isKeyframe && !this.staticDataReady && dynamicCount > 0) {
      // Build cache arrays from the slots we just decoded
      const cachePositions = new Float32Array(dynamicCount * 3);
      const cacheMotions = new Float32Array(dynamicCount * 3);
      const cacheQuats = new Uint8Array(dynamicCount * 4);
      const cacheScales = new Uint8Array(dynamicCount * 4);
      const cacheSh0 = new Uint8Array(dynamicCount * 4);
      const cacheLifetimes = new Int16Array(dynamicCount);

      let cacheIdx = 0;
      for (let i = 0; i < decodedCount; i++) {
        const motionMag = this.tempMotionMags[i];
        if (motionMag >= DeltaSplatDecoder.STATIC_MOTION_THRESHOLD) {
          const slot = this.tempSlots[i];
          const srcPosBase = slot * 3;
          const srcAttrBase = slot * 4;
          const dstPosBase = cacheIdx * 3;
          const dstAttrBase = cacheIdx * 4;

          cachePositions[dstPosBase] = this.positions[srcPosBase];
          cachePositions[dstPosBase + 1] = this.positions[srcPosBase + 1];
          cachePositions[dstPosBase + 2] = this.positions[srcPosBase + 2];

          cacheMotions[dstPosBase] = this.motions[srcPosBase];
          cacheMotions[dstPosBase + 1] = this.motions[srcPosBase + 1];
          cacheMotions[dstPosBase + 2] = this.motions[srcPosBase + 2];

          cacheQuats[dstAttrBase] = this.quatsEncoded[srcAttrBase];
          cacheQuats[dstAttrBase + 1] = this.quatsEncoded[srcAttrBase + 1];
          cacheQuats[dstAttrBase + 2] = this.quatsEncoded[srcAttrBase + 2];
          cacheQuats[dstAttrBase + 3] = this.quatsEncoded[srcAttrBase + 3];

          cacheScales[dstAttrBase] = this.scalesEncoded[srcAttrBase];
          cacheScales[dstAttrBase + 1] = this.scalesEncoded[srcAttrBase + 1];
          cacheScales[dstAttrBase + 2] = this.scalesEncoded[srcAttrBase + 2];
          cacheScales[dstAttrBase + 3] = this.scalesEncoded[srcAttrBase + 3];

          cacheSh0[dstAttrBase] = this.sh0Encoded[srcAttrBase];
          cacheSh0[dstAttrBase + 1] = this.sh0Encoded[srcAttrBase + 1];
          cacheSh0[dstAttrBase + 2] = this.sh0Encoded[srcAttrBase + 2];
          cacheSh0[dstAttrBase + 3] = this.sh0Encoded[srcAttrBase + 3];

          cacheLifetimes[cacheIdx] = this.remainingFrames[slot];
          cacheIdx++;
        }
      }

      this.dynamicKeyframeBirthCache.set(frameIndex, {
        count: dynamicCount,
        lifetimes: cacheLifetimes,
        positions: cachePositions,
        motions: cacheMotions,
        quats: cacheQuats,
        scales: cacheScales,
        sh0: cacheSh0,
      });
    }

    // Log keyframe static/dynamic split
    if (isKeyframe && decodedCount > 0) {
      console.log(
        `Keyframe ${frameIndex}: ${decodedCount} births → ` +
          `${this.staticCount} static (total), ${this.activeCount} dynamic (active)`,
      );
    }

    this._updateActiveGaussians();
  }

  // Update positions and decrement lifetimes for active gaussians
  private _updateActiveGaussians(): void {
    let writeIdx = 0;
    for (let readIdx = 0; readIdx < this.activeIndices.length; readIdx++) {
      const slot = this.activeIndices[readIdx];

      // Free expired gaussians FIRST (check-then-decrement to match encoder's lifetime)
      // With lifetime=3: visible at frames B, B+1, B+2, then removed at B+3
      if (this.remainingFrames[slot] <= 0) {
        this.remainingFrames[slot] = -1; // Mark inactive
        this.freeSlots.push(slot);
        this.activeCount--;
        // Don't copy to writeIdx (effectively removes from activeIndices)
        continue;
      }

      // Skip motion update for newly-born gaussians (they just spawned)
      if (this.justBorn[slot]) {
        this.justBorn[slot] = 0;
      } else {
        // Position interpolation using SoA
        const posBase = slot * 3;
        this.positions[posBase] += this.motions[posBase];
        this.positions[posBase + 1] += this.motions[posBase + 1];
        this.positions[posBase + 2] += this.motions[posBase + 2];
      }

      // Decrement lifetime for next frame
      this.remainingFrames[slot]--;

      // Keep this slot in activeIndices (compacting in place)
      this.activeIndices[writeIdx++] = slot;
    }
    // Truncate activeIndices to remove expired entries
    this.activeIndices.length = writeIdx;
  }

  private _assembleSogTexture(): void {
    const ts = this.sogTileSize;
    const sogWidth = this.sogWidth;

    // Clear output
    this.sogTileData.fill(0);
    // Set alpha to 255 everywhere
    for (let i = 3; i < this.sogTileData.length; i += 4) {
      this.sogTileData[i] = 255;
    }

    // Iterate only active slots (much faster than sparse array iteration)
    for (let idx = 0; idx < this.activeIndices.length; idx++) {
      const slot = this.activeIndices[idx];
      const srcPosBase = slot * 3;
      const srcAttrBase = slot * 4;

      const row = Math.floor(idx / ts);
      const col = idx % ts;

      // Re-encode position from SoA
      this._encodePositionFromSoA(slot, this.posEncodeBuf);

      // means_l: tile (0,0)
      const meansLBase = (row * sogWidth + col) * 4;
      this.sogTileData[meansLBase] = this.posEncodeBuf[0];
      this.sogTileData[meansLBase + 1] = this.posEncodeBuf[1];
      this.sogTileData[meansLBase + 2] = this.posEncodeBuf[2];
      this.sogTileData[meansLBase + 3] = 255;

      // means_u: tile (1,0)
      const meansUBase = (row * sogWidth + ts + col) * 4;
      this.sogTileData[meansUBase] = this.posEncodeBuf[3];
      this.sogTileData[meansUBase + 1] = this.posEncodeBuf[4];
      this.sogTileData[meansUBase + 2] = this.posEncodeBuf[5];
      this.sogTileData[meansUBase + 3] = 255;

      // quats: tile (2,0)
      const quatsBase = (row * sogWidth + ts * 2 + col) * 4;
      this.sogTileData[quatsBase] = this.quatsEncoded[srcAttrBase];
      this.sogTileData[quatsBase + 1] = this.quatsEncoded[srcAttrBase + 1];
      this.sogTileData[quatsBase + 2] = this.quatsEncoded[srcAttrBase + 2];
      this.sogTileData[quatsBase + 3] = this.quatsEncoded[srcAttrBase + 3];

      // scales: tile (0,1)
      const scalesBase = ((ts + row) * sogWidth + col) * 4;
      this.sogTileData[scalesBase] = this.scalesEncoded[srcAttrBase];
      this.sogTileData[scalesBase + 1] = this.scalesEncoded[srcAttrBase + 1];
      this.sogTileData[scalesBase + 2] = this.scalesEncoded[srcAttrBase + 2];
      this.sogTileData[scalesBase + 3] = this.scalesEncoded[srcAttrBase + 3];

      // sh0: tile (1,1)
      const sh0Base = ((ts + row) * sogWidth + ts + col) * 4;
      this.sogTileData[sh0Base] = this.sh0Encoded[srcAttrBase];
      this.sogTileData[sh0Base + 1] = this.sh0Encoded[srcAttrBase + 1];
      this.sogTileData[sh0Base + 2] = this.sh0Encoded[srcAttrBase + 2];
      this.sogTileData[sh0Base + 3] = this.sh0Encoded[srcAttrBase + 3];

      // t_scale: tile (2,1) - leave as zeros (not needed for delta, all active)
    }
  }

  private _encodePositionFromSoA(slot: number, out: Uint8Array): void {
    const base = slot * 3;
    const posX = this.positions[base];
    const posY = this.positions[base + 1];
    const posZ = this.positions[base + 2];

    // Apply signed-log transform
    const logX = Math.sign(posX) * Math.log1p(Math.abs(posX));
    const logY = Math.sign(posY) * Math.log1p(Math.abs(posY));
    const logZ = Math.sign(posZ) * Math.log1p(Math.abs(posZ));

    // Normalize to [0, 65535]
    const normX = (logX - this.posMins[0]) / this.posRange[0];
    const normY = (logY - this.posMins[1]) / this.posRange[1];
    const normZ = (logZ - this.posMins[2]) / this.posRange[2];

    const u16X = Math.round(Math.max(0, Math.min(65535, normX * 65535)));
    const u16Y = Math.round(Math.max(0, Math.min(65535, normY * 65535)));
    const u16Z = Math.round(Math.max(0, Math.min(65535, normZ * 65535)));

    out[0] = u16X & 0xff;
    out[1] = u16Y & 0xff;
    out[2] = u16Z & 0xff;
    out[3] = u16X >> 8;
    out[4] = u16Y >> 8;
    out[5] = u16Z >> 8;
  }

  // Legacy method - kept for compatibility but unused in SoA path
  private _encodePositionInPlace(pos: Float32Array, out: Uint8Array): void {
    // Apply signed-log transform
    const logX = Math.sign(pos[0]) * Math.log1p(Math.abs(pos[0]));
    const logY = Math.sign(pos[1]) * Math.log1p(Math.abs(pos[1]));
    const logZ = Math.sign(pos[2]) * Math.log1p(Math.abs(pos[2]));

    // Normalize to [0, 65535]
    const normX = (logX - this.posMins[0]) / this.posRange[0];
    const normY = (logY - this.posMins[1]) / this.posRange[1];
    const normZ = (logZ - this.posMins[2]) / this.posRange[2];

    const u16X = Math.round(Math.max(0, Math.min(65535, normX * 65535)));
    const u16Y = Math.round(Math.max(0, Math.min(65535, normY * 65535)));
    const u16Z = Math.round(Math.max(0, Math.min(65535, normZ * 65535)));

    out[0] = u16X & 0xff;
    out[1] = u16Y & 0xff;
    out[2] = u16Z & 0xff;
    out[3] = u16X >> 8;
    out[4] = u16Y >> 8;
    out[5] = u16Z >> 8;
  }

  getTotalFrames(): number {
    return this.totalFrames;
  }

  hasKeyframes(): boolean {
    return this.keyframeIndices.size > 0;
  }

  /**
   * Set keyframe pixel data directly (for loading from File objects)
   */
  setKeyframeData(frameIndex: number, data: Uint8ClampedArray): void {
    if (!this.keyframeIndices.has(frameIndex)) {
      console.warn(
        `Frame ${frameIndex} is not registered as a keyframe, registering now`,
      );
      this.keyframeIndices.add(frameIndex);
    }
    this.keyframeData.set(frameIndex, data);
  }

  getSOGDimensions(): { width: number; height: number } {
    return { width: this.sogWidth, height: this.sogHeight };
  }

  /**
   * Load a .4dgs bundle file and extract its contents.
   * The bundle is a ZIP file containing video.webp, metadata.json, and keyframe files.
   */
  static async loadFromBundle(bundleBlob: Blob): Promise<{
    videoBlob: Blob;
    metadata: Delta4DGSMetadata;
    keyframeBlobs: Map<number, Blob>;
  }> {
    const buffer = await bundleBlob.arrayBuffer();
    const unzipped = unzipSync(new Uint8Array(buffer));

    // Extract metadata
    const metadataBytes = unzipped["metadata.json"];
    if (!metadataBytes) {
      throw new Error("Bundle missing metadata.json");
    }
    const metadataJson = new TextDecoder().decode(metadataBytes);
    const metadata = JSON.parse(metadataJson) as Delta4DGSMetadata;

    // Extract video
    const videoBytes = unzipped["video.webp"];
    if (!videoBytes) {
      throw new Error("Bundle missing video.webp");
    }
    const videoBlob = new Blob([videoBytes], { type: "image/webp" });

    // Extract keyframes
    const keyframeBlobs = new Map<number, Blob>();
    if (metadata.keyframes) {
      for (const kf of metadata.keyframes) {
        const filename = kf.path;
        const kfBytes = unzipped[filename];
        if (kfBytes) {
          keyframeBlobs.set(
            kf.frame_index,
            new Blob([kfBytes], { type: "image/webp" }),
          );
        } else {
          console.warn(`Bundle missing keyframe: ${filename}`);
        }
      }
    }

    console.log(
      `Loaded bundle: ${metadata.video.frames} video frames, ${keyframeBlobs.size} keyframes`,
    );
    return { videoBlob, metadata, keyframeBlobs };
  }
}
