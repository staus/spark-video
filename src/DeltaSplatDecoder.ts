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

interface ActiveGaussian {
  quatsEncoded: Uint8Array;
  scalesEncoded: Uint8Array;
  sh0Encoded: Uint8Array;
  position: Float32Array;
  motion: Float32Array;
  remainingFrames: number;
  justBorn: boolean;
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
  private activeGaussians: (ActiveGaussian | null)[];
  private activeIndices: number[]; // Compact list of occupied slot indices
  private freeSlots: number[];
  private activeCount: number;
  currentFrameIndex: number;

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
    this.activeGaussians = new Array(maxActive).fill(null);
    this.activeIndices = [];
    this.freeSlots = [];
    for (let i = maxActive - 1; i >= 0; i--) {
      this.freeSlots.push(i);
    }
    this.activeCount = 0;
    this.currentFrameIndex = 0;

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
   * Process a single frame: decode births, update positions, assemble SOG texture.
   * Returns { data: Uint8Array, count: number }
   */
  processFrame(frameIndex: number): { data: Uint8Array; count: number } {
    // Reset if we're starting over
    if (frameIndex === 0 && this.currentFrameIndex !== 0) {
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
    // Reset if we're starting over
    if (frameIndex === 0 && this.currentFrameIndex !== 0) {
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

    // Iterate active gaussians
    for (let idx = 0; idx < this.activeIndices.length; idx++) {
      const g = this.activeGaussians[this.activeIndices[idx]];
      if (!g) continue;

      // Position buffer: RGBA32F, 2D grid layout (alpha = 1.0)
      const row = Math.floor(idx / texSize);
      const col = idx % texSize;
      const posBase = (row * texSize + col) * 4;
      this.gpuPositionBuffer[posBase] = g.position[0];
      this.gpuPositionBuffer[posBase + 1] = g.position[1];
      this.gpuPositionBuffer[posBase + 2] = g.position[2];
      this.gpuPositionBuffer[posBase + 3] = 1.0;

      // Attribute buffer: 3 planes (quats, scales, sh0), same 2D layout as position
      const pixelOffset = (row * texSize + col) * 4;

      // Plane 0: quats
      const quatsBase = pixelOffset;
      this.gpuAttributeBuffer[quatsBase] = g.quatsEncoded[0];
      this.gpuAttributeBuffer[quatsBase + 1] = g.quatsEncoded[1];
      this.gpuAttributeBuffer[quatsBase + 2] = g.quatsEncoded[2];
      this.gpuAttributeBuffer[quatsBase + 3] = g.quatsEncoded[3];

      // Plane 1: scales
      const scalesBase = planeSize + pixelOffset;
      this.gpuAttributeBuffer[scalesBase] = g.scalesEncoded[0];
      this.gpuAttributeBuffer[scalesBase + 1] = g.scalesEncoded[1];
      this.gpuAttributeBuffer[scalesBase + 2] = g.scalesEncoded[2];
      this.gpuAttributeBuffer[scalesBase + 3] = g.scalesEncoded[3];

      // Plane 2: sh0
      const sh0Base = planeSize * 2 + pixelOffset;
      this.gpuAttributeBuffer[sh0Base] = g.sh0Encoded[0];
      this.gpuAttributeBuffer[sh0Base + 1] = g.sh0Encoded[1];
      this.gpuAttributeBuffer[sh0Base + 2] = g.sh0Encoded[2];
      this.gpuAttributeBuffer[sh0Base + 3] = g.sh0Encoded[3];
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

  reset(): void {
    this.activeGaussians.fill(null);
    this.activeIndices = [];
    this.freeSlots = [];
    for (let i = this.maxActive - 1; i >= 0; i--) {
      this.freeSlots.push(i);
    }
    this.activeCount = 0;
    this.currentFrameIndex = 0;
  }

  private _processOneFrame(frameIndex: number): void {
    // 1. Decode births from delta frame
    const births = this._decodeBirths(frameIndex);

    // 2. Add births to active set
    for (const birth of births) {
      if (this.freeSlots.length === 0) {
        console.warn(`No free slots for birth at frame ${frameIndex}`);
        break;
      }
      const slot = this.freeSlots.pop();
      if (slot === undefined) break;
      this.activeGaussians[slot] = birth;
      this.activeIndices.push(slot);
      this.activeCount++;
    }

    // 3. Update positions and decrement lifetimes (iterate only active slots)
    let writeIdx = 0;
    for (let readIdx = 0; readIdx < this.activeIndices.length; readIdx++) {
      const slot = this.activeIndices[readIdx];
      const g = this.activeGaussians[slot];
      if (!g) continue;

      // Skip motion update for newly-born gaussians (they just spawned)
      if (g.justBorn) {
        g.justBorn = false;
      } else {
        // Position interpolation
        g.position[0] += g.motion[0];
        g.position[1] += g.motion[1];
        g.position[2] += g.motion[2];
      }

      // Decrement lifetime
      g.remainingFrames--;

      // Free expired gaussians
      if (g.remainingFrames < 0) {
        this.activeGaussians[slot] = null;
        this.freeSlots.push(slot);
        this.activeCount--;
        // Don't copy to writeIdx (effectively removes from activeIndices)
      } else {
        // Keep this slot in activeIndices (compacting in place)
        this.activeIndices[writeIdx++] = slot;
      }
    }
    // Truncate activeIndices to remove expired entries
    this.activeIndices.length = writeIdx;
  }

  private _decodeBirths(frameIndex: number): ActiveGaussian[] {
    const birthCount = this.metadata["4dgs"].birth_counts[frameIndex];
    if (birthCount === 0) return [];

    // Determine if this is a keyframe or video frame
    const isKeyframe = this.keyframeIndices.has(frameIndex);
    let frameData: Uint8ClampedArray;
    let ts: number;
    let frameWidth: number;

    if (isKeyframe) {
      // Use keyframe data with its own tile size
      const kfData = this.keyframeData.get(frameIndex);
      if (!kfData) {
        console.warn(`Keyframe ${frameIndex} not loaded`);
        return [];
      }
      frameData = kfData;
      ts = this.keyframeTileSizes.get(frameIndex) ?? this.tileSize;
      frameWidth = ts * this.metadata.grid[0];
    } else {
      // Map original frame index to video frame index (O(1) lookup)
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
        // No keyframes, direct mapping
        videoFrameIndex = frameIndex;
      }

      if (videoFrameIndex >= this.framePixelData.length) {
        console.warn(
          `Video frame index ${videoFrameIndex} out of bounds (${this.framePixelData.length} frames)`,
        );
        return [];
      }

      frameData = this.framePixelData[videoFrameIndex];
      ts = this.tileSize;
      frameWidth = this.frameWidth;
    }

    const births: ActiveGaussian[] = [];
    const layout = this.metadata.layout;

    // Tile offsets (layout values are [col, row] which map to [x/ts, y/ts])
    const meansLOff = [layout.means_l[0] * ts, layout.means_l[1] * ts];
    const meansUOff = [layout.means_u[0] * ts, layout.means_u[1] * ts];
    const quatsOff = [layout.quats[0] * ts, layout.quats[1] * ts];
    const motionLOff = [layout.motion_l[0] * ts, layout.motion_l[1] * ts];
    const scalesOff = [layout.scales[0] * ts, layout.scales[1] * ts];
    const sh0Off = [layout.sh0[0] * ts, layout.sh0[1] * ts];
    const motionUOff = [layout.motion_u[0] * ts, layout.motion_u[1] * ts];
    const metaOff = [layout.meta[0] * ts, layout.meta[1] * ts];

    for (let i = 0; i < birthCount; i++) {
      const col = i % ts;
      const row = Math.floor(i / ts);

      // Read pixels from each tile
      const meansL = this._getPixel(
        frameData,
        meansLOff[0] + col,
        meansLOff[1] + row,
        frameWidth,
      );
      const meansU = this._getPixel(
        frameData,
        meansUOff[0] + col,
        meansUOff[1] + row,
        frameWidth,
      );
      const quats = this._getPixel(
        frameData,
        quatsOff[0] + col,
        quatsOff[1] + row,
        frameWidth,
      );
      const motionL = this._getPixel(
        frameData,
        motionLOff[0] + col,
        motionLOff[1] + row,
        frameWidth,
      );
      const scales = this._getPixel(
        frameData,
        scalesOff[0] + col,
        scalesOff[1] + row,
        frameWidth,
      );
      const sh0 = this._getPixel(
        frameData,
        sh0Off[0] + col,
        sh0Off[1] + row,
        frameWidth,
      );
      const motionU = this._getPixel(
        frameData,
        motionUOff[0] + col,
        motionUOff[1] + row,
        frameWidth,
      );
      const meta = this._getPixel(
        frameData,
        metaOff[0] + col,
        metaOff[1] + row,
        frameWidth,
      );

      // Decode position (16-bit signed-log)
      const posU16X = meansL[0] + meansU[0] * 256;
      const posU16Y = meansL[1] + meansU[1] * 256;
      const posU16Z = meansL[2] + meansU[2] * 256;
      const posLogX = this.posMins[0] + (posU16X / 65535) * this.posRange[0];
      const posLogY = this.posMins[1] + (posU16Y / 65535) * this.posRange[1];
      const posLogZ = this.posMins[2] + (posU16Z / 65535) * this.posRange[2];
      // Apply inverse signed-log: sign(x) * (exp(|x|) - 1)
      const position = new Float32Array([
        Math.sign(posLogX) * (Math.exp(Math.abs(posLogX)) - 1),
        Math.sign(posLogY) * (Math.exp(Math.abs(posLogY)) - 1),
        Math.sign(posLogZ) * (Math.exp(Math.abs(posLogZ)) - 1),
      ]);

      // Decode motion (16-bit signed-log)
      const motU16X = motionL[0] + motionU[0] * 256;
      const motU16Y = motionL[1] + motionU[1] * 256;
      const motU16Z = motionL[2] + motionU[2] * 256;
      const motLogX =
        this.motionMins[0] + (motU16X / 65535) * this.motionRange[0];
      const motLogY =
        this.motionMins[1] + (motU16Y / 65535) * this.motionRange[1];
      const motLogZ =
        this.motionMins[2] + (motU16Z / 65535) * this.motionRange[2];
      const motion = new Float32Array([
        Math.sign(motLogX) * (Math.exp(Math.abs(motLogX)) - 1),
        Math.sign(motLogY) * (Math.exp(Math.abs(motLogY)) - 1),
        Math.sign(motLogZ) * (Math.exp(Math.abs(motLogZ)) - 1),
      ]);

      // Decode lifetime (16-bit) and clamp to remaining frames
      const rawLifetime = meta[0] + meta[1] * 256;
      const maxLifetime = this.totalFrames - frameIndex;
      const lifetime = Math.min(rawLifetime, maxLifetime);

      // Debug: log first birth of first frame
      if (frameIndex === 0 && i === 0) {
        console.log(
          `First birth: pos=(${position[0].toFixed(3)}, ${position[1].toFixed(3)}, ${position[2].toFixed(3)})`,
        );
        console.log(
          `  lifetime=${lifetime}, quats=[${quats.join(",")}], scales=[${scales.join(",")}]`,
        );
      }

      // Store pre-encoded tiles (copied directly to SOG each frame)
      births.push({
        quatsEncoded: new Uint8Array([quats[0], quats[1], quats[2], quats[3]]),
        scalesEncoded: new Uint8Array([
          scales[0],
          scales[1],
          scales[2],
          scales[3],
        ]),
        sh0Encoded: new Uint8Array([sh0[0], sh0[1], sh0[2], sh0[3]]),
        position,
        motion,
        remainingFrames: lifetime,
        justBorn: true, // Flag to skip first motion update
      });
    }

    return births;
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
      const g = this.activeGaussians[this.activeIndices[idx]];
      if (!g) continue;

      const row = Math.floor(idx / ts);
      const col = idx % ts;

      // Re-encode position
      this._encodePositionInPlace(g.position, this.posEncodeBuf);

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
      this.sogTileData.set(g.quatsEncoded, quatsBase);

      // scales: tile (0,1)
      const scalesBase = ((ts + row) * sogWidth + col) * 4;
      this.sogTileData.set(g.scalesEncoded, scalesBase);

      // sh0: tile (1,1)
      const sh0Base = ((ts + row) * sogWidth + ts + col) * 4;
      this.sogTileData.set(g.sh0Encoded, sh0Base);

      // t_scale: tile (2,1) - leave as zeros (not needed for delta, all active)
    }
  }

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
