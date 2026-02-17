/**
 * Delta-encoded 4DGS video decoder.
 *
 * Each frame contains only newly-born gaussians with motion vectors and lifetimes.
 * The decoder accumulates births, interpolates positions, and reconstructs full SOG textures.
 *
 * Uses canvas 2D with careful color space handling to read raw pixel data.
 */

// WebCodecs ImageDecoder API types (not yet in lib.dom.d.ts)
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
 * Metadata format for delta-encoded 4DGS video files (JSON sidecar)
 */
export interface Delta4DGSMetadata {
  tile_size: number;
  grid: [number, number]; // [cols, rows] for delta frame layout
  layout: Record<string, [number, number]>; // tile name -> [col, row]
  video: {
    frames: number;
    fps: number;
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
  };
  encoding: "delta";
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
  private freeSlots: number[];
  private activeCount: number;
  currentFrameIndex: number;

  // Store raw pixel data for each frame (pre-decoded at load time)
  private framePixelData: Uint8ClampedArray[];
  private frameWidth: number;
  private frameHeight: number;

  // SOG output: 3x2 tiles (standard SOG layout for GPU decode)
  private sogWidth: number;
  private sogHeight: number;
  private sogTileData: Uint8Array;

  // Pre-allocated position encoding buffer
  private posEncodeBuf: Uint8Array;

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

    console.log(
      `DeltaSplatDecoder initialized: maxActive=${maxActive}, deltaTile=${this.tileSize}, sogTile=${this.sogTileSize}`,
    );
  }

  async loadDeltaFrames(webpBlob: Blob): Promise<void> {
    const arrayBuffer = await webpBlob.arrayBuffer();
    const decoder = new ImageDecoder({ data: arrayBuffer, type: "image/webp" });
    await decoder.tracks.ready;

    const frameCount =
      decoder.tracks.selectedTrack?.frameCount || this.metadata.video.frames;
    this.framePixelData = [];

    // Create canvas for pixel extraction
    const canvas = new OffscreenCanvas(this.frameWidth, this.frameHeight);
    const ctx = canvas.getContext("2d", {
      willReadFrequently: true,
      colorSpace: "srgb",
    });
    if (!ctx) {
      throw new Error("Failed to get 2D context");
    }

    console.log(`Decoding ${frameCount} delta frames...`);
    console.log(`Frame dimensions: ${this.frameWidth}x${this.frameHeight}`);

    for (let i = 0; i < frameCount; i++) {
      const result = await decoder.decode({ frameIndex: i });
      const frame = result.image;

      // Create ImageBitmap without color conversion
      const bitmap = await createImageBitmap(frame, {
        premultiplyAlpha: "none",
        colorSpaceConversion: "none",
      });

      // Draw to canvas and extract pixels
      ctx.drawImage(bitmap, 0, 0);
      const imageData = ctx.getImageData(
        0,
        0,
        this.frameWidth,
        this.frameHeight,
        {
          colorSpace: "srgb",
        },
      );

      this.framePixelData.push(imageData.data);

      bitmap.close();
      frame.close();

      // Debug: log first frame's first few pixels
      if (i === 0) {
        const d = imageData.data;
        console.log(
          `Frame 0 first pixel: R=${d[0]} G=${d[1]} B=${d[2]} A=${d[3]}`,
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
   * Get pixel value from pre-decoded frame data.
   */
  private _getPixel(
    frameData: Uint8ClampedArray,
    x: number,
    y: number,
  ): [number, number, number, number] {
    const idx = (y * this.frameWidth + x) * 4;
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

  reset(): void {
    this.activeGaussians.fill(null);
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
      this.activeCount++;
    }

    // 3. Update positions and decrement lifetimes
    for (let i = 0; i < this.activeGaussians.length; i++) {
      const g = this.activeGaussians[i];
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
      if (g.remainingFrames <= 0) {
        this.activeGaussians[i] = null;
        this.freeSlots.push(i);
        this.activeCount--;
      }
    }

    // Debug log every 10 frames
    if (frameIndex % 10 === 0) {
      console.log(
        `Frame ${frameIndex}: ${births.length} births, ${this.activeCount} active`,
      );
    }
  }

  private _decodeBirths(frameIndex: number): ActiveGaussian[] {
    const birthCount = this.metadata["4dgs"].birth_counts[frameIndex];
    if (birthCount === 0) return [];

    const frameData = this.framePixelData[frameIndex];
    const births: ActiveGaussian[] = [];
    const layout = this.metadata.layout;
    const ts = this.tileSize;

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
      );
      const meansU = this._getPixel(
        frameData,
        meansUOff[0] + col,
        meansUOff[1] + row,
      );
      const quats = this._getPixel(
        frameData,
        quatsOff[0] + col,
        quatsOff[1] + row,
      );
      const motionL = this._getPixel(
        frameData,
        motionLOff[0] + col,
        motionLOff[1] + row,
      );
      const scales = this._getPixel(
        frameData,
        scalesOff[0] + col,
        scalesOff[1] + row,
      );
      const sh0 = this._getPixel(frameData, sh0Off[0] + col, sh0Off[1] + row);
      const motionU = this._getPixel(
        frameData,
        motionUOff[0] + col,
        motionUOff[1] + row,
      );
      const meta = this._getPixel(
        frameData,
        metaOff[0] + col,
        metaOff[1] + row,
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
      const maxLifetime = this.framePixelData.length - frameIndex;
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

    let idx = 0;
    for (const g of this.activeGaussians) {
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

      idx++;
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
    return this.framePixelData.length;
  }

  getSOGDimensions(): { width: number; height: number } {
    return { width: this.sogWidth, height: this.sogHeight };
  }
}
