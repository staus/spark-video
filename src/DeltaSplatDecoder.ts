/**
 * Delta-encoded 4DGS video decoder.
 *
 * Each frame contains only newly-born gaussians with motion vectors and lifetimes.
 * The decoder accumulates births, interpolates positions, and reconstructs full SOG textures.
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
  private tileSize: number;

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

  private deltaFrames: ImageBitmap[];
  private deltaCanvas: OffscreenCanvas;
  private deltaCtx: OffscreenCanvasRenderingContext2D;

  // SOG output: 3x2 tiles (standard SOG layout for GPU decode)
  private sogWidth: number;
  private sogHeight: number;
  private sogTileData: Uint8Array;

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

    // Delta frames (ImageBitmaps)
    this.deltaFrames = [];

    // Canvas for reading delta frame pixels
    this.deltaCanvas = new OffscreenCanvas(
      this.tileSize * metadata.grid[0],
      this.tileSize * metadata.grid[1],
    );
    const ctx = this.deltaCanvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) {
      throw new Error("Failed to get 2D context for delta canvas");
    }
    this.deltaCtx = ctx;

    // SOG output: 3x2 tiles
    const sogWidth = this.tileSize * 3;
    const sogHeight = this.tileSize * 2;
    this.sogWidth = sogWidth;
    this.sogHeight = sogHeight;
    this.sogTileData = new Uint8Array(sogWidth * sogHeight * 4);

    console.log(
      `DeltaSplatDecoder initialized: maxActive=${maxActive}, tileSize=${this.tileSize}`,
    );
  }

  async loadDeltaFrames(webpBlob: Blob): Promise<void> {
    const arrayBuffer = await webpBlob.arrayBuffer();
    const decoder = new ImageDecoder({ data: arrayBuffer, type: "image/webp" });
    await decoder.tracks.ready;

    const frameCount =
      decoder.tracks.selectedTrack?.frameCount || this.metadata.video.frames;
    this.deltaFrames = [];

    console.log(`Decoding ${frameCount} delta frames...`);
    for (let i = 0; i < frameCount; i++) {
      const result = await decoder.decode({ frameIndex: i });
      const bitmap = await createImageBitmap(result.image, {
        premultiplyAlpha: "none",
        colorSpaceConversion: "none",
      });
      this.deltaFrames.push(bitmap);
      result.image.close();
    }
    decoder.close();
    console.log(`Decoded ${this.deltaFrames.length} delta frames`);
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
  }

  private _decodeBirths(frameIndex: number): ActiveGaussian[] {
    const birthCount = this.metadata["4dgs"].birth_counts[frameIndex];
    if (birthCount === 0) return [];

    // Draw delta frame to canvas for pixel access
    const frame = this.deltaFrames[frameIndex];
    this.deltaCtx.drawImage(frame, 0, 0);

    const births: ActiveGaussian[] = [];
    const layout = this.metadata.layout;
    const ts = this.tileSize;

    // Read tile data
    const readTile = (col: number, row: number): Uint8ClampedArray => {
      const x = col * ts;
      const y = row * ts;
      return this.deltaCtx.getImageData(x, y, ts, ts).data;
    };

    const meansL = readTile(layout.means_l[0], layout.means_l[1]);
    const meansU = readTile(layout.means_u[0], layout.means_u[1]);
    const quats = readTile(layout.quats[0], layout.quats[1]);
    const motionL = readTile(layout.motion_l[0], layout.motion_l[1]);
    const scales = readTile(layout.scales[0], layout.scales[1]);
    const sh0 = readTile(layout.sh0[0], layout.sh0[1]);
    const motionU = readTile(layout.motion_u[0], layout.motion_u[1]);
    const meta = readTile(layout.meta[0], layout.meta[1]);

    for (let i = 0; i < birthCount; i++) {
      const p = i * 4; // pixel offset (RGBA per pixel)

      // Decode position (16-bit signed-log)
      const posU16X = meansL[p] + meansU[p] * 256;
      const posU16Y = meansL[p + 1] + meansU[p + 1] * 256;
      const posU16Z = meansL[p + 2] + meansU[p + 2] * 256;
      const posLog = new Float32Array([
        this.posMins[0] + (posU16X / 65535) * this.posRange[0],
        this.posMins[1] + (posU16Y / 65535) * this.posRange[1],
        this.posMins[2] + (posU16Z / 65535) * this.posRange[2],
      ]);
      // Apply inverse signed-log: sign(x) * (exp(|x|) - 1)
      const position = new Float32Array([
        Math.sign(posLog[0]) * (Math.exp(Math.abs(posLog[0])) - 1),
        Math.sign(posLog[1]) * (Math.exp(Math.abs(posLog[1])) - 1),
        Math.sign(posLog[2]) * (Math.exp(Math.abs(posLog[2])) - 1),
      ]);

      // Decode motion (16-bit signed-log)
      const motU16X = motionL[p] + motionU[p] * 256;
      const motU16Y = motionL[p + 1] + motionU[p + 1] * 256;
      const motU16Z = motionL[p + 2] + motionU[p + 2] * 256;
      const motLog = new Float32Array([
        this.motionMins[0] + (motU16X / 65535) * this.motionRange[0],
        this.motionMins[1] + (motU16Y / 65535) * this.motionRange[1],
        this.motionMins[2] + (motU16Z / 65535) * this.motionRange[2],
      ]);
      const motion = new Float32Array([
        Math.sign(motLog[0]) * (Math.exp(Math.abs(motLog[0])) - 1),
        Math.sign(motLog[1]) * (Math.exp(Math.abs(motLog[1])) - 1),
        Math.sign(motLog[2]) * (Math.exp(Math.abs(motLog[2])) - 1),
      ]);

      // Decode lifetime (16-bit)
      const lifetime = meta[p] + meta[p + 1] * 256;

      // Store pre-encoded tiles (copied directly to SOG each frame)
      births.push({
        quatsEncoded: new Uint8Array([
          quats[p],
          quats[p + 1],
          quats[p + 2],
          quats[p + 3],
        ]),
        scalesEncoded: new Uint8Array([
          scales[p],
          scales[p + 1],
          scales[p + 2],
          scales[p + 3],
        ]),
        sh0Encoded: new Uint8Array([
          sh0[p],
          sh0[p + 1],
          sh0[p + 2],
          sh0[p + 3],
        ]),
        position,
        motion,
        remainingFrames: lifetime,
        justBorn: true, // Flag to skip first motion update
      });
    }

    return births;
  }

  private _assembleSogTexture(): void {
    const ts = this.tileSize;
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
      const posEncoded = this._encodePosition(g.position);

      // means_l: tile (0,0)
      const meansLBase = (row * sogWidth + col) * 4;
      this.sogTileData[meansLBase] = posEncoded[0];
      this.sogTileData[meansLBase + 1] = posEncoded[1];
      this.sogTileData[meansLBase + 2] = posEncoded[2];
      this.sogTileData[meansLBase + 3] = 255;

      // means_u: tile (1,0)
      const meansUBase = (row * sogWidth + ts + col) * 4;
      this.sogTileData[meansUBase] = posEncoded[3];
      this.sogTileData[meansUBase + 1] = posEncoded[4];
      this.sogTileData[meansUBase + 2] = posEncoded[5];
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

  private _encodePosition(pos: Float32Array): Uint8Array {
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

    return new Uint8Array([
      u16X & 0xff,
      u16Y & 0xff,
      u16Z & 0xff, // low bytes
      u16X >> 8,
      u16Y >> 8,
      u16Z >> 8, // high bytes
    ]);
  }

  getTotalFrames(): number {
    return this.deltaFrames.length;
  }

  getSOGDimensions(): { width: number; height: number } {
    return { width: this.sogWidth, height: this.sogHeight };
  }
}
