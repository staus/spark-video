import * as THREE from "three";
import type { GPUVideoTileUVs, SOGVideoMetadata } from "./PackedSplats";
import { SplatMesh, type SplatMeshOptions } from "./SplatMesh";

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
}

declare const ImageDecoder: {
  new (init: ImageDecoderInit): ImageDecoderInterface;
};

/**
 * Metadata format for 4DGS video files (JSON sidecar)
 */
export interface Video4DGSMetadata {
  tile_size: number;
  layout: Record<string, [number, number]>; // tile name -> [col, row]
  video: {
    frames: number;
    fps: number;
  };
  sog: {
    count: number;
    bounds: {
      min: [number, number, number];
      max: [number, number, number];
    };
    scales: {
      codebook: number[];
    };
    sh0: {
      codebook: number[];
    };
  };
  "4dgs"?: {
    frame_gaussian_counts?: number[];
  };
}

/**
 * VideoSplatMesh - Extends SplatMesh with animated WebP video playback
 *
 * Uses ImageBitmap directly as texture source to avoid canvas color space conversion.
 * This preserves raw pixel values needed for GPU decode shader.
 *
 * Usage:
 *   const videoMesh = new VideoSplatMesh();
 *   await videoMesh.loadVideo(webpBlob, jsonMetadata);
 *   scene.add(videoMesh);
 *
 *   // In render loop:
 *   videoMesh.tick(renderer, performance.now());
 *   renderer.render(scene, camera);
 *
 *   // Playback control:
 *   videoMesh.play();
 *   videoMesh.pause();
 */
export class VideoSplatMesh extends SplatMesh {
  // Frame data
  private frameData: ImageBitmap[] = [];
  private totalFrames = 0;
  private fps = 30;
  private frameInterval = 1000 / 30;
  private videoWidth = 0;
  private videoHeight = 0;

  // Texture created directly from ImageBitmap (no canvas color conversion)
  private frameTexture: THREE.Texture | null = null;

  // Tile UV coordinates for GPU decode
  private tileUVs: GPUVideoTileUVs | null = null;

  // Per-frame gaussian counts (optional - if not provided, uses static count)
  private frameGaussianCounts: number[] | null = null;
  private staticCount = 0;

  // Metadata for validation (stored for validateDecode)
  private validationMetadata: {
    positionMins: [number, number, number];
    positionMaxs: [number, number, number];
    scaleCodebook: number[];
    sh0Codebook: number[];
    lnScaleMin: number;
    lnScaleMax: number;
  } | null = null;

  // Playback state
  currentFrameIndex = 0;
  isPlaying = false;
  private lastFrameTime = 0;
  private accumulatedTime = 0;

  // Callback for frame changes (optional, for UI updates)
  onFrameChange: ((frameIndex: number, totalFrames: number) => void) | null =
    null;

  constructor(options: SplatMeshOptions = {}) {
    super(options);
  }

  /**
   * Check if ImageDecoder API is available
   */
  static isSupported(): boolean {
    return "ImageDecoder" in window;
  }

  /**
   * Load an animated WebP video with JSON metadata
   */
  async loadVideo(
    webpBlob: Blob,
    metadata: Video4DGSMetadata,
  ): Promise<{ loadTime: number }> {
    const loadStart = performance.now();

    // Validate metadata
    if (!metadata.sog || !metadata.layout || !metadata.video) {
      throw new Error("Invalid video metadata");
    }

    this.fps = metadata.video.fps || 30;
    this.frameInterval = 1000 / this.fps;
    this.totalFrames = metadata.video.frames || 1;

    // Create ImageDecoder
    const arrayBuffer = await webpBlob.arrayBuffer();
    const decoder = new ImageDecoder({
      data: arrayBuffer,
      type: "image/webp",
    });

    await decoder.tracks.ready;
    const selectedTrack = decoder.tracks.selectedTrack;

    // Use track frame count if available
    if (selectedTrack?.frameCount) {
      this.totalFrames = selectedTrack.frameCount;
    }

    console.log(`VideoSplatMesh: ${this.totalFrames} frames @ ${this.fps}fps`);

    // Pre-decode all frames to ImageBitmap with no color conversion
    this.frameData = [];
    for (let i = 0; i < this.totalFrames; i++) {
      const result = await decoder.decode({ frameIndex: i });
      const frame = result.image;

      if (i === 0) {
        this.videoWidth = frame.displayWidth;
        this.videoHeight = frame.displayHeight;
      }

      // Create ImageBitmap with NO color space conversion to preserve raw values
      // Do NOT use imageOrientation: 'flipY' here - the shader handles coordinate conversion
      const bitmap = await createImageBitmap(frame, {
        premultiplyAlpha: "none",
        colorSpaceConversion: "none",
      });
      this.frameData.push(bitmap);
      frame.close();
    }

    // Calculate tile UVs
    this.tileUVs = this.calculateTileUVs(metadata);

    // Store per-frame gaussian counts if available
    this.staticCount = metadata.sog.count;
    if (metadata["4dgs"]?.frame_gaussian_counts) {
      this.frameGaussianCounts = metadata["4dgs"].frame_gaussian_counts;
    }

    // Initialize GPU video mode in PackedSplats
    const sparkMetadata: SOGVideoMetadata = {
      count: metadata.sog.count,
      mins: metadata.sog.bounds.min,
      maxs: metadata.sog.bounds.max,
      scaleCodebook: metadata.sog.scales.codebook,
      sh0Codebook: metadata.sog.sh0.codebook,
    };

    this.packedSplats.initVideoModeGPU(sparkMetadata, metadata.tile_size);

    // Store metadata for validation
    const scaleCodebook = metadata.sog.scales.codebook;
    const lnScaleMin = Math.min(...scaleCodebook);
    const lnScaleMax = Math.max(...scaleCodebook);
    this.validationMetadata = {
      positionMins: metadata.sog.bounds.min,
      positionMaxs: metadata.sog.bounds.max,
      scaleCodebook: scaleCodebook,
      sh0Codebook: metadata.sog.sh0.codebook,
      lnScaleMin,
      lnScaleMax,
    };

    // Use first frame's count if available, otherwise static count
    const initialCount = this.frameGaussianCounts?.[0] ?? this.staticCount;
    this.numSplats = initialCount;

    const loadTime = performance.now() - loadStart;

    // Log comprehensive load summary
    console.log("[VideoSplatMesh] === Load Summary ===");
    console.log(
      `  Frames: ${this.totalFrames} @ ${this.fps}fps (${(this.totalFrames / this.fps).toFixed(2)}s)`,
    );
    console.log(`  Texture: ${this.videoWidth}x${this.videoHeight}`);
    console.log(`  Tile size: ${metadata.tile_size}px`);
    console.log(`  Max splat capacity: ${this.staticCount.toLocaleString()}`);
    if (this.frameGaussianCounts) {
      const minCount = Math.min(...this.frameGaussianCounts);
      const maxCount = Math.max(...this.frameGaussianCounts);
      console.log(
        `  Dynamic counts: ${minCount.toLocaleString()} - ${maxCount.toLocaleString()} splats/frame`,
      );
      console.log(`  Frame counts: [${this.frameGaussianCounts.join(", ")}]`);
    } else {
      console.log(
        `  Static count: ${this.staticCount.toLocaleString()} splats/frame`,
      );
    }
    console.log(
      `  Bounds: [${sparkMetadata.mins.map((v) => v.toFixed(3)).join(", ")}] to [${sparkMetadata.maxs.map((v) => v.toFixed(3)).join(", ")}]`,
    );
    console.log(`  Load time: ${loadTime.toFixed(0)}ms`);
    console.log("[VideoSplatMesh] === Ready ===");

    return { loadTime };
  }

  private calculateTileUVs(metadata: Video4DGSMetadata): GPUVideoTileUVs {
    const tileSize = metadata.tile_size;
    const layout = metadata.layout;
    const w = this.videoWidth;
    const h = this.videoHeight;

    const getTileUV = (name: string) => {
      const [col, row] = layout[name];
      const x = col * tileSize;
      const y = row * tileSize;
      return {
        u0: x / w,
        v0: y / h,
        u1: (x + tileSize) / w,
        v1: (y + tileSize) / h,
      };
    };

    return {
      means_l: getTileUV("means_l"),
      means_u: getTileUV("means_u"),
      quats: getTileUV("quats"),
      scales: getTileUV("scales"),
      sh0: getTileUV("sh0"),
    };
  }

  // WebGL texture handle for raw uploads (bypasses THREE.js color management)
  private glTexture: WebGLTexture | null = null;

  /**
   * Upload frame to GPU using raw WebGL, bypassing THREE.js color management.
   * Guarantees no color space conversion, no alpha premultiplication.
   */
  private uploadFrameRawWebGL(
    renderer: THREE.WebGLRenderer,
    index: number,
  ): void {
    if (index < 0 || index >= this.frameData.length) return;

    const bitmap = this.frameData[index];
    const gl = renderer.getContext() as WebGL2RenderingContext;
    this.currentFrameIndex = index;

    // Create THREE.Texture container on first use
    if (!this.frameTexture) {
      this.frameTexture = new THREE.Texture();
      this.frameTexture.minFilter = THREE.NearestFilter;
      this.frameTexture.magFilter = THREE.NearestFilter;
      this.frameTexture.generateMipmaps = false;
      this.frameTexture.colorSpace = THREE.NoColorSpace;

      // Create raw WebGL texture
      this.glTexture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this.glTexture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

      // Inject our WebGL texture into THREE.js texture properties
      const texProps = renderer.properties.get(this.frameTexture) as {
        __webglTexture: WebGLTexture | null;
        __webglInit: boolean;
      };
      texProps.__webglTexture = this.glTexture;
      texProps.__webglInit = true;
    }

    // Upload with explicit raw settings - no color conversion
    gl.bindTexture(gl.TEXTURE_2D, this.glTexture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);

    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA8, // Raw RGBA, not SRGB8_ALPHA8
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      bitmap,
    );
  }

  // Logging state
  private lastLoggedFrame = -1;
  private frameDecodeCount = 0;

  /**
   * Decode a frame to GPU. Single path for all frame updates.
   */
  private decodeFrame(renderer: THREE.WebGLRenderer, frameIndex: number) {
    if (!this.tileUVs) return;

    // Upload frame using raw WebGL (bypasses THREE.js color management)
    this.uploadFrameRawWebGL(renderer, frameIndex);

    // frameTexture is guaranteed to exist after uploadFrameRawWebGL
    if (!this.frameTexture) return;

    const expectedCount =
      this.frameGaussianCounts?.[frameIndex] ?? this.staticCount;
    this.packedSplats.updateVideoSplatCount(expectedCount);
    this.numSplats = expectedCount;

    this.packedSplats.updateFromVideoTextureGPU(
      renderer,
      this.frameTexture,
      this.tileUVs,
      this.videoWidth,
      this.videoHeight,
    );

    this.updateVersion();
    this.frameDecodeCount++;

    // Log frame info (throttled to avoid spam)
    if (frameIndex !== this.lastLoggedFrame) {
      const hasDynamicCounts = this.frameGaussianCounts !== null;
      console.log(
        `[VideoSplatMesh] Frame ${frameIndex}/${this.totalFrames - 1}: ` +
          `splats=${expectedCount.toLocaleString()} ` +
          `(${hasDynamicCounts ? "dynamic" : "static"}) ` +
          `texture=${this.videoWidth}x${this.videoHeight} ` +
          `tile=${this.tileUVs ? "ready" : "missing"}`,
      );
      this.lastLoggedFrame = frameIndex;
    }

    this.onFrameChange?.(this.currentFrameIndex, this.totalFrames);
  }

  /**
   * Call each frame from the render loop.
   * Returns true if a new frame was decoded.
   */
  tick(
    renderer: THREE.WebGLRenderer,
    now: number = performance.now(),
  ): boolean {
    if (!this.isPlaying || !this.frameTexture || !this.tileUVs) {
      return false;
    }

    if (this.lastFrameTime === 0) {
      this.lastFrameTime = now;
      this.accumulatedTime = 0;
    }

    const deltaTime = now - this.lastFrameTime;
    this.lastFrameTime = now;
    this.accumulatedTime += deltaTime;

    if (this.accumulatedTime < this.frameInterval) {
      return false;
    }

    this.accumulatedTime -= this.frameInterval;
    this.currentFrameIndex = (this.currentFrameIndex + 1) % this.totalFrames;
    this.decodeFrame(renderer, this.currentFrameIndex);

    return true;
  }

  /**
   * Decode first frame without starting playback.
   */
  decodeFirstFrame(renderer: THREE.WebGLRenderer) {
    this.decodeFrame(renderer, 0);
  }

  play() {
    this.isPlaying = true;
    this.lastFrameTime = 0;
    this.accumulatedTime = 0;
  }

  pause() {
    this.isPlaying = false;
  }

  toggle() {
    if (this.isPlaying) {
      this.pause();
    } else {
      this.play();
    }
  }

  seekToFrame(frame: number, renderer: THREE.WebGLRenderer) {
    const frameIndex = Math.max(0, Math.min(frame, this.totalFrames - 1));
    this.decodeFrame(renderer, frameIndex);
  }

  getTotalFrames(): number {
    return this.totalFrames;
  }

  getFPS(): number {
    return this.fps;
  }

  /**
   * Validate the decode pipeline by reading back raw pixels and decoded splat data.
   * Traces through ALL shader math step-by-step with actual codebook values.
   * Call this from browser console: videoMesh.validateDecode(renderer)
   */
  async validateDecode(
    renderer: THREE.WebGLRenderer,
    splatIndex = 0,
  ): Promise<void> {
    if (!this.frameData.length || !this.tileUVs) {
      console.error("[Validate] No frame data or tile UVs");
      return;
    }
    if (!this.validationMetadata) {
      console.error("[Validate] No validation metadata");
      return;
    }

    const meta = this.validationMetadata;
    const gl = renderer.getContext() as WebGL2RenderingContext;
    const frameIndex = this.currentFrameIndex;
    const bitmap = this.frameData[frameIndex];

    console.log(
      "[Validate] ═══════════════════════════════════════════════════",
    );
    console.log("[Validate] COMPREHENSIVE SHADER PIPELINE VALIDATION");
    console.log(
      "[Validate] ═══════════════════════════════════════════════════",
    );
    console.log(`  Frame: ${frameIndex}, Splat: ${splatIndex}`);
    console.log(`  Texture: ${this.videoWidth}x${this.videoHeight}`);
    console.log(
      `  Position bounds: [${meta.positionMins.join(", ")}] to [${meta.positionMaxs.join(", ")}]`,
    );
    console.log(
      `  Scale range: ln(${meta.lnScaleMin.toFixed(3)}) to ln(${meta.lnScaleMax.toFixed(3)})`,
    );

    // Step 1: Read raw pixels from ImageBitmap using a temp canvas
    const canvas = document.createElement("canvas");
    canvas.width = this.videoWidth;
    canvas.height = this.videoHeight;
    const ctx = canvas.getContext("2d", {
      colorSpace: "srgb",
      willReadFrequently: true,
    });
    if (!ctx) {
      console.error("[Validate] Failed to get canvas 2D context");
      return;
    }
    ctx.drawImage(bitmap, 0, 0);

    const tileSize = Math.round(
      this.tileUVs.means_l.u1 * this.videoWidth -
        this.tileUVs.means_l.u0 * this.videoWidth,
    );
    const tileX = splatIndex % tileSize;
    const tileY = Math.floor(splatIndex / tileSize);

    const readTilePixel = (tileUV: { u0: number; v0: number }) => {
      const tileStartX = Math.round(tileUV.u0 * this.videoWidth);
      const tileStartY = Math.round(tileUV.v0 * this.videoHeight);
      const px = tileStartX + tileX;
      const py = tileStartY + tileY;
      const data = ctx.getImageData(px, py, 1, 1).data;
      return { r: data[0], g: data[1], b: data[2], a: data[3], px, py };
    };

    const meansL = readTilePixel(this.tileUVs.means_l);
    const meansU = readTilePixel(this.tileUVs.means_u);
    const quats = readTilePixel(this.tileUVs.quats);
    const scales = readTilePixel(this.tileUVs.scales);
    const sh0 = readTilePixel(this.tileUVs.sh0);

    console.log("\n[Validate] ── STEP 1: RAW TILE PIXELS ──");
    console.log(
      `  means_l[${meansL.px},${meansL.py}]: R=${meansL.r} G=${meansL.g} B=${meansL.b} A=${meansL.a}`,
    );
    console.log(
      `  means_u[${meansU.px},${meansU.py}]: R=${meansU.r} G=${meansU.g} B=${meansU.b} A=${meansU.a}`,
    );
    console.log(
      `  quats[${quats.px},${quats.py}]: R=${quats.r} G=${quats.g} B=${quats.b} A=${quats.a}`,
    );
    console.log(
      `  scales[${scales.px},${scales.py}]: R=${scales.r} G=${scales.g} B=${scales.b} A=${scales.a}`,
    );
    console.log(
      `  sh0[${sh0.px},${sh0.py}]: R=${sh0.r} G=${sh0.g} B=${sh0.b} A=${sh0.a}`,
    );

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 2: POSITION DECODE (shader: decodePosition)
    // ═══════════════════════════════════════════════════════════════════════
    console.log("\n[Validate] ── STEP 2: POSITION DECODE ──");

    // Combine low and high bytes to uint16
    const posX_u16 = meansL.r + meansU.r * 256;
    const posY_u16 = meansL.g + meansU.g * 256;
    const posZ_u16 = meansL.b + meansU.b * 256;
    console.log(`  uint16: X=${posX_u16}, Y=${posY_u16}, Z=${posZ_u16}`);

    // Normalize to 0-1
    const posXnorm = posX_u16 / 65535.0;
    const posYnorm = posY_u16 / 65535.0;
    const posZnorm = posZ_u16 / 65535.0;
    console.log(
      `  normalized: X=${posXnorm.toFixed(6)}, Y=${posYnorm.toFixed(6)}, Z=${posZnorm.toFixed(6)}`,
    );

    // Interpolate in log space
    const posXlog =
      meta.positionMins[0] +
      (meta.positionMaxs[0] - meta.positionMins[0]) * posXnorm;
    const posYlog =
      meta.positionMins[1] +
      (meta.positionMaxs[1] - meta.positionMins[1]) * posYnorm;
    const posZlog =
      meta.positionMins[2] +
      (meta.positionMaxs[2] - meta.positionMins[2]) * posZnorm;
    console.log(
      `  log-space: X=${posXlog.toFixed(6)}, Y=${posYlog.toFixed(6)}, Z=${posZlog.toFixed(6)}`,
    );

    // Apply exp transform: sign(x) * (exp(abs(x)) - 1)
    const expTransform = (v: number) =>
      Math.sign(v) * (Math.exp(Math.abs(v)) - 1);
    const expectedPosX = expTransform(posXlog);
    const expectedPosY = expTransform(posYlog);
    const expectedPosZ = expTransform(posZlog);
    console.log(
      `  EXPECTED position: X=${expectedPosX.toFixed(6)}, Y=${expectedPosY.toFixed(6)}, Z=${expectedPosZ.toFixed(6)}`,
    );

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 3: QUATERNION DECODE (shader: decodeQuaternion - smallest-three)
    // ═══════════════════════════════════════════════════════════════════════
    console.log(
      "\n[Validate] ── STEP 3: QUATERNION DECODE (smallest-three) ──",
    );

    const SQRT2 = Math.sqrt(2);
    const r0 = (quats.r / 255.0 - 0.5) * SQRT2;
    const r1 = (quats.g / 255.0 - 0.5) * SQRT2;
    const r2 = (quats.b / 255.0 - 0.5) * SQRT2;
    console.log(
      `  decoded components: r0=${r0.toFixed(6)}, r1=${r1.toFixed(6)}, r2=${r2.toFixed(6)}`,
    );

    const rr = Math.sqrt(Math.max(0, 1 - r0 * r0 - r1 * r1 - r2 * r2));
    console.log(`  reconstructed component: rr=${rr.toFixed(6)}`);

    const rOrder = quats.a - 252;
    console.log(`  order index: ${rOrder} (alpha=${quats.a})`);

    let expectedQuat: [number, number, number, number];
    if (rOrder === 0) {
      expectedQuat = [r0, r1, r2, rr]; // w was largest
    } else if (rOrder === 1) {
      expectedQuat = [rr, r1, r2, r0]; // x was largest
    } else if (rOrder === 2) {
      expectedQuat = [r1, rr, r2, r0]; // y was largest
    } else {
      expectedQuat = [r1, r2, rr, r0]; // z was largest
    }

    // Normalize
    const qLen = Math.sqrt(
      expectedQuat[0] ** 2 +
        expectedQuat[1] ** 2 +
        expectedQuat[2] ** 2 +
        expectedQuat[3] ** 2,
    );
    expectedQuat = expectedQuat.map((q) => q / qLen) as [
      number,
      number,
      number,
      number,
    ];
    console.log(
      `  EXPECTED quaternion: [${expectedQuat.map((q) => q.toFixed(6)).join(", ")}]`,
    );

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 4: SCALE DECODE (shader: decodeScales - codebook lookup)
    // ═══════════════════════════════════════════════════════════════════════
    console.log("\n[Validate] ── STEP 4: SCALE DECODE (codebook) ──");

    const scaleIdxX = scales.r;
    const scaleIdxY = scales.g;
    const scaleIdxZ = scales.b;
    console.log(
      `  codebook indices: X=${scaleIdxX}, Y=${scaleIdxY}, Z=${scaleIdxZ}`,
    );

    const logScaleX = meta.scaleCodebook[scaleIdxX] ?? meta.scaleCodebook[0];
    const logScaleY = meta.scaleCodebook[scaleIdxY] ?? meta.scaleCodebook[0];
    const logScaleZ = meta.scaleCodebook[scaleIdxZ] ?? meta.scaleCodebook[0];
    console.log(
      `  log-scale values: X=${logScaleX.toFixed(6)}, Y=${logScaleY.toFixed(6)}, Z=${logScaleZ.toFixed(6)}`,
    );

    const expectedScaleX = Math.exp(logScaleX);
    const expectedScaleY = Math.exp(logScaleY);
    const expectedScaleZ = Math.exp(logScaleZ);
    console.log(
      `  EXPECTED scales: X=${expectedScaleX.toExponential(6)}, Y=${expectedScaleY.toExponential(6)}, Z=${expectedScaleZ.toExponential(6)}`,
    );

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 5: COLOR DECODE (shader: decodeRGBA - SH0 codebook + alpha passthrough)
    // ═══════════════════════════════════════════════════════════════════════
    console.log("\n[Validate] ── STEP 5: COLOR DECODE (SH0 codebook) ──");

    const SH_C0 = 0.28209479177387814;
    const sh0IdxR = sh0.r;
    const sh0IdxG = sh0.g;
    const sh0IdxB = sh0.b;
    console.log(`  codebook indices: R=${sh0IdxR}, G=${sh0IdxG}, B=${sh0IdxB}`);

    const sh0R = meta.sh0Codebook[sh0IdxR] ?? meta.sh0Codebook[0];
    const sh0G = meta.sh0Codebook[sh0IdxG] ?? meta.sh0Codebook[0];
    const sh0B = meta.sh0Codebook[sh0IdxB] ?? meta.sh0Codebook[0];
    console.log(
      `  SH0 values: R=${sh0R.toFixed(6)}, G=${sh0G.toFixed(6)}, B=${sh0B.toFixed(6)}`,
    );

    // SH0 to RGB: rgb = SH_C0 * sh0 + 0.5
    const colorR = Math.max(0, Math.min(1, SH_C0 * sh0R + 0.5));
    const colorG = Math.max(0, Math.min(1, SH_C0 * sh0G + 0.5));
    const colorB = Math.max(0, Math.min(1, SH_C0 * sh0B + 0.5));
    const colorA = sh0.a / 255.0; // Opacity passthrough (already 0-1 normalized)
    console.log(
      `  EXPECTED RGBA (0-1): R=${colorR.toFixed(6)}, G=${colorG.toFixed(6)}, B=${colorB.toFixed(6)}, A=${colorA.toFixed(6)}`,
    );
    console.log(
      `  EXPECTED RGBA (0-255): R=${Math.round(colorR * 255)}, G=${Math.round(colorG * 255)}, B=${Math.round(colorB * 255)}, A=${sh0.a}`,
    );

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 6: EXPECTED PACKED FORMAT (shader: packSplatEncoding)
    // ═══════════════════════════════════════════════════════════════════════
    console.log("\n[Validate] ── STEP 6: EXPECTED PACKED FORMAT ──");

    // Pack RGBA (rgbMin=0, rgbMax=1 means no re-encoding needed)
    const packedR = Math.round(Math.max(0, Math.min(255, colorR * 255)));
    const packedG = Math.round(Math.max(0, Math.min(255, colorG * 255)));
    const packedB = Math.round(Math.max(0, Math.min(255, colorB * 255)));
    const packedA = sh0.a;
    const expectedWord0 =
      packedR | (packedG << 8) | (packedB << 16) | (packedA << 24);
    console.log(
      `  Expected word0 (RGBA): 0x${expectedWord0.toString(16).padStart(8, "0")}`,
    );
    console.log(
      `    -> R=${packedR}, G=${packedG}, B=${packedB}, A=${packedA}`,
    );

    // Pack position as float16
    const packF16 = (val: number) => {
      if (val === 0) return 0;
      const sign = val < 0 ? 1 : 0;
      const absVal = Math.abs(val);
      const exp = Math.floor(Math.log2(absVal));
      const expBiased = exp + 15;
      if (expBiased <= 0) return sign << 15; // Denorm -> 0
      if (expBiased >= 31) return (sign << 15) | 0x7c00; // Inf
      const frac = Math.round((absVal / 2 ** exp - 1) * 1024);
      return (sign << 15) | (expBiased << 10) | (frac & 0x3ff);
    };

    const posXf16 = packF16(expectedPosX);
    const posYf16 = packF16(expectedPosY);
    const posZf16 = packF16(expectedPosZ);
    const expectedWord1 = posXf16 | (posYf16 << 16);
    console.log(
      `  Expected word1 (pos XY): 0x${expectedWord1.toString(16).padStart(8, "0")}`,
    );
    console.log(
      `    -> posX_f16=0x${posXf16.toString(16)}, posY_f16=0x${posYf16.toString(16)}`,
    );

    // Pack quaternion using octahedral encoding (encodeQuatOctXy88R8)
    // This is complex - let me implement it
    const encodeQuatOctXy88R8 = (q: [number, number, number, number]) => {
      let [qx, qy, qz, qw] = q;
      // Ensure minimal representation
      if (qw < 0) {
        qx = -qx;
        qy = -qy;
        qz = -qz;
        qw = -qw;
      }

      const theta = 2 * Math.acos(Math.min(1, qw));
      const halfTheta = theta * 0.5;
      const s = Math.sin(halfTheta);

      let axis: [number, number, number];
      if (Math.abs(s) < 1e-6) {
        axis = [1, 0, 0];
      } else {
        axis = [qx / s, qy / s, qz / s];
      }

      // Folded octahedral mapping
      const sum = Math.abs(axis[0]) + Math.abs(axis[1]) + Math.abs(axis[2]);
      let px = axis[0] / sum;
      let py = axis[1] / sum;

      if (axis[2] < 0) {
        const oldPx = px;
        px = (1 - Math.abs(py)) * (px >= 0 ? 1 : -1);
        py = (1 - Math.abs(oldPx)) * (py >= 0 ? 1 : -1);
      }

      const u_f = px * 0.5 + 0.5;
      const v_f = py * 0.5 + 0.5;
      const quantU = Math.round(Math.max(0, Math.min(255, u_f * 255)));
      const quantV = Math.round(Math.max(0, Math.min(255, v_f * 255)));
      const angleInt = Math.round(
        Math.max(0, Math.min(255, (theta / Math.PI) * 255)),
      );

      return (angleInt << 16) | (quantV << 8) | quantU;
    };

    const uQuat = encodeQuatOctXy88R8(expectedQuat);
    const uQuat0 = uQuat & 0xff;
    const uQuat1 = (uQuat >> 8) & 0xff;
    const uQuat2 = (uQuat >> 16) & 0xff;
    console.log(
      `  Quaternion octahedral: 0x${uQuat.toString(16).padStart(6, "0")}`,
    );
    console.log(`    -> bytes: [${uQuat0}, ${uQuat1}, ${uQuat2}]`);

    // Pack scales (log encoding)
    const lnScaleScale = 254.0 / (meta.lnScaleMax - meta.lnScaleMin);
    const packScale = (s: number) => {
      if (s === 0) return 0;
      const encoded =
        Math.round(
          Math.max(
            0,
            Math.min(254, (Math.log(s) - meta.lnScaleMin) * lnScaleScale),
          ),
        ) + 1;
      return encoded;
    };
    const uScaleX = packScale(expectedScaleX);
    const uScaleY = packScale(expectedScaleY);
    const uScaleZ = packScale(expectedScaleZ);
    console.log(`  Packed scales: X=${uScaleX}, Y=${uScaleY}, Z=${uScaleZ}`);

    // Assemble word2 and word3
    const expectedWord2 = posZf16 | (uQuat0 << 16) | (uQuat1 << 24);
    const expectedWord3 =
      uScaleX | (uScaleY << 8) | (uScaleZ << 16) | (uQuat2 << 24);
    console.log(
      `  Expected word2: 0x${expectedWord2.toString(16).padStart(8, "0")}`,
    );
    console.log(
      `  Expected word3: 0x${expectedWord3.toString(16).padStart(8, "0")}`,
    );

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 7: READ ACTUAL GPU OUTPUT AND COMPARE
    // ═══════════════════════════════════════════════════════════════════════
    console.log("\n[Validate] ── STEP 7: ACTUAL GPU OUTPUT ──");

    const target = this.packedSplats.target;
    if (!target) {
      console.error("[Validate] No render target");
      return;
    }

    const SPLAT_TEX_WIDTH = 2048;
    const SPLAT_TEX_HEIGHT = 2048;
    const layerSize = SPLAT_TEX_WIDTH * SPLAT_TEX_HEIGHT;
    const layer = Math.floor(splatIndex / layerSize);
    const indexInLayer = splatIndex % layerSize;
    const splatX = indexInLayer % SPLAT_TEX_WIDTH;
    const splatY = Math.floor(indexInLayer / SPLAT_TEX_WIDTH);

    const readBuffer = new Uint32Array(4);
    renderer.setRenderTarget(target, layer);
    gl.readPixels(
      splatX,
      splatY,
      1,
      1,
      gl.RGBA_INTEGER,
      gl.UNSIGNED_INT,
      readBuffer,
    );
    renderer.setRenderTarget(null);

    console.log(
      `  Actual word0: 0x${readBuffer[0].toString(16).padStart(8, "0")}`,
    );
    console.log(
      `  Actual word1: 0x${readBuffer[1].toString(16).padStart(8, "0")}`,
    );
    console.log(
      `  Actual word2: 0x${readBuffer[2].toString(16).padStart(8, "0")}`,
    );
    console.log(
      `  Actual word3: 0x${readBuffer[3].toString(16).padStart(8, "0")}`,
    );

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 8: DETAILED COMPARISON
    // ═══════════════════════════════════════════════════════════════════════
    console.log("\n[Validate] ── STEP 8: COMPARISON ──");

    const compareWord = (name: string, expected: number, actual: number) => {
      const match = expected === actual;
      const status = match ? "✓" : "✗";
      console.log(
        `  ${status} ${name}: expected=0x${expected.toString(16).padStart(8, "0")}, actual=0x${actual.toString(16).padStart(8, "0")}`,
      );
      if (!match) {
        console.log(
          `     XOR diff: 0x${(expected ^ actual).toString(16).padStart(8, "0")}`,
        );
      }
      return match;
    };

    const w0Match = compareWord("word0 (RGBA)", expectedWord0, readBuffer[0]);
    const w1Match = compareWord("word1 (pos XY)", expectedWord1, readBuffer[1]);
    const w2Match = compareWord(
      "word2 (pos Z + quat)",
      expectedWord2,
      readBuffer[2],
    );
    const w3Match = compareWord(
      "word3 (scales + quat)",
      expectedWord3,
      readBuffer[3],
    );

    // Decode actual values for comparison
    const unpackF16 = (bits: number) => {
      const sign = (bits >> 15) & 1;
      const exp = (bits >> 10) & 0x1f;
      const frac = bits & 0x3ff;
      if (exp === 0) return sign ? -0 : 0;
      if (exp === 31)
        return sign ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY;
      return (sign ? -1 : 1) * 2 ** (exp - 15) * (1 + frac / 1024);
    };

    const actualPosX = unpackF16(readBuffer[1] & 0xffff);
    const actualPosY = unpackF16((readBuffer[1] >> 16) & 0xffff);
    const actualPosZ = unpackF16(readBuffer[2] & 0xffff);

    console.log("\n[Validate] ── POSITION COMPARISON ──");
    console.log(
      `  Expected: [${expectedPosX.toFixed(6)}, ${expectedPosY.toFixed(6)}, ${expectedPosZ.toFixed(6)}]`,
    );
    console.log(
      `  Actual:   [${actualPosX.toFixed(6)}, ${actualPosY.toFixed(6)}, ${actualPosZ.toFixed(6)}]`,
    );
    console.log(
      `  Delta:    [${(actualPosX - expectedPosX).toFixed(6)}, ${(actualPosY - expectedPosY).toFixed(6)}, ${(actualPosZ - expectedPosZ).toFixed(6)}]`,
    );

    const actualR = readBuffer[0] & 0xff;
    const actualG = (readBuffer[0] >> 8) & 0xff;
    const actualB = (readBuffer[0] >> 16) & 0xff;
    const actualA = (readBuffer[0] >> 24) & 0xff;

    console.log("\n[Validate] ── RGBA COMPARISON ──");
    console.log(`  Expected: [${packedR}, ${packedG}, ${packedB}, ${packedA}]`);
    console.log(`  Actual:   [${actualR}, ${actualG}, ${actualB}, ${actualA}]`);
    console.log(
      `  Delta:    [${actualR - packedR}, ${actualG - packedG}, ${actualB - packedB}, ${actualA - packedA}]`,
    );

    // Final verdict
    console.log(
      "\n[Validate] ═══════════════════════════════════════════════════",
    );
    if (w0Match && w1Match && w2Match && w3Match) {
      console.log(
        "[Validate] ✓ ALL WORDS MATCH - Pipeline is working correctly!",
      );
    } else {
      console.log(
        "[Validate] ✗ MISMATCH DETECTED - Check comparison above for details",
      );
      if (
        readBuffer[0] === 0 &&
        readBuffer[1] === 0 &&
        readBuffer[2] === 0 &&
        readBuffer[3] === 0
      ) {
        console.log(
          "[Validate] ⚠️ All zeros - shader may not have written to this splat!",
        );
        console.log(`  splatIndex=${splatIndex}, splatCount=${this.numSplats}`);
      }
    }
    console.log(
      "[Validate] ═══════════════════════════════════════════════════",
    );
  }

  dispose() {
    super.dispose();
    if (this.frameTexture) {
      this.frameTexture.dispose();
      this.frameTexture = null;
    }
    // glTexture is deleted by THREE.js when frameTexture.dispose() is called
    // (we injected it into texture properties)
    this.glTexture = null;
    for (const bitmap of this.frameData) {
      bitmap.close();
    }
    this.frameData = [];
  }
}
