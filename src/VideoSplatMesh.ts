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
