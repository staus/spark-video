import * as THREE from "three";
import type { GPUVideoTileUVs, SOGVideoMetadata } from "./PackedSplats";
import { SparkRenderer } from "./SparkRenderer";
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
    means: {
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
  "4dgs"?: {
    frame_gaussian_counts?: number[];
  };
}

/**
 * VideoSplatMesh - Extends SplatMesh with animated WebP video playback
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

  // Canvas and texture for frame rendering
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private canvasTexture: THREE.CanvasTexture | null = null;

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

    this.canvas = document.createElement("canvas");
    const ctx = this.canvas.getContext("2d", {
      willReadFrequently: false,
      alpha: true,
      colorSpace: "srgb",
    });
    if (!ctx) throw new Error("Failed to create 2D context");
    this.ctx = ctx;
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

    // Pre-decode all frames
    this.frameData = [];
    for (let i = 0; i < this.totalFrames; i++) {
      const result = await decoder.decode({ frameIndex: i });
      const frame = result.image;

      if (i === 0) {
        this.videoWidth = frame.displayWidth;
        this.videoHeight = frame.displayHeight;
        this.canvas.width = this.videoWidth;
        this.canvas.height = this.videoHeight;
      }

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

    // Create texture from first frame
    this.drawFrame(0);
    this.createTexture();

    // Initialize GPU video mode in PackedSplats
    const sparkMetadata: SOGVideoMetadata = {
      count: metadata.sog.count,
      mins: metadata.sog.means.mins,
      maxs: metadata.sog.means.maxs,
      scaleCodebook: metadata.sog.scales.codebook,
      sh0Codebook: metadata.sog.sh0.codebook,
    };

    this.packedSplats.initVideoModeGPU(sparkMetadata, metadata.tile_size);

    // Use first frame's count if available, otherwise static count
    const initialCount = this.frameGaussianCounts?.[0] ?? this.staticCount;
    this.numSplats = initialCount;

    const loadTime = performance.now() - loadStart;
    console.log(`VideoSplatMesh loaded in ${loadTime.toFixed(0)}ms`);

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

  private drawFrame(index: number) {
    if (index >= 0 && index < this.frameData.length) {
      this.ctx.drawImage(this.frameData[index], 0, 0);
      this.currentFrameIndex = index;
    }
  }

  private createTexture() {
    if (this.canvasTexture) {
      this.canvasTexture.dispose();
    }

    this.canvasTexture = new THREE.CanvasTexture(this.canvas);
    this.canvasTexture.minFilter = THREE.NearestFilter;
    this.canvasTexture.magFilter = THREE.NearestFilter;
    this.canvasTexture.generateMipmaps = false;
    this.canvasTexture.colorSpace = THREE.LinearSRGBColorSpace;
    this.canvasTexture.flipY = true;
    this.canvasTexture.needsUpdate = true;
  }

  /**
   * Get the gaussian count for a specific frame
   */
  private getFrameSplatCount(frameIndex: number): number {
    if (
      this.frameGaussianCounts &&
      frameIndex < this.frameGaussianCounts.length
    ) {
      return this.frameGaussianCounts[frameIndex];
    }
    return this.staticCount;
  }

  /**
   * Call each frame from the render loop.
   * Returns true if a new frame was decoded.
   */
  tick(
    renderer: THREE.WebGLRenderer,
    now: number = performance.now(),
  ): boolean {
    if (!this.isPlaying || !this.canvasTexture || !this.tileUVs) {
      return false;
    }

    // Initialize timing on first tick
    if (this.lastFrameTime === 0) {
      this.lastFrameTime = now;
      this.accumulatedTime = 0;
    }

    const deltaTime = now - this.lastFrameTime;
    this.lastFrameTime = now;
    this.accumulatedTime += deltaTime;

    // Check if it's time for a new frame
    if (this.accumulatedTime < this.frameInterval) {
      return false;
    }

    this.accumulatedTime -= this.frameInterval;

    // Advance to next frame
    this.currentFrameIndex = (this.currentFrameIndex + 1) % this.totalFrames;

    // Draw frame to canvas
    this.drawFrame(this.currentFrameIndex);

    // Update splat count for this frame (handles varying gaussian counts per frame)
    const frameCount = this.getFrameSplatCount(this.currentFrameIndex);
    this.packedSplats.updateVideoSplatCount(frameCount);
    this.numSplats = frameCount;

    // Upload texture to GPU
    this.canvasTexture.needsUpdate = true;
    renderer.initTexture(this.canvasTexture);

    // GPU decode: video texture -> packed splats
    this.packedSplats.updateFromVideoTextureGPU(
      renderer,
      this.canvasTexture,
      this.tileUVs,
      this.videoWidth,
      this.videoHeight,
    );

    // Flush GPU to ensure decode completes before reading
    const gl = renderer.getContext() as WebGL2RenderingContext;
    gl.flush();

    // Trigger SparkRenderer regeneration by incrementing version
    this.updateVersion();

    // Force immediate regeneration by finding SparkRenderer and triggering sync update
    this.triggerImmediateRegeneration(renderer);

    // Notify callback
    if (this.onFrameChange) {
      this.onFrameChange(this.currentFrameIndex, this.totalFrames);
    }

    return true;
  }

  /**
   * Find SparkRenderer in scene and trigger immediate regeneration
   */
  private triggerImmediateRegeneration(_renderer: THREE.WebGLRenderer) {
    // Walk up to find the scene
    let current: THREE.Object3D | null = this as THREE.Object3D;
    while (current && !(current instanceof THREE.Scene)) {
      current = current.parent;
    }
    if (!current) return;

    const scene = current as THREE.Scene;

    // Find SparkRenderer in the scene
    let spark: SparkRenderer | null = null;
    scene.traverse((node) => {
      if (node instanceof SparkRenderer) {
        spark = node;
      }
    });

    if (spark) {
      const sr = spark as SparkRenderer;
      // Force synchronous update by setting needsUpdate and preUpdate
      sr.needsUpdate = true;
      const savedPreUpdate = sr.preUpdate;
      sr.preUpdate = true;
      sr.update({ scene, viewToWorld: sr.defaultView.viewToWorld });
      sr.preUpdate = savedPreUpdate;
    }
  }

  /**
   * Decode first frame without starting playback.
   * Call after loadVideo() to show initial frame.
   */
  decodeFirstFrame(renderer: THREE.WebGLRenderer) {
    if (!this.canvasTexture || !this.tileUVs) return;

    // Update splat count for first frame
    const frameCount = this.getFrameSplatCount(0);
    this.packedSplats.updateVideoSplatCount(frameCount);
    this.numSplats = frameCount;

    renderer.initTexture(this.canvasTexture);
    this.packedSplats.updateFromVideoTextureGPU(
      renderer,
      this.canvasTexture,
      this.tileUVs,
      this.videoWidth,
      this.videoHeight,
    );

    // Flush GPU to ensure decode completes
    const gl = renderer.getContext() as WebGL2RenderingContext;
    gl.flush();

    this.updateVersion();
    this.triggerImmediateRegeneration(renderer);
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

  seekToFrame(frame: number, renderer?: THREE.WebGLRenderer) {
    this.currentFrameIndex = Math.max(0, Math.min(frame, this.totalFrames - 1));
    this.drawFrame(this.currentFrameIndex);

    // Update splat count for this frame
    const frameCount = this.getFrameSplatCount(this.currentFrameIndex);
    this.packedSplats.updateVideoSplatCount(frameCount);
    this.numSplats = frameCount;

    if (this.canvasTexture && this.tileUVs) {
      this.canvasTexture.needsUpdate = true;

      // GPU decode the frame if renderer is provided
      if (renderer) {
        renderer.initTexture(this.canvasTexture);
        this.packedSplats.updateFromVideoTextureGPU(
          renderer,
          this.canvasTexture,
          this.tileUVs,
          this.videoWidth,
          this.videoHeight,
        );

        // Flush GPU to ensure decode completes
        const gl = renderer.getContext() as WebGL2RenderingContext;
        gl.flush();

        this.updateVersion();
        this.triggerImmediateRegeneration(renderer);
      }
    }

    // Notify callback
    if (this.onFrameChange) {
      this.onFrameChange(this.currentFrameIndex, this.totalFrames);
    }
  }

  getTotalFrames(): number {
    return this.totalFrames;
  }

  getFPS(): number {
    return this.fps;
  }

  dispose() {
    super.dispose();
    if (this.canvasTexture) {
      this.canvasTexture.dispose();
      this.canvasTexture = null;
    }
    for (const bitmap of this.frameData) {
      bitmap.close();
    }
    this.frameData = [];
  }
}
