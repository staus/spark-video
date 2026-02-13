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
    this.numSplats = sparkMetadata.count;

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

    // Trigger SparkRenderer regeneration
    this.needsUpdate = true;

    // Notify callback
    if (this.onFrameChange) {
      this.onFrameChange(this.currentFrameIndex, this.totalFrames);
    }

    return true;
  }

  /**
   * Decode first frame without starting playback.
   * Call after loadVideo() to show initial frame.
   */
  decodeFirstFrame(renderer: THREE.WebGLRenderer) {
    if (!this.canvasTexture || !this.tileUVs) return;

    renderer.initTexture(this.canvasTexture);
    this.packedSplats.updateFromVideoTextureGPU(
      renderer,
      this.canvasTexture,
      this.tileUVs,
      this.videoWidth,
      this.videoHeight,
    );
    this.needsUpdate = true;
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
        this.needsUpdate = true;
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
