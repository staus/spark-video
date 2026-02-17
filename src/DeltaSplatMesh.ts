import * as THREE from "three";
import { type Delta4DGSMetadata, DeltaSplatDecoder } from "./DeltaSplatDecoder";
import type { GPUVideoTileUVs, SOGVideoMetadata } from "./PackedSplats";
import { SplatMesh, type SplatMeshOptions } from "./SplatMesh";

/**
 * DeltaSplatMesh - Playback class for delta-encoded 4DGS video.
 * Extends SplatMesh and uses DeltaSplatDecoder to reconstruct frames.
 *
 * Usage:
 *   const deltaMesh = new DeltaSplatMesh();
 *   await deltaMesh.loadDelta(webpBlob, jsonMetadata);
 *   scene.add(deltaMesh);
 *
 *   // In render loop:
 *   deltaMesh.tick(renderer, performance.now());
 *   renderer.render(scene, camera);
 *
 *   // Playback control:
 *   deltaMesh.play();
 *   deltaMesh.pause();
 */
export class DeltaSplatMesh extends SplatMesh {
  // Quaternion transform names for debugging UI
  static readonly QUAT_TRANSFORM_NAMES = [
    "identity",
    "rotX+90",
    "rotX-90",
    "rotX+180",
    "rotY+90",
    "rotY-90",
    "rotY+180",
    "rotZ+90",
    "rotZ-90",
    "rotZ+180",
    "swapXY",
    "swapXZ",
    "swapYZ",
    "negX",
    "negY",
    "negZ",
    "negW",
    "negXY",
    "negXZ",
    "negYZ",
    "rotX90+swapYZ",
    "rotX-90+swapYZ",
    "conjugate",
    "blenderZ→Y (v1)",
    "blenderZ→Y (v2)",
    "rotX90+negZ",
    "rotX-90+negZ",
    "rotX90+negY",
    "rotX-90+negY",
    "WXYZ→XYZW",
    "XYZW→WXYZ",
    "cycleYZWX",
  ];
  private decoder: DeltaSplatDecoder | null = null;
  private frameTexture: THREE.DataTexture | null = null;
  private tileUVs: GPUVideoTileUVs | null = null;
  private metadata: Delta4DGSMetadata | null = null;

  currentFrameIndex = 0;
  isPlaying = false;
  private lastFrameTime = 0;
  private frameInterval = 1000 / 30;

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
   * Load a delta-encoded animated WebP video with JSON metadata
   */
  async loadDelta(
    webpBlob: Blob,
    metadata: Delta4DGSMetadata,
  ): Promise<{ loadTime: number }> {
    const loadStart = performance.now();

    this.metadata = metadata;

    // Create decoder
    this.decoder = new DeltaSplatDecoder(metadata);
    await this.decoder.loadDeltaFrames(webpBlob);

    const tileSize = metadata.tile_size;
    const { width: sogWidth, height: sogHeight } =
      this.decoder.getSOGDimensions();

    // Initialize GPU video mode on PackedSplats
    const sparkMetadata: SOGVideoMetadata = {
      count: metadata["4dgs"].max_active_gaussians,
      mins: metadata.sog.means.mins,
      maxs: metadata.sog.means.maxs,
      scaleCodebook: metadata.sog.scales.codebook,
      sh0Codebook: metadata.sog.sh0.codebook,
    };
    this.packedSplats.initVideoModeGPU(sparkMetadata, tileSize);

    // Calculate 3x2 tile UVs for reconstructed SOG
    this.tileUVs = {
      means_l: {
        u0: 0,
        v0: 0,
        u1: tileSize / sogWidth,
        v1: tileSize / sogHeight,
      },
      means_u: {
        u0: tileSize / sogWidth,
        v0: 0,
        u1: (2 * tileSize) / sogWidth,
        v1: tileSize / sogHeight,
      },
      quats: {
        u0: (2 * tileSize) / sogWidth,
        v0: 0,
        u1: 1,
        v1: tileSize / sogHeight,
      },
      scales: {
        u0: 0,
        v0: tileSize / sogHeight,
        u1: tileSize / sogWidth,
        v1: 1,
      },
      sh0: {
        u0: tileSize / sogWidth,
        v0: tileSize / sogHeight,
        u1: (2 * tileSize) / sogWidth,
        v1: 1,
      },
    };

    // Create reusable DataTexture
    const texData = new Uint8Array(sogWidth * sogHeight * 4);
    this.frameTexture = new THREE.DataTexture(texData, sogWidth, sogHeight);
    this.frameTexture.format = THREE.RGBAFormat;
    this.frameTexture.type = THREE.UnsignedByteType;
    this.frameTexture.minFilter = THREE.NearestFilter;
    this.frameTexture.magFilter = THREE.NearestFilter;
    this.frameTexture.generateMipmaps = false;
    this.frameTexture.colorSpace = THREE.LinearSRGBColorSpace;

    this.frameInterval = 1000 / metadata.video.fps;

    const loadTime = performance.now() - loadStart;

    console.log(
      `DeltaSplatMesh loaded: ${this.decoder.getTotalFrames()} frames @ ${metadata.video.fps}fps`,
    );

    return { loadTime };
  }

  /**
   * Decode first frame without starting playback.
   */
  decodeFirstFrame(renderer: THREE.WebGLRenderer): void {
    if (!this.decoder) return;

    this.decoder.reset();
    this.currentFrameIndex = 0;
    const { data, count } = this.decoder.processFrame(0);

    this._uploadFrame(renderer, data, count);
  }

  /**
   * Seek to a specific frame.
   * Note: Delta decoding requires sequential processing.
   * If seeking backwards, resets and processes from beginning.
   */
  seekToFrame(frame: number, renderer: THREE.WebGLRenderer): void {
    if (!this.decoder) return;

    // Delta decoding requires sequential processing
    // If seeking backwards, reset and process from beginning
    if (frame < this.decoder.currentFrameIndex) {
      this.decoder.reset();
    }

    const { data, count } = this.decoder.processFrame(frame);
    this.currentFrameIndex = frame;

    this._uploadFrame(renderer, data, count);
  }

  /**
   * Call each frame from the render loop.
   * Returns true if a new frame was decoded.
   */
  tick(
    renderer: THREE.WebGLRenderer,
    now: number = performance.now(),
  ): boolean {
    if (
      !this.isPlaying ||
      !this.decoder ||
      !this.frameTexture ||
      !this.tileUVs
    ) {
      return false;
    }

    // Frame timing
    if (this.lastFrameTime === 0) this.lastFrameTime = now;
    if (now - this.lastFrameTime < this.frameInterval) return false;
    this.lastFrameTime = now;

    // Advance frame
    this.currentFrameIndex =
      (this.currentFrameIndex + 1) % this.decoder.getTotalFrames();

    // Process delta frame
    const { data, count } = this.decoder.processFrame(this.currentFrameIndex);

    this._uploadFrame(renderer, data, count);

    return true;
  }

  private _uploadFrame(
    renderer: THREE.WebGLRenderer,
    data: Uint8Array,
    count: number,
  ): void {
    if (!this.frameTexture || !this.tileUVs) return;

    // Update texture data
    const imageData = this.frameTexture.image as {
      data: Uint8Array;
      width: number;
      height: number;
    };
    imageData.data.set(data);
    this.frameTexture.needsUpdate = true;

    // GPU decode
    this.packedSplats.updateVideoSplatCount(count);
    this.numSplats = count;

    this.packedSplats.updateFromVideoTextureGPU(
      renderer,
      this.frameTexture,
      this.tileUVs,
      imageData.width,
      imageData.height,
    );

    this.updateVersion();

    this.onFrameChange?.(this.currentFrameIndex, this.getTotalFrames());
  }

  play(): void {
    this.isPlaying = true;
    this.lastFrameTime = 0;
  }

  pause(): void {
    this.isPlaying = false;
  }

  toggle(): void {
    if (this.isPlaying) {
      this.pause();
    } else {
      this.play();
    }
  }

  getTotalFrames(): number {
    return this.decoder?.getTotalFrames() || 0;
  }

  getFPS(): number {
    return this.metadata?.video?.fps || 30;
  }

  // Stub methods for API compatibility with VideoSplatMesh
  // Delta encoding applies transforms at encode time

  setQuatTransformMode(_mode: number): void {
    // Delta encoding applies quat transform at encode time
    // GPU decode should use identity (mode 0)
    console.log("Delta encoding: quat transform applied at encode time");
  }

  getQuatTransformMode(): number {
    return 0;
  }

  setMaxScaleFilter(_maxScale: number): void {
    // Not implemented for delta encoding
  }

  getMaxScaleFilter(): number {
    return 0;
  }

  setStaticVizMode(_enabled: boolean): void {
    // Not applicable to delta encoding (no t_scale)
  }

  getStaticVizMode(): boolean {
    return false;
  }

  setStaticThreshold(_threshold: number): void {
    // Not applicable to delta encoding
  }

  getStaticThreshold(): number {
    return 0;
  }

  getTScaleRange(): [number, number] {
    return [0, 1];
  }

  dispose(): void {
    if (this.frameTexture) {
      this.frameTexture.dispose();
      this.frameTexture = null;
    }
    super.dispose();
  }

  static getQuatTransformName(mode: number): string {
    return DeltaSplatMesh.QUAT_TRANSFORM_NAMES[mode] ?? `unknown(${mode})`;
  }

  static getQuatTransformCount(): number {
    return DeltaSplatMesh.QUAT_TRANSFORM_NAMES.length;
  }
}
