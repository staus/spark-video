import * as THREE from "three";
import { type Delta4DGSMetadata, DeltaSplatDecoder } from "./DeltaSplatDecoder";
import type { DeltaModeMetadata } from "./PackedSplats";
import { SparkRenderer } from "./SparkRenderer";
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
  private positionTexture: THREE.DataTexture | null = null;
  private attributeTexture: THREE.DataTexture | null = null;
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

    // Initialize GPU delta mode on PackedSplats (float positions, no CPU encoding)
    const deltaMetadata: DeltaModeMetadata = {
      maxCount: metadata["4dgs"].max_active_gaussians,
      scaleCodebook: metadata.sog.scales.codebook,
      sh0Codebook: metadata.sog.sh0.codebook,
    };
    this.packedSplats.initDeltaModeGPU(deltaMetadata);

    // Create GPU textures for position and attributes
    const { positionSize, attributeWidth, attributeHeight } =
      this.decoder.getGPUTextureDimensions();

    // Position texture: RGBA32F (4 floats per texel, alpha unused)
    const posData = new Float32Array(positionSize * positionSize * 4);
    this.positionTexture = new THREE.DataTexture(
      posData,
      positionSize,
      positionSize,
      THREE.RGBAFormat,
      THREE.FloatType,
    );
    this.positionTexture.minFilter = THREE.NearestFilter;
    this.positionTexture.magFilter = THREE.NearestFilter;
    this.positionTexture.generateMipmaps = false;
    this.positionTexture.colorSpace = THREE.NoColorSpace;

    // Attribute texture: RGBA8, 3 planes stacked vertically (quats, scales, sh0)
    // Each plane is positionSize x positionSize
    // CRITICAL: NoColorSpace prevents sRGB conversion which would corrupt codebook indices
    const attrData = new Uint8Array(attributeWidth * attributeHeight * 4);
    this.attributeTexture = new THREE.DataTexture(
      attrData,
      attributeWidth,
      attributeHeight,
      THREE.RGBAFormat,
      THREE.UnsignedByteType,
    );
    this.attributeTexture.minFilter = THREE.NearestFilter;
    this.attributeTexture.magFilter = THREE.NearestFilter;
    this.attributeTexture.generateMipmaps = false;
    this.attributeTexture.colorSpace = THREE.NoColorSpace;

    this.frameInterval = 1000 / metadata.video.fps;

    const loadTime = performance.now() - loadStart;

    console.log(
      `DeltaSplatMesh loaded: ${this.decoder.getTotalFrames()} frames @ ${metadata.video.fps}fps (GPU float mode)`,
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
    const { positions, attributes, count } = this.decoder.processFrameGPU(0);

    this._uploadFrameGPU(renderer, positions, attributes, count);
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

    const { positions, attributes, count } =
      this.decoder.processFrameGPU(frame);
    this.currentFrameIndex = frame;

    this._uploadFrameGPU(renderer, positions, attributes, count);
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
      !this.positionTexture ||
      !this.attributeTexture
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

    // Process delta frame (GPU mode - no CPU encoding)
    const { positions, attributes, count } = this.decoder.processFrameGPU(
      this.currentFrameIndex,
    );

    this._uploadFrameGPU(renderer, positions, attributes, count);

    return true;
  }

  private _uploadFrameGPU(
    renderer: THREE.WebGLRenderer,
    positions: Float32Array,
    attributes: Uint8Array,
    count: number,
  ): void {
    if (!this.positionTexture || !this.attributeTexture) return;

    // Update position texture data
    const posImageData = this.positionTexture.image as {
      data: Float32Array;
      width: number;
      height: number;
    };
    posImageData.data.set(positions);
    this.positionTexture.needsUpdate = true;

    // Update attribute texture data
    const attrImageData = this.attributeTexture.image as {
      data: Uint8Array;
      width: number;
      height: number;
    };
    attrImageData.data.set(attributes);
    this.attributeTexture.needsUpdate = true;

    // GPU decode (float positions, no signed-log encoding)
    this.numSplats = count;

    this.packedSplats.updateFromDeltaTextureGPU(
      renderer,
      this.positionTexture,
      this.attributeTexture,
      count,
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

  // Current quaternion transform mode
  private quatTransformMode = 0;

  /**
   * Set quaternion transform mode for debugging orientation issues.
   * The transform is applied in the render shader for instant updates.
   */
  setQuatTransformMode(mode: number): void {
    this.quatTransformMode = mode;

    // Find the SparkRenderer in the scene and update its uniform
    let scene: THREE.Object3D | null = this.parent;
    while (scene && !(scene instanceof THREE.Scene)) {
      scene = scene.parent;
    }
    if (scene) {
      scene.traverse((obj) => {
        if (obj instanceof SparkRenderer) {
          // biome-ignore lint/suspicious/noExplicitAny: accessing uniforms
          const uniforms = (obj as any).uniforms;
          if (uniforms?.quatTransformMode) {
            uniforms.quatTransformMode.value = mode;
          }
        }
      });
    }
  }

  getQuatTransformMode(): number {
    return this.quatTransformMode;
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
    if (this.positionTexture) {
      this.positionTexture.dispose();
      this.positionTexture = null;
    }
    if (this.attributeTexture) {
      this.attributeTexture.dispose();
      this.attributeTexture = null;
    }
    this.packedSplats.disposeDeltaModeGPU();
    super.dispose();
  }

  static getQuatTransformName(mode: number): string {
    return DeltaSplatMesh.QUAT_TRANSFORM_NAMES[mode] ?? `unknown(${mode})`;
  }

  static getQuatTransformCount(): number {
    return DeltaSplatMesh.QUAT_TRANSFORM_NAMES.length;
  }
}
