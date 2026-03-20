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
  // Dynamic textures (uploaded each frame)
  private positionTexture: THREE.DataTexture | null = null;
  private attributeTexture: THREE.DataTexture | null = null;
  // Static textures (uploaded once after first frame)
  private staticPositionTexture: THREE.DataTexture | null = null;
  private staticAttributeTexture: THREE.DataTexture | null = null;
  private staticTexturesUploaded = false;
  private staticCount = 0;

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
   * @param webpBlob Blob containing the animated WebP video
   * @param metadata Parsed JSON metadata
   * @param baseUrl Optional base URL for loading keyframe PNG files (if any)
   */
  async loadDelta(
    webpBlob: Blob,
    metadata: Delta4DGSMetadata,
    baseUrl?: string,
  ): Promise<{ loadTime: number }> {
    const loadStart = performance.now();

    this.metadata = metadata;

    // Create decoder
    this.decoder = new DeltaSplatDecoder(metadata);
    await this.decoder.loadDeltaFrames(webpBlob);

    // Load keyframes if present and base URL provided
    if (metadata.keyframes && metadata.keyframes.length > 0) {
      if (baseUrl) {
        await this.decoder.loadKeyframes(baseUrl);
      } else {
        console.warn(
          `Metadata has ${metadata.keyframes.length} keyframe(s) but no baseUrl provided - keyframes won't be loaded`,
        );
      }
    }

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
   * Also initializes and uploads static textures (keyframe gaussians with zero motion).
   */
  decodeFirstFrame(renderer: THREE.WebGLRenderer): void {
    if (!this.decoder) return;

    this.decoder.reset();
    this.currentFrameIndex = 0;
    const { positions, attributes, count } = this.decoder.processFrameGPU(0);

    // Initialize static buffers after first frame (classifies keyframe births)
    this.decoder.initStaticBuffers();
    this._initStaticTextures(renderer);

    this._uploadFrameGPU(renderer, positions, attributes, count);
  }

  /**
   * Initialize static textures for keyframe gaussians with zero motion.
   * These are uploaded once and never updated.
   */
  private _initStaticTextures(renderer: THREE.WebGLRenderer): void {
    if (this.staticTexturesUploaded || !this.decoder) return;

    const staticData = this.decoder.getStaticData();
    if (!staticData) {
      console.log("No static gaussians - all gaussians are dynamic");
      this.staticTexturesUploaded = true;
      return;
    }

    const { positions, attributes, count, textureSize } = staticData;
    this.staticCount = count;

    // Create static position texture
    this.staticPositionTexture = new THREE.DataTexture(
      positions,
      textureSize,
      textureSize,
      THREE.RGBAFormat,
      THREE.FloatType,
    );
    this.staticPositionTexture.minFilter = THREE.NearestFilter;
    this.staticPositionTexture.magFilter = THREE.NearestFilter;
    this.staticPositionTexture.generateMipmaps = false;
    this.staticPositionTexture.colorSpace = THREE.NoColorSpace;
    this.staticPositionTexture.needsUpdate = true;

    // Create static attribute texture
    this.staticAttributeTexture = new THREE.DataTexture(
      attributes,
      textureSize,
      textureSize * 3,
      THREE.RGBAFormat,
      THREE.UnsignedByteType,
    );
    this.staticAttributeTexture.minFilter = THREE.NearestFilter;
    this.staticAttributeTexture.magFilter = THREE.NearestFilter;
    this.staticAttributeTexture.generateMipmaps = false;
    this.staticAttributeTexture.colorSpace = THREE.NoColorSpace;
    this.staticAttributeTexture.needsUpdate = true;

    // Upload static textures to GPU (one-time upload)
    this.packedSplats.setStaticDeltaTextures(
      renderer,
      this.staticPositionTexture,
      this.staticAttributeTexture,
      count,
    );

    this.staticTexturesUploaded = true;

    const staticKB = (
      (positions.byteLength + attributes.byteLength) /
      1024
    ).toFixed(1);
    console.log(
      `Static textures uploaded: ${count} gaussians, ${textureSize}x${textureSize}, ${staticKB} KB`,
    );
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

    // Frame timing - use fixed-interval advancement to prevent drift
    if (this.lastFrameTime === 0) this.lastFrameTime = now;
    if (now - this.lastFrameTime < this.frameInterval) return false;

    const totalFrames = this.decoder.getTotalFrames();

    // Calculate how many frames we need to process to catch up
    const framesBehind = Math.floor(
      (now - this.lastFrameTime) / this.frameInterval,
    );

    // Process up to 3 frames per tick when catching up
    // CRITICAL: We upload EACH frame to ensure all gaussians are displayed
    // (not just the final state). Short-lived gaussians would otherwise
    // be born and die within the catch-up loop without ever being visible.
    const framesToProcess = Math.min(framesBehind, 3);

    for (let i = 0; i < framesToProcess; i++) {
      this.lastFrameTime += this.frameInterval;
      this.currentFrameIndex = (this.currentFrameIndex + 1) % totalFrames;

      // Process delta frame (GPU mode - no CPU encoding)
      const { positions, attributes, count } = this.decoder.processFrameGPU(
        this.currentFrameIndex,
      );

      // Upload each frame - ensures all gaussians get displayed
      this._uploadFrameGPU(renderer, positions, attributes, count);
    }

    return framesToProcess > 0;
  }

  private _uploadFrameGPU(
    renderer: THREE.WebGLRenderer,
    positions: Float32Array,
    attributes: Uint8Array,
    dynamicCount: number,
  ): void {
    if (!this.positionTexture || !this.attributeTexture) return;

    // Update position texture data (dynamic gaussians only)
    const posImageData = this.positionTexture.image as {
      data: Float32Array;
      width: number;
      height: number;
    };
    posImageData.data.set(positions);
    this.positionTexture.needsUpdate = true;

    // Update attribute texture data (dynamic gaussians only)
    const attrImageData = this.attributeTexture.image as {
      data: Uint8Array;
      width: number;
      height: number;
    };
    attrImageData.data.set(attributes);
    this.attributeTexture.needsUpdate = true;

    // Total count = static + dynamic
    const totalCount = this.staticCount + dynamicCount;
    this.numSplats = totalCount;

    // GPU decode (float positions, no signed-log encoding)
    // Passes static count so shader knows where dynamic data starts
    this.packedSplats.updateFromDeltaTextureGPU(
      renderer,
      this.positionTexture,
      this.attributeTexture,
      dynamicCount,
      this.staticCount,
    );

    // Log upload stats (every 30 frames to avoid spam)
    if (this.currentFrameIndex % 30 === 0) {
      // Dynamic data actually used (not full texture size)
      const dynamicPosBytes = dynamicCount * 16; // 4 floats × 4 bytes
      const dynamicAttrBytes = dynamicCount * 12; // 3 planes × 4 bytes
      const dynamicKB = ((dynamicPosBytes + dynamicAttrBytes) / 1024).toFixed(
        1,
      );
      // Full texture upload size (what GPU actually receives)
      const texPosBytes = this.positionTexture.image.width ** 2 * 16;
      const texAttrBytes =
        this.attributeTexture.image.width *
        this.attributeTexture.image.height *
        4;
      const texKB = ((texPosBytes + texAttrBytes) / 1024).toFixed(1);
      console.log(
        `Frame ${this.currentFrameIndex}: ${dynamicCount} dynamic + ${this.staticCount} static = ${totalCount} total ` +
          `(${dynamicKB} KB data, ${texKB} KB texture)`,
      );
    }

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

  /**
   * Get static/dynamic gaussian counts for performance monitoring
   */
  getGaussianCounts(): { static: number; dynamic: number; total: number } {
    const dynamicCount = this.numSplats - this.staticCount;
    return {
      static: this.staticCount,
      dynamic: dynamicCount,
      total: this.numSplats,
    };
  }

  /**
   * Get the underlying decoder for direct access (e.g., loading keyframes from File objects)
   */
  getDecoder(): DeltaSplatDecoder {
    if (!this.decoder) {
      throw new Error(
        "Decoder not initialized - call loadDelta() first or create decoder manually",
      );
    }
    return this.decoder;
  }

  /**
   * Create decoder without loading frames (for manual loading flow)
   */
  createDecoder(metadata: Delta4DGSMetadata): DeltaSplatDecoder {
    this.metadata = metadata;
    this.decoder = new DeltaSplatDecoder(metadata);
    return this.decoder;
  }

  /**
   * Initialize GPU mode after frames are loaded (for manual loading flow)
   */
  initGPUMode(metadata: Delta4DGSMetadata): void {
    if (!this.decoder) {
      throw new Error("Decoder not created - call createDecoder() first");
    }

    // Initialize GPU delta mode on PackedSplats
    const deltaMetadata: DeltaModeMetadata = {
      maxCount: metadata["4dgs"].max_active_gaussians,
      scaleCodebook: metadata.sog.scales.codebook,
      sh0Codebook: metadata.sog.sh0.codebook,
    };
    this.packedSplats.initDeltaModeGPU(deltaMetadata);

    // Create GPU textures
    const { positionSize, attributeWidth, attributeHeight } =
      this.decoder.getGPUTextureDimensions();

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

    console.log(
      `DeltaSplatMesh GPU mode initialized: ${this.decoder.getTotalFrames()} frames @ ${metadata.video.fps}fps`,
    );
  }

  getFPS(): number {
    return this.metadata?.video?.fps || 30;
  }

  // Current quaternion transform mode
  private quatTransformMode = 0;
  private maxScaleFilter = 0;

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

  setMaxScaleFilter(maxScale: number): void {
    this.maxScaleFilter = maxScale;
    this.packedSplats.setDeltaMaxScaleFilter(maxScale);
  }

  getMaxScaleFilter(): number {
    return this.maxScaleFilter ?? 0;
  }

  /**
   * Re-run the GPU decode pass on existing texture data.
   * Use after updating uniforms (e.g. maxScaleFilter) to apply changes
   * without re-processing delta frames.
   */
  redecodeGPU(renderer: THREE.WebGLRenderer): void {
    if (!this.positionTexture || !this.attributeTexture) return;

    this.positionTexture.needsUpdate = true;
    this.attributeTexture.needsUpdate = true;

    // Dynamic count = total - static
    const dynamicCount = this.numSplats - this.staticCount;

    this.packedSplats.updateFromDeltaTextureGPU(
      renderer,
      this.positionTexture,
      this.attributeTexture,
      dynamicCount,
      this.staticCount,
    );

    this.updateVersion();
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
    if (this.staticPositionTexture) {
      this.staticPositionTexture.dispose();
      this.staticPositionTexture = null;
    }
    if (this.staticAttributeTexture) {
      this.staticAttributeTexture.dispose();
      this.staticAttributeTexture = null;
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
