import { Delta4DGSMetadata, DeltaSplatDecoder } from './DeltaSplatDecoder';
import { SplatMesh, SplatMeshOptions } from './SplatMesh';
import * as THREE from "three";
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
export declare class DeltaSplatMesh extends SplatMesh {
    static readonly QUAT_TRANSFORM_NAMES: string[];
    private decoder;
    private positionTexture;
    private attributeTexture;
    private staticPositionTexture;
    private staticAttributeTexture;
    private staticTexturesUploaded;
    private staticCount;
    private metadata;
    currentFrameIndex: number;
    isPlaying: boolean;
    private lastFrameTime;
    private frameInterval;
    onFrameChange: ((frameIndex: number, totalFrames: number) => void) | null;
    constructor(options?: SplatMeshOptions);
    /**
     * Check if ImageDecoder API is available
     */
    static isSupported(): boolean;
    /**
     * Load a delta-encoded animated WebP video with JSON metadata
     * @param webpBlob Blob containing the animated WebP video
     * @param metadata Parsed JSON metadata
     * @param baseUrl Optional base URL for loading keyframe PNG files (if any)
     */
    loadDelta(webpBlob: Blob, metadata: Delta4DGSMetadata, baseUrl?: string): Promise<{
        loadTime: number;
    }>;
    /**
     * Decode first frame without starting playback.
     * Also initializes and uploads static textures (keyframe gaussians with zero motion).
     */
    decodeFirstFrame(renderer: THREE.WebGLRenderer): void;
    /**
     * Initialize static textures for keyframe gaussians with zero motion.
     * These are uploaded once and never updated.
     */
    private _initStaticTextures;
    /**
     * Seek to a specific frame.
     * Note: Delta decoding requires sequential processing.
     * If seeking backwards, resets and processes from beginning.
     */
    seekToFrame(frame: number, renderer: THREE.WebGLRenderer): void;
    /**
     * Call each frame from the render loop.
     * Returns true if a new frame was decoded.
     */
    tick(renderer: THREE.WebGLRenderer, now?: number): boolean;
    private _uploadFrameGPU;
    play(): void;
    pause(): void;
    toggle(): void;
    getTotalFrames(): number;
    /**
     * Get static/dynamic gaussian counts for performance monitoring
     */
    getGaussianCounts(): {
        static: number;
        dynamic: number;
        total: number;
    };
    /**
     * Get the underlying decoder for direct access (e.g., loading keyframes from File objects)
     */
    getDecoder(): DeltaSplatDecoder;
    /**
     * Create decoder without loading frames (for manual loading flow)
     */
    createDecoder(metadata: Delta4DGSMetadata): DeltaSplatDecoder;
    /**
     * Initialize GPU mode after frames are loaded (for manual loading flow)
     */
    initGPUMode(metadata: Delta4DGSMetadata): void;
    getFPS(): number;
    private quatTransformMode;
    private maxScaleFilter;
    /**
     * Set quaternion transform mode for debugging orientation issues.
     * The transform is applied in the render shader for instant updates.
     */
    setQuatTransformMode(mode: number): void;
    getQuatTransformMode(): number;
    setMaxScaleFilter(maxScale: number): void;
    getMaxScaleFilter(): number;
    /**
     * Re-run the GPU decode pass on existing texture data.
     * Use after updating uniforms (e.g. maxScaleFilter) to apply changes
     * without re-processing delta frames.
     */
    redecodeGPU(renderer: THREE.WebGLRenderer): void;
    setStaticVizMode(_enabled: boolean): void;
    getStaticVizMode(): boolean;
    setStaticThreshold(_threshold: number): void;
    getStaticThreshold(): number;
    getTScaleRange(): [number, number];
    dispose(): void;
    static getQuatTransformName(mode: number): string;
    static getQuatTransformCount(): number;
}
