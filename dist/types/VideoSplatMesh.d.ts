import { SplatMesh, SplatMeshOptions } from './SplatMesh';
import * as THREE from "three";
/**
 * Metadata format for 4DGS video files (JSON sidecar)
 */
export interface Video4DGSMetadata {
    tile_size: number;
    layout: Record<string, [number, number]>;
    video: {
        frames: number;
        fps: number;
    };
    sog: {
        count: number;
        means?: {
            mins: [number, number, number];
            maxs: [number, number, number];
        };
        scales: {
            codebook: number[];
        };
        sh0: {
            codebook: number[];
        };
        t_scale?: {
            codebook: number[];
        };
    };
    "4dgs"?: {
        frame_gaussian_counts?: number[];
        t_scale_range?: [number, number];
        static_threshold?: number;
        static_frame_index?: number;
        dynamic_frame_start?: number;
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
export declare class VideoSplatMesh extends SplatMesh {
    private frameData;
    private totalFrames;
    private fps;
    private frameInterval;
    private videoWidth;
    private videoHeight;
    private frameTexture;
    private tileUVs;
    private frameGaussianCounts;
    private staticCount;
    private validationMetadata;
    currentFrameIndex: number;
    isPlaying: boolean;
    private lastFrameTime;
    private accumulatedTime;
    onFrameChange: ((frameIndex: number, totalFrames: number) => void) | null;
    private quatTransformMode;
    private maxScaleFilter;
    private staticVizMode;
    private staticThreshold;
    private tScaleRange;
    private hasStaticDynamicSplit;
    private staticFrameIndex;
    private dynamicFrameStart;
    private staticFrameTexture;
    private staticGaussianCount;
    static readonly QUAT_TRANSFORM_NAMES: string[];
    constructor(options?: SplatMeshOptions);
    /**
     * Set the quaternion transform mode for debugging coordinate system issues.
     * The transform is applied to each gaussian's rotation quaternion after decoding.
     */
    setQuatTransformMode(mode: number): void;
    /**
     * Get the current quaternion transform mode.
     */
    getQuatTransformMode(): number;
    /**
     * Get the name of a quaternion transform mode.
     */
    static getQuatTransformName(mode: number): string;
    /**
     * Get the total number of quaternion transform modes.
     */
    static getQuatTransformCount(): number;
    /**
     * Set the max scale filter. Gaussians with any axis larger than this will be hidden.
     * Set to 0 to disable filtering.
     */
    setMaxScaleFilter(maxScale: number): void;
    /**
     * Get the current max scale filter value.
     */
    getMaxScaleFilter(): number;
    /**
     * Enable/disable static visualization mode.
     * When enabled, static gaussians (t_scale >= threshold) are rendered in green.
     */
    setStaticVizMode(enabled: boolean): void;
    /**
     * Get whether static visualization mode is enabled.
     */
    getStaticVizMode(): boolean;
    /**
     * Set the t_scale threshold for static/dynamic classification.
     * Gaussians with t_scale >= threshold are considered static.
     */
    setStaticThreshold(threshold: number): void;
    /**
     * Get the current static threshold value.
     */
    getStaticThreshold(): number;
    /**
     * Get the t_scale range from metadata (for UI slider bounds).
     * Returns [min, max] t_scale values.
     */
    getTScaleRange(): [number, number];
    /**
     * Check if ImageDecoder API is available
     */
    static isSupported(): boolean;
    /**
     * Load an animated WebP video with JSON metadata
     */
    loadVideo(webpBlob: Blob, metadata: Video4DGSMetadata): Promise<{
        loadTime: number;
    }>;
    private calculateTileUVs;
    private glTexture;
    private staticGlTexture;
    /**
     * Upload frame to GPU using raw WebGL, bypassing THREE.js color management.
     * Guarantees no color space conversion, no alpha premultiplication.
     */
    private uploadFrameRawWebGL;
    private lastLoggedFrame;
    private frameDecodeCount;
    /**
     * Upload static frame to a separate texture (called once during first decode)
     */
    private uploadStaticFrameTexture;
    /**
     * Decode a frame to GPU. Single path for all frame updates.
     */
    private decodeFrame;
    /**
     * Call each frame from the render loop.
     * Returns true if a new frame was decoded.
     */
    tick(renderer: THREE.WebGLRenderer, now?: number): boolean;
    /**
     * Decode first frame without starting playback.
     * For static/dynamic split, decodes the first dynamic frame (which composites with static).
     */
    decodeFirstFrame(renderer: THREE.WebGLRenderer): void;
    play(): void;
    pause(): void;
    toggle(): void;
    seekToFrame(frame: number, renderer: THREE.WebGLRenderer): void;
    getTotalFrames(): number;
    getFPS(): number;
    /**
     * Validate the decode pipeline by reading back raw pixels and decoded splat data.
     * Traces through ALL shader math step-by-step with actual codebook values.
     * Call this from browser console: videoMesh.validateDecode(renderer)
     */
    validateDecode(renderer: THREE.WebGLRenderer, splatIndex?: number): Promise<void>;
    dispose(): void;
}
