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
        source_duration?: number;
        source_fps?: number;
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
    };
    "4dgs"?: {
        frame_gaussian_counts?: number[];
        frame_times?: number[];
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
    private frameTextureA;
    private frameTextureB;
    private glTextureA;
    private glTextureB;
    private currentFrameA;
    private currentFrameB;
    private tileUVs;
    private frameGaussianCounts;
    private staticCount;
    private validationMetadata;
    currentFrameIndex: number;
    isPlaying: boolean;
    private lastFrameTime;
    private accumulatedTime;
    interpolationEnabled: boolean;
    private sourceDuration;
    private frameTimes;
    private currentPlaybackTime;
    interpAlpha: number;
    onFrameChange: ((frameIndex: number, totalFrames: number) => void) | null;
    onInterpolationUpdate: ((alpha: number, frameA: number, frameB: number, time: number) => void) | null;
    onTimeChange: ((currentTime: number, totalDuration: number) => void) | null;
    constructor(options?: SplatMeshOptions);
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
    /**
     * Upload frame to GPU using raw WebGL, bypassing THREE.js color management.
     * Guarantees no color space conversion, no alpha premultiplication.
     */
    private uploadFrameRawWebGL;
    private lastLoggedFrame;
    private frameDecodeCount;
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
     */
    decodeFirstFrame(renderer: THREE.WebGLRenderer): void;
    play(): void;
    pause(): void;
    toggle(): void;
    seekToFrame(frame: number, renderer: THREE.WebGLRenderer): void;
    getTotalFrames(): number;
    getFPS(): number;
    getSourceDuration(): number;
    getPlaybackTime(): number;
    /**
     * Upload a frame to texture A for dual-frame interpolation.
     */
    private uploadFrameToTextureA;
    /**
     * Upload a frame to texture B for dual-frame interpolation.
     */
    private uploadFrameToTextureB;
    /**
     * Interpolated tick - advances playback time and decodes interpolated frames.
     * Uses source_duration for correct timing instead of frame-based stepping.
     * Returns true if the display was updated.
     */
    tickInterpolated(renderer: THREE.WebGLRenderer, now?: number): boolean;
    /**
     * Seek to a specific time in seconds.
     */
    seekToTime(timeSeconds: number, renderer: THREE.WebGLRenderer): void;
    /**
     * Validate the decode pipeline by reading back raw pixels and decoded splat data.
     * Traces through ALL shader math step-by-step with actual codebook values.
     * Call this from browser console: videoMesh.validateDecode(renderer)
     */
    validateDecode(renderer: THREE.WebGLRenderer, splatIndex?: number): Promise<void>;
    dispose(): void;
}
