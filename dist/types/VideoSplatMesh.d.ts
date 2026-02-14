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
    currentFrameIndex: number;
    isPlaying: boolean;
    private lastFrameTime;
    private accumulatedTime;
    onFrameChange: ((frameIndex: number, totalFrames: number) => void) | null;
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
    /**
     * Update texture with a specific frame's ImageBitmap
     * Uses THREE.Texture directly from ImageBitmap to avoid canvas color conversion
     */
    private updateFrameTexture;
    /**
     * Get the gaussian count for a specific frame
     */
    private getFrameSplatCount;
    /**
     * Call each frame from the render loop.
     * Returns true if a new frame was decoded.
     */
    tick(renderer: THREE.WebGLRenderer, now?: number): boolean;
    /**
     * Find SparkRenderer in scene and trigger immediate regeneration
     */
    private triggerImmediateRegeneration;
    /**
     * Decode first frame without starting playback.
     * Call after loadVideo() to show initial frame.
     */
    decodeFirstFrame(renderer: THREE.WebGLRenderer): void;
    play(): void;
    pause(): void;
    toggle(): void;
    seekToFrame(frame: number, renderer?: THREE.WebGLRenderer): void;
    getTotalFrames(): number;
    getFPS(): number;
    dispose(): void;
}
