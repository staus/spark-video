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
export declare class VideoSplatMesh extends SplatMesh {
    private frameData;
    private totalFrames;
    private fps;
    private frameInterval;
    private videoWidth;
    private videoHeight;
    private canvas;
    private ctx;
    private canvasTexture;
    private tileUVs;
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
    private drawFrame;
    private createTexture;
    /**
     * Call each frame from the render loop.
     * Returns true if a new frame was decoded.
     */
    tick(renderer: THREE.WebGLRenderer, now?: number): boolean;
    /**
     * Decode first frame without starting playback.
     * Call after loadVideo() to show initial frame.
     */
    decodeFirstFrame(renderer: THREE.WebGLRenderer): void;
    play(): void;
    pause(): void;
    toggle(): void;
    seekToFrame(frame: number): void;
    getTotalFrames(): number;
    getFPS(): number;
    dispose(): void;
}
