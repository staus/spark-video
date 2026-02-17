import { Delta4DGSMetadata } from './DeltaSplatDecoder';
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
     */
    loadDelta(webpBlob: Blob, metadata: Delta4DGSMetadata): Promise<{
        loadTime: number;
    }>;
    /**
     * Decode first frame without starting playback.
     */
    decodeFirstFrame(renderer: THREE.WebGLRenderer): void;
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
    getFPS(): number;
    private quatTransformMode;
    /**
     * Set quaternion transform mode for debugging orientation issues.
     * The transform is applied in the render shader for instant updates.
     */
    setQuatTransformMode(mode: number): void;
    getQuatTransformMode(): number;
    setMaxScaleFilter(_maxScale: number): void;
    getMaxScaleFilter(): number;
    setStaticVizMode(_enabled: boolean): void;
    getStaticVizMode(): boolean;
    setStaticThreshold(_threshold: number): void;
    getStaticThreshold(): number;
    getTScaleRange(): [number, number];
    dispose(): void;
    static getQuatTransformName(mode: number): string;
    static getQuatTransformCount(): number;
}
