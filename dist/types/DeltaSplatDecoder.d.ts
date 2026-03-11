/**
 * Delta-encoded 4DGS video decoder.
 *
 * Each frame contains only newly-born gaussians with motion vectors and lifetimes.
 * The decoder accumulates births, interpolates positions, and reconstructs full SOG textures.
 *
 * Uses canvas 2D with careful color space handling to read raw pixel data.
 */
/**
 * Keyframe metadata - frames extracted separately from the video stream
 */
export interface KeyframeInfo {
    frame_index: number;
    path: string;
    tile_size: number;
    birth_count: number;
}
/**
 * Metadata format for delta-encoded 4DGS video files (JSON sidecar)
 */
export interface Delta4DGSMetadata {
    tile_size: number;
    grid: [number, number];
    layout: Record<string, [number, number]>;
    video: {
        frames: number;
        fps: number;
        start_frame?: number;
        frame_map?: number[];
    };
    sog: {
        means: {
            mins: [number, number, number];
            maxs: [number, number, number];
        };
        motion: {
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
    "4dgs": {
        max_active_gaussians: number;
        birth_counts: number[];
        total_frames?: number;
    };
    encoding: "delta";
    keyframes?: KeyframeInfo[];
}
export declare class DeltaSplatDecoder {
    private metadata;
    private tileSize;
    private sogTileSize;
    private posMins;
    private posMaxs;
    private posRange;
    private motionMins;
    private motionMaxs;
    private motionRange;
    private maxActive;
    private activeGaussians;
    private activeIndices;
    private freeSlots;
    private activeCount;
    currentFrameIndex: number;
    private framePixelData;
    private frameWidth;
    private frameHeight;
    private keyframeData;
    private keyframeTileSizes;
    private keyframeIndices;
    private totalFrames;
    private sogWidth;
    private sogHeight;
    private sogTileData;
    private posEncodeBuf;
    private gpuPositionBuffer;
    private gpuAttributeBuffer;
    private gpuTextureSize;
    constructor(metadata: Delta4DGSMetadata);
    loadDeltaFrames(webpBlob: Blob): Promise<void>;
    /**
     * Load keyframe PNG images.
     * Call this after loadDeltaFrames() if metadata contains keyframes.
     *
     * @param baseUrl Base URL for fetching keyframe files (directory containing the JSON)
     */
    loadKeyframes(baseUrl: string): Promise<void>;
    /**
     * Get pixel value from pre-decoded frame data.
     * @param frameData Raw pixel data
     * @param x X coordinate
     * @param y Y coordinate
     * @param width Frame width (optional, defaults to this.frameWidth)
     */
    private _getPixel;
    /**
     * Process a single frame: decode births, update positions, assemble SOG texture.
     * Returns { data: Uint8Array, count: number }
     */
    processFrame(frameIndex: number): {
        data: Uint8Array;
        count: number;
    };
    /**
     * GPU-optimized frame processing: returns float positions and uint8 attributes.
     * Skips CPU-side signed-log encoding - positions uploaded as floats directly.
     */
    processFrameGPU(frameIndex: number): {
        positions: Float32Array;
        attributes: Uint8Array;
        count: number;
        textureSize: number;
    };
    /**
     * Fill GPU output buffers with float positions and uint8 attributes.
     * Much faster than _assembleSogTexture() - no position encoding.
     */
    private _fillGPUBuffers;
    /**
     * Get GPU texture dimensions for creating THREE.DataTexture
     */
    getGPUTextureDimensions(): {
        positionSize: number;
        attributeWidth: number;
        attributeHeight: number;
    };
    reset(): void;
    private _processOneFrame;
    private _decodeBirths;
    private _assembleSogTexture;
    private _encodePositionInPlace;
    getTotalFrames(): number;
    hasKeyframes(): boolean;
    /**
     * Set keyframe pixel data directly (for loading from File objects)
     */
    setKeyframeData(frameIndex: number, data: Uint8ClampedArray): void;
    getSOGDimensions(): {
        width: number;
        height: number;
    };
}
