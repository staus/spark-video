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
    private activeIndices;
    private freeSlots;
    private activeCount;
    currentFrameIndex: number;
    private positions;
    private motions;
    private quatsEncoded;
    private scalesEncoded;
    private sh0Encoded;
    private remainingFrames;
    private justBorn;
    private tempSlots;
    private tempMotionMags;
    private framePixelData;
    private frameWidth;
    private frameHeight;
    private keyframeData;
    private keyframeTileSizes;
    private keyframeIndices;
    private totalFrames;
    private frameIndexToVideoIndex;
    private sogWidth;
    private sogHeight;
    private sogTileData;
    private posEncodeBuf;
    private gpuPositionBuffer;
    private gpuAttributeBuffer;
    private gpuTextureSize;
    private staticSlots;
    private staticCount;
    private staticPositionBuffer;
    private staticAttributeBuffer;
    private staticTextureSize;
    private staticDataReady;
    private dynamicTextureSize;
    private dynamicPositionBuffer;
    private dynamicAttributeBuffer;
    private static readonly STATIC_MOTION_THRESHOLD;
    private dynamicKeyframeBirthCache;
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
     * Decode births directly into SoA buffers at specified slots.
     * Zero per-frame allocations - all data written to pre-allocated arrays.
     * @returns Number of births decoded, motion magnitudes stored in motionMags parameter
     */
    private _decodeBirthsToSoA;
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
    /**
     * Initialize and fill static buffers after keyframe processing.
     * Call this after processFrameGPU(0) to finalize static data.
     */
    initStaticBuffers(): void;
    /**
     * Get static gaussian data for one-time GPU upload.
     * Returns null if no static gaussians or not yet initialized.
     */
    getStaticData(): {
        positions: Float32Array;
        attributes: Uint8Array;
        count: number;
        textureSize: number;
    } | null;
    /**
     * Get static texture dimensions for creating THREE.DataTexture.
     * Returns null if no static gaussians.
     */
    getStaticTextureDimensions(): {
        positionSize: number;
        attributeWidth: number;
        attributeHeight: number;
    } | null;
    /**
     * Get counts for static vs dynamic gaussians.
     */
    getGaussianCounts(): {
        static: number;
        dynamic: number;
        total: number;
    };
    reset(): void;
    private _processOneFrame;
    private _updateActiveGaussians;
    private _assembleSogTexture;
    private _encodePositionFromSoA;
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
    /**
     * Load a .4dgs bundle file and extract its contents.
     * The bundle is a ZIP file containing video.webp, metadata.json, and keyframe files.
     */
    static loadFromBundle(bundleBlob: Blob): Promise<{
        videoBlob: Blob;
        metadata: Delta4DGSMetadata;
        keyframeBlobs: Map<number, Blob>;
    }>;
}
