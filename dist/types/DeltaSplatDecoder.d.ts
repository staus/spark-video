/**
 * Delta-encoded 4DGS video decoder.
 *
 * Each frame contains only newly-born gaussians with motion vectors and lifetimes.
 * The decoder accumulates births, interpolates positions, and reconstructs full SOG textures.
 *
 * Uses canvas 2D with careful color space handling to read raw pixel data.
 */
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
    };
    encoding: "delta";
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
    private freeSlots;
    private activeCount;
    currentFrameIndex: number;
    private framePixelData;
    private frameWidth;
    private frameHeight;
    private sogWidth;
    private sogHeight;
    private sogTileData;
    private posEncodeBuf;
    constructor(metadata: Delta4DGSMetadata);
    loadDeltaFrames(webpBlob: Blob): Promise<void>;
    /**
     * Get pixel value from pre-decoded frame data.
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
    reset(): void;
    private _processOneFrame;
    private _decodeBirths;
    private _assembleSogTexture;
    private _encodePositionInPlace;
    getTotalFrames(): number;
    getSOGDimensions(): {
        width: number;
        height: number;
    };
}
