import { FullScreenQuad } from 'three/addons/postprocessing/Pass.js';
import { GsplatGenerator } from './SplatGenerator';
import { SplatFileType } from './SplatLoader';
import { DynoProgram, DynoProgramTemplate, DynoUniform } from './dyno';
import { TPackedSplats } from './dyno/splats';
import * as THREE from "three";
export type SplatEncoding = {
    rgbMin?: number;
    rgbMax?: number;
    lnScaleMin?: number;
    lnScaleMax?: number;
    sh1Min?: number;
    sh1Max?: number;
    sh2Min?: number;
    sh2Max?: number;
    sh3Min?: number;
    sh3Max?: number;
};
export declare const DEFAULT_SPLAT_ENCODING: SplatEncoding;
export type PackedSplatsOptions = {
    url?: string;
    fileBytes?: Uint8Array | ArrayBuffer;
    fileType?: SplatFileType;
    fileName?: string;
    maxSplats?: number;
    packedArray?: Uint32Array;
    numSplats?: number;
    construct?: (splats: PackedSplats) => Promise<void> | void;
    extra?: Record<string, unknown>;
    splatEncoding?: SplatEncoding;
};
export declare class PackedSplats {
    maxSplats: number;
    numSplats: number;
    packedArray: Uint32Array | null;
    extra: Record<string, unknown>;
    splatEncoding?: SplatEncoding;
    initialized: Promise<PackedSplats>;
    isInitialized: boolean;
    target: THREE.WebGLArrayRenderTarget | null;
    source: THREE.DataArrayTexture | null;
    needsUpdate: boolean;
    dyno: DynoUniform<typeof TPackedSplats, "packedSplats">;
    dynoRgbMinMaxLnScaleMinMax: DynoUniform<"vec4", "rgbMinMaxLnScaleMinMax">;
    dynoSh1MinMax: DynoUniform<"vec2", "sh1MinMax">;
    dynoSh2MinMax: DynoUniform<"vec2", "sh2MinMax">;
    dynoSh3MinMax: DynoUniform<"vec2", "sh3MinMax">;
    constructor(options?: PackedSplatsOptions);
    reinitialize(options: PackedSplatsOptions): void;
    initialize(options: PackedSplatsOptions): void;
    asyncInitialize(options: PackedSplatsOptions): Promise<void>;
    dispose(): void;
    ensureSplats(numSplats: number): Uint32Array;
    ensureSplatsSh(level: number, numSplats: number): Uint32Array;
    getSplat(index: number): {
        center: THREE.Vector3;
        scales: THREE.Vector3;
        quaternion: THREE.Quaternion;
        opacity: number;
        color: THREE.Color;
    };
    setSplat(index: number, center: THREE.Vector3, scales: THREE.Vector3, quaternion: THREE.Quaternion, opacity: number, color: THREE.Color): void;
    pushSplat(center: THREE.Vector3, scales: THREE.Vector3, quaternion: THREE.Quaternion, opacity: number, color: THREE.Color): void;
    forEachSplat(callback: (index: number, center: THREE.Vector3, scales: THREE.Vector3, quaternion: THREE.Quaternion, opacity: number, color: THREE.Color) => void): void;
    ensureGenerate(maxSplats: number): boolean;
    generateMapping(splatCounts: number[]): {
        maxSplats: number;
        mapping: {
            base: number;
            count: number;
        }[];
    };
    getTexture(): THREE.DataArrayTexture;
    private maybeUpdateSource;
    private static emptySource;
    static getEmpty(): THREE.DataArrayTexture;
    prepareProgramMaterial(generator: GsplatGenerator): {
        program: DynoProgram;
        material: THREE.RawShaderMaterial;
    };
    private saveRenderState;
    private resetRenderState;
    generate({ generator, base, count, renderer, }: {
        generator: GsplatGenerator;
        base: number;
        count: number;
        renderer: THREE.WebGLRenderer;
    }): {
        nextBase: number;
    };
    static programTemplate: DynoProgramTemplate | null;
    static generatorProgram: WeakMap<GsplatGenerator, DynoProgram>;
    private static clearValue;
    static fullScreenQuad: FullScreenQuad;
    /**
     * Pre-computed video mode data for fast frame updates
     */
    private videoModeData;
    /**
     * Initialize for video frame updates with full SOG metadata
     * Pre-computes all lookup tables for maximum frame update performance
     */
    initVideoMode(metadata: SOGVideoMetadata): void;
    /**
     * Update splat data from SOG-format video tiles (optimized path)
     * Uses pre-computed lookup tables for maximum performance
     */
    updateFromVideoTiles(tiles: SOGVideoTiles): void;
    /**
     * Update directly from raw packed array data
     * Fastest path when data is already in Spark format
     */
    updateFromPackedArray(data: Uint32Array, numSplats?: number): void;
    /**
     * GPU video mode data - shader materials and textures
     */
    private gpuVideoModeData;
    /**
     * Initialize GPU video mode with shader-based decoding
     * Creates codebook textures and decode shader material
     */
    initVideoModeGPU(metadata: SOGVideoMetadata, tileSize: number): void;
    /**
     * Update splat data from video texture using GPU shader (zero CPU path)
     * This is the fastest possible path - no getImageData, no CPU loops
     */
    updateFromVideoTextureGPU(renderer: THREE.WebGLRenderer, videoTexture: THREE.Texture, tileUVs: GPUVideoTileUVs, videoWidth: number, videoHeight: number): void;
    /**
     * Dispose GPU video mode resources
     */
    disposeVideoModeGPU(): void;
}
/**
 * SOG video metadata for initializing video mode
 */
export type SOGVideoMetadata = {
    count: number;
    mins: [number, number, number];
    maxs: [number, number, number];
    scaleCodebook: number[];
    sh0Codebook: number[];
};
/**
 * SOG video tile data from a single frame
 */
export type SOGVideoTiles = {
    means_l: Uint8ClampedArray;
    means_u: Uint8ClampedArray;
    quats: Uint8ClampedArray;
    scales: Uint8ClampedArray;
    sh0: Uint8ClampedArray;
};
/**
 * Tile UV coordinates for GPU video decoding
 */
export type GPUVideoTileUV = {
    u0: number;
    v0: number;
    u1: number;
    v1: number;
};
/**
 * All tile UVs needed for GPU video decoding
 */
export type GPUVideoTileUVs = {
    means_l: GPUVideoTileUV;
    means_u: GPUVideoTileUV;
    quats: GPUVideoTileUV;
    scales: GPUVideoTileUV;
    sh0: GPUVideoTileUV;
};
export declare const dynoPackedSplats: (packedSplats?: PackedSplats) => DynoPackedSplats;
export declare class DynoPackedSplats extends DynoUniform<typeof TPackedSplats, "packedSplats", {
    texture: THREE.DataArrayTexture;
    numSplats: number;
    rgbMinMaxLnScaleMinMax: THREE.Vector4;
}> {
    packedSplats?: PackedSplats;
    constructor({ packedSplats }?: {
        packedSplats?: PackedSplats;
    });
}
