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
     * Update directly from raw packed array data
     * Fastest path when data is already in Spark format
     */
    updateFromPackedArray(data: Uint32Array, numSplats?: number): void;
    private gpuDeltaModeData;
    /**
     * Initialize GPU delta mode with float position input.
     * Uses a simpler shader that reads float positions directly.
     */
    initDeltaModeGPU(metadata: DeltaModeMetadata): void;
    /**
     * Update splats from float positions and uint8 attributes.
     * This bypasses the CPU signed-log encoding entirely.
     *
     * @param dynamicCount Number of dynamic gaussians in the textures
     * @param staticCount Number of static gaussians (already uploaded via setStaticDeltaTextures)
     */
    updateFromDeltaTextureGPU(renderer: THREE.WebGLRenderer, positionTexture: THREE.DataTexture, attributeTexture: THREE.DataTexture, dynamicCount: number, staticCount?: number): void;
    /**
     * Set static delta textures (uploaded once for keyframe gaussians with zero motion).
     * These are rendered first, before dynamic gaussians.
     */
    setStaticDeltaTextures(renderer: THREE.WebGLRenderer, positionTexture: THREE.DataTexture, attributeTexture: THREE.DataTexture, count: number): void;
    /**
     * Update the quaternion transform mode for delta mode
     */
    setDeltaQuatTransformMode(mode: number): void;
    /**
     * Update the max scale filter for delta mode
     */
    setDeltaMaxScaleFilter(maxScale: number): void;
    /**
     * Dispose GPU delta mode resources
     */
    disposeDeltaModeGPU(): void;
}
/**
 * Metadata for initializing delta mode (float positions)
 */
export type DeltaModeMetadata = {
    maxCount: number;
    scaleCodebook: number[];
    sh0Codebook: number[];
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
