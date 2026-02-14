import * as THREE from "three";
import { FullScreenQuad } from "three/addons/postprocessing/Pass.js";

import type { GsplatGenerator } from "./SplatGenerator";
import { type SplatFileType, SplatLoader, unpackSplats } from "./SplatLoader";
import {
  LN_SCALE_MAX,
  LN_SCALE_MIN,
  SPLAT_TEX_HEIGHT,
  SPLAT_TEX_WIDTH,
} from "./defines";
import {
  DynoProgram,
  DynoProgramTemplate,
  DynoSampler2D,
  type DynoType,
  DynoUniform,
  DynoVec2,
  DynoVec4,
  dynoBlock,
  outputPackedSplat,
} from "./dyno";
import { TPackedSplats, definePackedSplats } from "./dyno/splats";
import { getShaders } from "./shaders";
import { getTextureSize, setPackedSplat, unpackSplat } from "./utils";

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

export const DEFAULT_SPLAT_ENCODING: SplatEncoding = {
  rgbMin: 0,
  rgbMax: 1,
  lnScaleMin: LN_SCALE_MIN,
  lnScaleMax: LN_SCALE_MAX,
  sh1Min: -1,
  sh1Max: 1,
  sh2Min: -1,
  sh2Max: 1,
  sh3Min: -1,
  sh3Max: 1,
};

// Initialize a PackedSplats collection from source data via
// url, fileBytes, or packedArray. Creates an empty array if none are set,
// and splat data can be constructed using pushSplat()/setSplat(). The maximum
// splat size allocation will grow automatically, starting from maxSplats.
export type PackedSplatsOptions = {
  // URL to fetch a Gaussian splat file from (supports .ply, .splat, .ksplat,
  // .spz formats). (default: undefined)
  url?: string;
  // Raw bytes of a Gaussian splat file to decode directly instead of fetching
  // from URL. (default: undefined)
  fileBytes?: Uint8Array | ArrayBuffer;
  // Override the file type detection for formats that can't be reliably
  // auto-detected (.splat, .ksplat). (default: undefined auto-detects other
  // formats from file contents)
  fileType?: SplatFileType;
  // File name to use for type detection. (default: undefined)
  fileName?: string;
  // Reserve space for at least this many splats when constructing the collection
  // initially. The array will automatically resize past maxSplats so setting it is
  // an optional optimization. (default: 0)
  maxSplats?: number;
  // Use provided packed data array, where each 4 consecutive uint32 values
  // encode one "packed" Gsplat. (default: undefined)
  packedArray?: Uint32Array;
  // Override number of splats in packed array to use only a subset.
  // (default: length of packed array / 4)
  numSplats?: number;
  // Callback function to programmatically create splats at initialization.
  // (default: undefined)
  construct?: (splats: PackedSplats) => Promise<void> | void;
  // Additional splat data, such as spherical harmonics components (sh1, sh2, sh3). (default: {})
  extra?: Record<string, unknown>;
  // Override the default splat encoding ranges for the PackedSplats.
  // (default: undefined)
  splatEncoding?: SplatEncoding;
};

// A PackedSplats is a collection of Gaussian splats, packed into a format that
// takes exactly 16 bytes per Gsplat to maximize memory and cache efficiency.
// The center xyz coordinates are encoded as float16 (3 x 2 bytes), scale xyz
// as 3 x uint8 that encode a log scale from e^-12 to e^9, rgba as 4 x uint8,
// and quaternion encoded via axis+angle using 2 x uint8 for octahedral encoding
// of the axis direction and a uint8 to encode rotation amount from 0..Pi.

export class PackedSplats {
  maxSplats = 0;
  numSplats = 0;
  packedArray: Uint32Array | null = null;
  extra: Record<string, unknown>;
  splatEncoding?: SplatEncoding;

  initialized: Promise<PackedSplats>;
  isInitialized = false;

  // Either target or source will be non-null, depending on whether the PackedSplats
  // is being used as a data source or generated to.
  target: THREE.WebGLArrayRenderTarget | null = null;
  source: THREE.DataArrayTexture | null = null;
  // Set to true if source packedArray is updated to have it upload to GPU
  needsUpdate = true;

  // A PackedSplats can be used in a dyno graph using the below property dyno:
  // const gsplat = dyno.readPackedSplats(this.dyno, dynoIndex);
  dyno: DynoUniform<typeof TPackedSplats, "packedSplats">;
  dynoRgbMinMaxLnScaleMinMax: DynoUniform<"vec4", "rgbMinMaxLnScaleMinMax">;
  dynoSh1MinMax: DynoUniform<"vec2", "sh1MinMax">;
  dynoSh2MinMax: DynoUniform<"vec2", "sh2MinMax">;
  dynoSh3MinMax: DynoUniform<"vec2", "sh3MinMax">;

  constructor(options: PackedSplatsOptions = {}) {
    this.extra = {};
    this.dyno = new DynoPackedSplats({ packedSplats: this });
    this.dynoRgbMinMaxLnScaleMinMax = new DynoVec4({
      key: "rgbMinMaxLnScaleMinMax",
      value: new THREE.Vector4(0.0, 1.0, LN_SCALE_MIN, LN_SCALE_MAX),
      update: (value) => {
        value.set(
          this.splatEncoding?.rgbMin ?? 0.0,
          this.splatEncoding?.rgbMax ?? 1.0,
          this.splatEncoding?.lnScaleMin ?? LN_SCALE_MIN,
          this.splatEncoding?.lnScaleMax ?? LN_SCALE_MAX,
        );
        return value;
      },
    });
    this.dynoSh1MinMax = new DynoVec2({
      key: "sh1MinMax",
      value: new THREE.Vector2(-1, 1),
      update: (value) => {
        value.set(
          this.splatEncoding?.sh1Min ?? -1,
          this.splatEncoding?.sh1Max ?? 1,
        );
        return value;
      },
    });
    this.dynoSh2MinMax = new DynoVec2({
      key: "sh2MinMax",
      value: new THREE.Vector2(-1, 1),
      update: (value) => {
        value.set(
          this.splatEncoding?.sh2Min ?? -1,
          this.splatEncoding?.sh2Max ?? 1,
        );
        return value;
      },
    });
    this.dynoSh3MinMax = new DynoVec2({
      key: "sh3MinMax",
      value: new THREE.Vector2(-1, 1),
      update: (value) => {
        value.set(
          this.splatEncoding?.sh3Min ?? -1,
          this.splatEncoding?.sh3Max ?? 1,
        );
        return value;
      },
    });

    // The following line will be overridden by reinitialize()
    this.initialized = Promise.resolve(this);
    this.reinitialize(options);
  }

  reinitialize(options: PackedSplatsOptions) {
    this.isInitialized = false;

    this.extra = {};
    this.splatEncoding = options.splatEncoding;

    if (options.url || options.fileBytes || options.construct) {
      // We need to initialize asynchronously given the options
      this.initialized = this.asyncInitialize(options).then(() => {
        this.isInitialized = true;
        return this;
      });
    } else {
      this.initialize(options);
      this.isInitialized = true;
      this.initialized = Promise.resolve(this);
    }
  }

  initialize(options: PackedSplatsOptions) {
    if (options.packedArray) {
      this.packedArray = options.packedArray;
      // Calculate number of horizontal texture rows that could fit in array.
      // A properly initialized packedArray should already take into account the
      // width and height of the texture and be rounded up with padding.
      this.maxSplats = Math.floor(this.packedArray.length / 4);
      this.maxSplats =
        Math.floor(this.maxSplats / SPLAT_TEX_WIDTH) * SPLAT_TEX_WIDTH;
      this.numSplats = Math.min(
        this.maxSplats,
        options.numSplats ?? Number.POSITIVE_INFINITY,
      );
    } else {
      this.maxSplats = options.maxSplats ?? 0;
      this.numSplats = 0;
    }
    this.extra = options.extra ?? {};
  }

  async asyncInitialize(options: PackedSplatsOptions) {
    const { url, fileBytes, construct } = options;
    if (url) {
      const loader = new SplatLoader();
      loader.packedSplats = this;
      await loader.loadAsync(url);
    } else if (fileBytes) {
      const unpacked = await unpackSplats({
        input: fileBytes,
        fileType: options.fileType,
        pathOrUrl: options.fileName ?? url,
        splatEncoding: options.splatEncoding ?? DEFAULT_SPLAT_ENCODING,
      });
      this.initialize(unpacked);
    }

    if (construct) {
      const maybePromise = construct(this);
      // If construct returns a promise, wait for it to complete
      if (maybePromise instanceof Promise) {
        await maybePromise;
      }
    }
  }

  // Call this when you are finished with the PackedSplats and want to free
  // any buffers it holds.
  dispose() {
    if (this.target) {
      this.target.dispose();
      this.target.texture.source.data = null;
      this.target = null;
    }
    if (this.source) {
      this.source.dispose();
      this.source.source.data = null;
      this.source = null;
    }

    this.packedArray = null;

    for (const key in this.extra) {
      const dyno = this.extra[key] as DynoUniform<
        DynoType,
        string,
        THREE.Texture
      >;
      if (dyno instanceof DynoUniform) {
        const texture = dyno.value;
        if (texture?.isTexture) {
          texture.dispose();
          texture.source.data = null;
        }
      }
    }
    this.extra = {};
  }

  // Ensures that this.packedArray can fit numSplats Gsplats. If it's too small,
  // resize exponentially and copy over the original data.
  //
  // Typically you don't need to call this, because calling this.setSplat(index, ...)
  // and this.pushSplat(...) will automatically call ensureSplats() so we have
  // enough splats.
  ensureSplats(numSplats: number): Uint32Array {
    const targetSize =
      numSplats <= this.maxSplats
        ? this.maxSplats
        : // Grow exponentially to avoid frequent reallocations
          Math.max(numSplats, 2 * this.maxSplats);
    const currentSize = !this.packedArray ? 0 : this.packedArray.length / 4;

    if (!this.packedArray || targetSize > currentSize) {
      this.maxSplats = getTextureSize(targetSize).maxSplats;
      const newArray = new Uint32Array(this.maxSplats * 4);
      if (this.packedArray) {
        // Copy over existing data
        newArray.set(this.packedArray);
      }
      this.packedArray = newArray;
    }
    return this.packedArray;
  }

  // Ensure the extra array for the given level is large enough to hold numSplats
  ensureSplatsSh(level: number, numSplats: number): Uint32Array {
    let wordsPerSplat: number;
    let key: string;
    if (level === 0) {
      return this.ensureSplats(numSplats);
    }
    if (level === 1) {
      // 3 x 3 uint7 = 63 bits = 2 uint32
      wordsPerSplat = 2;
      key = "sh1";
    } else if (level === 2) {
      // 5 x 3 uint8 = 120 bits = 4 uint32
      wordsPerSplat = 4;
      key = "sh2";
    } else if (level === 3) {
      // 7 x 3 uint6 = 126 bits = 4 uint32
      wordsPerSplat = 4;
      key = "sh3";
    } else {
      throw new Error(`Invalid level: ${level}`);
    }

    // Figure out our current and desired maxSplats
    let maxSplats: number = !this.extra[key]
      ? 0
      : (this.extra[key] as Uint32Array).length / wordsPerSplat;
    const targetSize =
      numSplats <= maxSplats ? maxSplats : Math.max(numSplats, 2 * maxSplats);

    if (!this.extra[key] || targetSize > maxSplats) {
      // Reallocate the array
      maxSplats = getTextureSize(targetSize).maxSplats;
      const newArray = new Uint32Array(maxSplats * wordsPerSplat);
      if (this.extra[key]) {
        // Copy over existing data
        newArray.set(this.extra[key] as Uint32Array);
      }
      this.extra[key] = newArray;
    }
    return this.extra[key] as Uint32Array;
  }

  // Unpack the 16-byte Gsplat data at index into the Three.js components
  // center: THREE.Vector3, scales: THREE.Vector3, quaternion: THREE.Quaternion,
  // opacity: number 0..1, color: THREE.Color 0..1.
  getSplat(index: number): {
    center: THREE.Vector3;
    scales: THREE.Vector3;
    quaternion: THREE.Quaternion;
    opacity: number;
    color: THREE.Color;
  } {
    if (!this.packedArray || index >= this.numSplats) {
      throw new Error("Invalid index");
    }
    return unpackSplat(this.packedArray, index, this.splatEncoding);
  }

  // Set all PackedSplat components at index with the provided Gsplat attributes
  // (can be the same objects returned by getSplat). Ensures there is capacity
  // for at least index+1 Gsplats.
  setSplat(
    index: number,
    center: THREE.Vector3,
    scales: THREE.Vector3,
    quaternion: THREE.Quaternion,
    opacity: number,
    color: THREE.Color,
  ) {
    const packedSplats = this.ensureSplats(index + 1);
    setPackedSplat(
      packedSplats,
      index,
      center.x,
      center.y,
      center.z,
      scales.x,
      scales.y,
      scales.z,
      quaternion.x,
      quaternion.y,
      quaternion.z,
      quaternion.w,
      opacity,
      color.r,
      color.g,
      color.b,
    );
    this.numSplats = Math.max(this.numSplats, index + 1);
  }

  // Effectively calls this.setSplat(this.numSplats++, center, ...), useful on
  // construction where you just want to iterate and create a collection of Gsplats.
  pushSplat(
    center: THREE.Vector3,
    scales: THREE.Vector3,
    quaternion: THREE.Quaternion,
    opacity: number,
    color: THREE.Color,
  ) {
    const packedSplats = this.ensureSplats(this.numSplats + 1);
    setPackedSplat(
      packedSplats,
      this.numSplats,
      center.x,
      center.y,
      center.z,
      scales.x,
      scales.y,
      scales.z,
      quaternion.x,
      quaternion.y,
      quaternion.z,
      quaternion.w,
      opacity,
      color.r,
      color.g,
      color.b,
    );
    ++this.numSplats;
  }

  // Iterate over Gsplats index 0..=(this.numSplats-1), unpack each Gsplat
  // and invoke the callback function with the Gsplat attributes.
  forEachSplat(
    callback: (
      index: number,
      center: THREE.Vector3,
      scales: THREE.Vector3,
      quaternion: THREE.Quaternion,
      opacity: number,
      color: THREE.Color,
    ) => void,
  ) {
    if (!this.packedArray || !this.numSplats) {
      return;
    }
    for (let i = 0; i < this.numSplats; ++i) {
      const unpacked = unpackSplat(this.packedArray, i, this.splatEncoding);
      callback(
        i,
        unpacked.center,
        unpacked.scales,
        unpacked.quaternion,
        unpacked.opacity,
        unpacked.color,
      );
    }
  }

  // Ensures our PackedSplats.target render target has enough space to generate
  // maxSplats total Gsplats, and reallocate if not large enough.
  ensureGenerate(maxSplats: number): boolean {
    if (this.target && (maxSplats ?? 1) <= this.maxSplats) {
      return false;
    }
    if (this.target) {
      this.target.dispose();
    }

    const textureSize = getTextureSize(maxSplats ?? 1);
    const { width, height, depth } = textureSize;
    this.maxSplats = textureSize.maxSplats;

    // The packed Gsplats are stored in a 2D array texture of max size
    // 2048 x 2048 x 2048, one RGBA32UI pixel = 4 uint32 = one Gsplat
    this.target = new THREE.WebGLArrayRenderTarget(width, height, depth, {
      depthBuffer: false,
      stencilBuffer: false,
      generateMipmaps: false,
      magFilter: THREE.NearestFilter,
      minFilter: THREE.NearestFilter,
    });
    this.target.texture.format = THREE.RGBAIntegerFormat;
    this.target.texture.type = THREE.UnsignedIntType;
    this.target.texture.internalFormat = "RGBA32UI";
    this.target.scissorTest = true;
    return true;
  }

  // Given an array of splatCounts (.numSplats for each
  // SplatGenerator/SplatMesh in the scene), compute a
  // "mapping layout" in the composite array of generated outputs.
  generateMapping(splatCounts: number[]): {
    maxSplats: number;
    mapping: { base: number; count: number }[];
  } {
    let maxSplats = 0;
    const mapping = splatCounts.map((numSplats) => {
      const base = maxSplats;
      // Generation happens in horizontal row chunks, so round up to full width
      const rounded = Math.ceil(numSplats / SPLAT_TEX_WIDTH) * SPLAT_TEX_WIDTH;
      maxSplats += rounded;
      return { base, count: numSplats };
    });
    return { maxSplats, mapping };
  }

  // Returns a THREE.DataArrayTexture representing the PackedSplats content as
  // a Uint32x4 data array texture (2048 x 2048 x depth in size)
  getTexture(): THREE.DataArrayTexture {
    if (this.target) {
      // Return the render target's texture
      return this.target.texture;
    }
    if (this.source || this.packedArray) {
      // Update source texture if needed and return
      const source = this.maybeUpdateSource();
      return source;
    }

    return PackedSplats.getEmpty();
  }

  // Check if source texture needs to be created/updated
  private maybeUpdateSource(): THREE.DataArrayTexture {
    if (!this.packedArray) {
      throw new Error("No packed splats");
    }

    if (this.needsUpdate || !this.source) {
      this.needsUpdate = false;

      if (this.source) {
        const { width, height, depth } = this.source.image;
        if (this.maxSplats !== width * height * depth) {
          // The existing source texture isn't the right size, so dispose it
          this.source.dispose();
          this.source = null;
        }
      }
      if (!this.source) {
        // Allocate a new source texture of the right size
        const { width, height, depth } = getTextureSize(this.maxSplats);
        this.source = new THREE.DataArrayTexture(
          this.packedArray,
          width,
          height,
          depth,
        );
        this.source.format = THREE.RGBAIntegerFormat;
        this.source.type = THREE.UnsignedIntType;
        this.source.internalFormat = "RGBA32UI";
        this.source.needsUpdate = true;
      } else if (this.packedArray.buffer !== this.source.image.data.buffer) {
        // The source texture is the right size, update the data
        this.source.image.data = new Uint8Array(this.packedArray.buffer);
      }
      // Indicate to Three.js that the source texture needs to be uploaded to the GPU
      this.source.needsUpdate = true;
    }
    return this.source;
  }

  private static emptySource: THREE.DataArrayTexture | null = null;

  // Can be used where you need an uninitialized THREE.DataArrayTexture like
  // a uniform you will update with the result of this.getTexture() later.
  static getEmpty(): THREE.DataArrayTexture {
    if (!PackedSplats.emptySource) {
      const { width, height, depth, maxSplats } = getTextureSize(1);
      const emptyArray = new Uint32Array(maxSplats * 4);
      PackedSplats.emptySource = new THREE.DataArrayTexture(
        emptyArray,
        width,
        height,
        depth,
      );
      PackedSplats.emptySource.format = THREE.RGBAIntegerFormat;
      PackedSplats.emptySource.type = THREE.UnsignedIntType;
      PackedSplats.emptySource.internalFormat = "RGBA32UI";
      PackedSplats.emptySource.needsUpdate = true;
    }
    return PackedSplats.emptySource;
  }

  // Get a program and THREE.RawShaderMaterial for a given GsplatGenerator,
  // generating it if necessary and caching the result.
  prepareProgramMaterial(generator: GsplatGenerator): {
    program: DynoProgram;
    material: THREE.RawShaderMaterial;
  } {
    let program = PackedSplats.generatorProgram.get(generator);
    if (!program) {
      // A Gsplat needs to be turned into a packed uvec4 for the dyno graph
      const graph = dynoBlock(
        { index: "int" },
        { output: "uvec4" },
        ({ index }) => {
          generator.inputs.index = index;
          const gsplat = generator.outputs.gsplat;
          const output = outputPackedSplat(
            gsplat,
            this.dynoRgbMinMaxLnScaleMinMax,
          );
          return { output };
        },
      );
      if (!PackedSplats.programTemplate) {
        PackedSplats.programTemplate = new DynoProgramTemplate(
          getShaders().computeUvec4Template,
        );
      }
      // Create a program from the template and graph
      program = new DynoProgram({
        graph,
        inputs: { index: "index" },
        outputs: { output: "target" },
        template: PackedSplats.programTemplate,
      });
      Object.assign(program.uniforms, {
        targetLayer: { value: 0 },
        targetBase: { value: 0 },
        targetCount: { value: 0 },
      });
      PackedSplats.generatorProgram.set(generator, program);
    }

    // Prepare and update our material we'll use to render the Gsplats
    const material = program.prepareMaterial();
    PackedSplats.fullScreenQuad.material = material;
    return { program, material };
  }

  private saveRenderState(renderer: THREE.WebGLRenderer) {
    return {
      xrEnabled: renderer.xr.enabled,
      autoClear: renderer.autoClear,
    };
  }

  private resetRenderState(
    renderer: THREE.WebGLRenderer,
    state: {
      xrEnabled: boolean;
      autoClear: boolean;
    },
  ) {
    renderer.setRenderTarget(null);
    renderer.xr.enabled = state.xrEnabled;
    renderer.autoClear = state.autoClear;
  }

  // Executes a dyno program specified by generator which is any DynoBlock that
  // maps { index: "int" } to { gsplat: Gsplat }. This is called in
  // SparkRenderer.updateInternal() to re-generate Gsplats in the scene for
  // SplatGenerator instances whose version is newer than what was generated
  // for it last time.
  generate({
    generator,
    base,
    count,
    renderer,
  }: {
    generator: GsplatGenerator;
    base: number;
    count: number;
    renderer: THREE.WebGLRenderer;
  }): { nextBase: number } {
    if (!this.target) {
      throw new Error("Target must be initialized with ensureSplats");
    }
    if (base + count > this.maxSplats) {
      throw new Error("Base + count exceeds maxSplats");
    }

    const { program, material } = this.prepareProgramMaterial(generator);
    program.update();

    const renderState = this.saveRenderState(renderer);

    // Generate the Gsplats in "layer" chunks, in horizontal row ranges,
    // that cover the total count of Gsplats.
    const nextBase =
      Math.ceil((base + count) / SPLAT_TEX_WIDTH) * SPLAT_TEX_WIDTH;
    const layerSize = SPLAT_TEX_WIDTH * SPLAT_TEX_HEIGHT;
    material.uniforms.targetBase.value = base;
    material.uniforms.targetCount.value = count;

    // Keep generating layers until we've reached the next generation's base
    while (base < nextBase) {
      const layer = Math.floor(base / layerSize);
      material.uniforms.targetLayer.value = layer;

      const layerBase = layer * layerSize;
      const layerYStart = Math.floor((base - layerBase) / SPLAT_TEX_WIDTH);
      const layerYEnd = Math.min(
        SPLAT_TEX_HEIGHT,
        Math.ceil((nextBase - layerBase) / SPLAT_TEX_WIDTH),
      );

      // Render the desired portion of the layer
      this.target.scissor.set(
        0,
        layerYStart,
        SPLAT_TEX_WIDTH,
        layerYEnd - layerYStart,
      );
      renderer.setRenderTarget(this.target, layer);
      renderer.xr.enabled = false;
      renderer.autoClear = false;
      // Clear integer framebuffer with proper WebGL2 call
      const gl = renderer.getContext() as WebGL2RenderingContext;
      gl.clearBufferuiv(gl.COLOR, 0, PackedSplats.clearValue);
      PackedSplats.fullScreenQuad.render(renderer);

      base += SPLAT_TEX_WIDTH * (layerYEnd - layerYStart);
    }

    this.resetRenderState(renderer, renderState);
    return { nextBase };
  }

  static programTemplate: DynoProgramTemplate | null = null;

  // Cache for GsplatGenerator programs
  static generatorProgram = new WeakMap<GsplatGenerator, DynoProgram>();

  // Clear value for integer framebuffers
  private static clearValue = new Uint32Array([0, 0, 0, 0]);

  // Static full-screen quad for pseudo-compute shader rendering
  static fullScreenQuad = new FullScreenQuad(
    new THREE.RawShaderMaterial({ visible: false }),
  );

  // =============================================
  // Video Frame Update API
  // Enables real-time updates from video gaussian splat formats
  // =============================================

  /**
   * Pre-computed video mode data for fast frame updates
   */
  private videoModeData: VideoModeData | null = null;

  /**
   * Initialize for video frame updates with full SOG metadata
   * Pre-computes all lookup tables for maximum frame update performance
   */
  initVideoMode(metadata: SOGVideoMetadata) {
    const SH_C0 = 0.28209479177387814;
    const SQRT2 = Math.sqrt(2);

    // Pre-compute scale lookup: codebook index -> Spark's uint8 scale format
    const scaleLookup = new Uint8Array(256);
    const lnScaleScale = 254.0 / (LN_SCALE_MAX - LN_SCALE_MIN);
    for (let i = 0; i < 256; i++) {
      const logScale = metadata.scaleCodebook[i] ?? metadata.scaleCodebook[0];
      const sparkScale = Math.min(
        255,
        Math.max(1, Math.round((logScale - LN_SCALE_MIN) * lnScaleScale) + 1),
      );
      scaleLookup[i] = sparkScale;
    }

    // Pre-compute SH0/color lookup: codebook index -> RGB uint8
    const sh0Lookup = new Uint8Array(256);
    for (let i = 0; i < 256; i++) {
      const shVal = metadata.sh0Codebook[i] ?? metadata.sh0Codebook[0];
      const rgb = Math.min(
        255,
        Math.max(0, Math.round((SH_C0 * shVal + 0.5) * 255)),
      );
      sh0Lookup[i] = rgb;
    }

    // Pre-compute quaternion component lookup: pixel value -> float component
    const quatLookup = new Float32Array(256);
    for (let i = 0; i < 256; i++) {
      quatLookup[i] = (i / 255 - 0.5) * SQRT2;
    }

    // Pre-compute position float16 lookup for uint16 values
    // This is a 64KB table but eliminates per-splat float16 conversion
    const posFloat16LookupX = new Uint16Array(65536);
    const posFloat16LookupY = new Uint16Array(65536);
    const posFloat16LookupZ = new Uint16Array(65536);

    const rangeX = metadata.maxs[0] - metadata.mins[0];
    const rangeY = metadata.maxs[1] - metadata.mins[1];
    const rangeZ = metadata.maxs[2] - metadata.mins[2];

    for (let i = 0; i < 65536; i++) {
      const fx = i / 65535;
      // World coord with exp transform
      let posX = metadata.mins[0] + rangeX * fx;
      let posY = metadata.mins[1] + rangeY * fx;
      let posZ = metadata.mins[2] + rangeZ * fx;
      posX = Math.sign(posX) * (Math.exp(Math.abs(posX)) - 1);
      posY = Math.sign(posY) * (Math.exp(Math.abs(posY)) - 1);
      posZ = Math.sign(posZ) * (Math.exp(Math.abs(posZ)) - 1);
      posFloat16LookupX[i] = toFloat16(posX);
      posFloat16LookupY[i] = toFloat16(posY);
      posFloat16LookupZ[i] = toFloat16(posZ);
    }

    // Pre-compute octahedral quaternion encoding lookup
    // Maps (r0, r1, r2, order) -> Spark's 24-bit octahedral format
    // This is too large for a full table, so we use partial lookups

    this.videoModeData = {
      count: metadata.count,
      scaleLookup,
      sh0Lookup,
      quatLookup,
      posFloat16LookupX,
      posFloat16LookupY,
      posFloat16LookupZ,
    };

    this.ensureSplats(metadata.count);
    this.numSplats = metadata.count;
  }

  /**
   * Update splat data from SOG-format video tiles (optimized path)
   * Uses pre-computed lookup tables for maximum performance
   */
  updateFromVideoTiles(tiles: SOGVideoTiles) {
    if (!this.videoModeData) {
      throw new Error("Call initVideoMode() before updateFromVideoTiles()");
    }
    if (!this.packedArray) {
      throw new Error("PackedArray not initialized");
    }

    const {
      count,
      scaleLookup,
      sh0Lookup,
      quatLookup,
      posFloat16LookupX,
      posFloat16LookupY,
      posFloat16LookupZ,
    } = this.videoModeData;
    const packed = this.packedArray;

    // Use local references for tighter inner loop
    const meansL = tiles.means_l;
    const meansU = tiles.means_u;
    const quats = tiles.quats;
    const scales = tiles.scales;
    const sh0 = tiles.sh0;

    // Main processing loop - optimized for V8
    for (let i = 0; i < count; i++) {
      const p = i << 2; // pixelOffset = i * 4

      // Position: combine bytes -> lookup float16
      const posXU16 = meansL[p] | (meansU[p] << 8);
      const posYU16 = meansL[p + 1] | (meansU[p + 1] << 8);
      const posZU16 = meansL[p + 2] | (meansU[p + 2] << 8);
      const posXF16 = posFloat16LookupX[posXU16];
      const posYF16 = posFloat16LookupY[posYU16];
      const posZF16 = posFloat16LookupZ[posZU16];

      // Quaternion: decode smallest-three, re-encode octahedral
      const r0 = quatLookup[quats[p]];
      const r1 = quatLookup[quats[p + 1]];
      const r2 = quatLookup[quats[p + 2]];
      const rr = Math.sqrt(Math.max(0, 1.0 - r0 * r0 - r1 * r1 - r2 * r2));
      const rOrder = quats[p + 3] - 252;
      const qx = rOrder === 0 ? r0 : rOrder === 1 ? rr : r1;
      const qy = rOrder <= 1 ? r1 : rOrder === 2 ? rr : r2;
      const qz = rOrder <= 2 ? r2 : rr;
      const qw = rOrder === 0 ? rr : r0;
      const uQuat = encodeQuatOct(qx, qy, qz, qw);

      // Scales: direct lookup
      const uScaleX = scaleLookup[scales[p]];
      const uScaleY = scaleLookup[scales[p + 1]];
      const uScaleZ = scaleLookup[scales[p + 2]];

      // Colors: lookup SH0 -> RGB
      const colorR = sh0Lookup[sh0[p]];
      const colorG = sh0Lookup[sh0[p + 1]];
      const colorB = sh0Lookup[sh0[p + 2]];
      const opacity = sh0[p + 3];

      // Pack into Spark format
      packed[p] = colorR | (colorG << 8) | (colorB << 16) | (opacity << 24);
      packed[p + 1] = posXF16 | (posYF16 << 16);
      packed[p + 2] =
        posZF16 | ((uQuat & 0xff) << 16) | (((uQuat >> 8) & 0xff) << 24);
      packed[p + 3] =
        uScaleX |
        (uScaleY << 8) |
        (uScaleZ << 16) |
        (((uQuat >> 16) & 0xff) << 24);
    }

    this.needsUpdate = true;
  }

  /**
   * Update directly from raw packed array data
   * Fastest path when data is already in Spark format
   */
  updateFromPackedArray(data: Uint32Array, numSplats?: number) {
    if (!this.packedArray || this.packedArray.length < data.length) {
      this.ensureSplats(data.length / 4);
    }
    this.packedArray?.set(data);
    this.numSplats = numSplats ?? data.length / 4;
    this.needsUpdate = true;
  }

  // =============================================
  // GPU Video Frame Update API
  // Zero CPU involvement - decode directly in GPU shader
  // =============================================

  /**
   * GPU video mode data - shader materials and textures
   */
  private gpuVideoModeData: GPUVideoModeData | null = null;

  /**
   * Initialize GPU video mode with shader-based decoding
   * Creates codebook textures and decode shader material
   */
  initVideoModeGPU(metadata: SOGVideoMetadata, tileSize: number) {
    // Create scale codebook texture (256x1, R32F format)
    // Stores original log scale values for GPU decode
    const scaleData = new Float32Array(256);
    let lnScaleMin = Number.POSITIVE_INFINITY;
    let lnScaleMax = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < 256; i++) {
      const val = metadata.scaleCodebook[i] ?? metadata.scaleCodebook[0];
      scaleData[i] = val;
      if (val < lnScaleMin) lnScaleMin = val;
      if (val > lnScaleMax) lnScaleMax = val;
    }
    const scaleCodebookTexture = new THREE.DataTexture(
      scaleData,
      256,
      1,
      THREE.RedFormat,
      THREE.FloatType,
    );
    scaleCodebookTexture.minFilter = THREE.NearestFilter;
    scaleCodebookTexture.magFilter = THREE.NearestFilter;
    scaleCodebookTexture.needsUpdate = true;

    // Create SH0/color codebook texture (256x1, R32F format)
    // Stores original SH0 values for GPU decode
    const sh0Data = new Float32Array(256);
    for (let i = 0; i < 256; i++) {
      sh0Data[i] = metadata.sh0Codebook[i] ?? metadata.sh0Codebook[0];
    }
    const sh0CodebookTexture = new THREE.DataTexture(
      sh0Data,
      256,
      1,
      THREE.RedFormat,
      THREE.FloatType,
    );
    sh0CodebookTexture.minFilter = THREE.NearestFilter;
    sh0CodebookTexture.magFilter = THREE.NearestFilter;
    sh0CodebookTexture.needsUpdate = true;

    // Set splatEncoding with the actual scale range from the codebook
    // This ensures the render shader uses the same range for unpacking
    this.splatEncoding = {
      rgbMin: 0,
      rgbMax: 1,
      lnScaleMin,
      lnScaleMax,
    };

    // Create decode shader material
    const shaderCode = getShaders().videoDecodeUvec4;
    const vertexShader = `
      in vec3 position;
      void main() {
        gl_Position = vec4(position, 1.0);
      }
    `;

    const material = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader,
      fragmentShader: shaderCode,
      uniforms: {
        targetLayer: { value: 0 },
        targetBase: { value: 0 },
        targetCount: { value: metadata.count },
        videoTexture: { value: null },
        videoTextureB: { value: null }, // Second texture for interpolation
        interpAlpha: { value: 0.0 }, // Interpolation factor (0 = A only, >0 = blend)
        videoSize: { value: new THREE.Vector2(0, 0) },
        tileUV_means_l: { value: new THREE.Vector4(0, 0, 0, 0) },
        tileUV_means_u: { value: new THREE.Vector4(0, 0, 0, 0) },
        tileUV_quats: { value: new THREE.Vector4(0, 0, 0, 0) },
        tileUV_scales: { value: new THREE.Vector4(0, 0, 0, 0) },
        tileUV_sh0: { value: new THREE.Vector4(0, 0, 0, 0) },
        tileSize: { value: tileSize },
        positionMins: {
          value: new THREE.Vector3(
            metadata.mins[0],
            metadata.mins[1],
            metadata.mins[2],
          ),
        },
        positionMaxs: {
          value: new THREE.Vector3(
            metadata.maxs[0],
            metadata.maxs[1],
            metadata.maxs[2],
          ),
        },
        scaleCodebook: { value: scaleCodebookTexture },
        sh0Codebook: { value: sh0CodebookTexture },
        splatCount: { value: metadata.count },
        rgbMinMaxLnScaleMinMax: {
          value: new THREE.Vector4(0, 1, lnScaleMin, lnScaleMax),
        },
      },
    });

    this.gpuVideoModeData = {
      count: metadata.count,
      tileSize,
      scaleCodebookTexture,
      sh0CodebookTexture,
      material,
      positionMins: metadata.mins,
      positionMaxs: metadata.maxs,
    };

    // Ensure we have enough space for the splats
    this.ensureGenerate(metadata.count);
    this.numSplats = metadata.count;
  }

  /**
   * Update splat data from video texture using GPU shader (zero CPU path)
   * This is the fastest possible path - no getImageData, no CPU loops
   */
  updateFromVideoTextureGPU(
    renderer: THREE.WebGLRenderer,
    videoTexture: THREE.Texture,
    tileUVs: GPUVideoTileUVs,
    videoWidth: number,
    videoHeight: number,
  ) {
    if (!this.gpuVideoModeData) {
      throw new Error("Call initVideoModeGPU() first");
    }
    if (!this.target) {
      throw new Error("Render target not initialized");
    }

    const { material, count, tileSize } = this.gpuVideoModeData;

    // Update uniforms
    material.uniforms.videoTexture.value = videoTexture;
    material.uniforms.videoSize.value.set(videoWidth, videoHeight);
    material.uniforms.tileSize.value = tileSize;

    // Set tile UV coordinates
    material.uniforms.tileUV_means_l.value.set(
      tileUVs.means_l.u0,
      tileUVs.means_l.v0,
      tileUVs.means_l.u1,
      tileUVs.means_l.v1,
    );
    material.uniforms.tileUV_means_u.value.set(
      tileUVs.means_u.u0,
      tileUVs.means_u.v0,
      tileUVs.means_u.u1,
      tileUVs.means_u.v1,
    );
    material.uniforms.tileUV_quats.value.set(
      tileUVs.quats.u0,
      tileUVs.quats.v0,
      tileUVs.quats.u1,
      tileUVs.quats.v1,
    );
    material.uniforms.tileUV_scales.value.set(
      tileUVs.scales.u0,
      tileUVs.scales.v0,
      tileUVs.scales.u1,
      tileUVs.scales.v1,
    );
    material.uniforms.tileUV_sh0.value.set(
      tileUVs.sh0.u0,
      tileUVs.sh0.v0,
      tileUVs.sh0.u1,
      tileUVs.sh0.v1,
    );

    // Render to packed splat texture
    const renderState = this.saveRenderState(renderer);

    const layerSize = SPLAT_TEX_WIDTH * SPLAT_TEX_HEIGHT;
    const numLayers = Math.ceil(count / layerSize);

    PackedSplats.fullScreenQuad.material = material;

    for (let layer = 0; layer < numLayers; layer++) {
      const layerBase = layer * layerSize;
      const layerCount = Math.min(count - layerBase, layerSize);
      const layerYEnd = Math.ceil(layerCount / SPLAT_TEX_WIDTH);

      material.uniforms.targetLayer.value = layer;
      material.uniforms.targetBase.value = layerBase;
      material.uniforms.targetCount.value = layerCount;

      this.target.scissor.set(0, 0, SPLAT_TEX_WIDTH, layerYEnd);
      renderer.setRenderTarget(this.target, layer);
      renderer.xr.enabled = false;
      renderer.autoClear = false;
      // Clear integer framebuffer with proper WebGL2 call
      const gl = renderer.getContext() as WebGL2RenderingContext;
      gl.clearBufferuiv(gl.COLOR, 0, PackedSplats.clearValue);
      PackedSplats.fullScreenQuad.render(renderer);
    }

    this.resetRenderState(renderer, renderState);
  }

  /**
   * Update splat data from two video textures with interpolation.
   * Blends between frame A and frame B based on interpAlpha (0-1).
   * Used for smooth playback between keyframes.
   */
  updateFromDualVideoTextureGPU(
    renderer: THREE.WebGLRenderer,
    videoTextureA: THREE.Texture,
    videoTextureB: THREE.Texture,
    interpAlpha: number,
    tileUVs: GPUVideoTileUVs,
    videoWidth: number,
    videoHeight: number,
  ) {
    if (!this.gpuVideoModeData) {
      throw new Error("Call initVideoModeGPU() first");
    }
    if (!this.target) {
      throw new Error("Render target not initialized");
    }

    const { material, count, tileSize } = this.gpuVideoModeData;

    // Update uniforms for dual-texture interpolation
    material.uniforms.videoTexture.value = videoTextureA;
    material.uniforms.videoTextureB.value = videoTextureB;
    material.uniforms.interpAlpha.value = interpAlpha;
    material.uniforms.videoSize.value.set(videoWidth, videoHeight);
    material.uniforms.tileSize.value = tileSize;

    // Set tile UV coordinates
    material.uniforms.tileUV_means_l.value.set(
      tileUVs.means_l.u0,
      tileUVs.means_l.v0,
      tileUVs.means_l.u1,
      tileUVs.means_l.v1,
    );
    material.uniforms.tileUV_means_u.value.set(
      tileUVs.means_u.u0,
      tileUVs.means_u.v0,
      tileUVs.means_u.u1,
      tileUVs.means_u.v1,
    );
    material.uniforms.tileUV_quats.value.set(
      tileUVs.quats.u0,
      tileUVs.quats.v0,
      tileUVs.quats.u1,
      tileUVs.quats.v1,
    );
    material.uniforms.tileUV_scales.value.set(
      tileUVs.scales.u0,
      tileUVs.scales.v0,
      tileUVs.scales.u1,
      tileUVs.scales.v1,
    );
    material.uniforms.tileUV_sh0.value.set(
      tileUVs.sh0.u0,
      tileUVs.sh0.v0,
      tileUVs.sh0.u1,
      tileUVs.sh0.v1,
    );

    // Render to packed splat texture
    const renderState = this.saveRenderState(renderer);

    const layerSize = SPLAT_TEX_WIDTH * SPLAT_TEX_HEIGHT;
    const numLayers = Math.ceil(count / layerSize);

    PackedSplats.fullScreenQuad.material = material;

    for (let layer = 0; layer < numLayers; layer++) {
      const layerBase = layer * layerSize;
      const layerCount = Math.min(count - layerBase, layerSize);
      const layerYEnd = Math.ceil(layerCount / SPLAT_TEX_WIDTH);

      material.uniforms.targetLayer.value = layer;
      material.uniforms.targetBase.value = layerBase;
      material.uniforms.targetCount.value = layerCount;

      this.target.scissor.set(0, 0, SPLAT_TEX_WIDTH, layerYEnd);
      renderer.setRenderTarget(this.target, layer);
      renderer.xr.enabled = false;
      renderer.autoClear = false;
      const gl = renderer.getContext() as WebGL2RenderingContext;
      gl.clearBufferuiv(gl.COLOR, 0, PackedSplats.clearValue);
      PackedSplats.fullScreenQuad.render(renderer);
    }

    this.resetRenderState(renderer, renderState);
  }

  /**
   * Update the splat count for GPU video mode
   * Call this before updateFromVideoTextureGPU when frame has different count
   */
  updateVideoSplatCount(count: number) {
    if (!this.gpuVideoModeData) {
      return;
    }
    this.gpuVideoModeData.count = count;
    this.gpuVideoModeData.material.uniforms.splatCount.value = count;
    this.numSplats = count;
  }

  /**
   * Dispose GPU video mode resources
   */
  disposeVideoModeGPU() {
    if (this.gpuVideoModeData) {
      this.gpuVideoModeData.scaleCodebookTexture.dispose();
      this.gpuVideoModeData.sh0CodebookTexture.dispose();
      this.gpuVideoModeData.material.dispose();
      this.gpuVideoModeData = null;
    }
  }
}

// =============================================
// Video Mode Types
// =============================================

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
 * Pre-computed video mode data for fast updates
 */
type VideoModeData = {
  count: number;
  scaleLookup: Uint8Array;
  sh0Lookup: Uint8Array;
  quatLookup: Float32Array;
  posFloat16LookupX: Uint16Array;
  posFloat16LookupY: Uint16Array;
  posFloat16LookupZ: Uint16Array;
};

/**
 * GPU video mode data - shader materials and codebook textures
 */
type GPUVideoModeData = {
  count: number;
  tileSize: number;
  scaleCodebookTexture: THREE.DataTexture;
  sh0CodebookTexture: THREE.DataTexture;
  material: THREE.RawShaderMaterial;
  positionMins: [number, number, number];
  positionMaxs: [number, number, number];
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

// =============================================
// Video Frame Helpers
// =============================================

// Shared typed array views for float16 conversion
const f32View = new Float32Array(1);
const i32View = new Int32Array(f32View.buffer);

/**
 * Convert float32 to float16 (half precision)
 */
function toFloat16(value: number): number {
  f32View[0] = value;
  const f = i32View[0];

  const sign = (f >> 16) & 0x8000;
  const exponent = ((f >> 23) & 0xff) - 127 + 15;
  const mantissa = f & 0x7fffff;

  if (exponent <= 0) {
    if (exponent < -10) return sign;
    const m = (mantissa | 0x800000) >> (1 - exponent);
    return sign | (m >> 13);
  }
  if (exponent >= 31) return sign | 0x7c00;
  return sign | (exponent << 10) | (mantissa >> 13);
}

/**
 * Encode quaternion to Spark's octahedral XY88R8 format
 * Returns 24-bit value: [quantU:8][quantV:8][angleInt:8]
 */
function encodeQuatOct(qx: number, qy: number, qz: number, qw: number): number {
  // Normalize
  const len = Math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
  let nx = qx / len;
  let ny = qy / len;
  let nz = qz / len;
  let nw = qw / len;

  // Force minimal representation (w >= 0)
  if (nw < 0) {
    nx = -nx;
    ny = -ny;
    nz = -nz;
    nw = -nw;
  }

  // Compute rotation angle
  const theta = 2 * Math.acos(Math.min(1, nw));

  // Recover rotation axis
  const xyzNorm = Math.sqrt(nx * nx + ny * ny + nz * nz);
  let axisX: number;
  let axisY: number;
  let axisZ: number;
  if (xyzNorm < 1e-6) {
    axisX = 1;
    axisY = 0;
    axisZ = 0;
  } else {
    const invNorm = 1 / xyzNorm;
    axisX = nx * invNorm;
    axisY = ny * invNorm;
    axisZ = nz * invNorm;
  }

  // Folded octahedral mapping
  const sum = Math.abs(axisX) + Math.abs(axisY) + Math.abs(axisZ);
  let px = axisX / sum;
  let py = axisY / sum;

  // Fold lower hemisphere
  if (axisZ < 0) {
    const tmp = px;
    px = (1 - Math.abs(py)) * (px >= 0 ? 1 : -1);
    py = (1 - Math.abs(tmp)) * (py >= 0 ? 1 : -1);
  }

  // Quantize to 8 bits each
  const quantU = Math.round((px * 0.5 + 0.5) * 255);
  const quantV = Math.round((py * 0.5 + 0.5) * 255);
  const angleInt = Math.round((theta / Math.PI) * 255);

  return quantU | (quantV << 8) | (angleInt << 16);
}

// You can use a PackedSplats as a dyno block using the function
// dyno.readPackedSplats(packedSplats.dyno, dynoIndex) where
// dynoIndex is of type DynoVal<"int">. If you need to be able to change
// the input PackedSplats dynamically, however, you should create a
// DynoPackedSplats, whose property packedSplats you can change to any
// PackedSplats and that will be used in the dyno shader program.

export const dynoPackedSplats = (packedSplats?: PackedSplats) =>
  new DynoPackedSplats({ packedSplats });

export class DynoPackedSplats extends DynoUniform<
  typeof TPackedSplats,
  "packedSplats",
  {
    texture: THREE.DataArrayTexture;
    numSplats: number;
    rgbMinMaxLnScaleMinMax: THREE.Vector4;
  }
> {
  packedSplats?: PackedSplats;

  constructor({ packedSplats }: { packedSplats?: PackedSplats } = {}) {
    super({
      key: "packedSplats",
      type: TPackedSplats,
      globals: () => [definePackedSplats],
      value: {
        texture: PackedSplats.getEmpty(),
        numSplats: 0,
        rgbMinMaxLnScaleMinMax: new THREE.Vector4(
          0,
          1,
          LN_SCALE_MIN,
          LN_SCALE_MAX,
        ),
      },
      update: (value) => {
        value.texture =
          this.packedSplats?.getTexture() ?? PackedSplats.getEmpty();
        value.numSplats = this.packedSplats?.numSplats ?? 0;
        value.rgbMinMaxLnScaleMinMax.set(
          this.packedSplats?.splatEncoding?.rgbMin ?? 0,
          this.packedSplats?.splatEncoding?.rgbMax ?? 1,
          this.packedSplats?.splatEncoding?.lnScaleMin ?? LN_SCALE_MIN,
          this.packedSplats?.splatEncoding?.lnScaleMax ?? LN_SCALE_MAX,
        );
        return value;
      },
    });
    this.packedSplats = packedSplats;
  }
}
