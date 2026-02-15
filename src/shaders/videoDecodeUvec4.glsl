precision highp float;
precision highp int;
precision highp sampler2D;
precision highp usampler2D;

#include <splatDefines>

// Target texture parameters
uniform uint targetLayer;
uniform int targetBase;
uniform int targetCount;

// Video texture (combined frame with all tiles)
uniform sampler2D videoTexture;
uniform vec2 videoSize;  // Width, height of video frame

// Tile layout - each vec4 contains (u0, v0, u1, v1) for the tile
uniform vec4 tileUV_means_l;
uniform vec4 tileUV_means_u;
uniform vec4 tileUV_quats;
uniform vec4 tileUV_scales;
uniform vec4 tileUV_sh0;

// Tile size in pixels (square)
uniform float tileSize;

// Position metadata for decoding
uniform vec3 positionMins;
uniform vec3 positionMaxs;

// Codebook textures (256x1, RG32F format - stores (logScale, sh0Value))
uniform sampler2D scaleCodebook;   // Lookup: index -> log scale value
uniform sampler2D sh0Codebook;     // Lookup: index -> SH0 value

// Splat count
uniform int splatCount;

// Encoding range for pack/unpack (must match render shader)
// vec4(rgbMin, rgbMax, lnScaleMin, lnScaleMax)
uniform vec4 rgbMinMaxLnScaleMinMax;

// Quaternion transform mode for debugging coordinate system issues
// 0 = identity, 1+ = various axis swaps/rotations
uniform int quatTransformMode;

out uvec4 target;

// Constants for quaternion decoding
const float SQRT2 = 1.41421356237;
const float SH_C0 = 0.28209479177387814;
// Note: PI is already defined in splatDefines

// Quaternion multiplication: result = a * b
vec4 quatMul(vec4 a, vec4 b) {
    return vec4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

// Apply quaternion transformation based on mode
// Returns transformed quaternion (x, y, z, w)
vec4 applyQuatTransform(vec4 q) {
    // Pre-computed rotation quaternions
    // rotX90 = rotation 90° around X axis
    float s45 = 0.70710678118; // sin(45°) = cos(45°) = sqrt(2)/2

    vec4 rotX90 = vec4(s45, 0.0, 0.0, s45);      // 90° around X
    vec4 rotX180 = vec4(1.0, 0.0, 0.0, 0.0);     // 180° around X
    vec4 rotX270 = vec4(s45, 0.0, 0.0, -s45);    // -90° around X
    vec4 rotY90 = vec4(0.0, s45, 0.0, s45);      // 90° around Y
    vec4 rotY180 = vec4(0.0, 1.0, 0.0, 0.0);     // 180° around Y
    vec4 rotY270 = vec4(0.0, s45, 0.0, -s45);    // -90° around Y
    vec4 rotZ90 = vec4(0.0, 0.0, s45, s45);      // 90° around Z
    vec4 rotZ180 = vec4(0.0, 0.0, 1.0, 0.0);     // 180° around Z
    vec4 rotZ270 = vec4(0.0, 0.0, s45, -s45);    // -90° around Z

    vec4 result = q;

    if (quatTransformMode == 0) {
        // identity
        result = q;
    } else if (quatTransformMode == 1) {
        // rotX+90
        result = quatMul(rotX90, q);
    } else if (quatTransformMode == 2) {
        // rotX-90
        result = quatMul(rotX270, q);
    } else if (quatTransformMode == 3) {
        // rotX+180
        result = quatMul(rotX180, q);
    } else if (quatTransformMode == 4) {
        // rotY+90
        result = quatMul(rotY90, q);
    } else if (quatTransformMode == 5) {
        // rotY-90
        result = quatMul(rotY270, q);
    } else if (quatTransformMode == 6) {
        // rotY+180
        result = quatMul(rotY180, q);
    } else if (quatTransformMode == 7) {
        // rotZ+90
        result = quatMul(rotZ90, q);
    } else if (quatTransformMode == 8) {
        // rotZ-90
        result = quatMul(rotZ270, q);
    } else if (quatTransformMode == 9) {
        // rotZ+180
        result = quatMul(rotZ180, q);
    } else if (quatTransformMode == 10) {
        // Swap XY components
        result = vec4(q.y, q.x, q.z, q.w);
    } else if (quatTransformMode == 11) {
        // Swap XZ components
        result = vec4(q.z, q.y, q.x, q.w);
    } else if (quatTransformMode == 12) {
        // Swap YZ components
        result = vec4(q.x, q.z, q.y, q.w);
    } else if (quatTransformMode == 13) {
        // Negate X
        result = vec4(-q.x, q.y, q.z, q.w);
    } else if (quatTransformMode == 14) {
        // Negate Y
        result = vec4(q.x, -q.y, q.z, q.w);
    } else if (quatTransformMode == 15) {
        // Negate Z
        result = vec4(q.x, q.y, -q.z, q.w);
    } else if (quatTransformMode == 16) {
        // Negate W
        result = vec4(q.x, q.y, q.z, -q.w);
    } else if (quatTransformMode == 17) {
        // Negate XY
        result = vec4(-q.x, -q.y, q.z, q.w);
    } else if (quatTransformMode == 18) {
        // Negate XZ
        result = vec4(-q.x, q.y, -q.z, q.w);
    } else if (quatTransformMode == 19) {
        // Negate YZ
        result = vec4(q.x, -q.y, -q.z, q.w);
    } else if (quatTransformMode == 20) {
        // rotX+90, then swap YZ
        result = quatMul(rotX90, q);
        result = vec4(result.x, result.z, result.y, result.w);
    } else if (quatTransformMode == 21) {
        // rotX-90, then swap YZ
        result = quatMul(rotX270, q);
        result = vec4(result.x, result.z, result.y, result.w);
    } else if (quatTransformMode == 22) {
        // Conjugate (invert rotation)
        result = vec4(-q.x, -q.y, -q.z, q.w);
    } else if (quatTransformMode == 23) {
        // Blender Z-up to Y-up: swap Y and Z, negate new Z
        result = vec4(q.x, q.z, -q.y, q.w);
    } else if (quatTransformMode == 24) {
        // Blender Z-up to Y-up variant 2
        result = vec4(q.x, -q.z, q.y, q.w);
    } else if (quatTransformMode == 25) {
        // rotX90 then negate Z
        result = quatMul(rotX90, q);
        result = vec4(result.x, result.y, -result.z, result.w);
    } else if (quatTransformMode == 26) {
        // rotX-90 then negate Z
        result = quatMul(rotX270, q);
        result = vec4(result.x, result.y, -result.z, result.w);
    } else if (quatTransformMode == 27) {
        // rotX90 then negate Y
        result = quatMul(rotX90, q);
        result = vec4(result.x, -result.y, result.z, result.w);
    } else if (quatTransformMode == 28) {
        // rotX-90 then negate Y
        result = quatMul(rotX270, q);
        result = vec4(result.x, -result.y, result.z, result.w);
    } else if (quatTransformMode == 29) {
        // WXYZ to XYZW reorder (different quaternion conventions)
        result = vec4(q.w, q.x, q.y, q.z);
    } else if (quatTransformMode == 30) {
        // XYZW to WXYZ reorder
        result = vec4(q.y, q.z, q.w, q.x);
    } else if (quatTransformMode == 31) {
        // Full cycle reorder YZWX
        result = vec4(q.z, q.w, q.x, q.y);
    }

    return normalize(result);
}

// Sample a tile at the given splat index
vec4 sampleTile(vec4 tileUV, int splatIndex) {
    // Calculate which pixel in the tile this splat maps to
    int tileSizeInt = int(tileSize);
    int tileX = splatIndex % tileSizeInt;
    int tileY = splatIndex / tileSizeInt;

    // Convert to UV coordinates - tile UV coordinates are in normalized space
    // tileUV.xy is the top-left corner of the tile in UV space
    float u = tileUV.x + (float(tileX) + 0.5) / videoSize.x;
    float v = tileUV.y + (float(tileY) + 0.5) / videoSize.y;

    // WebGL texture coordinates: V=0 is the first row of uploaded image data
    // For ImageBitmap without flipY, this corresponds to the top of the image
    // Tile UVs are calculated in top-left origin, which matches directly
    // No V-flip needed

    return texture(videoTexture, vec2(u, v));
}

// Decode position from means_l and means_u tiles
vec3 decodePosition(int splatIndex) {
    vec4 meansL = sampleTile(tileUV_means_l, splatIndex);
    vec4 meansU = sampleTile(tileUV_means_u, splatIndex);

    // Combine low and high bytes to get uint16 values (0-65535 range)
    // Texture samples are raw bytes normalized to 0-1, multiply by 255 to recover original bytes
    vec3 posU16 = vec3(
        floor(meansL.r * 255.0 + 0.5) + floor(meansU.r * 255.0 + 0.5) * 256.0,
        floor(meansL.g * 255.0 + 0.5) + floor(meansU.g * 255.0 + 0.5) * 256.0,
        floor(meansL.b * 255.0 + 0.5) + floor(meansU.b * 255.0 + 0.5) * 256.0
    );

    // Normalize to 0-1 range
    vec3 posNorm = posU16 / 65535.0;

    // Interpolate in log space
    vec3 posLog = positionMins + (positionMaxs - positionMins) * posNorm;

    // Apply exp transform: sign(x) * (exp(abs(x)) - 1)
    vec3 pos;
    pos.x = sign(posLog.x) * (exp(abs(posLog.x)) - 1.0);
    pos.y = sign(posLog.y) * (exp(abs(posLog.y)) - 1.0);
    pos.z = sign(posLog.z) * (exp(abs(posLog.z)) - 1.0);

    return pos;
}

// Decode quaternion from quats tile (smallest-three format)
vec4 decodeQuaternion(int splatIndex) {
    vec4 quatsRaw = sampleTile(tileUV_quats, splatIndex);

    // Get uint8 values
    float qr = floor(quatsRaw.r * 255.0 + 0.5);
    float qg = floor(quatsRaw.g * 255.0 + 0.5);
    float qb = floor(quatsRaw.b * 255.0 + 0.5);
    float qa = floor(quatsRaw.a * 255.0 + 0.5);

    // Decode smallest-three components
    float r0 = (qr / 255.0 - 0.5) * SQRT2;
    float r1 = (qg / 255.0 - 0.5) * SQRT2;
    float r2 = (qb / 255.0 - 0.5) * SQRT2;

    // Compute fourth component
    float rr = sqrt(max(0.0, 1.0 - r0*r0 - r1*r1 - r2*r2));

    // Decode order from alpha channel (252, 253, 254, 255 -> 0, 1, 2, 3)
    int rOrder = int(qa) - 252;

    // Reconstruct quaternion based on which component was dropped
    // CPU reference: qx = rOrder===0 ? r0 : rOrder===1 ? rr : r1
    //                qy = rOrder<=1 ? r1 : rOrder===2 ? rr : r2
    //                qz = rOrder<=2 ? r2 : rr
    //                qw = rOrder===0 ? rr : r0
    vec4 quat;
    if (rOrder == 0) {
        quat = vec4(r0, r1, r2, rr);  // w was largest
    } else if (rOrder == 1) {
        quat = vec4(rr, r1, r2, r0);  // x was largest
    } else if (rOrder == 2) {
        quat = vec4(r1, rr, r2, r0);  // y was largest
    } else {
        quat = vec4(r1, r2, rr, r0);  // z was largest
    }

    return normalize(quat);
}

// Decode scales from codebook lookup
vec3 decodeScales(int splatIndex) {
    vec4 scalesRaw = sampleTile(tileUV_scales, splatIndex);

    // Get codebook indices as uint8
    float idxX = floor(scalesRaw.r * 255.0 + 0.5);
    float idxY = floor(scalesRaw.g * 255.0 + 0.5);
    float idxZ = floor(scalesRaw.b * 255.0 + 0.5);

    // Look up log scale values from codebook (stored as R32F)
    float logScaleX = texture(scaleCodebook, vec2((idxX + 0.5) / 256.0, 0.5)).r;
    float logScaleY = texture(scaleCodebook, vec2((idxY + 0.5) / 256.0, 0.5)).r;
    float logScaleZ = texture(scaleCodebook, vec2((idxZ + 0.5) / 256.0, 0.5)).r;

    // Convert log scale to linear scale
    return vec3(exp(logScaleX), exp(logScaleY), exp(logScaleZ));
}

// Decode color and opacity from sh0 tile and codebook
vec4 decodeRGBA(int splatIndex) {
    vec4 sh0Raw = sampleTile(tileUV_sh0, splatIndex);

    // Get codebook indices as uint8
    float idxR = floor(sh0Raw.r * 255.0 + 0.5);
    float idxG = floor(sh0Raw.g * 255.0 + 0.5);
    float idxB = floor(sh0Raw.b * 255.0 + 0.5);

    // Look up SH0 values from codebook
    float sh0R = texture(sh0Codebook, vec2((idxR + 0.5) / 256.0, 0.5)).r;
    float sh0G = texture(sh0Codebook, vec2((idxG + 0.5) / 256.0, 0.5)).r;
    float sh0B = texture(sh0Codebook, vec2((idxB + 0.5) / 256.0, 0.5)).r;

    // Convert SH0 to RGB: rgb = SH_C0 * sh0 + 0.5
    float colorR = SH_C0 * sh0R + 0.5;
    float colorG = SH_C0 * sh0G + 0.5;
    float colorB = SH_C0 * sh0B + 0.5;

    // Opacity is direct from the alpha channel (not through codebook)
    float opacity = sh0Raw.a;

    return vec4(clamp(colorR, 0.0, 1.0), clamp(colorG, 0.0, 1.0), clamp(colorB, 0.0, 1.0), opacity);
}

void main() {
    // Calculate which splat this fragment corresponds to
    int targetIndex = int(targetLayer << SPLAT_TEX_LAYER_BITS) +
                     int(uint(gl_FragCoord.y) << SPLAT_TEX_WIDTH_BITS) +
                     int(gl_FragCoord.x);
    int splatIndex = targetIndex - targetBase;

    if (splatIndex >= 0 && splatIndex < targetCount && splatIndex < splatCount) {
        // Decode all splat attributes from video tiles
        vec3 center = decodePosition(splatIndex);
        vec4 quaternion = decodeQuaternion(splatIndex);
        vec3 scales = decodeScales(splatIndex);
        vec4 rgba = decodeRGBA(splatIndex);

        // Apply quaternion transformation (for coordinate system debugging)
        quaternion = applyQuatTransform(quaternion);

        // Pack into Spark's uvec4 format using dynamic encoding range
        target = packSplatEncoding(center, scales, quaternion, rgba, rgbMinMaxLnScaleMinMax);
    } else {
        target = uvec4(0u, 0u, 0u, 0u);
    }
}
