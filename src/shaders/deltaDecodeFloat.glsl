precision highp float;
precision highp int;
precision highp sampler2D;

#include <splatDefines>

// Target texture parameters
uniform uint targetLayer;
uniform int targetBase;
uniform int targetCount;

// Dynamic textures (uploaded each frame)
uniform sampler2D positionTexture;
uniform int positionTextureSize;  // Width/height of square texture
uniform sampler2D attributeTexture;
uniform int attributeTextureSize;  // Width of texture (height = attributeTextureSize * 3)

// Static textures (uploaded once for keyframe gaussians with zero motion)
uniform sampler2D staticPositionTexture;
uniform int staticPositionTextureSize;
uniform sampler2D staticAttributeTexture;
uniform int staticAttributeTextureSize;
uniform int staticCount;  // Number of static gaussians (indices 0..staticCount-1)

// Combined codebook texture (256x1, RG32F format)
// R = exp(logScale) pre-computed, G = sh0 value
uniform sampler2D codebook;

// Splat count (total = static + dynamic)
uniform int splatCount;

// Encoding range for pack/unpack
uniform vec4 rgbMinMaxLnScaleMinMax;

// Quaternion transform mode
uniform int quatTransformMode;

out uvec4 target;

// Constants
const float SQRT2 = 1.41421356237;
const float SH_C0 = 0.28209479177387814;

// Quaternion multiplication
vec4 quatMul(vec4 a, vec4 b) {
    return vec4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

// Apply quaternion transform
vec4 applyQuatTransformLocal(vec4 q) {
    float s45 = 0.70710678118;
    vec4 rotX90 = vec4(s45, 0.0, 0.0, s45);
    vec4 rotX180 = vec4(1.0, 0.0, 0.0, 0.0);
    vec4 rotX270 = vec4(s45, 0.0, 0.0, -s45);
    vec4 rotY90 = vec4(0.0, s45, 0.0, s45);
    vec4 rotY180 = vec4(0.0, 1.0, 0.0, 0.0);
    vec4 rotY270 = vec4(0.0, s45, 0.0, -s45);
    vec4 rotZ90 = vec4(0.0, 0.0, s45, s45);
    vec4 rotZ180 = vec4(0.0, 0.0, 1.0, 0.0);
    vec4 rotZ270 = vec4(0.0, 0.0, s45, -s45);

    vec4 result = q;

    if (quatTransformMode == 0) { result = q; }
    else if (quatTransformMode == 1) { result = quatMul(rotX90, q); }
    else if (quatTransformMode == 2) { result = quatMul(rotX270, q); }
    else if (quatTransformMode == 3) { result = quatMul(rotX180, q); }
    else if (quatTransformMode == 4) { result = quatMul(rotY90, q); }
    else if (quatTransformMode == 5) { result = quatMul(rotY270, q); }
    else if (quatTransformMode == 6) { result = quatMul(rotY180, q); }
    else if (quatTransformMode == 7) { result = quatMul(rotZ90, q); }
    else if (quatTransformMode == 8) { result = quatMul(rotZ270, q); }
    else if (quatTransformMode == 9) { result = quatMul(rotZ180, q); }
    else if (quatTransformMode == 10) { result = vec4(q.y, q.x, q.z, q.w); }
    else if (quatTransformMode == 11) { result = vec4(q.z, q.y, q.x, q.w); }
    else if (quatTransformMode == 12) { result = vec4(q.x, q.z, q.y, q.w); }
    else if (quatTransformMode == 13) { result = vec4(-q.x, q.y, q.z, q.w); }
    else if (quatTransformMode == 14) { result = vec4(q.x, -q.y, q.z, q.w); }
    else if (quatTransformMode == 15) { result = vec4(q.x, q.y, -q.z, q.w); }
    else if (quatTransformMode == 16) { result = vec4(q.x, q.y, q.z, -q.w); }
    else if (quatTransformMode == 17) { result = vec4(-q.x, -q.y, q.z, q.w); }
    else if (quatTransformMode == 18) { result = vec4(-q.x, q.y, -q.z, q.w); }
    else if (quatTransformMode == 19) { result = vec4(q.x, -q.y, -q.z, q.w); }
    else if (quatTransformMode == 20) { result = quatMul(rotX90, q); result = vec4(result.x, result.z, result.y, result.w); }
    else if (quatTransformMode == 21) { result = quatMul(rotX270, q); result = vec4(result.x, result.z, result.y, result.w); }
    else if (quatTransformMode == 22) { result = vec4(-q.x, -q.y, -q.z, q.w); }
    else if (quatTransformMode == 23) { result = vec4(q.x, q.z, -q.y, q.w); }
    else if (quatTransformMode == 24) { result = vec4(q.x, -q.z, q.y, q.w); }
    else if (quatTransformMode == 25) { result = quatMul(rotX90, q); result = vec4(result.x, result.y, -result.z, result.w); }
    else if (quatTransformMode == 26) { result = quatMul(rotX270, q); result = vec4(result.x, result.y, -result.z, result.w); }
    else if (quatTransformMode == 27) { result = quatMul(rotX90, q); result = vec4(result.x, -result.y, result.z, result.w); }
    else if (quatTransformMode == 28) { result = quatMul(rotX270, q); result = vec4(result.x, -result.y, result.z, result.w); }
    else if (quatTransformMode == 29) { result = vec4(q.w, q.x, q.y, q.z); }
    else if (quatTransformMode == 30) { result = vec4(q.y, q.z, q.w, q.x); }
    else if (quatTransformMode == 31) { result = vec4(q.z, q.w, q.x, q.y); }

    return normalize(result);
}

// Sample position from static texture
vec3 sampleStaticPosition(int idx) {
    int x = idx % staticPositionTextureSize;
    int y = idx / staticPositionTextureSize;
    return texelFetch(staticPositionTexture, ivec2(x, y), 0).rgb;
}

// Sample position from dynamic texture
vec3 sampleDynamicPosition(int idx) {
    int x = idx % positionTextureSize;
    int y = idx / positionTextureSize;
    return texelFetch(positionTexture, ivec2(x, y), 0).rgb;
}

// Sample position based on splat index (static vs dynamic)
vec3 samplePosition(int splatIndex) {
    if (splatIndex < staticCount) {
        return sampleStaticPosition(splatIndex);
    } else {
        return sampleDynamicPosition(splatIndex - staticCount);
    }
}

// Sample attribute from static texture
vec4 sampleStaticAttribute(int idx, int row) {
    int x = idx % staticAttributeTextureSize;
    int y = idx / staticAttributeTextureSize;
    return texelFetch(staticAttributeTexture, ivec2(x, row * staticAttributeTextureSize + y), 0);
}

// Sample attribute from dynamic texture
vec4 sampleDynamicAttribute(int idx, int row) {
    int x = idx % attributeTextureSize;
    int y = idx / attributeTextureSize;
    return texelFetch(attributeTexture, ivec2(x, row * attributeTextureSize + y), 0);
}

// Sample attribute based on splat index (static vs dynamic)
vec4 sampleAttribute(int splatIndex, int row) {
    if (splatIndex < staticCount) {
        return sampleStaticAttribute(splatIndex, row);
    } else {
        return sampleDynamicAttribute(splatIndex - staticCount, row);
    }
}

// Decode quaternion from smallest-three encoding
vec4 decodeQuaternion(int splatIndex) {
    vec4 quatsRaw = sampleAttribute(splatIndex, 0);
    float qr = floor(quatsRaw.r * 255.0 + 0.5);
    float qg = floor(quatsRaw.g * 255.0 + 0.5);
    float qb = floor(quatsRaw.b * 255.0 + 0.5);
    float qa = floor(quatsRaw.a * 255.0 + 0.5);

    float r0 = (qr / 255.0 - 0.5) * SQRT2;
    float r1 = (qg / 255.0 - 0.5) * SQRT2;
    float r2 = (qb / 255.0 - 0.5) * SQRT2;
    float rr = sqrt(max(0.0, 1.0 - r0*r0 - r1*r1 - r2*r2));

    int rOrder = int(qa) - 252;
    vec4 quat;
    if (rOrder == 0) { quat = vec4(r0, r1, r2, rr); }
    else if (rOrder == 1) { quat = vec4(rr, r1, r2, r0); }
    else if (rOrder == 2) { quat = vec4(r1, rr, r2, r0); }
    else { quat = vec4(r1, r2, rr, r0); }

    return normalize(quat);
}

// Decode scales via codebook lookup (exp() pre-computed in R channel)
vec3 decodeScales(int splatIndex) {
    vec4 scalesRaw = sampleAttribute(splatIndex, 1);
    float idxX = floor(scalesRaw.r * 255.0 + 0.5);
    float idxY = floor(scalesRaw.g * 255.0 + 0.5);
    float idxZ = floor(scalesRaw.b * 255.0 + 0.5);

    // R channel = exp(logScale), already pre-computed
    float scaleX = texture(codebook, vec2((idxX + 0.5) / 256.0, 0.5)).r;
    float scaleY = texture(codebook, vec2((idxY + 0.5) / 256.0, 0.5)).r;
    float scaleZ = texture(codebook, vec2((idxZ + 0.5) / 256.0, 0.5)).r;

    return vec3(scaleX, scaleY, scaleZ);
}

// Decode RGBA via codebook lookup (sh0 values in G channel)
vec4 decodeRGBA(int splatIndex) {
    vec4 sh0Raw = sampleAttribute(splatIndex, 2);

    float idxR = floor(sh0Raw.r * 255.0 + 0.5);
    float idxG = floor(sh0Raw.g * 255.0 + 0.5);
    float idxB = floor(sh0Raw.b * 255.0 + 0.5);

    // G channel = sh0 value
    float sh0R = texture(codebook, vec2((idxR + 0.5) / 256.0, 0.5)).g;
    float sh0G = texture(codebook, vec2((idxG + 0.5) / 256.0, 0.5)).g;
    float sh0B = texture(codebook, vec2((idxB + 0.5) / 256.0, 0.5)).g;

    float colorR = SH_C0 * sh0R + 0.5;
    float colorG = SH_C0 * sh0G + 0.5;
    float colorB = SH_C0 * sh0B + 0.5;
    float opacity = sh0Raw.a;

    return vec4(clamp(colorR, 0.0, 1.0), clamp(colorG, 0.0, 1.0), clamp(colorB, 0.0, 1.0), opacity);
}

void main() {
    int targetIndex = int(targetLayer << SPLAT_TEX_LAYER_BITS) +
                     int(uint(gl_FragCoord.y) << SPLAT_TEX_WIDTH_BITS) +
                     int(gl_FragCoord.x);
    int splatIndex = targetIndex - targetBase;

    if (splatIndex >= 0 && splatIndex < targetCount && splatIndex < splatCount) {
        // Read float position directly
        vec3 center = samplePosition(splatIndex);

        // Decode attributes via codebook
        vec4 quaternion = decodeQuaternion(splatIndex);
        vec3 scales = decodeScales(splatIndex);
        vec4 rgba = decodeRGBA(splatIndex);

        // Apply quaternion transform
        quaternion = applyQuatTransformLocal(quaternion);

        // Pack into Spark format
        target = packSplatEncoding(center, scales, quaternion, rgba, rgbMinMaxLnScaleMinMax);
    } else {
        target = uvec4(0u, 0u, 0u, 0u);
    }
}
