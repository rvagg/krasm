//! Four-lane wasm32 SIMD SHA-256 adapted from Hugo Dias's `commp` revision a37a5b2bb1f272d0862e3d59accd318c84ea00e6.

use super::{IV, NODE_SIZE, truncated_hash_64};
use core::arch::wasm32::*;

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
    0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
    0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
    0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
];

/// `K[i] + W[i]` for the padding block of a 64-byte message, which is
/// always the same: `0x80`, zeros, then the bit length 512
const PAD_KW: [u32; 64] = {
    let mut w = [0u32; 64];
    w[0] = 0x8000_0000;
    w[15] = 512;
    let mut i = 16;
    while i < 64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        i += 1;
    }
    let mut kw = [0u32; 64];
    let mut i = 0;
    while i < 64 {
        kw[i] = K[i].wrapping_add(w[i]);
        i += 1;
    }
    kw
};

/// wasm SIMD has no rotate instruction
#[inline(always)]
fn rotr(x: v128, n: u32) -> v128 {
    v128_or(u32x4_shr(x, n), u32x4_shl(x, 32 - n))
}

/// Swap the bytes of each `u32` lane (SHA-256 words are big-endian)
#[inline(always)]
fn bswap(x: v128) -> v128 {
    i8x16_shuffle::<3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12>(x, x)
}

/// Transpose a 4x4 matrix of `u32`s held in 4 rows
#[inline(always)]
fn transpose(r0: v128, r1: v128, r2: v128, r3: v128) -> [v128; 4] {
    let t0 = u32x4_shuffle::<0, 4, 1, 5>(r0, r1);
    let t1 = u32x4_shuffle::<2, 6, 3, 7>(r0, r1);
    let t2 = u32x4_shuffle::<0, 4, 1, 5>(r2, r3);
    let t3 = u32x4_shuffle::<2, 6, 3, 7>(r2, r3);
    [
        u32x4_shuffle::<0, 1, 4, 5>(t0, t2),
        u32x4_shuffle::<2, 3, 6, 7>(t0, t2),
        u32x4_shuffle::<0, 1, 4, 5>(t1, t3),
        u32x4_shuffle::<2, 3, 6, 7>(t1, t3),
    ]
}

/// One SHA-256 round; callers rotate the variable names instead of
/// moving the state
macro_rules! round {
    ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident, $h:ident, $kw:expr) => {
        let s1 = v128_xor(v128_xor(rotr($e, 6), rotr($e, 11)), rotr($e, 25));
        let ch = v128_xor($g, v128_and($e, v128_xor($f, $g)));
        let t1 = u32x4_add(u32x4_add($h, s1), u32x4_add(ch, $kw));
        let s0 = v128_xor(v128_xor(rotr($a, 2), rotr($a, 13)), rotr($a, 22));
        let maj = v128_or(v128_and($a, $b), v128_and($c, v128_or($a, $b)));
        $d = u32x4_add($d, t1);
        $h = u32x4_add(t1, u32x4_add(s0, maj));
    };
}

/// Eight rounds starting at `i`, rotating the names back to the start
macro_rules! rounds8 {
    ($s:ident, $i:expr, |$j:ident| $kw:expr) => {{
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = $s;
        let kw = |$j: usize| $kw;
        round!(a, b, c, d, e, f, g, h, kw($i));
        round!(h, a, b, c, d, e, f, g, kw($i + 1));
        round!(g, h, a, b, c, d, e, f, kw($i + 2));
        round!(f, g, h, a, b, c, d, e, kw($i + 3));
        round!(e, f, g, h, a, b, c, d, kw($i + 4));
        round!(d, e, f, g, h, a, b, c, kw($i + 5));
        round!(c, d, e, f, g, h, a, b, kw($i + 6));
        round!(b, c, d, e, f, g, h, a, kw($i + 7));
        $s = [a, b, c, d, e, f, g, h];
    }};
}

/// Add the input state back after 64 rounds
#[inline(always)]
fn finish(input: [v128; 8], s: [v128; 8]) -> [v128; 8] {
    core::array::from_fn(|k| u32x4_add(input[k], s[k]))
}

/// Compress the message block; `w` holds its 16 words and is used as the
/// ring buffer for the message schedule
#[inline(always)]
fn compress_message(state: [v128; 8], w: &mut [v128; 16]) -> [v128; 8] {
    let mut s = state;
    for i in (0..64).step_by(8) {
        if i >= 16 {
            for j in i..i + 8 {
                let w15 = w[(j - 15) & 15];
                let w2 = w[(j - 2) & 15];
                let s0 = v128_xor(v128_xor(rotr(w15, 7), rotr(w15, 18)), u32x4_shr(w15, 3));
                let s1 = v128_xor(v128_xor(rotr(w2, 17), rotr(w2, 19)), u32x4_shr(w2, 10));
                w[j & 15] = u32x4_add(u32x4_add(w[j & 15], s0), u32x4_add(w[(j - 7) & 15], s1));
            }
        }
        rounds8!(s, i, |j| u32x4_add(w[j & 15], u32x4_splat(K[j])));
    }
    finish(state, s)
}

/// Compress the constant padding block, whose schedule is precomputed
#[inline(always)]
fn compress_padding(state: [v128; 8]) -> [v128; 8] {
    let mut s = state;
    for i in (0..64).step_by(8) {
        rounds8!(s, i, |j| u32x4_splat(PAD_KW[j]));
    }
    finish(state, s)
}

/// Hash 4 independent 64-byte messages into truncated nodes
#[inline(always)]
fn hash4(msgs: &[[u8; 64]; 4], out: &mut [[u8; NODE_SIZE]; 4]) {
    // w[i % 16] holds schedule word i of all 4 messages
    let mut w = [u32x4_splat(0); 16];
    for g in 0..4 {
        // SAFETY: each message is 64 bytes, so offset 16 * g + 16 is in bounds;
        // wasm loads have no alignment requirement
        let [r0, r1, r2, r3] =
            core::array::from_fn(|m| unsafe { bswap(v128_load(msgs[m].as_ptr().add(16 * g) as *const v128)) });
        w[4 * g..4 * g + 4].copy_from_slice(&transpose(r0, r1, r2, r3));
    }

    let mid = compress_message(IV.map(|x| u32x4_splat(x)), &mut w);
    let s = compress_padding(mid);

    let lo = transpose(s[0], s[1], s[2], s[3]);
    let hi = transpose(s[4], s[5], s[6], s[7]);
    for m in 0..4 {
        let node = &mut out[m];
        // SAFETY: each node is 32 bytes; wasm stores have no alignment requirement
        unsafe {
            v128_store(node.as_mut_ptr() as *mut v128, bswap(lo[m]));
            v128_store(node.as_mut_ptr().add(16) as *mut v128, bswap(hi[m]));
        }
        node[NODE_SIZE - 1] &= 0b0011_1111;
    }
}

/// Hash many 64-byte messages into truncated nodes, 4 at a time
pub fn hash_many(msgs: &[[u8; 64]], out: &mut [[u8; NODE_SIZE]]) {
    let mut msg_groups = msgs.chunks_exact(4);
    let mut out_groups = out[..msgs.len()].chunks_exact_mut(4);
    for (m, o) in (&mut msg_groups).zip(&mut out_groups) {
        hash4(m.try_into().unwrap(), o.try_into().unwrap());
    }
    // A lone message is cheaper scalar; 2 or 3 share one padded group,
    // hashed by recursing so `hash4` is only inlined into the loop above
    match (msg_groups.remainder(), out_groups.into_remainder()) {
        ([], _) => {}
        ([msg], [node]) => *node = truncated_hash_64(msg),
        (rest, nodes) => {
            let mut group = [[0u8; 64]; 4];
            let mut out = [[0u8; NODE_SIZE]; 4];
            group[..rest.len()].copy_from_slice(rest);
            hash_many(&group, &mut out);
            nodes.copy_from_slice(&out[..nodes.len()]);
        }
    }
}
