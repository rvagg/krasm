//! Filecoin CommP core for WASI.
//!
//! Adapted from Hugo Dias's `commp` revision
//! a37a5b2bb1f272d0862e3d59accd318c84ea00e6:
//! https://github.com/hugomrdias/commp/tree/a37a5b2bb1f272d0862e3d59accd318c84ea00e6/rs/commp
//!
//! Licensed MIT OR Apache-2.0. This retains the upstream four-lane wasm SIMD
//! SHA-256, batched FR32 processing, and O(log n) streaming Merkle tree.

use std::fmt;

#[cfg(all(target_arch = "wasm32", not(target_feature = "simd128")))]
compile_error!("build with -C target-feature=+simd128");

/// A write would exceed the largest representable CommP payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PayloadTooLarge;

impl fmt::Display for PayloadTooLarge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("payload exceeds maximum CommP size")
    }
}

impl std::error::Error for PayloadTooLarge {}

/// Size of a merkle tree node (32 bytes)
const NODE_SIZE: usize = 32;

/// Input bytes per quad (127 bytes = 4 * 254 bits / 8)
const IN_BYTES_PER_QUAD: usize = 127;

/// Output bytes per quad after FR32 padding (128 bytes)
const OUT_BYTES_PER_QUAD: usize = 128;

/// Maximum tree levels
const MAX_LEVEL: usize = 64;

/// Largest payload accepted, in bytes: 127 * 2^47 (~15.9 PiB)
///
/// data-segment allows up to tree height 255, far beyond `u64`. This is the
/// largest payload for which every derived size (padding, piece size) stays
/// below 2^53, so the digest fields are exact as JS numbers.
pub const MAX_PAYLOAD_SIZE: u64 = (IN_BYTES_PER_QUAD as u64) << 47;

/// Whether writing `len` more bytes would exceed `MAX_PAYLOAD_SIZE`
#[inline]
fn exceeds_max_payload(bytes_written: u64, len: usize) -> bool {
    bytes_written.saturating_add(len as u64) > MAX_PAYLOAD_SIZE
}

/// Quads FR32-padded and hashed together before reducing to one subtree
const BATCH_QUADS: usize = 128;

/// Leaves per batch (2 per quad); must be a power of two
const BATCH_LEAVES: usize = BATCH_QUADS * 2;

/// Stack slot of a full batch's subtree root (`BATCH_LEAVES = 2^BATCH_LEVEL`)
const BATCH_LEVEL: usize = BATCH_LEAVES.trailing_zeros() as usize;

/// Pre-computed zero commitment nodes for each level.
fn get_zero_comm(level: usize) -> [u8; NODE_SIZE] {
    static ZERO_COMMS: std::sync::OnceLock<[[u8; NODE_SIZE]; MAX_LEVEL]> = std::sync::OnceLock::new();
    ZERO_COMMS.get_or_init(|| {
        let mut comms = [[0u8; NODE_SIZE]; MAX_LEVEL];
        let mut concat = [0u8; NODE_SIZE * 2];
        for i in 1..MAX_LEVEL {
            concat[..NODE_SIZE].copy_from_slice(&comms[i - 1]);
            concat[NODE_SIZE..].copy_from_slice(&comms[i - 1]);
            comms[i] = truncated_hash_64(&concat);
        }
        comms
    })[level]
}

/// SHA-256 initial state
const IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Padding block of a 64-byte message: `0x80`, zeros, then the bit length 512
const PAD_BLOCK: [u8; 64] = {
    let mut block = [0u8; 64];
    block[0] = 0x80;
    block[62] = 0x02;
    block
};

/// Compute truncated SHA256 hash for 64-byte input
///
/// Calls the raw compression function directly: the input is always one
/// block, so `Digest`'s buffering and padding logic is dead weight.
#[inline(never)]
fn truncated_hash_64(data: &[u8; 64]) -> [u8; NODE_SIZE] {
    let mut state = IV;
    sha2::block_api::compress256(&mut state, &[*data, PAD_BLOCK]);
    let mut result = [0u8; NODE_SIZE];
    for (bytes, word) in result.as_chunks_mut::<4>().0.iter_mut().zip(state) {
        bytes.copy_from_slice(&word.to_be_bytes());
    }
    result[NODE_SIZE - 1] &= 0b00111111;
    result
}

/// Hash many 64-byte messages into truncated nodes (portable fallback)
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
fn hash_many(msgs: &[[u8; 64]], out: &mut [[u8; NODE_SIZE]]) {
    for (msg, node) in msgs.iter().zip(out.iter_mut()) {
        *node = truncated_hash_64(msg);
    }
}

/// 4-lane SHA-256 for wasm32 SIMD.
///
/// sha2's `wasm32_simd128` backend only vectorizes the message schedule; the
/// rounds of each compression stay scalar. Every leaf and every node on a tree
/// level is independent, so here 4 messages are hashed at once, one per `u32`
/// lane of a `v128`.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use simd::hash_many;

/// View a level of nodes as the 64-byte messages formed by adjacent pairs
#[inline(always)]
fn as_pairs(nodes: &[[u8; NODE_SIZE]]) -> &[[u8; 64]] {
    debug_assert!(nodes.len().is_multiple_of(2));
    // SAFETY: [[u8; 32]; 2] and [u8; 64] have the same size and alignment (1)
    unsafe { core::slice::from_raw_parts(nodes.as_ptr() as *const [u8; 64], nodes.len() / 2) }
}

/// Reduce `len` nodes at tree `height` by `steps` levels, pairing an odd
/// trailing node with a zero commitment. The result is left at the start of
/// `nodes`; returns the remaining count.
#[inline(never)]
fn reduce(nodes: &mut [[u8; NODE_SIZE]], mut len: usize, mut height: usize, steps: usize) -> usize {
    let mut next = [[0u8; NODE_SIZE]; BATCH_LEAVES / 2];
    for _ in 0..steps {
        if len % 2 == 1 {
            nodes[len] = get_zero_comm(height);
            len += 1;
        }
        len /= 2;
        hash_many(as_pairs(&nodes[..len * 2]), &mut next[..len]);
        nodes[..len].copy_from_slice(&next[..len]);
        height += 1;
    }
    len
}

/// FR32 pad a 127-byte quad into a 128-byte output buffer
///
/// FR32 inserts 2 zero bits every 254 bits (31.75 bytes).
#[inline(always)]
fn fr32_pad(source: &[u8], output: &mut [u8; OUT_BYTES_PER_QUAD]) {
    // First Fr element (bytes 0-31): copy directly, clear top 2 bits
    output[..32].copy_from_slice(&source[..32]);
    output[31] &= 0b00111111;

    // Second Fr element (bytes 32-63): shift left by 2 bits
    for i in 32..64 {
        output[i] = (source[i] << 2) | (source[i - 1] >> 6);
    }
    output[63] &= 0b00111111;

    // Third Fr element (bytes 64-95): shift left by 4 bits
    for i in 64..96 {
        output[i] = (source[i] << 4) | (source[i - 1] >> 4);
    }
    output[95] &= 0b00111111;

    // Fourth Fr element (bytes 96-127): shift left by 6 bits
    for i in 96..127 {
        output[i] = (source[i] << 6) | (source[i - 1] >> 2);
    }
    // Last byte: just the top 6 bits of source[126] shifted right
    output[127] = source[126] >> 2;
}

/// Compute parent node from two children using a pre-allocated buffer
#[inline(always)]
fn compute_node_into(left: &[u8; NODE_SIZE], right: &[u8; NODE_SIZE], concat: &mut [u8; 64]) -> [u8; NODE_SIZE] {
    concat[..NODE_SIZE].copy_from_slice(left);
    concat[NODE_SIZE..].copy_from_slice(right);
    truncated_hash_64(concat)
}

/// Pending subtree roots, one per tree level, updated like a binary counter
///
/// Bit `k` of `count` is set when `nodes[k]` holds the root of a complete
/// subtree of `2^k` leaves that is still waiting for its right sibling. Memory
/// is O(log n) regardless of input size.
#[derive(Clone, Copy)]
struct Stack {
    /// `nodes[k]` is a pending node at tree level `k + 1`
    nodes: [[u8; NODE_SIZE]; MAX_LEVEL],
    /// Number of leaves pushed so far
    count: u64,
}

impl Stack {
    fn new() -> Self {
        Stack {
            // A memset: the `[[0; 32]; 64]` literal is unrolled into 128 SIMD
            // stores at every call site, ~3KB each
            // SAFETY: all-zero bytes are a valid [[u8; 32]; 64]
            nodes: unsafe { core::mem::zeroed() },
            count: 0,
        }
    }

    /// Push the root of a complete subtree of `2^level` leaves, merging
    /// completed subtrees upward. `count` must be a multiple of `2^level`.
    #[inline]
    fn push_at(&mut self, root: [u8; NODE_SIZE], mut level: usize, concat: &mut [u8; 64]) {
        debug_assert!(self.count.is_multiple_of(1 << level));
        self.count += 1 << level;
        let mut node = root;
        while self.count >> level & 1 == 0 {
            node = compute_node_into(&self.nodes[level], &node, concat);
            level += 1;
        }
        self.nodes[level] = node;
    }

    /// Fold pending nodes into the root, padding with zero commitments
    ///
    /// Our leaves hash 64-byte halves of a quad, so they are level 1 of the
    /// reference tree (level 0 is the raw 32-byte FR32 chunks). Requires at
    /// least two leaves. Returns (height, root).
    fn fold(&self) -> (u8, [u8; NODE_SIZE]) {
        let n = self.count;
        let top = (u64::BITS - 1 - n.leading_zeros()) as usize;
        if n.is_power_of_two() {
            return ((top + 1) as u8, self.nodes[top]);
        }

        // Carry the right-most partial subtree up to the level of `top`,
        // pairing it with a pending left sibling or a zero commitment
        let mut concat = [0u8; 64];
        let lowest = n.trailing_zeros() as usize;
        let mut acc = compute_node_into(&self.nodes[lowest], &get_zero_comm(lowest + 1), &mut concat);
        for level in lowest + 1..top {
            acc = if n >> level & 1 == 1 {
                compute_node_into(&self.nodes[level], &acc, &mut concat)
            } else {
                compute_node_into(&acc, &get_zero_comm(level + 1), &mut concat)
            };
        }

        ((top + 2) as u8, compute_node_into(&self.nodes[top], &acc, &mut concat))
    }
}

/// View FR32-padded quads as their 64-byte halves (one message per leaf)
#[inline(always)]
fn as_messages(quads: &[[u8; OUT_BYTES_PER_QUAD]]) -> &[[u8; 64]] {
    // SAFETY: [u8; 128] and [[u8; 64]; 2] have the same size and alignment (1)
    unsafe { core::slice::from_raw_parts(quads.as_ptr() as *const [u8; 64], quads.len() * 2) }
}

/// Streaming CommP hasher with O(log n) memory.
pub struct CommPHasher {
    /// Buffer for accumulating partial quads
    buffer: [u8; IN_BYTES_PER_QUAD],
    /// Current offset into buffer
    offset: usize,
    /// Total bytes written
    bytes_written: u64,
    /// FR32-padded quads waiting to be hashed as one batch. Allocated on the
    /// first full quad, so small inputs and new hashers stay cheap.
    batch: Vec<[u8; OUT_BYTES_PER_QUAD]>,
    /// Pending tree nodes; full batches enter at `BATCH_LEVEL`
    stack: Stack,
    /// Reusable buffer for hashing node pairs
    concat_buffer: [u8; 64],
}

impl CommPHasher {
    /// Create a new hasher
    pub fn new() -> Self {
        CommPHasher {
            buffer: [0u8; IN_BYTES_PER_QUAD],
            offset: 0,
            bytes_written: 0,
            batch: Vec::new(),
            stack: Stack::new(),
            concat_buffer: [0u8; 64],
        }
    }

    /// FR32 pad a full quad into the batch, flushing it when full
    #[inline]
    fn push_quad(&mut self, quad: &[u8]) {
        if self.batch.capacity() == 0 {
            self.batch.reserve_exact(BATCH_QUADS);
        }
        let mut padded = [0u8; OUT_BYTES_PER_QUAD];
        fr32_pad(quad, &mut padded);
        self.batch.push(padded);
        if self.batch.len() == BATCH_QUADS {
            self.flush_batch();
        }
    }

    /// Hash a full batch into one subtree root and push it onto the stack
    fn flush_batch(&mut self) {
        let mut nodes = [[0u8; NODE_SIZE]; BATCH_LEAVES];
        hash_many(as_messages(&self.batch), &mut nodes);
        reduce(&mut nodes, BATCH_LEAVES, 1, BATCH_LEVEL);
        self.stack.push_at(nodes[0], BATCH_LEVEL, &mut self.concat_buffer);
        self.batch.clear();
    }

    /// Write bytes into the hasher
    ///
    /// Returns `PayloadTooLarge` without changing the hasher if the total
    /// would exceed `MAX_PAYLOAD_SIZE`.
    pub fn write(&mut self, bytes: &[u8]) -> Result<(), PayloadTooLarge> {
        let len = bytes.len();
        if exceeds_max_payload(self.bytes_written, len) {
            return Err(PayloadTooLarge);
        }
        if len == 0 {
            return Ok(());
        }

        self.bytes_written += len as u64;

        // Fast path: if we can't complete a quad, just buffer
        if self.offset + len < IN_BYTES_PER_QUAD {
            self.buffer[self.offset..self.offset + len].copy_from_slice(bytes);
            self.offset += len;
            return Ok(());
        }

        let mut read_pos = 0;

        // Complete the buffered quad if we have partial data
        if self.offset > 0 {
            let bytes_needed = IN_BYTES_PER_QUAD - self.offset;
            self.buffer[self.offset..].copy_from_slice(&bytes[..bytes_needed]);
            read_pos = bytes_needed;

            let buffer = self.buffer;
            self.push_quad(&buffer);
            self.offset = 0;
        }

        // Process full quads directly from input
        while read_pos + IN_BYTES_PER_QUAD <= len {
            self.push_quad(&bytes[read_pos..read_pos + IN_BYTES_PER_QUAD]);
            read_pos += IN_BYTES_PER_QUAD;
        }

        // Buffer remaining bytes
        let remaining = len - read_pos;
        if remaining > 0 {
            self.buffer[..remaining].copy_from_slice(&bytes[read_pos..]);
            self.offset = remaining;
        }
        Ok(())
    }

    /// Build final tree and return (height, root)
    ///
    /// Works on copies, so it does not modify the hasher and can be called
    /// repeatedly and interleaved with `write()`.
    fn build(&self) -> (u8, [u8; NODE_SIZE]) {
        // Leaves of the partial batch, plus the buffered partial quad (or an
        // all-zero quad for empty input). The batch is never full at rest, so
        // everything fits in one batch.
        let mut nodes = [[0u8; NODE_SIZE]; BATCH_LEAVES];
        let mut len = self.batch.len() * 2;
        hash_many(as_messages(&self.batch), &mut nodes[..len]);
        if self.offset > 0 || self.bytes_written == 0 {
            let mut buffer = self.buffer;
            buffer[self.offset..].fill(0);
            let mut tail = [[0u8; OUT_BYTES_PER_QUAD]; 1];
            fr32_pad(&buffer, &mut tail[0]);
            hash_many(as_messages(&tail), &mut nodes[len..len + 2]);
            len += 2;
        }

        // Small input: the whole tree is this partial batch. Our leaves hash
        // 64-byte halves of a quad, so they are level 1 of the reference tree
        // (level 0 is the raw 32-byte FR32 chunks).
        if self.stack.count == 0 {
            let steps = len.next_power_of_two().trailing_zeros() as usize;
            reduce(&mut nodes, len, 1, steps);
            return ((steps + 1) as u8, nodes[0]);
        }

        // Pad the partial batch with zero leaves to a full batch. Since the
        // stack already holds at least one batch, this doesn't change the
        // padded tree size, so folding gives the same root and height.
        let mut stack = self.stack;
        if len > 0 {
            let mut concat = [0u8; 64];
            reduce(&mut nodes, len, 1, BATCH_LEVEL);
            stack.push_at(nodes[0], BATCH_LEVEL, &mut concat);
        }
        stack.fold()
    }

    /// Return the 32-byte Merkle root without changing the hasher.
    pub fn root(&self) -> [u8; NODE_SIZE] {
        let (_, root) = self.build();
        root
    }

    /// Get the tree height
    pub fn height(&self) -> u8 {
        let (height, _) = self.build();
        height
    }

    /// Get bytes written count
    pub fn count(&self) -> u64 {
        self.bytes_written
    }

    /// Reset the hasher for reuse
    pub fn reset(&mut self) {
        self.buffer.fill(0);
        self.offset = 0;
        self.bytes_written = 0;
        self.batch.clear();
        self.stack.count = 0;
    }
}

impl Default for CommPHasher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i * 7 + i / 251) as u8).collect()
    }

    fn expected(hex: &str) -> [u8; NODE_SIZE] {
        let mut root = [0; NODE_SIZE];
        for (i, byte) in root.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        root
    }

    fn root_from_chunks(data: &[u8], chunk_size: usize) -> [u8; NODE_SIZE] {
        let mut hasher = CommPHasher::new();
        for chunk in data.chunks(chunk_size) {
            hasher.write(chunk).unwrap();
        }
        hasher.root()
    }

    #[test]
    fn fixed_reference_roots() {
        let vectors = [
            (0, "3731bb99ac689f66eef5973e4a94da188f4ddcae580724fc6f3fd60dfd488333"),
            (126, "e7174a9e17b4eab0299c33fb7a5376d9a780ab93cc4873777a7452ed32ecaf14"),
            (127, "b3c22ed2e440d87a1dfcf2c7c21bc887742acfbba9d00954a0132b99445dc523"),
            (128, "b2bba11fa929911e8a2f530c1a3838d65276f8f6d15f70ca298770f5dca4ca2b"),
            (
                BATCH_QUADS * IN_BYTES_PER_QUAD - 1,
                "1a343fdc6331d54cd9b38e3031ee2a0abb99c491c957d5a31246e174475bcf0c",
            ),
            (
                BATCH_QUADS * IN_BYTES_PER_QUAD,
                "924382943c160fd556a7be4f523d58b2c52e6c15defd5595808fc201c2ab6129",
            ),
            (
                BATCH_QUADS * IN_BYTES_PER_QUAD + 1,
                "7aed196494e2870b40b7edca2645a64acccabd5d6e31f697c8bdf784ae73ca24",
            ),
        ];
        for (len, root) in vectors {
            assert_eq!(
                root_from_chunks(&sample(len), len.max(1)),
                expected(root),
                "{len} bytes"
            );
        }

        let legacy_data = [0x42; 32_768];
        assert_eq!(
            root_from_chunks(&legacy_data, 127),
            expected("cb042c314b73c00e5e1743099e0abb01937bd3e1b8f37d5b4b76f44c6ee9a30c")
        );
    }

    #[test]
    fn chunk_boundaries_preserve_batched_root() {
        let data = sample(BATCH_QUADS * IN_BYTES_PER_QUAD + 1);
        let root = expected("7aed196494e2870b40b7edca2645a64acccabd5d6e31f697c8bdf784ae73ca24");
        for chunk_size in [1, 126, 127, 128, 8192] {
            assert_eq!(root_from_chunks(&data, chunk_size), root, "{chunk_size}-byte chunks");
        }
    }

    #[test]
    fn root_is_repeatable_and_reset_reuses_hasher() {
        let data = sample(BATCH_QUADS * IN_BYTES_PER_QUAD + 1);
        let mut hasher = CommPHasher::new();
        hasher.write(&data).unwrap();
        let root = hasher.root();
        assert_eq!(hasher.root(), root);
        hasher.reset();
        for chunk in data.chunks(127) {
            hasher.write(chunk).unwrap();
        }
        assert_eq!(hasher.root(), root);
    }

    #[test]
    fn oversized_write_preserves_state() {
        let mut hasher = CommPHasher::new();
        hasher.write(&[0x42; 127]).unwrap();
        let root = hasher.root();
        hasher.bytes_written = MAX_PAYLOAD_SIZE;
        assert_eq!(hasher.write(&[1]), Err(PayloadTooLarge));
        assert_eq!(hasher.count(), MAX_PAYLOAD_SIZE);
        assert_eq!(hasher.root(), root);
    }
}
