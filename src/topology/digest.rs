// SPDX-License-Identifier: MIT OR Apache-2.0

//! Deterministic topology digest for [`SynapticGraph`].
//!
//! Replay, checkpoint provenance, and cross-runtime comparisons need a stable
//! identifier of the *logical* graph that does not depend on CSR insertion
//! order, map iteration, pointer layout, host endianness, or JSON formatting.
//!
//! # Schema v1
//!
//! The printable value is:
//!
//! ```text
//! synaptic-wiring.topology.digest.v1:sha256:<64 lowercase hex chars>
//! ```
//!
//! The SHA-256 preimage is the concatenation of:
//!
//! 1. Domain separator [`TOPOLOGY_DIGEST_DOMAIN`] as UTF-8, then a NUL byte
//!    so the separator cannot prefix-collide with the payload.
//! 2. Schema version [`TOPOLOGY_DIGEST_SCHEMA_VERSION`] as little-endian `u16`.
//! 3. Neuron count as little-endian `u64`.
//! 4. Edge count as little-endian `u64`.
//! 5. One 23-byte record per logical edge, sorted in canonical order
//!    `(source, target, delay, polarity_tag, weight_bits)`:
//!    - `source`: little-endian `u64` (CSR row index)
//!    - `target`: little-endian `u64` (zero-extended [`crate::NeuronId`])
//!    - `weight_bits`: little-endian `u32` of [`f32::to_bits`] (IEEE 754
//!      bit pattern of the **signed** CSR weight)
//!    - `delay`: little-endian [`crate::DelayTicks`]
//!    - `polarity_tag`: `0` = [`Polarity::Excitatory`], `1` =
//!      [`Polarity::Inhibitory`]
//!
//! Bumping the schema version or domain separator produces a new printable
//! prefix; v1 values stay comparable forever.
//!
//! # Float canonicalization
//!
//! Schema v1 hashes the IEEE bit pattern. Construction and serde already
//! reject NaN and infinities, so a valid graph never contains them.
//! IEEE signed zero (`+0.0` vs `-0.0`) is **not** collapsed: the two bit
//! patterns digest differently, which is tested.

use std::fmt;
use std::str::FromStr;

use serde::de::{Deserializer, Error as DeError};
use serde::{Deserialize, Serialize, Serializer};
use sha2::{Digest, Sha256};

use super::graph::SynapticGraph;
use crate::types::Polarity;

/// Schema version mixed into the v1 preimage and reported by
/// [`TopologyDigest::schema_version`].
pub const TOPOLOGY_DIGEST_SCHEMA_VERSION: u16 = 1;

/// Domain separator (UTF-8) that prefixes the SHA-256 preimage and the
/// printable manifest form. Changing this is a new digest family.
pub const TOPOLOGY_DIGEST_DOMAIN: &str = "synaptic-wiring.topology.digest.v1";

/// Hash algorithm name embedded in the printable manifest form.
pub const TOPOLOGY_DIGEST_ALGORITHM: &str = "sha256";

/// NUL terminator after the domain separator in the hash preimage.
const DOMAIN_TERMINATOR: u8 = 0;

/// Printable, versioned digest of a graph's logical topology.
///
/// The [`fmt::Display`] form is stable and safe to store in manifests:
/// `{domain}:{algorithm}:{hex}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TopologyDigest {
    schema_version: u16,
    hash: [u8; 32],
}

impl TopologyDigest {
    /// Compute the schema-v1 digest of `graph`.
    ///
    /// Edges are collected from the CSR and sorted into canonical order, so
    /// two graphs built from the same logical synapses in different insertion
    /// orders compare equal. Runtime mesh state (tick, delay buffer) is not
    /// part of the digest.
    #[must_use]
    pub fn from_graph(graph: &SynapticGraph) -> Self {
        let edges = canonical_edges(graph);
        Self {
            schema_version: TOPOLOGY_DIGEST_SCHEMA_VERSION,
            hash: hash_v1(graph.neuron_count() as u64, &edges),
        }
    }

    /// Schema version that produced this digest.
    #[must_use]
    pub fn schema_version(&self) -> u16 {
        self.schema_version
    }

    /// Hash algorithm name (`sha256` for schema v1).
    #[must_use]
    pub fn algorithm(&self) -> &'static str {
        TOPOLOGY_DIGEST_ALGORITHM
    }

    /// Domain separator mixed into this digest family.
    #[must_use]
    pub fn domain(&self) -> &'static str {
        TOPOLOGY_DIGEST_DOMAIN
    }

    /// Raw 32-byte SHA-256 hash.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.hash
    }

    /// Lowercase hex encoding of the SHA-256 hash (64 characters).
    #[must_use]
    pub fn to_hex(&self) -> String {
        hex_encode(&self.hash)
    }
}

impl fmt::Display for TopologyDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{TOPOLOGY_DIGEST_DOMAIN}:{TOPOLOGY_DIGEST_ALGORITHM}:{}",
            self.to_hex()
        )
    }
}

impl FromStr for TopologyDigest {
    type Err = TopologyDigestParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (domain, algorithm, hex) = split_manifest(s)?;
        if domain != TOPOLOGY_DIGEST_DOMAIN {
            return Err(TopologyDigestParseError(format!(
                "unsupported digest domain {domain:?}"
            )));
        }
        if algorithm != TOPOLOGY_DIGEST_ALGORITHM {
            return Err(TopologyDigestParseError(format!(
                "unsupported digest algorithm {algorithm:?}"
            )));
        }
        Ok(Self {
            schema_version: TOPOLOGY_DIGEST_SCHEMA_VERSION,
            hash: hex_decode_sha256(hex)?,
        })
    }
}

impl Serialize for TopologyDigest {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for TopologyDigest {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(DeError::custom)
    }
}

/// Error from parsing a printable [`TopologyDigest`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyDigestParseError(String);

impl TopologyDigestParseError {
    fn format() -> Self {
        Self(format!(
            "expected {TOPOLOGY_DIGEST_DOMAIN}:{TOPOLOGY_DIGEST_ALGORITHM}:<64 hex chars>"
        ))
    }
}

impl fmt::Display for TopologyDigestParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TopologyDigestParseError {}

impl SynapticGraph {
    /// Deterministic, versioned digest of this graph's logical topology.
    ///
    /// See [`TopologyDigest`] for the schema, domain separator, and float
    /// bit-pattern rules. Equivalent graphs built through different insertion
    /// orders produce the same value.
    ///
    /// ```
    /// use synaptic_wiring::topology::SynapticGraph;
    /// use synaptic_wiring::types::{Polarity, SynapseDescriptor};
    ///
    /// let a = [
    ///     SynapseDescriptor {
    ///         source: 0,
    ///         target: 1,
    ///         weight: 0.5,
    ///         delay: 2,
    ///         polarity: Polarity::Excitatory,
    ///     },
    ///     SynapseDescriptor {
    ///         source: 1,
    ///         target: 0,
    ///         weight: 0.25,
    ///         delay: 1,
    ///         polarity: Polarity::Inhibitory,
    ///     },
    /// ];
    /// let mut b = a;
    /// b.swap(0, 1);
    ///
    /// let ga = SynapticGraph::from_descriptors(2, &a).unwrap();
    /// let gb = SynapticGraph::from_descriptors(2, &b).unwrap();
    /// assert_eq!(ga.topology_digest(), gb.topology_digest());
    /// ```
    #[must_use]
    pub fn topology_digest(&self) -> TopologyDigest {
        TopologyDigest::from_graph(self)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CanonicalEdge {
    source: u64,
    target: u64,
    delay: u16,
    polarity: u8,
    weight_bits: u32,
}

fn canonical_edges(graph: &SynapticGraph) -> Vec<CanonicalEdge> {
    let mut edges = Vec::with_capacity(graph.synapse_count());
    for src in 0..graph.neuron_count() {
        let source = src as u64;
        for (target, weight, delay, polarity) in graph.outgoing(src) {
            debug_assert!(
                weight.is_finite(),
                "valid graphs reject non-finite weights before digest"
            );
            edges.push(CanonicalEdge {
                source,
                target: u64::from(target),
                delay,
                polarity: polarity_tag(polarity),
                weight_bits: weight.to_bits(),
            });
        }
    }
    edges.sort_unstable();
    edges
}

fn hash_v1(neuron_count: u64, edges: &[CanonicalEdge]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(TOPOLOGY_DIGEST_DOMAIN.as_bytes());
    hasher.update([DOMAIN_TERMINATOR]);
    hasher.update(TOPOLOGY_DIGEST_SCHEMA_VERSION.to_le_bytes());
    hasher.update(neuron_count.to_le_bytes());
    hasher.update((edges.len() as u64).to_le_bytes());
    for edge in edges {
        hasher.update(edge.source.to_le_bytes());
        hasher.update(edge.target.to_le_bytes());
        hasher.update(edge.weight_bits.to_le_bytes());
        hasher.update(edge.delay.to_le_bytes());
        hasher.update([edge.polarity]);
    }
    hasher.finalize().into()
}

fn polarity_tag(polarity: Polarity) -> u8 {
    match polarity {
        Polarity::Excitatory => 0,
        Polarity::Inhibitory => 1,
    }
}

fn split_manifest(s: &str) -> Result<(&str, &str, &str), TopologyDigestParseError> {
    let (domain, rest) = s
        .split_once(':')
        .ok_or_else(TopologyDigestParseError::format)?;
    let (algorithm, hex) = rest
        .split_once(':')
        .ok_or_else(TopologyDigestParseError::format)?;
    Ok((domain, algorithm, hex))
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

fn hex_decode_sha256(hex: &str) -> Result<[u8; 32], TopologyDigestParseError> {
    if hex.len() != 64 {
        return Err(TopologyDigestParseError(format!(
            "digest hex must be 64 characters, got {}",
            hex.len()
        )));
    }
    let mut out = [0u8; 32];
    let bytes = hex.as_bytes();
    for (i, slot) in out.iter_mut().enumerate() {
        let hi = hex_nibble(bytes[i * 2])?;
        let lo = hex_nibble(bytes[i * 2 + 1])?;
        *slot = (hi << 4) | lo;
    }
    Ok(out)
}

fn hex_nibble(b: u8) -> Result<u8, TopologyDigestParseError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(TopologyDigestParseError(format!(
            "invalid hex digit {:?}",
            b as char
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SynapseDescriptor;

    fn desc(
        source: u32,
        target: u32,
        weight: f32,
        delay: u16,
        polarity: Polarity,
    ) -> SynapseDescriptor {
        SynapseDescriptor {
            source,
            target,
            weight,
            delay,
            polarity,
        }
    }

    /// Golden v1 digest for a 0-neuron empty graph.
    const GOLDEN_EMPTY_0: &str = "synaptic-wiring.topology.digest.v1:sha256:14f893289199426a03b83570fda63b60872dedfe03c0b52a46a2061445d7a608";

    /// Golden v1 digest for a 3-neuron empty graph (neuron count is in the preimage).
    const GOLDEN_EMPTY_3: &str = "synaptic-wiring.topology.digest.v1:sha256:a2c1014ab2a24a342e01878c5dbe69972e8ab4152cb36d2faf88ec210d0b6203";

    /// Golden v1 digest for the three-edge fixture used across digest tests.
    const GOLDEN_SMALL: &str = "synaptic-wiring.topology.digest.v1:sha256:be1786c690cda6d02d9c6fa06d1844642398ecebab3d19ccec981062711cd36f";

    fn small_fixture(order: &[usize]) -> Vec<SynapseDescriptor> {
        let base = [
            desc(0, 1, 0.9, 3, Polarity::Excitatory),
            desc(0, 2, 0.15, 1, Polarity::Inhibitory),
            desc(1, 0, 0.5, 2, Polarity::Excitatory),
        ];
        order.iter().map(|&i| base[i]).collect()
    }

    #[test]
    fn topology_digest_golden_empty_graphs() {
        assert_eq!(
            SynapticGraph::new(0).topology_digest().to_string(),
            GOLDEN_EMPTY_0
        );
        assert_eq!(
            SynapticGraph::new(3).topology_digest().to_string(),
            GOLDEN_EMPTY_3
        );
        assert_ne!(GOLDEN_EMPTY_0, GOLDEN_EMPTY_3);
    }

    #[test]
    fn topology_digest_golden_small_fixture() {
        let graph = SynapticGraph::from_descriptors(3, &small_fixture(&[0, 1, 2])).unwrap();
        assert_eq!(graph.topology_digest().to_string(), GOLDEN_SMALL);
    }

    #[test]
    fn topology_digest_insertion_order_permutations_agree() {
        let orders = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let expected = SynapticGraph::from_descriptors(3, &small_fixture(&orders[0]))
            .unwrap()
            .topology_digest();
        for order in orders {
            let graph = SynapticGraph::from_descriptors(3, &small_fixture(&order)).unwrap();
            assert_eq!(
                graph.topology_digest(),
                expected,
                "insertion order {order:?} must not change the digest"
            );
            assert_eq!(graph.topology_digest().to_string(), GOLDEN_SMALL);
        }
    }

    fn digest_of(descriptors: &[SynapseDescriptor]) -> TopologyDigest {
        SynapticGraph::from_descriptors(3, descriptors)
            .unwrap()
            .topology_digest()
    }

    #[test]
    fn topology_digest_changes_with_endpoint_delay_polarity_or_weight() {
        let base = digest_of(&small_fixture(&[0, 1, 2]));
        let mut endpoint = small_fixture(&[0, 1, 2]);
        endpoint[0].target = 0;
        let mut delay = small_fixture(&[0, 1, 2]);
        delay[0].delay = 9;
        let mut polarity = small_fixture(&[0, 1, 2]);
        polarity[2].polarity = Polarity::Inhibitory;
        let mut weight = small_fixture(&[0, 1, 2]);
        weight[1].weight = 0.16;

        assert_ne!(digest_of(&endpoint), base);
        assert_ne!(digest_of(&delay), base);
        assert_ne!(digest_of(&polarity), base);
        assert_ne!(digest_of(&weight), base);
    }

    #[test]
    fn topology_digest_signed_zero_is_hashed_by_bit_pattern() {
        let plus = SynapticGraph::from_descriptors(2, &[desc(0, 1, 0.0, 0, Polarity::Excitatory)])
            .unwrap()
            .topology_digest();
        let minus =
            SynapticGraph::from_descriptors(2, &[desc(0, 1, -0.0, 0, Polarity::Excitatory)])
                .unwrap()
                .topology_digest();
        assert_ne!(
            plus, minus,
            "+0.0 and -0.0 are distinct IEEE bit patterns and must digest differently"
        );
        assert_eq!(
            plus.to_string(),
            "synaptic-wiring.topology.digest.v1:sha256:72c7a0686739be822c0dedf46f71dee35a2c154b3debc37019afe9ece19ae9d0"
        );
        assert_eq!(
            minus.to_string(),
            "synaptic-wiring.topology.digest.v1:sha256:1b9b1789b7aed14363f73ed4f3c5708fc05180166ee7b9f620212f57d2a5812f"
        );
    }

    #[test]
    fn topology_digest_nan_cannot_appear_in_a_valid_graph() {
        assert!(
            SynapticGraph::from_descriptors(2, &[desc(0, 1, f32::NAN, 0, Polarity::Excitatory)])
                .is_err()
        );
        assert!(
            SynapticGraph::from_descriptors(
                2,
                &[desc(0, 1, f32::INFINITY, 0, Polarity::Excitatory)]
            )
            .is_err()
        );
    }

    #[test]
    fn topology_digest_is_printable_and_round_trips() {
        let digest = SynapticGraph::from_descriptors(3, &small_fixture(&[2, 0, 1]))
            .unwrap()
            .topology_digest();
        let printed = digest.to_string();
        assert!(printed.starts_with(TOPOLOGY_DIGEST_DOMAIN));
        assert!(printed.contains(TOPOLOGY_DIGEST_ALGORITHM));
        assert_eq!(digest.schema_version(), TOPOLOGY_DIGEST_SCHEMA_VERSION);
        assert_eq!(printed.parse::<TopologyDigest>().unwrap(), digest);

        let json = serde_json::to_string(&digest).unwrap();
        let restored: TopologyDigest = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, digest);
    }

    #[test]
    fn topology_digest_survives_graph_json_roundtrip() {
        let graph = SynapticGraph::from_descriptors(3, &small_fixture(&[1, 2, 0])).unwrap();
        let json = serde_json::to_string(&graph).unwrap();
        let restored: SynapticGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(graph.topology_digest(), restored.topology_digest());
        assert_eq!(restored.topology_digest().to_string(), GOLDEN_SMALL);
    }
}
