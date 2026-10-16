//! A compact binary file for tetrahedral meshes made offline (`tools/meshes`).
//!
//! Little-endian layout:
//!
//! | bytes | content |
//! |---|---|
//! | 8 | magic `EDMESH01` |
//! | 4 × u32 | point count, tet count, cell count, flags (bit 0: 16-bit indices) |
//! | 3 f32 per point | coordinates, µm |
//! | 4 u16 or u32 per tet | vertex indices |
//! | u16 per tet | subdomain tag (0 = extracellular, `1..=cells` = cell) |
//!
//! The file length must match the header exactly. Coordinates are f32, so a round trip
//! through a file rounds f64 points to f32.

use std::fmt;

use crate::mesh::{Mesh, MeshError};

pub const MAGIC: &[u8; 8] = b"EDMESH01";
const HEADER: usize = 8 + 16;
const SHORT_INDICES: u32 = 1;

/// Why a mesh file could not be read.
#[derive(Debug, Clone, PartialEq)]
pub enum MeshFileError {
    /// Wrong magic bytes or unknown flags.
    NotAMeshFile,
    /// More cells than a u16 tag can name.
    TooManyCells(usize),
    /// The file is shorter or longer than its header says.
    WrongLength { expected: usize, actual: usize },
    /// A coordinate is not finite.
    BadPoint(usize),
    /// The tets do not make a valid mesh.
    Mesh(MeshError),
}

impl fmt::Display for MeshFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAMeshFile => write!(f, "not an EDMESH01 mesh file"),
            Self::TooManyCells(n) => write!(f, "mesh file has {n} cells, at most 65534 allowed"),
            Self::WrongLength { expected, actual } => {
                write!(f, "mesh file has {actual} bytes, expected {expected}")
            }
            Self::BadPoint(p) => write!(f, "point {p} is not finite"),
            Self::Mesh(e) => write!(f, "invalid mesh: {e}"),
        }
    }
}

impl std::error::Error for MeshFileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Mesh(e) => Some(e),
            _ => None,
        }
    }
}

/// The raw contents of a mesh file.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshData {
    pub points: Vec<[f64; 3]>,
    pub tets: Vec<[u32; 4]>,
    pub tet_tag: Vec<u16>,
    pub n_cells: usize,
}

impl MeshData {
    /// Parses a mesh file without building the mesh.
    pub fn parse(bytes: &[u8]) -> Result<Self, MeshFileError> {
        if bytes.len() < HEADER || &bytes[..8] != MAGIC {
            return Err(MeshFileError::NotAMeshFile);
        }
        let word = |i: usize| u32::from_le_bytes(bytes[8 + 4 * i..12 + 4 * i].try_into().unwrap());
        let (n_points, n_tets, n_cells, flags) = (
            word(0) as usize,
            word(1) as usize,
            word(2) as usize,
            word(3),
        );
        if flags & !SHORT_INDICES != 0 {
            return Err(MeshFileError::NotAMeshFile);
        }
        if n_cells >= u16::MAX as usize {
            return Err(MeshFileError::TooManyCells(n_cells));
        }
        let index_size = if flags & SHORT_INDICES != 0 { 2 } else { 4 };
        let expected = n_points
            .checked_mul(12)
            .and_then(|p| n_tets.checked_mul(4 * index_size + 2).map(|t| (p, t)))
            .and_then(|(p, t)| p.checked_add(t))
            .and_then(|b| b.checked_add(HEADER));
        if expected != Some(bytes.len()) {
            return Err(MeshFileError::WrongLength {
                expected: expected.unwrap_or(usize::MAX),
                actual: bytes.len(),
            });
        }
        let mut at = HEADER;
        let mut take = |n: usize| {
            let s = &bytes[at..at + n];
            at += n;
            s
        };
        let mut points = Vec::with_capacity(n_points);
        for p in 0..n_points {
            let c = take(12);
            let xyz =
                [0, 1, 2].map(|k| f32::from_le_bytes(c[4 * k..4 * k + 4].try_into().unwrap()));
            if !xyz.iter().all(|x| x.is_finite()) {
                return Err(MeshFileError::BadPoint(p));
            }
            points.push(xyz.map(f64::from));
        }
        let idx = take(n_tets * 4 * index_size);
        let tets = (0..n_tets)
            .map(|t| {
                [0, 1, 2, 3].map(|k| {
                    let i = index_size * (4 * t + k);
                    if index_size == 2 {
                        u32::from(u16::from_le_bytes([idx[i], idx[i + 1]]))
                    } else {
                        u32::from_le_bytes(idx[i..i + 4].try_into().unwrap())
                    }
                })
            })
            .collect();
        let tags = take(2 * n_tets);
        let tet_tag = (0..n_tets)
            .map(|t| u16::from_le_bytes([tags[2 * t], tags[2 * t + 1]]))
            .collect();
        Ok(Self {
            points,
            tets,
            tet_tag,
            n_cells,
        })
    }

    /// Writes the file, with 16-bit indices when every index fits.
    ///
    /// Panics if the data cannot be written as a file `parse` accepts (counts beyond u32,
    /// a tag per tet missing, too many cells, or a point beyond f32).
    pub fn to_bytes(&self) -> Vec<u8> {
        let count = |n: usize| u32::try_from(n).expect("count fits in u32");
        assert_eq!(self.tet_tag.len(), self.tets.len(), "one tag per tet");
        assert!(self.n_cells < u16::MAX as usize, "too many cells");
        let short = self
            .tets
            .iter()
            .flatten()
            .all(|&i| i <= u32::from(u16::MAX));
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        for w in [
            count(self.points.len()),
            count(self.tets.len()),
            count(self.n_cells),
            if short { SHORT_INDICES } else { 0 },
        ] {
            out.extend_from_slice(&w.to_le_bytes());
        }
        for p in &self.points {
            for &x in p {
                let x = x as f32;
                assert!(x.is_finite(), "point {p:?} is not a finite f32");
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        for t in &self.tets {
            for &i in t {
                if short {
                    out.extend_from_slice(&(i as u16).to_le_bytes()); // checked above
                } else {
                    out.extend_from_slice(&i.to_le_bytes());
                }
            }
        }
        for &tag in &self.tet_tag {
            out.extend_from_slice(&tag.to_le_bytes());
        }
        out
    }

    /// Builds the mesh.
    pub fn into_mesh(self) -> Result<Mesh, MeshFileError> {
        Mesh::from_tets(self.points, &self.tets, self.tet_tag, self.n_cells)
            .map_err(MeshFileError::Mesh)
    }
}

/// Reads a mesh file and builds the mesh.
pub fn read(bytes: &[u8]) -> Result<Mesh, MeshFileError> {
    MeshData::parse(bytes)?.into_mesh()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube_corner() -> MeshData {
        // Two tets sharing a face: an extracellular one and a cell.
        MeshData {
            points: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            tets: vec![[0, 1, 2, 3], [1, 2, 3, 4]],
            tet_tag: vec![0, 1],
            n_cells: 1,
        }
    }

    #[test]
    fn round_trips_with_short_and_long_indices() {
        let data = cube_corner();
        let bytes = data.to_bytes();
        assert_eq!(bytes.len(), HEADER + 5 * 12 + 2 * (8 + 2));
        assert_eq!(MeshData::parse(&bytes).unwrap(), data);
        let mesh = read(&bytes).unwrap();
        assert_eq!(mesh.membrane.len(), 3);
        assert_eq!(mesh.membrane_facets.len(), 1);

        // Hand-written 32-bit variant.
        let mut long = bytes[..HEADER].to_vec();
        long[20..24].copy_from_slice(&0u32.to_le_bytes()); // flags
        long.extend_from_slice(&bytes[HEADER..HEADER + 60]);
        for t in &data.tets {
            t.iter()
                .for_each(|i| long.extend_from_slice(&i.to_le_bytes()));
        }
        long.extend_from_slice(&bytes[bytes.len() - 4..]);
        assert_eq!(MeshData::parse(&long).unwrap(), data);
    }

    #[test]
    fn rejects_damaged_files() {
        let bytes = cube_corner().to_bytes();
        assert_eq!(
            MeshData::parse(b"nope").unwrap_err(),
            MeshFileError::NotAMeshFile
        );
        let mut b = bytes.clone();
        b[0] = b'X';
        assert_eq!(
            MeshData::parse(&b).unwrap_err(),
            MeshFileError::NotAMeshFile
        );
        assert!(matches!(
            MeshData::parse(&bytes[..bytes.len() - 1]).unwrap_err(),
            MeshFileError::WrongLength { .. }
        ));
        let mut b = bytes.clone();
        b[HEADER..HEADER + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert_eq!(MeshData::parse(&b).unwrap_err(), MeshFileError::BadPoint(0));
        let mut b = bytes.clone();
        b[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            MeshData::parse(&b).unwrap_err(),
            MeshFileError::WrongLength { .. }
        ));
        // Unknown flags, a trailing byte, too many cells.
        let mut b = bytes.clone();
        b[20..24].copy_from_slice(&3u32.to_le_bytes());
        assert_eq!(
            MeshData::parse(&b).unwrap_err(),
            MeshFileError::NotAMeshFile
        );
        let mut b = bytes.clone();
        b.push(0);
        assert!(matches!(
            MeshData::parse(&b).unwrap_err(),
            MeshFileError::WrongLength { .. }
        ));
        let mut b = bytes.clone();
        b[16..20].copy_from_slice(&u32::from(u16::MAX).to_le_bytes());
        assert_eq!(
            read(&b).unwrap_err(),
            MeshFileError::TooManyCells(u16::MAX as usize)
        );
        // Tag 2 names no cell.
        let mut b = bytes.clone();
        let n = b.len();
        b[n - 2..].copy_from_slice(&2u16.to_le_bytes());
        assert_eq!(
            read(&b).unwrap_err(),
            MeshFileError::Mesh(MeshError::BadTet(1))
        );
        // Every truncation is an error, never a panic.
        for len in 0..bytes.len() {
            assert!(read(&bytes[..len]).is_err());
        }
        // Index 9 does not exist.
        let mut b = bytes;
        b[HEADER + 60..HEADER + 62].copy_from_slice(&9u16.to_le_bytes());
        assert_eq!(
            read(&b).unwrap_err(),
            MeshFileError::Mesh(MeshError::BadTet(0))
        );
    }
}
