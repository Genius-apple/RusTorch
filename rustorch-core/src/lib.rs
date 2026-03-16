#![allow(clippy::collapsible_if)]
#![allow(clippy::empty_line_after_outer_attr)]
#![allow(clippy::get_first)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::map_entry)]
#![allow(clippy::match_like_matches_macro)]
#![allow(clippy::missing_const_for_thread_local)]
#![allow(clippy::needless_return)]
#![allow(clippy::new_without_default)]
#![allow(clippy::redundant_pattern_matching)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::unnecessary_map_or)]
#![allow(clippy::unwrap_or_default)]

pub mod autograd;
pub mod backend;
pub mod broadcast;
pub mod graph;
pub mod jit;
pub mod ops;
pub mod optimizer;
pub mod storage;
pub mod tensor;

pub use backend::wgpu;
pub use storage::Storage;
pub use tensor::Tensor;
